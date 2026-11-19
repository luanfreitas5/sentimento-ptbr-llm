"""Calibração das probabilidades e predição com abstenção (parte do estágio ``evaluate``).

Para cada modelo que expõe probabilidades:

1. mede Brier, ECE e log-verossimilhança no teste, **com e sem** *temperature scaling*
   (temperatura ajustada na VALIDAÇÃO, nunca no teste), com IC bootstrap pareado da
   diferença de Brier e da log-verossimilhança;
2. grava a tabela de confiabilidade (``n`` e IC por faixa) antes e depois;
3. traça a curva risco-cobertura (macro-F1 contra a fração respondida);
4. fixa na validação o limiar que atinge a cobertura-alvo e o aplica ao teste, medindo a
   macro-F1 nos tweets respondidos e nos abstidos;
5. exporta, para o melhor modelo com probabilidades, os tweets de baixa confiança a
   encaminhar à revisão humana (somente ``id``, rótulo predito e confiança; sem texto).

Modelos sem probabilidades (os LLMs) ficam de fora e isso é registrado no log.
A temperatura não altera o ``argmax``: acurácia e F1 geral não mudam, só a confiança.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from constants.defaults import (
    DEFAULT_BOOTSTRAP_ITERATIONS,
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_RANDOM_SEED,
)
from constants.labels import SENTIMENT_CLASSES
from evaluation.abstention import (
    DEFAULT_COVERAGES,
    calculate_risk_coverage_curve,
    evaluate_abstention_at_threshold,
    select_confidence_threshold,
)
from evaluation.calibration import (
    DEFAULT_N_CALIBRATION_BINS,
    apply_temperature_scaling,
    calculate_calibration_metrics,
    calculate_negative_log_likelihood,
    calculate_reliability_table,
    fit_temperature_scaling,
)
from evaluation.predictions import (
    CATEGORY_LLM,
    ModelCheckpoint,
    ModelPredictions,
    predict_with_checkpoint,
)

logger = logging.getLogger(__name__)

CALIBRATION_FILE_NAME = "calibracao_por_modelo.csv"
RELIABILITY_FILE_NAME = "calibracao_confiabilidade.csv"
RISK_COVERAGE_FILE_NAME = "risco_cobertura.csv"
ABSTENTION_FILE_NAME = "abstencao_por_modelo.csv"
HUMAN_REVIEW_FILE_NAME = "revisao_humana.csv"
DEFAULT_TARGET_COVERAGE = 0.8
VARIANT_RAW = "original"
VARIANT_CALIBRATED = "temperature_scaling"


@dataclass(frozen=True)
class CalibrationAbstentionResult:
    """Resultado da calibração e da abstenção.

    Attributes
    ----------
    tables : dict[str, pl.DataFrame]
        Tabelas indexadas pelo nome de arquivo (``*_FILE_NAME``); só as não vazias.
    summary : dict[str, Any]
        Resumo para ``reports/metrics/avaliacao.json``.
    """

    tables: dict[str, pl.DataFrame]
    summary: dict[str, Any]


def _bootstrap_mean_difference(
    differences: np.ndarray,
    *,
    n_bootstrap: int,
    confidence_level: float,
    random_seed: int,
) -> tuple[float, float]:
    """IC percentil da média de diferenças pareadas (depois menos antes)."""
    indices = np.random.default_rng(random_seed).integers(
        0, len(differences), size=(n_bootstrap, len(differences))
    )
    means = differences[indices].mean(axis=1)
    alpha = 1 - confidence_level
    return (
        float(np.percentile(means, 100 * alpha / 2)),
        float(np.percentile(means, 100 * (1 - alpha / 2))),
    )


def _evaluate_one_model(
    test_predictions: ModelPredictions,
    y_true_test: Sequence[str],
    validation_scores: np.ndarray,
    y_true_validation: Sequence[str],
    *,
    config: Mapping[str, Any],
    n_bootstrap: int,
    confidence_level: float,
    random_seed: int,
) -> dict[str, Any]:
    """Calibra e avalia a abstenção de um modelo; devolve as linhas de cada tabela."""
    name = test_predictions.name
    y_true = [y_true_test[index] for index in test_predictions.indices]
    raw_scores = np.asarray(test_predictions.y_score)
    n_bins = int(config.get("n_bins", DEFAULT_N_CALIBRATION_BINS))

    temperature = 1.0
    if config.get("temperature_scaling", True):
        temperature = fit_temperature_scaling(y_true_validation, validation_scores)
    calibrated_scores = apply_temperature_scaling(raw_scores, temperature)

    y_pred = test_predictions.y_pred
    variants = {VARIANT_RAW: raw_scores, VARIANT_CALIBRATED: calibrated_scores}
    metrics = {
        variant: calculate_calibration_metrics(y_true, scores, n_bins=n_bins)
        for variant, scores in variants.items()
    }
    nll = {
        variant: calculate_negative_log_likelihood(y_true, scores)
        for variant, scores in variants.items()
    }
    one_hot = (np.array(y_true)[:, None] == np.array(SENTIMENT_CLASSES)[None, :]).astype(float)
    brier = {
        variant: np.sum((one_hot - scores) ** 2, axis=1) for variant, scores in variants.items()
    }
    options = {
        "n_bootstrap": n_bootstrap,
        "confidence_level": confidence_level,
        "random_seed": random_seed,
    }
    brier_difference = brier[VARIANT_CALIBRATED] - brier[VARIANT_RAW]
    nll_difference = nll[VARIANT_CALIBRATED] - nll[VARIANT_RAW]
    brier_low, brier_high = _bootstrap_mean_difference(brier_difference, **options)
    nll_low, nll_high = _bootstrap_mean_difference(nll_difference, **options)
    calibration_row = {
        "model": name,
        "category": test_predictions.category,
        "n_test": len(y_true),
        "n_validation": len(y_true_validation),
        "temperature": temperature,
        "brier_original": metrics[VARIANT_RAW]["brier_score"],
        "brier_calibrated": metrics[VARIANT_CALIBRATED]["brier_score"],
        "brier_difference": float(brier_difference.mean()),
        "brier_difference_ci_low": brier_low,
        "brier_difference_ci_high": brier_high,
        "ece_original": metrics[VARIANT_RAW]["expected_calibration_error"],
        "ece_calibrated": metrics[VARIANT_CALIBRATED]["expected_calibration_error"],
        "nll_original": float(nll[VARIANT_RAW].mean()),
        "nll_calibrated": float(nll[VARIANT_CALIBRATED].mean()),
        "nll_difference": float(nll_difference.mean()),
        "nll_difference_ci_low": nll_low,
        "nll_difference_ci_high": nll_high,
    }

    reliability = [
        calculate_reliability_table(y_true, y_pred, np.max(scores, axis=1), n_bins=n_bins)
        .with_columns(pl.lit(name).alias("model"), pl.lit(variant).alias("variant"))
        .select("model", "variant", pl.exclude("model", "variant"))
        for variant, scores in variants.items()
    ]

    final_confidences = np.max(calibrated_scores, axis=1)
    coverages = tuple(config.get("coverages", DEFAULT_COVERAGES))
    risk_coverage = calculate_risk_coverage_curve(
        y_true,
        y_pred,
        final_confidences,
        coverages=coverages,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=random_seed,
    ).with_columns(pl.lit(name).alias("model"))

    target_coverage = float(config.get("target_coverage", DEFAULT_TARGET_COVERAGE))
    threshold = select_confidence_threshold(
        np.max(apply_temperature_scaling(validation_scores, temperature), axis=1),
        target_coverage,
    )
    abstention = {
        "model": name,
        "category": test_predictions.category,
        "target_coverage": target_coverage,
        **evaluate_abstention_at_threshold(
            y_true,
            y_pred,
            final_confidences,
            threshold,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_state=random_seed,
        ),
    }
    return {
        "calibration": calibration_row,
        "reliability": pl.concat(reliability),
        "risk_coverage": risk_coverage,
        "abstention": abstention,
        "confidences": final_confidences,
        "threshold": threshold,
    }


def _build_human_review_table(
    best: ModelPredictions,
    test_ids: Sequence[str],
    confidences: np.ndarray,
    threshold: float,
) -> pl.DataFrame:
    """Tweets de teste abaixo do limiar (somente id, rótulo predito e confiança)."""
    return (
        pl.DataFrame(
            {
                "id": [test_ids[index] for index in best.indices],
                "model": best.name,
                "y_pred": best.y_pred,
                "confidence": confidences,
            }
        )
        .filter(pl.col("confidence") < threshold)
        .sort("confidence")
    )


def _evaluate_calibratable_models(
    checkpoints: Sequence[ModelCheckpoint],
    test_predictions: Sequence[ModelPredictions],
    test: pl.DataFrame,
    validation: pl.DataFrame,
    *,
    vectorizer: Any | None,
    config: Mapping[str, Any],
    n_bootstrap: int,
    confidence_level: float,
    random_seed: int,
) -> dict[str, dict[str, Any]]:
    """Calibra e avalia cada modelo que expõe probabilidades (LLMs ficam de fora)."""
    y_true_test = test["sentiment_label"].to_list()
    y_true_validation = validation["sentiment_label"].to_list()
    validation_texts = validation["text"].to_list()

    outputs: dict[str, dict[str, Any]] = {}
    for checkpoint, predictions in zip(checkpoints, test_predictions, strict=True):
        if predictions.y_score is None or checkpoint.category == CATEGORY_LLM:
            logger.info("Modelo '%s' sem probabilidades: fora da calibração.", checkpoint.name)
            continue
        validation_scores = predict_with_checkpoint(
            checkpoint, validation_texts, vectorizer=vectorizer
        ).y_score
        if validation_scores is None:
            continue
        outputs[checkpoint.name] = _evaluate_one_model(
            predictions,
            y_true_test,
            validation_scores,
            y_true_validation,
            config=config,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_seed=random_seed,
        )
    return outputs


def run_calibration_abstention(
    checkpoints: Sequence[ModelCheckpoint],
    test_predictions: Sequence[ModelPredictions],
    test: pl.DataFrame,
    validation: pl.DataFrame,
    *,
    vectorizer: Any | None,
    config: Mapping[str, Any],
    n_bootstrap: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> CalibrationAbstentionResult:
    """Calibra os modelos com probabilidades e avalia a predição com abstenção.

    Parameters
    ----------
    checkpoints : Sequence[ModelCheckpoint]
        Modelos persistidos (usados para predizer a validação, onde a temperatura é ajustada).
    test_predictions : Sequence[ModelPredictions]
        Predições já calculadas no teste, na mesma ordem de ``checkpoints``.
    test : pl.DataFrame
        Conjunto de teste (``id`` e ``sentiment_label``).
    validation : pl.DataFrame
        Conjunto de validação (``text`` e ``sentiment_label``), disjunto do teste.
    vectorizer : Any | None
        Vetorizador TF-IDF do treino, exigido pelos modelos clássicos.
    config : Mapping[str, Any]
        ``configs/evaluation.yaml -> calibration`` (``n_bins``, ``temperature_scaling``,
        ``coverages``, ``target_coverage``).
    n_bootstrap : int, optional
        Reamostragens dos ICs, by default 1000.
    confidence_level : float, optional
        Nível dos ICs, by default 0.95.
    random_seed : int, optional
        Semente do bootstrap, by default 42.

    Returns
    -------
    CalibrationAbstentionResult
        Tabelas e resumo; vazios se nenhum modelo expõe probabilidades.

    Examples
    --------
    >>> run_calibration_abstention(  # doctest: +SKIP
    ...     checkpoints, predictions, test, validation, vectorizer=None, config={}
    ... )
    """
    outputs = _evaluate_calibratable_models(
        checkpoints,
        test_predictions,
        test,
        validation,
        vectorizer=vectorizer,
        config=config,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_seed=random_seed,
    )
    if not outputs:
        logger.warning("Nenhum modelo com probabilidades; calibração e abstenção puladas.")
        return CalibrationAbstentionResult(tables={}, summary={})

    tables: dict[str, pl.DataFrame] = {
        CALIBRATION_FILE_NAME: pl.DataFrame([item["calibration"] for item in outputs.values()]),
        RELIABILITY_FILE_NAME: pl.concat([item["reliability"] for item in outputs.values()]),
        RISK_COVERAGE_FILE_NAME: pl.concat([item["risk_coverage"] for item in outputs.values()]),
        ABSTENTION_FILE_NAME: pl.DataFrame([item["abstention"] for item in outputs.values()]),
    }
    best_name = max(outputs, key=lambda key: outputs[key]["abstention"]["f1_macro_all"])
    best_predictions = next(item for item in test_predictions if item.name == best_name)
    review = _build_human_review_table(
        best_predictions,
        test["id"].to_list(),
        outputs[best_name]["confidences"],
        outputs[best_name]["threshold"],
    )
    tables[HUMAN_REVIEW_FILE_NAME] = review

    best_abstention = outputs[best_name]["abstention"]
    summary = {
        "n_models": len(outputs),
        "temperature_scaling": bool(config.get("temperature_scaling", True)),
        "target_coverage": best_abstention["target_coverage"],
        "human_review": {
            "model": best_name,
            "threshold": best_abstention["threshold"],
            "n_to_review": review.height,
            "test_coverage": best_abstention["coverage"],
        },
    }
    logger.info(
        "Calibração e abstenção: %d modelo(s); revisão humana (%s): %d tweet(s) abaixo de %.3f.",
        len(outputs),
        best_name,
        review.height,
        best_abstention["threshold"],
    )
    return CalibrationAbstentionResult(tables=tables, summary=summary)
