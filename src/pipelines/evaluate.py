"""Estágio ``evaluate``: avaliação no teste, comparação estatística e ablação.

Aplica todos os modelos persistidos (``models/checkpoints``) ao conjunto de
TESTE, que só é usado aqui. Para cada modelo reporta as métricas com intervalo
de confiança bootstrap, o relatório por classe, a calibração (quando há
probabilidades) e o desempenho por faixa de comprimento do texto. Compara os
modelos par a par com o teste de McNemar (correção de Holm) sobre os tweets
preditos por ambos e executa a ablação do pipeline clássico no conjunto de
VALIDAÇÃO (nunca no teste, para não orientar decisões de projeto com o conjunto
final).

Saídas: ``reports/tables/avaliacao/`` (métricas, por classe, por fatia,
predições), ``reports/statistics/`` (McNemar), ``reports/ablation/`` (ablação)
e ``reports/metrics/avaliacao.json`` (resumo). A etapa ``report`` consome essas
saídas.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from config.paths import ProjectPaths
from constants.defaults import (
    DEFAULT_BOOTSTRAP_ITERATIONS,
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SIGNIFICANCE_ALPHA,
)
from constants.labels import SENTIMENT_CLASSES
from constants.metrics import PRIMARY_METRIC
from data.loader import load_training_example_dataset
from data.writer import write_dataset
from evaluation.ablation import identify_most_impactful_component, run_pipeline_ablation
from evaluation.calibration import calculate_calibration_metrics
from evaluation.evaluator import evaluate_classifier
from evaluation.predictions import (
    CATEGORY_CLASSICAL,
    CATEGORY_LABELS,
    ModelCheckpoint,
    ModelPredictions,
    discover_checkpoints,
    predict_with_checkpoint,
)
from evaluation.significance import run_mcnemar_test
from evaluation.slice_evaluation import evaluate_metrics_by_slice
from exceptions.data import DataNotFoundError
from io_utils.csv import write_csv
from io_utils.json import write_json
from models.persistence import load_classifier
from pipelines.features import TFIDF_VECTORIZER_FILE_NAME
from pipelines.training_llm import select_sample_indices

logger = logging.getLogger(__name__)

EVALUATION_SUBDIR = "avaliacao"
PREDICTIONS_FILE_NAME = "predicoes_teste.parquet"
METRICS_TABLE_FILE_NAME = "metricas_por_modelo.csv"
PER_CLASS_TABLE_FILE_NAME = "relatorio_por_classe.csv"
SLICE_TABLE_FILE_NAME = "metricas_por_comprimento.csv"
MCNEMAR_FILE_NAME = "comparacao_mcnemar.csv"
ABLATION_FILE_NAME = "ablacao_pipeline_classico.csv"
SUMMARY_FILE_NAME = "avaliacao.json"
_LENGTH_SLICES: tuple[str, ...] = ("1_curto", "2_medio", "3_longo")


@dataclass(frozen=True)
class EvaluateResult:
    """Resultado da etapa ``evaluate``.

    Attributes
    ----------
    metrics_table : pl.DataFrame
        Uma linha por modelo, com métricas, ICs, calibração e custo de inferência.
    mcnemar_table : pl.DataFrame
        Comparações par a par (vazia com menos de dois modelos).
    ablation_table : pl.DataFrame | None
        Impacto de cada componente do pipeline clássico (``None`` se dispensada).
    summary : dict[str, Any]
        Conteúdo de ``reports/metrics/avaliacao.json``.
    files : dict[str, Path]
        Arquivos gravados, indexados pelo nome.
    """

    metrics_table: pl.DataFrame
    mcnemar_table: pl.DataFrame
    ablation_table: pl.DataFrame | None
    summary: dict[str, Any]
    files: dict[str, Path]


def adjust_p_values_holm(p_values: Sequence[float]) -> list[float]:
    """Aplica a correção de Holm-Bonferroni a uma família de p-valores.

    Parameters
    ----------
    p_values : Sequence[float]
        P-valores brutos.

    Returns
    -------
    list[float]
        P-valores ajustados, na ordem original (monotônicos e limitados a 1).

    Examples
    --------
    >>> adjust_p_values_holm([0.01, 0.04, 0.03])
    [0.03, 0.06, 0.06]
    """
    n_tests = len(p_values)
    order = np.argsort(p_values)
    adjusted = [0.0] * n_tests
    running_maximum = 0.0
    for rank, index in enumerate(order):
        running_maximum = max(running_maximum, (n_tests - rank) * p_values[index])
        adjusted[index] = min(1.0, running_maximum)
    return adjusted


def _build_length_slices(texts: Sequence[str]) -> list[str]:
    """Classifica cada texto em curto/médio/longo pelos tercis do número de palavras."""
    lengths = np.array([len(text.split()) for text in texts])
    lower, upper = np.quantile(lengths, [1 / 3, 2 / 3])
    return [
        _LENGTH_SLICES[0]
        if length <= lower
        else _LENGTH_SLICES[1]
        if length <= upper
        else _LENGTH_SLICES[2]
        for length in lengths
    ]


def _evaluate_model(
    predictions: ModelPredictions,
    y_true_all: Sequence[str],
    *,
    n_bootstrap: int,
    confidence_level: float,
    random_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Avalia um modelo: devolve a linha da tabela de métricas e as linhas por classe."""
    y_true = [y_true_all[index] for index in predictions.indices]
    result = evaluate_classifier(
        y_true,
        predictions.y_pred,
        y_score=predictions.y_score,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=random_seed,
    )
    row: dict[str, Any] = {
        "model": predictions.name,
        "category": predictions.category,
        "category_label": CATEGORY_LABELS[predictions.category],
        "n_test": len(y_true),
        **result.point_metrics,
    }
    for metric_name, (lower, upper) in result.confidence_intervals.items():
        row[f"{metric_name}_ci_low"] = lower
        row[f"{metric_name}_ci_high"] = upper
    if predictions.y_score is not None:
        row.update(calculate_calibration_metrics(y_true, predictions.y_score))
    row["inference_ms_per_sample"] = predictions.inference_ms_per_sample

    per_class_rows = [
        {"model": predictions.name, "category": predictions.category, "class": label, **values}
        for label, values in result.per_class_report.items()
    ]
    return row, per_class_rows


def _compare_models_mcnemar(
    predictions: Sequence[ModelPredictions], y_true_all: Sequence[str], alpha: float
) -> pl.DataFrame:
    """McNemar par a par (tweets preditos por ambos), com correção de Holm."""
    rows: list[dict[str, Any]] = []
    for first, second in combinations(predictions, 2):
        first_by_index = dict(zip(first.indices, first.y_pred, strict=True))
        second_by_index = dict(zip(second.indices, second.y_pred, strict=True))
        common = sorted(set(first_by_index) & set(second_by_index))
        if not common:
            continue
        y_true = [y_true_all[index] for index in common]
        pred_a = [first_by_index[index] for index in common]
        pred_b = [second_by_index[index] for index in common]
        test = run_mcnemar_test(y_true, pred_a, pred_b)
        rows.append(
            {
                "model_a": first.name,
                "model_b": second.name,
                "n_common": len(common),
                "accuracy_a": float(np.mean(np.array(pred_a) == np.array(y_true))),
                "accuracy_b": float(np.mean(np.array(pred_b) == np.array(y_true))),
                "statistic": test["statistic"],
                "p_value": test["p_value"],
            }
        )
    if not rows:
        return pl.DataFrame()
    adjusted = adjust_p_values_holm([row["p_value"] for row in rows])
    for row, adjusted_value in zip(rows, adjusted, strict=True):
        row["p_value_holm"] = adjusted_value
        row["significant"] = adjusted_value < alpha
    return pl.DataFrame(rows)


def _build_predictions_table(
    test_ids: Sequence[str], y_true: Sequence[str], predictions: Sequence[ModelPredictions]
) -> pl.DataFrame:
    """Tabela larga ``id``, ``y_true`` e uma coluna ``pred_<modelo>`` (nula onde não prediz)."""
    columns: dict[str, list[Any]] = {"id": list(test_ids), "y_true": list(y_true)}
    for item in predictions:
        column: list[str | None] = [None] * len(test_ids)
        for index, label in zip(item.indices, item.y_pred, strict=True):
            column[index] = label
        columns[f"pred_{item.name}"] = column
    return pl.DataFrame(columns)


def _run_ablation(
    paths: ProjectPaths,
    train: pl.DataFrame,
    ablation_config: Mapping[str, Any],
    *,
    n_bootstrap: int,
    confidence_level: float,
    random_seed: int,
) -> pl.DataFrame:
    """Executa a ablação do pipeline clássico avaliando no conjunto de validação."""
    validation = load_training_example_dataset(paths.validation_corpus_file)
    return run_pipeline_ablation(
        train["text"].to_list(),
        train["sentiment_label"].to_list(),
        validation["text"].to_list(),
        validation["sentiment_label"].to_list(),
        components=ablation_config["components"],
        model_name=ablation_config.get("model", "logistic_regression"),
        model_params=ablation_config.get("model_params"),
        tfidf_params=ablation_config.get("tfidf_params"),
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=random_seed,
    )


def _load_vectorizer(paths: ProjectPaths, checkpoints: Sequence[ModelCheckpoint]) -> Any | None:
    """Carrega o vetorizador TF-IDF do treino, se algum modelo clássico será avaliado."""
    if not any(item.category == CATEGORY_CLASSICAL for item in checkpoints):
        return None
    vectorizer_path = paths.models_checkpoints_dir / TFIDF_VECTORIZER_FILE_NAME
    if not vectorizer_path.is_file():
        raise DataNotFoundError(f"{vectorizer_path} — rode `make features`")
    return load_classifier(vectorizer_path)


def _evaluate_all_models(
    all_predictions: Sequence[ModelPredictions],
    test: pl.DataFrame,
    *,
    n_bootstrap: int,
    confidence_level: float,
    random_seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Avalia cada modelo: tabela de métricas (melhor primeiro), por classe e por comprimento."""
    y_true_all = test["sentiment_label"].to_list()
    length_slices = _build_length_slices(test["text"].to_list())
    metric_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    slice_tables: list[pl.DataFrame] = []
    for item in all_predictions:
        row, class_rows = _evaluate_model(
            item,
            y_true_all,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_seed=random_seed,
        )
        metric_rows.append(row)
        per_class_rows.extend(class_rows)
        slice_tables.append(
            evaluate_metrics_by_slice(
                [y_true_all[index] for index in item.indices],
                item.y_pred,
                [length_slices[index] for index in item.indices],
            ).with_columns(pl.lit(item.name).alias("model"))
        )
    metrics_table = pl.DataFrame(metric_rows, infer_schema_length=None).sort(
        PRIMARY_METRIC, descending=True
    )
    return metrics_table, pl.DataFrame(per_class_rows), pl.concat(slice_tables)


def _build_summary(
    metrics_table: pl.DataFrame,
    ablation_table: pl.DataFrame | None,
    *,
    n_test: int,
    n_bootstrap: int,
    confidence_level: float,
    alpha: float,
) -> dict[str, Any]:
    """Resumo de ``reports/metrics/avaliacao.json``: melhor modelo geral e por categoria."""
    best = metrics_table.row(0, named=True)
    best_by_category = {
        category: metrics_table.filter(pl.col("category") == category).row(0, named=True)["model"]
        for category in metrics_table["category"].unique(maintain_order=True)
    }
    most_impactful = (
        identify_most_impactful_component(ablation_table) if ablation_table is not None else None
    )
    return {
        "primary_metric": PRIMARY_METRIC,
        "n_test": n_test,
        "n_models": metrics_table.height,
        "classes": list(SENTIMENT_CLASSES),
        "best_model": {
            "model": best["model"],
            "category": best["category"],
            PRIMARY_METRIC: best[PRIMARY_METRIC],
            "ci": [best[f"{PRIMARY_METRIC}_ci_low"], best[f"{PRIMARY_METRIC}_ci_high"]],
        },
        "best_by_category": best_by_category,
        "bootstrap": {"n_bootstrap": n_bootstrap, "confidence_level": confidence_level},
        "significance": {"test": "mcnemar", "correction": "holm", "alpha": alpha},
        "most_impactful_ablation": most_impactful,
    }


def _write_tables(
    paths: ProjectPaths,
    tables: Mapping[str, pl.DataFrame],
    predictions_table: pl.DataFrame,
) -> dict[str, Path]:
    """Grava as tabelas de ``evaluate`` e devolve os caminhos, indexados pelo nome."""
    tables_dir = paths.reports_tables_dir / EVALUATION_SUBDIR
    files = {
        "metrics": tables_dir / METRICS_TABLE_FILE_NAME,
        "per_class": tables_dir / PER_CLASS_TABLE_FILE_NAME,
        "slices": tables_dir / SLICE_TABLE_FILE_NAME,
        "predictions": tables_dir / PREDICTIONS_FILE_NAME,
    }
    for name, table in tables.items():
        write_csv(table, files[name])
    write_dataset(predictions_table, files["predictions"])
    return files


def run_evaluate_stage(
    paths: ProjectPaths,
    *,
    model_names: Sequence[str] | None = None,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    alpha: float = DEFAULT_SIGNIFICANCE_ALPHA,
    random_seed: int = DEFAULT_RANDOM_SEED,
    llm_max_test_samples: int | None = None,
    ablation_config: Mapping[str, Any] | None = None,
    skip_ablation: bool = False,
) -> EvaluateResult:
    """Avalia no teste todos os modelos treinados, compara-os e executa a ablação.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto.
    model_names : Sequence[str] | None, optional
        Restringe a avaliação a estes modelos, by default None (todos os checkpoints).
    n_bootstrap : int, optional
        Reamostragens dos ICs, by default 1000.
    confidence_level : float, optional
        Nível de confiança dos ICs, by default 0.95.
    alpha : float, optional
        Nível de significância do McNemar (após Holm), by default 0.05.
    random_seed : int, optional
        Semente do bootstrap e da subamostra dos LLMs, by default 42.
    llm_max_test_samples : int | None, optional
        Máximo de tweets de teste enviados a cada LLM (custo de inferência); a comparação
        estatística usa só os tweets comuns, by default None (todo o teste).
    ablation_config : Mapping[str, Any] | None, optional
        ``configs/evaluation.yaml -> ablation`` (``components``, ``model``, ``model_params``,
        ``tfidf_params``), by default None (sem ablação).
    skip_ablation : bool, optional
        Dispensa a ablação, by default False.

    Returns
    -------
    EvaluateResult
        Tabelas, resumo e arquivos gravados.

    Raises
    ------
    DataNotFoundError
        Se não houver nenhum modelo treinado ou faltar o vetorizador TF-IDF.

    Examples
    --------
    >>> run_evaluate_stage(paths)  # doctest: +SKIP
    """
    checkpoints = discover_checkpoints(paths.models_checkpoints_dir, model_names)
    if not checkpoints:
        raise DataNotFoundError(
            f"{paths.models_checkpoints_dir} — nenhum modelo treinado; rode `make classical`, "
            "`make deep`, `make transformer` e/ou `make llm`"
        )

    train = load_training_example_dataset(paths.training_corpus_file)
    test = load_training_example_dataset(paths.test_corpus_file)
    test_texts = test["text"].to_list()
    vectorizer = _load_vectorizer(paths, checkpoints)
    llm_indices = select_sample_indices(len(test_texts), llm_max_test_samples, random_seed)
    all_predictions = [
        predict_with_checkpoint(
            item,
            test_texts,
            vectorizer=vectorizer,
            train_texts=train["text"].to_list(),
            train_labels=train["sentiment_label"].to_list(),
            llm_indices=llm_indices,
        )
        for item in checkpoints
    ]

    metrics_table, per_class_table, slice_table = _evaluate_all_models(
        all_predictions,
        test,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_seed=random_seed,
    )
    y_true_all = test["sentiment_label"].to_list()
    mcnemar_table = _compare_models_mcnemar(all_predictions, y_true_all, alpha)
    files = _write_tables(
        paths,
        {"metrics": metrics_table, "per_class": per_class_table, "slices": slice_table},
        _build_predictions_table(test["id"].to_list(), y_true_all, all_predictions),
    )
    if not mcnemar_table.is_empty():
        files["mcnemar"] = paths.reports_statistics_dir / MCNEMAR_FILE_NAME
        write_csv(mcnemar_table, files["mcnemar"])

    ablation_table: pl.DataFrame | None = None
    if ablation_config and not skip_ablation:
        ablation_table = _run_ablation(
            paths,
            train,
            ablation_config,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_seed=random_seed,
        )
        files["ablation"] = paths.reports_ablation_dir / ABLATION_FILE_NAME
        write_csv(ablation_table, files["ablation"])

    summary = _build_summary(
        metrics_table,
        ablation_table,
        n_test=test.height,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        alpha=alpha,
    )
    files["summary"] = paths.reports_metrics_dir / SUMMARY_FILE_NAME
    write_json(summary, files["summary"])

    best = summary["best_model"]
    logger.info(
        "Avaliação concluída: %d modelo(s); melhor (%s) = %s (%s=%.4f).",
        metrics_table.height,
        best["category"],
        best["model"],
        PRIMARY_METRIC,
        best[PRIMARY_METRIC],
    )
    return EvaluateResult(
        metrics_table=metrics_table,
        mcnemar_table=mcnemar_table,
        ablation_table=ablation_table,
        summary=summary,
        files=files,
    )
