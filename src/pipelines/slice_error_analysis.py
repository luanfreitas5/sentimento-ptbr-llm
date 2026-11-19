"""Avaliação por fatias com IC e análise de erros (parte do estágio ``evaluate``).

Para cada modelo avaliado no teste:

1. mede a macro-F1 em cada fatia linguística (comprimento, emoji, hashtag, negação,
   ironia aparente e gíria) com IC de bootstrap e a diferença contra o restante;
2. lista os maiores erros do melhor modelo de cada paradigma, com categoria sugerida e
   coluna vazia para a classificação manual;
3. conta todos os erros desses modelos por categoria sugerida;
4. separa os erros em erro do modelo e erro do rótulo de treino nos tweets com *gold*.

O texto sai mascarado (sem @menções, URLs, e-mails, CPFs e telefones). As categorias
sugeridas são regras lexicais: a categoria final é a da coluna ``manual_category``.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import polars as pl

from constants.defaults import (
    DEFAULT_BOOTSTRAP_ITERATIONS,
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_RANDOM_SEED,
)
from evaluation.error_analysis import (
    ALL_DIMENSIONS,
    DEFAULT_MIN_SLICE_SIZE,
    DEFAULT_TOP_N_ERRORS,
    build_slice_flags,
    build_top_error_table,
    evaluate_slices_with_uncertainty,
    select_best_model_by_category,
    summarize_error_patterns,
    summarize_error_sources,
)
from evaluation.predictions import ModelPredictions

logger = logging.getLogger(__name__)

SLICE_UNCERTAINTY_FILE_NAME = "metricas_por_fatia.csv"
TOP_ERRORS_FILE_NAME = "erros_maiores_por_paradigma.csv"
ERROR_PATTERNS_FILE_NAME = "erros_padroes_por_paradigma.csv"
ERROR_SOURCES_FILE_NAME = "erros_origem_modelo_vs_rotulo.csv"


@dataclass(frozen=True)
class SliceErrorAnalysisResult:
    """Resultado da avaliação por fatias e da análise de erros.

    Attributes
    ----------
    tables : dict[str, pl.DataFrame]
        Tabelas indexadas pelo nome de arquivo (``*_FILE_NAME``); só as não vazias.
    summary : dict[str, Any]
        Resumo para ``reports/metrics/avaliacao.json``.
    """

    tables: dict[str, pl.DataFrame]
    summary: dict[str, Any]


def _evaluate_all_slices(
    predictions: Sequence[ModelPredictions],
    test: pl.DataFrame,
    slice_flags: pl.DataFrame,
    *,
    dimensions: Sequence[str],
    min_slice_size: int,
    n_bootstrap: int,
    confidence_level: float,
    random_seed: int,
) -> pl.DataFrame:
    """Macro-F1 por fatia de cada modelo, restrita aos tweets que ele predisse."""
    y_true_all = test["sentiment_label"].to_list()
    tables = [
        evaluate_slices_with_uncertainty(
            [y_true_all[index] for index in item.indices],
            item.y_pred,
            slice_flags[item.indices],
            dimensions=dimensions,
            min_slice_size=min_slice_size,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_seed=random_seed,
        ).with_columns(pl.lit(item.name).alias("model"), pl.lit(item.category).alias("category"))
        for item in predictions
    ]
    return pl.concat(tables, how="diagonal_relaxed")


def _summarize_worst_slices(slice_table: pl.DataFrame) -> list[dict[str, Any]]:
    """Fatias em que algum modelo é pior que o restante com o IC inteiro abaixo de zero."""
    if "significantly_worse" not in slice_table.columns:
        return []
    return (
        slice_table.filter(pl.col("significantly_worse").fill_null(False))
        .group_by("dimension", "slice")
        .agg(
            pl.len().alias("n_models_worse"),
            pl.col("f1_macro_difference").min().alias("worst_difference"),
        )
        .sort("n_models_worse", "worst_difference", descending=[True, False])
        .to_dicts()
    )


def run_slice_error_analysis(
    predictions: Sequence[ModelPredictions],
    metrics_table: pl.DataFrame,
    test: pl.DataFrame,
    references: Mapping[str, pl.DataFrame],
    *,
    config: Mapping[str, Any],
    n_bootstrap: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> SliceErrorAnalysisResult:
    """Executa a avaliação por fatias e a análise de erros sobre o teste.

    Parameters
    ----------
    predictions : Sequence[ModelPredictions]
        Predições de cada modelo no teste.
    metrics_table : pl.DataFrame
        Tabela de métricas por modelo (``model``, ``category``, ``f1_macro``), usada para
        escolher o melhor modelo de cada paradigma.
    test : pl.DataFrame
        Teste com ``id``, ``text`` e ``sentiment_label``.
    references : Mapping[str, pl.DataFrame]
        Referências *gold* (``id`` e ``label``); vazio dispensa a origem do erro.
    config : Mapping[str, Any]
        ``configs/evaluation.yaml -> slice_evaluation`` (``slices``, ``min_slice_size``,
        ``top_n_errors``).
    n_bootstrap : int, optional
        Reamostragens dos ICs, by default 1000.
    confidence_level : float, optional
        Nível de confiança dos ICs, by default 0.95.
    random_seed : int, optional
        Semente do bootstrap, by default 42.

    Returns
    -------
    SliceErrorAnalysisResult
        Tabelas e resumo.

    Raises
    ------
    ValueError
        Se ``config["slices"]`` citar uma dimensão desconhecida.

    Examples
    --------
    >>> run_slice_error_analysis(preds, metrics, test, {}, config={})  # doctest: +SKIP
    """
    dimensions = list(config.get("slices") or ALL_DIMENSIONS)
    unknown = [name for name in dimensions if name not in ALL_DIMENSIONS]
    if unknown:
        raise ValueError(f"Fatias desconhecidas em slice_evaluation.slices: {unknown}")
    min_slice_size = int(config.get("min_slice_size", DEFAULT_MIN_SLICE_SIZE))
    top_n = int(config.get("top_n_errors", DEFAULT_TOP_N_ERRORS))

    slice_flags = build_slice_flags(test["text"].to_list())
    slice_table = _evaluate_all_slices(
        predictions,
        test,
        slice_flags,
        dimensions=dimensions,
        min_slice_size=min_slice_size,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_seed=random_seed,
    )
    best_models = select_best_model_by_category(metrics_table)
    error_options = {"reference_name": config.get("reference")}
    tables = {
        SLICE_UNCERTAINTY_FILE_NAME: slice_table,
        TOP_ERRORS_FILE_NAME: build_top_error_table(
            predictions, best_models, test, slice_flags, references, top_n=top_n, **error_options
        ),
        ERROR_PATTERNS_FILE_NAME: summarize_error_patterns(
            predictions, best_models, test, slice_flags, references, **error_options
        ),
        ERROR_SOURCES_FILE_NAME: summarize_error_sources(
            predictions, test["sentiment_label"].to_list(), test["id"].to_list(), references
        ),
    }
    summary = {
        "dimensions": dimensions,
        "min_slice_size": min_slice_size,
        "best_model_by_category": best_models,
        "worst_slices": _summarize_worst_slices(slice_table),
    }
    logger.info(
        "Avaliação por fatias concluída: %d fatia(s) por modelo, %d fatia(s) pior(es) que o resto.",
        slice_table.filter(pl.col("model") == slice_table["model"][0]).height,
        len(summary["worst_slices"]),
    )
    return SliceErrorAnalysisResult(
        tables={name: table for name, table in tables.items() if not table.is_empty()},
        summary=summary,
    )
