"""Avaliação comparativa entre múltiplos classificadores de sentimento.

Implementa o estágio ``comparative_evaluation`` de ``configs/config.yaml ->
stages``: avalia cada modelo com
:func:`evaluation.evaluator.evaluate_classifier`, compara pares de modelos
com o teste de McNemar (``configs/evaluation.yaml ->
significance_tests.pairwise``), compara três ou mais modelos com Friedman +
post-hoc de Nemenyi quando fornecidos os scores por dobra, avalia por fatia
dos dados e consolida tudo em um único relatório salvo em disco.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl

from evaluation.evaluator import evaluate_classifier
from evaluation.reports import (
    build_evaluation_report,
    merge_evaluation_reports,
    save_evaluation_report,
)
from evaluation.significance import run_friedman_test, run_mcnemar_test, run_nemenyi_post_hoc_test
from evaluation.slice_evaluation import evaluate_metrics_by_slice

logger = logging.getLogger(__name__)

_MINIMUM_MODELS_FOR_FRIEDMAN_TEST = 3


@dataclass(frozen=True)
class ComparativeEvaluationResult:
    """Resultado consolidado da avaliação comparativa entre modelos.

    Parameters
    ----------
    merged_report : pl.DataFrame
        Relatório de métricas mesclado de todos os modelos (ver
        :func:`evaluation.reports.merge_evaluation_reports`).
    pairwise_significance_tests : dict[str, dict[str, float]]
        Resultado do teste de McNemar para cada par de modelos, indexado
        por ``"{modelo_a}_vs_{modelo_b}"``.
    friedman_test : dict[str, float] | None
        Resultado do teste de Friedman entre três ou mais modelos, quando
        ``cross_validation_fold_scores`` foi informado com ao menos três
        modelos; ``None`` caso contrário.
    nemenyi_test : dict[str, object] | None
        Resultado do post-hoc de Nemenyi, calculado junto ao teste de
        Friedman; ``None`` caso contrário.
    slice_reports : dict[str, pl.DataFrame]
        Avaliação por fatia dos dados, por modelo, quando ``slice_labels``
        foi informado; dicionário vazio caso contrário.
    """

    merged_report: pl.DataFrame
    pairwise_significance_tests: dict[str, dict[str, float]]
    friedman_test: dict[str, float] | None
    nemenyi_test: dict[str, object] | None
    slice_reports: dict[str, pl.DataFrame]


def run_comparative_evaluation_stage(
    model_predictions: Mapping[str, Sequence[str]],
    y_true: Sequence[str],
    *,
    slice_labels: Sequence[str] | None = None,
    cross_validation_fold_scores: Mapping[str, Sequence[float]] | None = None,
    output_path: Path,
) -> ComparativeEvaluationResult:
    """Avalia e compara estatisticamente múltiplos classificadores de sentimento.

    Parameters
    ----------
    model_predictions : Mapping[str, Sequence[str]]
        Rótulos preditos por cada modelo no mesmo conjunto de teste,
        indexados pelo nome do modelo. Ao menos dois modelos.
    y_true : Sequence[str]
        Rótulos de sentimento verdadeiros do conjunto de teste, mesmo
        tamanho de cada sequência em ``model_predictions``.
    slice_labels : Sequence[str] | None, optional
        Valor de fatia de cada amostra (ver
        :func:`evaluation.slice_evaluation.evaluate_metrics_by_slice`), by
        default None (nenhuma avaliação por fatia).
    cross_validation_fold_scores : Mapping[str, Sequence[float]] | None, optional
        Métrica principal de cada modelo por dobra de validação cruzada,
        indexada pelo nome do modelo; quando informado com três ou mais
        modelos, habilita o teste de Friedman e o post-hoc de Nemenyi, by
        default None.
    output_path : Path
        Caminho do arquivo CSV de destino do relatório mesclado (ver
        :func:`evaluation.reports.save_evaluation_report`).

    Returns
    -------
    ComparativeEvaluationResult
        Relatório mesclado, testes de significância par a par e (quando
        aplicável) multi-modelo, e avaliação por fatia.

    Raises
    ------
    EmptyDatasetError
        Se ``model_predictions`` estiver vazio ou algum modelo não tiver
        predições.

    Examples
    --------
    >>> run_comparative_evaluation_stage(
    ...     {"naive_bayes": y_pred_nb, "svm": y_pred_svm},
    ...     y_true,
    ...     output_path=Path("reports/metrics/comparativo.csv"),
    ... )  # doctest: +SKIP
    """
    reports = [
        build_evaluation_report(model_name, evaluate_classifier(y_true, y_pred))
        for model_name, y_pred in model_predictions.items()
    ]
    merged_report = merge_evaluation_reports(reports)
    save_evaluation_report(merged_report, output_path)

    pairwise_significance_tests: dict[str, dict[str, float]] = {
        f"{model_a}_vs_{model_b}": run_mcnemar_test(
            y_true, model_predictions[model_a], model_predictions[model_b]
        )
        for model_a, model_b in combinations(model_predictions, 2)
    }

    friedman_test: dict[str, float] | None = None
    nemenyi_test: dict[str, object] | None = None
    if (
        cross_validation_fold_scores is not None
        and len(cross_validation_fold_scores) >= _MINIMUM_MODELS_FOR_FRIEDMAN_TEST
    ):
        fold_scores = list(cross_validation_fold_scores.values())
        friedman_test = run_friedman_test(*fold_scores)
        nemenyi_test = run_nemenyi_post_hoc_test(np.array(fold_scores).T)

    slice_reports: dict[str, pl.DataFrame] = {}
    if slice_labels is not None:
        slice_reports = {
            model_name: evaluate_metrics_by_slice(y_true, y_pred, slice_labels)
            for model_name, y_pred in model_predictions.items()
        }

    logger.info(
        "Avaliação comparativa concluída para %d modelo(s), %d comparação(ões) par a par.",
        len(model_predictions),
        len(pairwise_significance_tests),
    )
    return ComparativeEvaluationResult(
        merged_report=merged_report,
        pairwise_significance_tests=pairwise_significance_tests,
        friedman_test=friedman_test,
        nemenyi_test=nemenyi_test,
        slice_reports=slice_reports,
    )
