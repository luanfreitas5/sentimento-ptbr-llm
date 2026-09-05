"""Consolidação e persistência de relatórios de avaliação.

Reúne as saídas de ``src/evaluation/evaluator.py`` em um único relatório
tabular, comparável entre modelos e salvo em ``reports/metrics/`` (ver
``config.paths.ProjectPaths.reports_metrics_dir`` e CLAUDE.md, "Model Cards
& Datasheets").
"""

import logging
from pathlib import Path

import polars as pl

from evaluation.evaluator import EvaluationResult
from exceptions.data import EmptyDatasetError
from io_utils.csv import write_csv

logger = logging.getLogger(__name__)


def build_evaluation_report(model_name: str, evaluation_result: EvaluationResult) -> pl.DataFrame:
    """Consolida o resultado de uma avaliação em um DataFrame tabular.

    Parameters
    ----------
    model_name : str
        Nome do modelo avaliado, incluído em cada linha do relatório.
    evaluation_result : EvaluationResult
        Resultado de :func:`evaluation.evaluator.evaluate_classifier`.

    Returns
    -------
    pl.DataFrame
        Uma linha por métrica, com ``model_name``, ``metric_name``,
        ``metric_value``, ``ci_lower`` e ``ci_upper`` (``None`` quando a
        métrica não tem intervalo de confiança calculado).

    Examples
    --------
    >>> import numpy as np
    >>> from evaluation.evaluator import EvaluationResult
    >>> resultado = EvaluationResult(
    ...     point_metrics={"f1_macro": 0.80},
    ...     confidence_intervals={"f1_macro": (0.75, 0.85)},
    ...     per_class_report={},
    ...     confusion_matrix=np.zeros((3, 3)),
    ... )
    >>> build_evaluation_report("naive_bayes", resultado)["metric_value"].to_list()
    [0.8]
    """
    rows = [
        {
            "model_name": model_name,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "ci_lower": evaluation_result.confidence_intervals.get(metric_name, (None, None))[0],
            "ci_upper": evaluation_result.confidence_intervals.get(metric_name, (None, None))[1],
        }
        for metric_name, metric_value in evaluation_result.point_metrics.items()
    ]
    return pl.DataFrame(rows)


def merge_evaluation_reports(reports: list[pl.DataFrame]) -> pl.DataFrame:
    """Concatena relatórios de avaliação de múltiplos modelos em uma tabela comparativa.

    Parameters
    ----------
    reports : list[pl.DataFrame]
        Relatórios individuais, no formato de :func:`build_evaluation_report`.

    Returns
    -------
    pl.DataFrame
        Concatenação vertical de todos os relatórios.

    Raises
    ------
    EmptyDatasetError
        Se ``reports`` estiver vazio.

    Examples
    --------
    >>> import polars as pl
    >>> relatorio_a = pl.DataFrame(
    ...     {"model_name": ["a"], "metric_name": ["f1_macro"], "metric_value": [0.8]}
    ... )
    >>> relatorio_b = pl.DataFrame(
    ...     {"model_name": ["b"], "metric_name": ["f1_macro"], "metric_value": [0.7]}
    ... )
    >>> merge_evaluation_reports([relatorio_a, relatorio_b])["model_name"].to_list()
    ['a', 'b']
    """
    if len(reports) == 0:
        raise EmptyDatasetError("reports")
    return pl.concat(reports, how="vertical")


def save_evaluation_report(report: pl.DataFrame, output_path: Path) -> None:
    """Salva um relatório de avaliação em CSV, criando diretórios pais se necessário.

    Parameters
    ----------
    report : pl.DataFrame
        Relatório a salvar, no formato de :func:`build_evaluation_report`
        ou :func:`merge_evaluation_reports`.
    output_path : Path
        Caminho do arquivo CSV de destino (ver
        ``config.paths.ProjectPaths.reports_metrics_dir``).

    Returns
    -------
    None

    Examples
    --------
    >>> import polars as pl
    >>> save_evaluation_report(
    ...     pl.DataFrame({"a": [1]}), Path("reports/metrics/exemplo.csv")
    ... )  # doctest: +SKIP
    """
    write_csv(report, output_path)
    logger.info("Relatório de avaliação salvo em '%s' (%d linha(s)).", output_path, report.height)
