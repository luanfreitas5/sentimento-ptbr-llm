"""Ciclo de vida de execuções de rastreamento de experimentos (MLflow).

Implementa CLAUDE.md, "Model & Data Versioning": cada execução de
treino/avaliação é registrada com parâmetros, métricas e artefatos,
rastreada pelo SHA do Git e pelo hash do dataset (ver
``src/experiment/reproducibility.py``) e validada contra
:mod:`schemas.experiment` antes de consolidação em relatórios
comparativos.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import polars as pl

from constants.metrics import validate_metric_name
from schemas.experiment import validate_experiment_run_metric

logger = logging.getLogger(__name__)


@contextmanager
def track_experiment_run(run_name: str, *, tags: dict[str, str] | None = None) -> Iterator[str]:
    """Abre uma execução do MLflow Tracking, com encerramento garantido ao final do bloco.

    Parameters
    ----------
    run_name : str
        Nome descritivo da execução (ex.: nome do modelo avaliado).
    tags : dict[str, str] | None, optional
        Tags adicionais associadas à execução (ex.: ``git_sha``), by
        default None.

    Yields
    ------
    str
        Identificador (``run_id``) da execução do MLflow criada.

    Examples
    --------
    >>> with track_experiment_run("naive_bayes") as run_id:  # doctest: +SKIP
    ...     pass
    """
    import mlflow

    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        logger.info("Execução MLflow iniciada: '%s' (run_id=%s).", run_name, run.info.run_id)
        yield run.info.run_id


def log_run_parameters(parameters: dict[str, Any]) -> None:
    """Registra parâmetros na execução MLflow ativa.

    Parameters
    ----------
    parameters : dict[str, Any]
        Parâmetros a registrar (ex.: hiperparâmetros, semente aleatória).

    Returns
    -------
    None

    Examples
    --------
    >>> log_run_parameters({"random_state": 42})  # doctest: +SKIP
    """
    import mlflow

    mlflow.log_params(parameters)


def log_run_metrics(metrics: dict[str, float]) -> None:
    """Registra métricas na execução MLflow ativa, validando seus nomes.

    Parameters
    ----------
    metrics : dict[str, float]
        Métricas a registrar; cada nome deve pertencer a
        :data:`constants.metrics.ALL_METRICS`.

    Returns
    -------
    None

    Raises
    ------
    DataValidationError
        Se algum nome de métrica não for reconhecido.

    Examples
    --------
    >>> log_run_metrics({"f1_macro": 0.82})  # doctest: +SKIP
    """
    for metric_name in metrics:
        validate_metric_name(metric_name)

    import mlflow

    mlflow.log_metrics(metrics)


def log_run_artifact(local_path: Path) -> None:
    """Registra um arquivo local como artefato da execução MLflow ativa.

    Parameters
    ----------
    local_path : Path
        Caminho do arquivo a anexar à execução.

    Returns
    -------
    None

    Examples
    --------
    >>> log_run_artifact(Path("reports/metrics/relatorio.csv"))  # doctest: +SKIP
    """
    import mlflow

    mlflow.log_artifact(str(local_path))


def build_experiment_run_metrics_dataframe(
    run_id: str, model_name: str, metrics: dict[str, float], *, git_sha: str, dataset_hash: str
) -> pl.DataFrame:
    """Monta e valida um DataFrame de métricas de execução para exportação/relatórios.

    Parameters
    ----------
    run_id : str
        Identificador da execução MLflow.
    model_name : str
        Nome do modelo avaliado nesta execução.
    metrics : dict[str, float]
        Métricas calculadas na execução; cada nome deve pertencer a
        :data:`constants.metrics.ALL_METRICS`.
    git_sha : str
        SHA do commit Git que produziu a execução (ver
        :func:`experiment.reproducibility.get_current_git_sha`).
    dataset_hash : str
        Hash do dataset usado na execução (ver
        :func:`utils.hashing.calculate_file_hash`).

    Returns
    -------
    pl.DataFrame
        DataFrame validado contra
        :class:`schemas.experiment.ExperimentRunMetricSchema`, uma linha
        por métrica.

    Raises
    ------
    DataValidationError
        Se o DataFrame resultante violar o contrato de dados.

    Examples
    --------
    >>> resultado = build_experiment_run_metrics_dataframe(
    ...     "run123", "naive_bayes", {"f1_macro": 0.82}, git_sha="abc123", dataset_hash="def456"
    ... )
    >>> resultado.height
    1
    """
    rows = [
        {
            "run_id": run_id,
            "model_name": model_name,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "git_sha": git_sha,
            "dataset_hash": dataset_hash,
        }
        for metric_name, metric_value in metrics.items()
    ]
    return validate_experiment_run_metric(pl.DataFrame(rows))
