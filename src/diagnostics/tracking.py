"""Registro dos experimentos de diagnóstico no MLflow.

O projeto declara ``experiment.tracking_uri``/``mlflow_tracking_dir`` em
configuração, mas nenhum código ligava esses valores ao cliente MLflow; este
módulo faz a ligação explícita (``set_tracking_uri`` + ``set_experiment``).
Usa a API do MLflow diretamente (import tardio) porque
:func:`experiment.tracker.log_run_metrics` só aceita métricas de classificação
de ``constants.metrics.ALL_METRICS``.

Cada execução registra: alvo, M, K, método de seleção, modelos, versão do
prompt, hash dos dados, SHA do Git e as métricas do passo.
"""

import logging
import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from config.paths import ProjectPaths
from experiment.reproducibility import get_current_git_sha

logger = logging.getLogger(__name__)

DIAGNOSTICS_EXPERIMENT_NAME = "sentimento-ptbr-llm/diagnostics"
_MAX_PARAM_LENGTH = 250


def configure_mlflow(
    paths: ProjectPaths, *, experiment_name: str = DIAGNOSTICS_EXPERIMENT_NAME
) -> None:
    """Aponta o MLflow para ``paths.mlflow_tracking_dir`` e seleciona o experimento.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos do projeto.
    experiment_name : str, optional
        Nome do experimento, by default :data:`DIAGNOSTICS_EXPERIMENT_NAME`.

    Examples
    --------
    >>> configure_mlflow(paths)  # doctest: +SKIP
    """
    import mlflow

    mlflow.set_tracking_uri(paths.mlflow_tracking_dir.resolve().as_uri())
    mlflow.set_experiment(experiment_name)


def flatten_params(params: Mapping[str, Any], *, prefix: str = "") -> dict[str, str]:
    """Achata um dicionário aninhado em ``chave.subchave -> str`` (ignora ``None``).

    Parameters
    ----------
    params : Mapping[str, Any]
        Parâmetros possivelmente aninhados.
    prefix : str, optional
        Prefixo das chaves, by default "".

    Returns
    -------
    dict[str, str]
        Parâmetros planos, com valores truncados ao limite do MLflow.

    Examples
    --------
    >>> flatten_params({"a": {"b": 1}, "c": None})
    {'a.b': '1'}
    """
    flat: dict[str, str] = {}
    for key, value in params.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten_params(value, prefix=f"{name}."))
        elif value is not None:
            flat[name] = str(value)[:_MAX_PARAM_LENGTH]
    return flat


def keep_finite_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    """Remove métricas ``nan``/``inf`` (o MLflow as rejeita ou distorce).

    Parameters
    ----------
    metrics : Mapping[str, float]
        Métricas candidatas.

    Returns
    -------
    dict[str, float]
        Apenas os valores finitos.

    Examples
    --------
    >>> keep_finite_metrics({"a": 1.0, "b": float("nan")})
    {'a': 1.0}
    """
    return {name: float(value) for name, value in metrics.items() if math.isfinite(value)}


@contextmanager
def track_diagnostics_run(
    run_name: str, *, params: Mapping[str, Any], tags: Mapping[str, str] | None = None
) -> Iterator[Any]:
    """Abre uma execução MLflow já com parâmetros e a versão do código (Git SHA).

    Parameters
    ----------
    run_name : str
        Nome da execução.
    params : Mapping[str, Any]
        Parâmetros (podem ser aninhados).
    tags : Mapping[str, str] | None, optional
        Tags adicionais, by default None.

    Yields
    ------
    Any
        A execução ativa do MLflow.

    Examples
    --------
    >>> with track_diagnostics_run("x", params={"a": 1}):  # doctest: +SKIP
    ...     pass
    """
    import mlflow

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(flatten_params(params))
        mlflow.set_tags({"git_sha": get_current_git_sha(), **(tags or {})})
        yield run


def log_diagnostics_metrics(metrics: Mapping[str, float]) -> None:
    """Registra métricas finitas na execução MLflow ativa.

    Parameters
    ----------
    metrics : Mapping[str, float]
        Métricas do passo.

    Examples
    --------
    >>> log_diagnostics_metrics({"auc": 0.7})  # doctest: +SKIP
    """
    import mlflow

    mlflow.log_metrics(keep_finite_metrics(metrics))


def log_diagnostics_artifact(path: Path) -> None:
    """Registra um arquivo como artefato da execução MLflow ativa.

    Parameters
    ----------
    path : Path
        Arquivo existente (ex.: tabela de hipóteses).

    Examples
    --------
    >>> log_diagnostics_artifact(Path("saida.csv"))  # doctest: +SKIP
    """
    import mlflow

    mlflow.log_artifact(str(path))
