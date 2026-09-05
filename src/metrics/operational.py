"""Métricas operacionais: latência de inferência e custo computacional.

Implementa ``configs/evaluation.yaml`` -> ``metrics.operational``: métricas
que não medem qualidade preditiva, mas viabilidade de uso em produção,
usadas para comparar classicos/deep learning/LLMs também sob a ótica de
custo (ver Seção 4.8 do documento mestre).
"""

import logging
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from exceptions.data import EmptyDatasetError
from exceptions.model import ModelError
from utils.timing import measure_execution_time

logger = logging.getLogger(__name__)


def calculate_latency_statistics(latencies_ms: Sequence[float]) -> dict[str, float]:
    """Resume uma amostra de latências de inferência em estatísticas-chave.

    Parameters
    ----------
    latencies_ms : Sequence[float]
        Latências individuais medidas, em milissegundos. Não vazia.

    Returns
    -------
    dict[str, float]
        Dicionário com as chaves ``mean_ms``, ``p50_ms``, ``p95_ms`` e
        ``p99_ms``.

    Raises
    ------
    EmptyDatasetError
        Se ``latencies_ms`` estiver vazia.

    Examples
    --------
    >>> resultado = calculate_latency_statistics([10.0, 12.0, 11.0, 50.0])
    >>> resultado["mean_ms"]
    20.75
    """
    if len(latencies_ms) == 0:
        raise EmptyDatasetError("latencies_ms")
    latencies_array = np.asarray(latencies_ms, dtype=float)
    return {
        "mean_ms": float(np.mean(latencies_array)),
        "p50_ms": float(np.percentile(latencies_array, 50)),
        "p95_ms": float(np.percentile(latencies_array, 95)),
        "p99_ms": float(np.percentile(latencies_array, 99)),
    }


def measure_inference_latency(
    predict_function: Callable[[Any], Any], inputs: Any, *, n_repeats: int = 10
) -> dict[str, float]:
    """Mede empiricamente a latência de inferência repetindo uma chamada de predição.

    Parameters
    ----------
    predict_function : Callable[[Any], Any]
        Função de predição a ser cronometrada (ex.: ``model.predict``).
    inputs : Any
        Entrada fixa repassada a cada chamada de ``predict_function``.
    n_repeats : int, optional
        Número de repetições para estimar a distribuição de latência, by
        default 10.

    Returns
    -------
    dict[str, float]
        Estatísticas de latência (ver :func:`calculate_latency_statistics`).

    Raises
    ------
    ValueError
        Se ``n_repeats`` for menor que 1.

    Examples
    --------
    >>> resultado = measure_inference_latency(lambda x: x + 1, 1, n_repeats=3)
    >>> resultado["mean_ms"] >= 0.0
    True
    """
    if n_repeats < 1:
        raise ValueError(f"n_repeats deve ser >= 1, recebido: {n_repeats}")

    latencies_ms: list[float] = []
    for _ in range(n_repeats):
        with measure_execution_time() as timing:
            predict_function(inputs)
        latencies_ms.append(timing.elapsed_seconds * 1000)

    statistics = calculate_latency_statistics(latencies_ms)
    logger.info(
        "Latência de inferência medida sobre %d repetições: média=%.2fms, p95=%.2fms.",
        n_repeats,
        statistics["mean_ms"],
        statistics["p95_ms"],
    )
    return statistics


def count_trainable_parameters(model: Any) -> int:
    """Conta o número de parâmetros treináveis de um modelo (proxy de custo computacional).

    Usa *duck typing* sobre a interface de ``torch.nn.Module``
    (``parameters()``/``requires_grad``/``numel()``), sem depender
    diretamente do ``torch`` como import de módulo.

    Parameters
    ----------
    model : Any
        Modelo a inspecionar. Deve expor um método ``parameters()`` que
        retorne um iterável de tensores com atributos ``requires_grad`` e
        ``numel()``.

    Returns
    -------
    int
        Número total de parâmetros com ``requires_grad=True``.

    Raises
    ------
    ModelError
        Se ``model`` não expuser o método ``parameters()``.

    Examples
    --------
    >>> class ModeloFalso:
    ...     def parameters(self):
    ...         return []
    >>> count_trainable_parameters(ModeloFalso())
    0
    """
    if not hasattr(model, "parameters"):
        raise ModelError(
            "count_trainable_parameters requer um modelo com o método 'parameters()' "
            "(ex.: torch.nn.Module); o modelo informado não o expõe."
        )
    return int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )


def calculate_operational_metrics(
    predict_function: Callable[[Any], Any], inputs: Any, *, n_repeats: int = 10, model: Any = None
) -> dict[str, float]:
    """Calcula o conjunto de métricas operacionais usado no projeto.

    Parameters
    ----------
    predict_function : Callable[[Any], Any]
        Função de predição a ser cronometrada.
    inputs : Any
        Entrada fixa repassada a cada chamada de ``predict_function``.
    n_repeats : int, optional
        Número de repetições da medição de latência, by default 10.
    model : Any, optional
        Modelo do qual extrair a contagem de parâmetros treináveis (ver
        :func:`count_trainable_parameters`); quando ``None``, o custo
        computacional não é incluído no resultado, by default None.

    Returns
    -------
    dict[str, float]
        Estatísticas de latência (``inference_time_ms_*``) e, quando
        ``model`` é informado, ``computational_cost`` (número de
        parâmetros treináveis).

    Examples
    --------
    >>> resultado = calculate_operational_metrics(lambda x: x, 1, n_repeats=3)
    >>> "inference_time_ms_mean" in resultado
    True
    """
    latency_statistics = measure_inference_latency(predict_function, inputs, n_repeats=n_repeats)
    metrics = {
        f"inference_time_ms_{suffix}": value
        for suffix, value in (
            ("mean", latency_statistics["mean_ms"]),
            ("p50", latency_statistics["p50_ms"]),
            ("p95", latency_statistics["p95_ms"]),
            ("p99", latency_statistics["p99_ms"]),
        )
    }
    if model is not None:
        metrics["computational_cost"] = float(count_trainable_parameters(model))
    return metrics
