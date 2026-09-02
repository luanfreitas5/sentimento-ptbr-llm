"""Paralelização de etapas de inferência/predição de modelos.

Usa múltiplas threads (``ThreadPoolExecutor``), adequado para inferência que
libera o GIL (ex.: chamadas a modelos em GPU via ``torch``) ou que é ligada
a I/O (ex.: requisições a uma API de modelo — ver ``src/inference/``).
"""

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

from parallel.core import ParallelExecutionResult, execute_parallel_tasks

InputType = TypeVar("InputType")
PredictionType = TypeVar("PredictionType")


def run_parallel_predictions(
    predict_func: Callable[[InputType], PredictionType],
    items: Iterable[InputType],
    *,
    max_workers: int | None = None,
    show_progress: bool = True,
) -> ParallelExecutionResult[InputType, PredictionType]:
    """Aplica uma função de predição a múltiplos itens em paralelo.

    Distribui chamadas de inferência entre múltiplas threads, adequado
    para modelos cuja inferência libera o GIL (ex.: ``torch`` em GPU) ou
    para chamadas a APIs externas de modelo.

    Parameters
    ----------
    predict_func : Callable[[InputType], PredictionType]
        Função de predição aplicada a cada item de entrada.
    items : Iterable[InputType]
        Itens (ex.: textos ou registros já vetorizados) a serem preditos.
    max_workers : int | None, optional
        Número máximo de threads usadas, by default None (o executor
        escolhe automaticamente).
    show_progress : bool, optional
        Se ``True``, exibe uma barra de progresso no console, by default
        True.

    Returns
    -------
    ParallelExecutionResult[InputType, PredictionType]
        Predições bem-sucedidas e falhas isoladas por item, cada uma
        preservando o item de entrada que causou o erro.

    Examples
    --------
    >>> resultado = run_parallel_predictions(len, ["ab", "cde"])  # doctest: +SKIP
    >>> sorted(resultado.successes)  # doctest: +SKIP
    [2, 3]
    """
    return execute_parallel_tasks(
        predict_func,
        items,
        executor_class=ThreadPoolExecutor,
        max_workers=max_workers,
        task_description="Inferência paralela",
        show_progress=show_progress,
    )
