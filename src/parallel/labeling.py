"""Paralelização da rotulagem automática de sentimento.

Usa múltiplos processos (``ProcessPoolExecutor``), adequado para o
rotulador heurístico-lexical (``src/labeling/automatic.py``): regex e
contagem léxica são operações ligadas a CPU, não a I/O.
"""

from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from typing import TypeVar

from parallel.core import ParallelExecutionResult, execute_parallel_tasks

ItemType = TypeVar("ItemType")
ResultType = TypeVar("ResultType")


def run_parallel_sentiment_labeling(
    label_func: Callable[[ItemType], ResultType],
    items: Iterable[ItemType],
    *,
    max_workers: int | None = None,
    show_progress: bool = True,
) -> ParallelExecutionResult[ItemType, ResultType]:
    """Aplica uma função de rotulagem de sentimento a múltiplos itens em paralelo.

    Parameters
    ----------
    label_func : Callable[[ItemType], ResultType]
        Função de rotulagem aplicada a cada item. Deve ser importável no
        nível de módulo (não local nem lambda), pois é serializada para os
        processos filhos.
    items : Iterable[ItemType]
        Itens a serem rotulados (textos simples, ou pares indexados — ver
        ``src/labeling/automatic.py``).
    max_workers : int | None, optional
        Número máximo de processos usados, by default None (o executor
        escolhe automaticamente).
    show_progress : bool, optional
        Se ``True``, exibe uma barra de progresso no console, by default
        True.

    Returns
    -------
    ParallelExecutionResult[ItemType, ResultType]
        Itens rotulados com sucesso e falhas isoladas por item, cada uma
        preservando o item original que causou o erro.

    Examples
    --------
    >>> resultado = run_parallel_sentiment_labeling(str.upper, ["a", "b"])  # doctest: +SKIP
    """
    return execute_parallel_tasks(
        label_func,
        items,
        executor_class=ProcessPoolExecutor,
        max_workers=max_workers,
        task_description="Rotulagem paralela de sentimento",
        show_progress=show_progress,
    )
