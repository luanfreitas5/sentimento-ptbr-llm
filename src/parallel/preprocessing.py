"""Paralelização de etapas de pré-processamento de texto.

Usa múltiplos processos (``ProcessPoolExecutor``) para distribuir a limpeza
e normalização de textos entre os núcleos disponíveis, já que essas
operações (regex, tokenização, remoção de acentos — ver
``src/preprocessing/``) são tipicamente ligadas a CPU, não a I/O.
"""

from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from typing import TypeVar

from parallel.core import ParallelExecutionResult, execute_parallel_tasks

ItemType = TypeVar("ItemType")
ResultType = TypeVar("ResultType")


def run_parallel_text_cleaning(
    clean_text_func: Callable[[ItemType], ResultType],
    texts: Iterable[ItemType],
    *,
    max_workers: int | None = None,
    show_progress: bool = True,
    chunk_size: int | None = None,
) -> ParallelExecutionResult[ItemType, ResultType]:
    """Aplica uma função de limpeza/normalização a múltiplos itens em paralelo.

    Distribui o processamento entre múltiplos processos, adequado para
    operações ligadas a CPU como remoção de acentos, normalização de
    espaços e aplicação de expressões regulares. Aceita tanto textos
    simples quanto itens indexados (``tuple[int, str]``), usados quando o
    chamador precisa realinhar os resultados à ordem original de entrada
    (ver ``src/preprocessing/pipeline.py``).

    Parameters
    ----------
    clean_text_func : Callable[[ItemType], ResultType]
        Função de limpeza aplicada a cada item. Deve ser importável no
        nível de módulo (não local nem lambda), pois é serializada para os
        processos filhos.
    texts : Iterable[ItemType]
        Itens a serem limpos (textos simples, ou pares indexados).
    max_workers : int | None, optional
        Número máximo de processos usados, by default None (o executor
        escolhe automaticamente com base nos núcleos disponíveis).
    show_progress : bool, optional
        Se ``True``, exibe uma barra de progresso no console, by default
        True.
    chunk_size : int | None, optional
        Repassado a :func:`parallel.core.execute_parallel_tasks`: agrupa
        os itens em lotes desse tamanho, uma ``Future`` por lote, reduzindo
        o overhead de IPC por item em corpora grandes, by default None
        (uma ``Future`` por item, comportamento original).

    Returns
    -------
    ParallelExecutionResult[ItemType, ResultType]
        Itens limpos com sucesso e falhas isoladas por item, cada uma
        preservando o item original que causou o erro.

    Examples
    --------
    >>> resultado = run_parallel_text_cleaning(str.strip, ["  a  ", " b "])  # doctest: +SKIP
    >>> sorted(resultado.successes)  # doctest: +SKIP
    ['a', 'b']
    """
    return execute_parallel_tasks(
        clean_text_func,
        texts,
        executor_class=ProcessPoolExecutor,
        max_workers=max_workers,
        task_description="Limpeza paralela de texto",
        show_progress=show_progress,
        chunk_size=chunk_size,
    )
