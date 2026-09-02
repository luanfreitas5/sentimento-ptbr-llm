"""Paralelização de etapas de pré-processamento de texto.

Usa múltiplos processos (``ProcessPoolExecutor``) para distribuir a limpeza
e normalização de textos entre os núcleos disponíveis, já que essas
operações (regex, tokenização, remoção de acentos — ver
``src/preprocessing/``) são tipicamente ligadas a CPU, não a I/O.
"""

from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor

from parallel.core import ParallelExecutionResult, execute_parallel_tasks


def run_parallel_text_cleaning(
    clean_text_func: Callable[[str], str],
    texts: Iterable[str],
    *,
    max_workers: int | None = None,
    show_progress: bool = True,
) -> ParallelExecutionResult[str, str]:
    """Aplica uma função de limpeza/normalização a múltiplos textos em paralelo.

    Distribui o processamento entre múltiplos processos, adequado para
    operações ligadas a CPU como remoção de acentos, normalização de
    espaços e aplicação de expressões regulares.

    Parameters
    ----------
    clean_text_func : Callable[[str], str]
        Função de limpeza aplicada a cada texto. Deve ser importável no
        nível de módulo (não local nem lambda), pois é serializada para os
        processos filhos.
    texts : Iterable[str]
        Textos a serem limpos.
    max_workers : int | None, optional
        Número máximo de processos usados, by default None (o executor
        escolhe automaticamente com base nos núcleos disponíveis).
    show_progress : bool, optional
        Se ``True``, exibe uma barra de progresso no console, by default
        True.

    Returns
    -------
    ParallelExecutionResult[str, str]
        Textos limpos com sucesso e falhas isoladas por item, cada uma
        preservando o texto original que causou o erro.

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
    )
