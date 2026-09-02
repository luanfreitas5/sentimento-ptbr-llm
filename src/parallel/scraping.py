"""Paralelização da coleta de dados (scraping).

Usa múltiplas threads (``ThreadPoolExecutor``), adequado para requisições de
rede concorrentes (ex.: múltiplas buscas via ``twscrape``, ver
``src/data/downloader.py``), que são ligadas a I/O.
"""

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

from parallel.core import ParallelExecutionResult, execute_parallel_tasks

QueryType = TypeVar("QueryType")
ScrapedDataType = TypeVar("ScrapedDataType")


def run_parallel_scraping(
    scrape_func: Callable[[QueryType], ScrapedDataType],
    queries: Iterable[QueryType],
    *,
    max_workers: int | None = None,
    show_progress: bool = True,
) -> ParallelExecutionResult[QueryType, ScrapedDataType]:
    """Executa múltiplas buscas/coletas de dados em paralelo.

    Distribui chamadas de coleta (ex.: buscas por termo ou usuário via
    ``twscrape``) entre múltiplas threads, adequado para requisições de
    rede concorrentes.

    Parameters
    ----------
    scrape_func : Callable[[QueryType], ScrapedDataType]
        Função que executa uma única coleta a partir de uma consulta (ex.:
        termo de busca, nome de usuário).
    queries : Iterable[QueryType]
        Consultas a serem executadas.
    max_workers : int | None, optional
        Número máximo de threads usadas, by default None (o executor
        escolhe automaticamente).
    show_progress : bool, optional
        Se ``True``, exibe uma barra de progresso no console, by default
        True.

    Returns
    -------
    ParallelExecutionResult[QueryType, ScrapedDataType]
        Dados coletados com sucesso e falhas isoladas por consulta, cada
        uma preservando a consulta que causou o erro (útil para nova
        tentativa).

    Examples
    --------
    >>> resultado = run_parallel_scraping(str.upper, ["python", "nlp"])  # doctest: +SKIP
    """
    return execute_parallel_tasks(
        scrape_func,
        queries,
        executor_class=ThreadPoolExecutor,
        max_workers=max_workers,
        task_description="Coleta paralela de dados",
        show_progress=show_progress,
    )
