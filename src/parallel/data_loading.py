"""Paralelização da leitura de lotes de arquivos Parquet.

Usa múltiplas threads (``ThreadPoolExecutor``): a leitura/decodificação
Parquet do ``polars`` é implementada em Rust e libera o GIL, então threads
evitam o custo de serializar DataFrames inteiros entre processos ao
carregar o lote de tweets brutos coletados por usuário (um arquivo por
usuário — ver ``data/raw/`` e ``src/data/loader.py``).
"""

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

from io_utils.parquet import read_parquet
from parallel.core import ParallelExecutionResult, execute_parallel_tasks


def run_parallel_parquet_loading(
    file_paths: Iterable[Path],
    *,
    max_workers: int | None = None,
    show_progress: bool = True,
) -> ParallelExecutionResult[Path, pl.DataFrame]:
    """Lê múltiplos arquivos Parquet em paralelo, isolando a falha de cada arquivo.

    Parameters
    ----------
    file_paths : Iterable[Path]
        Caminhos dos arquivos Parquet a serem lidos.
    max_workers : int | None, optional
        Número máximo de threads usadas, by default None (o executor
        escolhe automaticamente).
    show_progress : bool, optional
        Se ``True``, exibe uma barra de progresso no console, by default
        True.

    Returns
    -------
    ParallelExecutionResult[Path, pl.DataFrame]
        DataFrames lidos com sucesso e falhas isoladas por arquivo, cada
        uma preservando o caminho que causou o erro.

    Examples
    --------
    >>> run_parallel_parquet_loading([Path("data/raw/usuario1.parquet")])  # doctest: +SKIP
    """
    return execute_parallel_tasks(
        read_parquet,
        file_paths,
        executor_class=ThreadPoolExecutor,
        max_workers=max_workers,
        task_description="Leitura paralela de lote Parquet",
        show_progress=show_progress,
    )
