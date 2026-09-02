"""Leitura e escrita de arquivos Parquet como DataFrames Polars.

Parquet é o formato preferido para os estágios ``interim`` e ``processed``
(ver ``configs/paths.yaml``): colunar, tipado e eficiente para leitura
parcial.
"""

import logging
from pathlib import Path
from typing import Any

import polars as pl

from utils.validation import validate_file_exists

logger = logging.getLogger(__name__)


def read_parquet(file_path: Path, **kwargs: Any) -> pl.DataFrame:
    """Lê um arquivo Parquet como um DataFrame Polars.

    Parameters
    ----------
    file_path : Path
        Caminho do arquivo Parquet a ser lido.
    **kwargs : Any
        Argumentos adicionais repassados a :func:`polars.read_parquet`.

    Returns
    -------
    pl.DataFrame
        DataFrame com o conteúdo do arquivo Parquet.

    Raises
    ------
    DataNotFoundError
        Se o arquivo não existir.

    Examples
    --------
    >>> read_parquet(Path("data/processed/corpus_treino.parquet"))  # doctest: +SKIP
    """
    validate_file_exists(file_path)
    dataframe = pl.read_parquet(file_path, **kwargs)
    logger.debug("Arquivo Parquet lido: %s (%d linhas)", file_path, dataframe.height)
    return dataframe


def write_parquet(dataframe: pl.DataFrame, file_path: Path, **kwargs: Any) -> None:
    """Escreve um DataFrame Polars em um arquivo Parquet, criando diretórios pais se necessário.

    Parameters
    ----------
    dataframe : pl.DataFrame
        DataFrame a ser escrito.
    file_path : Path
        Caminho do arquivo Parquet de destino.
    **kwargs : Any
        Argumentos adicionais repassados a :meth:`polars.DataFrame.write_parquet`.

    Returns
    -------
    None

    Examples
    --------
    >>> write_parquet(
    ...     pl.DataFrame({"a": [1, 2]}), Path("data/processed/exemplo.parquet")
    ... )  # doctest: +SKIP
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.write_parquet(file_path, **kwargs)
    logger.debug("Arquivo Parquet escrito: %s (%d linhas)", file_path, dataframe.height)
