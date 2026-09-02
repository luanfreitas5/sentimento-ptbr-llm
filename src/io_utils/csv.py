"""Leitura e escrita de arquivos CSV como DataFrames Polars.

``polars`` é a biblioteca de manipulação de dados preferida do projeto (ver
CLAUDE.md, "Core Stack"), usada aqui em vez de ``pandas`` por desempenho.
"""

import logging
from pathlib import Path
from typing import Any

import polars as pl

from utils.validation import validate_file_exists

logger = logging.getLogger(__name__)


def read_csv(file_path: Path, **kwargs: Any) -> pl.DataFrame:
    """Lê um arquivo CSV como um DataFrame Polars.

    Parameters
    ----------
    file_path : Path
        Caminho do arquivo CSV a ser lido.
    **kwargs : Any
        Argumentos adicionais repassados a :func:`polars.read_csv`.

    Returns
    -------
    pl.DataFrame
        DataFrame com o conteúdo do arquivo CSV.

    Raises
    ------
    DataNotFoundError
        Se o arquivo não existir.

    Examples
    --------
    >>> read_csv(Path("data/raw/exemplo.csv"))  # doctest: +SKIP
    """
    validate_file_exists(file_path)
    dataframe = pl.read_csv(file_path, **kwargs)
    logger.debug("Arquivo CSV lido: %s (%d linhas)", file_path, dataframe.height)
    return dataframe


def write_csv(dataframe: pl.DataFrame, file_path: Path, **kwargs: Any) -> None:
    """Escreve um DataFrame Polars em um arquivo CSV, criando diretórios pais se necessário.

    Parameters
    ----------
    dataframe : pl.DataFrame
        DataFrame a ser escrito.
    file_path : Path
        Caminho do arquivo CSV de destino.
    **kwargs : Any
        Argumentos adicionais repassados a :meth:`polars.DataFrame.write_csv`.

    Returns
    -------
    None

    Examples
    --------
    >>> write_csv(pl.DataFrame({"a": [1, 2]}), Path("data/interim/exemplo.csv"))  # doctest: +SKIP
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.write_csv(file_path, **kwargs)
    logger.debug("Arquivo CSV escrito: %s (%d linhas)", file_path, dataframe.height)
