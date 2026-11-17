"""Chaves pseudonimizadas de autores para particionamento sem vazamento.

O CSV bronze identifica o **usuário** de cada tweet (``id_screen_name``), mas o
pipeline descarta essa coluna por minimização (LGPD). O custo é que o split por tweet
coloca o mesmo autor em treino e teste. Este módulo recupera só o necessário: uma
tabela ``id -> author_key``, em que ``author_key`` é um HMAC-SHA256 do usuário com
chave secreta do ``.env``. A tabela fica em ``data/silver`` (fora do Git), não contém
o identificador original e permite agrupar por autor sem reidentificá-lo.
"""

import logging
from pathlib import Path

import polars as pl

from exceptions.data import DataNotFoundError
from io_utils.parquet import read_parquet, write_parquet
from utils.privacy import pseudonymize_value

logger = logging.getLogger(__name__)

AUTHOR_KEYS_FILE_NAME = "autores_pseudonimizados.parquet"
AUTHOR_KEY_COLUMN = "author_key"
ID_COLUMN = "id"
RAW_AUTHOR_COLUMN = "id_screen_name"


def build_row_id_expression(row_number_column: str = "row_number") -> pl.Expr:
    """Cria a expressão do ``id`` do tweet a partir da posição da linha no CSV bronze.

    Mantém o mesmo formato (``t0000001``, ``t0000002``...) usado por
    ``scripts/run_tweets_data_pipeline.py``; o ``id`` é a única ponte entre o corpus
    e a tabela de autores.

    Parameters
    ----------
    row_number_column : str, optional
        Coluna com o número da linha (a partir de 1), by default "row_number".

    Returns
    -------
    pl.Expr
        Expressão do ``id`` com 7 dígitos.

    Examples
    --------
    >>> pl.DataFrame({"row_number": [1]}).select(build_row_id_expression()).item()
    't0000001'
    """
    return pl.format("t{}", pl.col(row_number_column).cast(pl.String).str.zfill(7)).alias(ID_COLUMN)


def build_author_key_table(
    raw: pl.DataFrame, *, key: str, author_column: str = RAW_AUTHOR_COLUMN
) -> pl.DataFrame:
    """Gera a tabela ``id -> author_key`` a partir do CSV bronze, sem expor o autor.

    Parameters
    ----------
    raw : pl.DataFrame
        CSV bronze lido na ordem original, com a coluna do autor.
    key : str
        Chave secreta de pseudonimização (``PSEUDONYMIZATION_KEY``).
    author_column : str, optional
        Coluna com o identificador do usuário, by default "id_screen_name".

    Returns
    -------
    pl.DataFrame
        Colunas ``id`` e ``author_key``, uma linha por linha do CSV.

    Raises
    ------
    KeyError
        Se ``author_column`` não existir em ``raw``.

    Examples
    --------
    >>> raw = pl.DataFrame({"id_screen_name": [10, 10, 11]})
    >>> tabela = build_author_key_table(raw, key="k")
    >>> tabela["author_key"].n_unique()
    2
    """
    if author_column not in raw.columns:
        raise KeyError(f"Coluna de autor ausente no CSV bronze: {author_column!r}")
    authors = raw.select(pl.col(author_column).cast(pl.String).alias("author")).with_row_index(
        "row_number", offset=1
    )
    unique_authors = authors["author"].drop_nulls().unique().to_list()
    pseudonyms = pl.DataFrame(
        {
            "author": unique_authors,
            AUTHOR_KEY_COLUMN: [pseudonymize_value(value, key) for value in unique_authors],
        },
        schema={"author": pl.String, AUTHOR_KEY_COLUMN: pl.String},
    )
    return authors.join(pseudonyms, on="author", how="left").select(
        build_row_id_expression(), pl.col(AUTHOR_KEY_COLUMN)
    )


def write_author_key_table(table: pl.DataFrame, directory: Path) -> Path:
    """Grava a tabela de chaves de autores em ``directory``.

    Parameters
    ----------
    table : pl.DataFrame
        Tabela de :func:`build_author_key_table`.
    directory : Path
        Pasta de destino (normalmente ``data/silver``).

    Returns
    -------
    Path
        Caminho do arquivo gravado.

    Examples
    --------
    >>> write_author_key_table(tabela, Path("data/silver"))  # doctest: +SKIP
    """
    file_path = directory / AUTHOR_KEYS_FILE_NAME
    write_parquet(table, file_path)
    logger.info("Chaves pseudonimizadas de autores gravadas: %d linhas.", table.height)
    return file_path


def load_author_key_table(directory: Path) -> pl.DataFrame:
    """Carrega a tabela ``id -> author_key``.

    Parameters
    ----------
    directory : Path
        Pasta que contém o arquivo (normalmente ``data/silver``).

    Returns
    -------
    pl.DataFrame
        Colunas ``id`` e ``author_key``.

    Raises
    ------
    DataNotFoundError
        Se o arquivo não existir (rode ``python scripts/run_tweets_data_pipeline.py
        --steps pseudonymize-authors``).

    Examples
    --------
    >>> load_author_key_table(Path("data/silver"))  # doctest: +SKIP
    """
    file_path = directory / AUTHOR_KEYS_FILE_NAME
    if not file_path.is_file():
        raise DataNotFoundError(
            f"{file_path} — rode `python scripts/run_tweets_data_pipeline.py "
            "--steps pseudonymize-authors`"
        )
    return read_parquet(file_path)
