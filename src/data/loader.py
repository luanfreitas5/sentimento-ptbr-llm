"""Carregamento de datasets do projeto, com validação de schema.

Cada função de carregamento despacha a leitura para o formato correto
(Parquet/CSV) com base na extensão do arquivo e valida o resultado contra o
contrato de dados (``pandera.polars``) apropriado, falhando cedo se o
arquivo carregado não corresponder ao formato esperado (ver CLAUDE.md,
"Data Contracts").
"""

import logging
from pathlib import Path

import polars as pl

from exceptions.data import DataError
from io_utils.csv import read_csv
from io_utils.parquet import read_parquet
from schemas.dataset import validate_labeled_corpus, validate_raw_tweet_dataset
from schemas.training import validate_training_example

logger = logging.getLogger(__name__)

_PARQUET_SUFFIXES = frozenset({".parquet"})
_CSV_SUFFIXES = frozenset({".csv"})


def read_dataset_file(file_path: Path) -> pl.DataFrame:
    """Lê um arquivo de dataset, despachando pelo formato conforme a extensão.

    Parameters
    ----------
    file_path : Path
        Caminho do arquivo a ser lido (``.csv`` ou ``.parquet``).

    Returns
    -------
    pl.DataFrame
        DataFrame com o conteúdo do arquivo, sem validação de schema.

    Raises
    ------
    DataNotFoundError
        Se o arquivo não existir.
    DataError
        Se a extensão do arquivo não for suportada.

    Examples
    --------
    >>> read_dataset_file(Path("data/processed/exemplo.parquet"))  # doctest: +SKIP
    """
    suffix = file_path.suffix.lower()
    if suffix in _PARQUET_SUFFIXES:
        return read_parquet(file_path)
    if suffix in _CSV_SUFFIXES:
        return read_csv(file_path)
    raise DataError(
        f"Formato de arquivo não suportado: '{suffix}'",
        context={"file_path": str(file_path)},
    )


def load_raw_tweet_dataset(file_path: Path) -> pl.DataFrame:
    """Carrega e valida um dataset de tweets brutos contra :class:`schemas.dataset.RawTweetSchema`.

    Parameters
    ----------
    file_path : Path
        Caminho do arquivo em ``data/raw`` ou ``data/external``.

    Returns
    -------
    pl.DataFrame
        DataFrame validado de tweets brutos.

    Raises
    ------
    DataNotFoundError
        Se o arquivo não existir.
    DataValidationError
        Se o conteúdo não satisfizer o contrato de dados.

    Examples
    --------
    >>> load_raw_tweet_dataset(Path("data/raw/tweets_coletados.parquet"))  # doctest: +SKIP
    """
    validated_df = validate_raw_tweet_dataset(read_dataset_file(file_path))
    logger.info("Dataset de tweets brutos carregado: %s (%d linhas)", file_path, validated_df.height)
    return validated_df


def load_labeled_corpus(file_path: Path) -> pl.DataFrame:
    """Carrega e valida o corpus rotulado contra :class:`schemas.dataset.LabeledCorpusSchema`.

    Parameters
    ----------
    file_path : Path
        Caminho do arquivo em ``data/processed``.

    Returns
    -------
    pl.DataFrame
        DataFrame validado do corpus rotulado.

    Raises
    ------
    DataNotFoundError
        Se o arquivo não existir.
    DataValidationError
        Se o conteúdo não satisfizer o contrato de dados.

    Examples
    --------
    >>> load_labeled_corpus(Path("data/processed/corpus_rotulado.parquet"))  # doctest: +SKIP
    """
    validated_corpus = validate_labeled_corpus(read_dataset_file(file_path))
    logger.info("Corpus rotulado carregado: %s (%d linhas)", file_path, validated_corpus.height)
    return validated_corpus


def load_training_example_dataset(file_path: Path) -> pl.DataFrame:
    """Carrega e valida um conjunto de treino/validação/teste já particionado.

    Parameters
    ----------
    file_path : Path
        Caminho do arquivo (ex.: ``data/processed/treino.parquet``).

    Returns
    -------
    pl.DataFrame
        DataFrame validado contra :class:`schemas.training.TrainingExampleSchema`.

    Raises
    ------
    DataNotFoundError
        Se o arquivo não existir.
    DataValidationError
        Se o conteúdo não satisfizer o contrato de dados.

    Examples
    --------
    >>> load_training_example_dataset(Path("data/processed/treino.parquet"))  # doctest: +SKIP
    """
    validated_df = validate_training_example(read_dataset_file(file_path))
    logger.info(
        "Conjunto de treino/validação/teste carregado: %s (%d linhas)", file_path, validated_df.height
    )
    return validated_df

