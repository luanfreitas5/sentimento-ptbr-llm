"""Escrita padronizada de datasets em ``data/interim`` e ``data/processed``.

As funções de escrita validada aplicam o contrato de dados apropriado antes
de persistir o DataFrame, garantindo que apenas dados conformes ao schema
cheguem às etapas seguintes do pipeline (ver CLAUDE.md, "Data Contracts").
"""

import logging
from pathlib import Path

import polars as pl

from io_utils.parquet import write_parquet
from schemas.dataset import validate_labeled_corpus
from schemas.training import validate_training_example
from utils.validation import validate_not_empty_collection

logger = logging.getLogger(__name__)


def write_dataset(dataframe: pl.DataFrame, file_path: Path) -> None:
    """Escreve um DataFrame em Parquet, o formato padrão dos estágios ``interim``/``processed``.

    Parameters
    ----------
    dataframe : pl.DataFrame
        DataFrame a ser escrito, não vazio.
    file_path : Path
        Caminho do arquivo de destino. Diretórios pais ausentes são criados
        automaticamente.

    Returns
    -------
    None

    Raises
    ------
    EmptyDatasetError
        Se ``dataframe`` estiver vazio.

    Examples
    --------
    >>> write_dataset(
    ...     pl.DataFrame({"id": ["1"]}), Path("data/interim/exemplo.parquet")
    ... )  # doctest: +SKIP
    """
    validate_not_empty_collection(dataframe, collection_name=str(file_path))
    write_parquet(dataframe, file_path)
    logger.info("Dataset escrito em: %s (%d linhas)", file_path, dataframe.height)


def write_labeled_corpus(dataframe: pl.DataFrame, file_path: Path) -> None:
    """Valida e escreve o corpus rotulado em ``data/processed``.

    Parameters
    ----------
    dataframe : pl.DataFrame
        DataFrame do corpus rotulado, validado contra
        :class:`schemas.dataset.LabeledCorpusSchema` antes da escrita.
    file_path : Path
        Caminho do arquivo de destino.

    Returns
    -------
    None

    Raises
    ------
    DataValidationError
        Se ``dataframe`` não satisfizer o contrato de dados.
    EmptyDatasetError
        Se ``dataframe`` estiver vazio.

    Examples
    --------
    >>> df = pl.DataFrame({"id": ["1"], "text": ["ótimo"], "sentiment_label": ["positivo"]})
    >>> write_labeled_corpus(df, Path("data/processed/exemplo.parquet"))  # doctest: +SKIP
    """
    write_dataset(validate_labeled_corpus(dataframe), file_path)


def write_training_example_dataset(dataframe: pl.DataFrame, file_path: Path) -> None:
    """Valida e escreve um conjunto de treino/validação/teste já particionado.

    Parameters
    ----------
    dataframe : pl.DataFrame
        DataFrame particionado, validado contra
        :class:`schemas.training.TrainingExampleSchema` antes da escrita.
    file_path : Path
        Caminho do arquivo de destino.

    Returns
    -------
    None

    Raises
    ------
    DataValidationError
        Se ``dataframe`` não satisfizer o contrato de dados.
    EmptyDatasetError
        Se ``dataframe`` estiver vazio.

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "id": ["1"],
    ...         "text": ["ótimo"],
    ...         "sentiment_label": ["positivo"],
    ...         "split": ["treino"],
    ...     }
    ... )
    >>> write_training_example_dataset(df, Path("data/processed/treino.parquet"))  # doctest: +SKIP
    """
    write_dataset(validate_training_example(dataframe), file_path)
