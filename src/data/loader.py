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
from polars.exceptions import PolarsError

from constants.columns import ID_COLUMN, TWEET_ID_COLUMN
from exceptions.data import DataError
from io_utils.csv import read_csv
from io_utils.parquet import read_parquet
from parallel.data_loading import run_parallel_parquet_loading
from schemas.dataset import validate_labeled_corpus, validate_raw_tweet_dataset
from schemas.training import validate_training_example
from utils.validation import validate_not_empty_collection

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


def load_raw_tweet_batch(
    directory: Path, *, max_workers: int | None = None, show_progress: bool = True
) -> pl.DataFrame:
    """Carrega, concatena e valida o lote de tweets brutos coletados por usuário.

    Lê em paralelo todos os arquivos ``*.parquet`` do diretório informado
    (um por usuário coletado — ver ``data/raw/``), isolando a falha de
    leitura de um arquivo corrompido sem abortar os demais (ver
    :func:`parallel.data_loading.run_parallel_parquet_loading`). A
    validação contra :class:`schemas.dataset.RawTweetSchema` ocorre uma
    única vez, sobre o lote já concatenado — não por arquivo — de forma
    que um ``tweet_id`` duplicado entre dois arquivos diferentes também
    seja detectado.

    Parameters
    ----------
    directory : Path
        Diretório contendo os arquivos Parquet brutos (``paths.data_raw_dir``).
    max_workers : int | None, optional
        Repassado a :func:`parallel.data_loading.run_parallel_parquet_loading`,
        by default None (o executor escolhe automaticamente).
    show_progress : bool, optional
        Se ``True``, exibe uma barra de progresso no console, by default True.

    Returns
    -------
    pl.DataFrame
        Lote de tweets brutos validado, com ``tweet_id`` renomeado para
        ``id`` (contrato usado pelo restante do pipeline).

    Raises
    ------
    EmptyDatasetError
        Se o diretório não contiver nenhum arquivo ``*.parquet``, ou se
        todos os arquivos encontrados falharem na leitura.
    DataError
        Se os arquivos lidos tiverem esquemas incompatíveis entre si (ex.:
        um arquivo produzido por uma versão antiga de ``ingestion``,
        indevidamente colocado no mesmo diretório do lote bruto por
        usuário), impedindo a concatenação em um único DataFrame.
    DataValidationError
        Se o lote concatenado violar o contrato de dados.

    Examples
    --------
    >>> load_raw_tweet_batch(Path("data/raw"))  # doctest: +SKIP
    """
    file_paths = sorted(directory.glob("*.parquet"))
    validate_not_empty_collection(file_paths, collection_name=str(directory))

    loading_result = run_parallel_parquet_loading(
        file_paths, max_workers=max_workers, show_progress=show_progress
    )
    for failure in loading_result.failures:
        logger.warning(
            "Falha ao ler arquivo Parquet do lote bruto: %s (%s)", failure.item, failure.error
        )
    validate_not_empty_collection(loading_result.successes, collection_name=str(directory))

    try:
        raw_batch = pl.concat(loading_result.successes, how="vertical")
    except PolarsError as exception:
        # Isolar por arquivo aqui exigiria refazer a leitura par a par (ou
        # inspecionar schemas individualmente) só para atribuir a mensagem
        # a um arquivo específico; uma mensagem genérica identificando o
        # diretório é suficiente para orientar o diagnóstico (ver docstring).
        logger.exception(
            "Falha ao concatenar o lote bruto de '%s': esquemas incompatíveis entre arquivos",
            directory,
        )
        raise DataError(
            f"arquivo(s) em '{directory}' têm esquema incompatível com o lote — "
            "verifique se algum arquivo não segue o formato de tweet bruto por usuário",
            context={"directory": str(directory)},
        ) from exception
    validated_batch = validate_raw_tweet_dataset(raw_batch).rename({TWEET_ID_COLUMN: ID_COLUMN})

    logger.info(
        "Lote de tweets brutos carregado: %d/%d arquivo(s), %d linha(s) (%s).",
        len(loading_result.successes),
        len(file_paths),
        validated_batch.height,
        directory,
    )
    return validated_batch


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
        "Conjunto de treino/validação/teste carregado: %s (%d linhas)",
        file_path,
        validated_df.height,
    )
    return validated_df
