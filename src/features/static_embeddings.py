"""Embeddings estáticos (FastText) de textos em português brasileiro.

Implementa a Seção 4.4 do documento mestre: representação densa por
documento a partir de vetores de palavra pré-treinados/treinados em pt-BR
(FastText), calculada como a média dos vetores das palavras conhecidas do
texto — abordagem clássica de "bag of word vectors", mais simples que os
embeddings contextuais (``src/features/contextual_embeddings.py``) e usada
como baseline de comparação (ver notebook ``03_engenharia_features.ipynb``).

A biblioteca ``fasttext`` é uma dependência pesada e opcional, ainda não
instalada no projeto (ver ``pyproject.toml``): o import ocorre de forma
tardia, dentro de :func:`load_fasttext_model`, para que o restante do
módulo permaneça importável sem ela.
"""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import polars as pl
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from exceptions.model import ModelError
from utils.validation import validate_file_exists, validate_not_empty_collection

logger = logging.getLogger(__name__)


class StaticEmbeddingModel(Protocol):
    """Interface mínima de um modelo de embeddings estáticos por palavra.

    Compatível com ``fasttext.FastText._FastText`` (retornado por
    ``fasttext.load_model``), permitindo injetar um modelo real ou um dublê
    de teste em :func:`extract_static_embeddings`, sem acoplamento à
    biblioteca ``fasttext``.
    """

    def get_word_vector(self, word: str) -> np.ndarray:
        """Retorna o vetor denso de uma palavra.

        Parameters
        ----------
        word : str
            Palavra de entrada.

        Returns
        -------
        np.ndarray
            Vetor denso da palavra, com a dimensão fixa do modelo.
        """
        ...

    def get_dimension(self) -> int:
        """Retorna a dimensão dos vetores produzidos pelo modelo.

        Returns
        -------
        int
            Dimensão dos vetores.
        """
        ...


def load_fasttext_model(model_path: Path) -> StaticEmbeddingModel:
    """Carrega um modelo FastText pré-treinado a partir do disco.

    Parameters
    ----------
    model_path : Path
        Caminho do arquivo do modelo (ex.: ``cc.pt.300.bin``, ver
        ``configs/model_params.yaml -> embeddings.static.model_name``).

    Returns
    -------
    StaticEmbeddingModel
        Modelo FastText carregado.

    Raises
    ------
    DataNotFoundError
        Se ``model_path`` não existir.
    ModelError
        Se a biblioteca ``fasttext`` não estiver instalada.

    Examples
    --------
    >>> load_fasttext_model(Path("models/fasttext/cc.pt.300.bin"))  # doctest: +SKIP
    """
    validate_file_exists(model_path)
    try:
        import fasttext
    except ImportError as exception:
        raise ModelError(
            "A biblioteca 'fasttext' não está instalada. Instale com "
            "`uv add fasttext-wheel` para extrair embeddings estáticos."
        ) from exception
    return fasttext.load_model(str(model_path))


def compute_document_embedding(tokens: Sequence[str], model: StaticEmbeddingModel) -> np.ndarray:
    """Calcula o embedding de um documento pela média dos vetores de suas palavras.

    Parameters
    ----------
    tokens : Sequence[str]
        Tokens do documento, tipicamente já normalizados por
        ``src/preprocessing/tokenization.py``.
    model : StaticEmbeddingModel
        Modelo de embeddings estáticos, via :func:`load_fasttext_model`.

    Returns
    -------
    np.ndarray
        Vetor médio dos tokens, ou vetor de zeros (na dimensão do modelo)
        se ``tokens`` estiver vazio.

    Examples
    --------
    >>> class _FakeModel:
    ...     def get_word_vector(self, word):
    ...         return np.array([1.0, 0.0]) if word == "bom" else np.array([0.0, 1.0])
    ...
    ...     def get_dimension(self):
    ...         return 2
    >>> compute_document_embedding(["bom", "dia"], _FakeModel())
    array([0.5, 0.5])
    """
    if not tokens:
        return np.zeros(model.get_dimension())
    word_vectors = [model.get_word_vector(token) for token in tokens]
    return np.mean(word_vectors, axis=0)


def extract_static_embeddings(
    dataframe: pl.DataFrame,
    model: StaticEmbeddingModel,
    *,
    id_column: str = "id",
    text_column: str = "text",
) -> pl.DataFrame:
    """Extrai embeddings estáticos para cada documento de um corpus.

    Parameters
    ----------
    dataframe : pl.DataFrame
        Corpus de entrada, contendo ao menos ``id_column`` e
        ``text_column``. Não vazio.
    model : StaticEmbeddingModel
        Modelo de embeddings estáticos, via :func:`load_fasttext_model`.
    id_column : str, optional
        Nome da coluna identificadora de cada documento, by default "id".
    text_column : str, optional
        Nome da coluna de texto (tokens separados por espaço), by default
        "text".

    Returns
    -------
    pl.DataFrame
        DataFrame largo com ``id_column`` e uma coluna ``embedding_<i>`` por
        dimensão do modelo.

    Raises
    ------
    EmptyDatasetError
        Se ``dataframe`` estiver vazio.

    Examples
    --------
    >>> class _FakeModel:
    ...     def get_word_vector(self, word):
    ...         return np.array([1.0, 0.0]) if word == "bom" else np.array([0.0, 1.0])
    ...
    ...     def get_dimension(self):
    ...         return 2
    >>> df = pl.DataFrame({"id": ["1"], "text": ["bom dia"]})
    >>> extract_static_embeddings(df, _FakeModel()).to_dicts()
    [{'id': '1', 'embedding_0': 0.5, 'embedding_1': 0.5}]
    """
    validate_not_empty_collection(dataframe, collection_name="dataframe")

    document_ids = dataframe[id_column].to_list()
    tokenized_documents = [text.split() for text in dataframe[text_column].to_list()]
    dimension = model.get_dimension()

    embeddings = np.zeros((len(tokenized_documents), dimension))
    progress_columns = (
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    with Progress(*progress_columns) as progress:
        task = progress.add_task("Extraindo embeddings estáticos", total=len(tokenized_documents))
        for row_index, tokens in enumerate(tokenized_documents):
            embeddings[row_index] = compute_document_embedding(tokens, model)
            progress.advance(task)

    embedding_columns = {
        f"embedding_{dimension_index}": embeddings[:, dimension_index]
        for dimension_index in range(dimension)
    }
    logger.info(
        "Embeddings estáticos extraídos: %d documento(s), dimensão %d.",
        len(tokenized_documents),
        dimension,
    )
    return pl.DataFrame({id_column: document_ids, **embedding_columns})
