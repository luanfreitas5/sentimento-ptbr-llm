"""Composição das etapas de pré-processamento em um pipeline reprodutível.

Encadeia a limpeza (``src/preprocessing/cleaning.py``), a normalização de
elementos estruturais (``src/preprocessing/text.py``) e de emojis
(``src/preprocessing/emojis.py``) em uma única função por texto e aplica o
resultado a um corpus inteiro, junto aos critérios de inclusão/exclusão
(``src/preprocessing/filtering.py``) e, opcionalmente, à tokenização
(``src/preprocessing/tokenization.py``), produzindo o dataset intermediário
usado pelas etapas seguintes (rotulagem e extração de features).
"""

import logging

import polars as pl

from exceptions.pipeline import PipelineStageError
from preprocessing.cleaning import clean_tweet_text
from preprocessing.emojis import normalize_emojis
from preprocessing.filtering import filter_by_inclusion_criteria
from preprocessing.text import (
    normalize_hashtags,
    normalize_mentions,
    normalize_repeated_characters,
    normalize_urls,
)
from preprocessing.tokenization import tokenize_and_normalize
from utils.text import normalize_whitespace
from utils.validation import validate_not_empty_collection

logger = logging.getLogger(__name__)


def normalize_tweet_text(text: str, *, keep_hashtag_word: bool = True) -> str:
    """Aplica a sequência completa de limpeza e normalização a um único tweet.

    Ordem das etapas: remoção do marcador de retweet, substituição de URLs
    e menções por tokens, normalização de hashtags, mapeamento de emojis
    para tokens semânticos, redução de repetições ortográficas e
    normalização final de espaçamento. A remoção do marcador de retweet
    precisa ocorrer antes da normalização de menções, pois o padrão de
    retweet depende do ``@usuario`` original no início do texto.

    Parameters
    ----------
    text : str
        Texto bruto de um tweet.
    keep_hashtag_word : bool, optional
        Repassado a :func:`preprocessing.text.normalize_hashtags`, by
        default True.

    Returns
    -------
    str
        Texto normalizado, pronto para tokenização ou extração de features.

    Examples
    --------
    >>> normalize_tweet_text("RT @exemplo: amei o produto!! 😍 #recomendo https://exemplo.com")
    'amei o produto!! [EMOJI_POSITIVO] recomendo [URL]'
    """
    normalized = clean_tweet_text(text)
    normalized = normalize_urls(normalized)
    normalized = normalize_mentions(normalized)
    normalized = normalize_hashtags(normalized, keep_word=keep_hashtag_word)
    normalized = normalize_emojis(normalized)
    normalized = normalize_repeated_characters(normalized)
    return normalize_whitespace(normalized)


def _normalize_row_text(text: str, *, keep_hashtag_word: bool) -> str:
    """Normaliza um texto individual, convertendo falhas em ``PipelineStageError``.

    Parameters
    ----------
    text : str
        Texto bruto de um tweet.
    keep_hashtag_word : bool
        Repassado a :func:`normalize_tweet_text`.

    Returns
    -------
    str
        Texto normalizado.

    Raises
    ------
    PipelineStageError
        Se a normalização falhar para o texto informado.
    """
    try:
        return normalize_tweet_text(text, keep_hashtag_word=keep_hashtag_word)
    except Exception as exception:  # captura ampla e proposital: isola a falha de uma linha
        logger.exception("Falha ao normalizar o texto de um tweet")
        raise PipelineStageError(
            stage_name="normalizacao_texto", detail=str(exception)
        ) from exception


def run_preprocessing_pipeline(
    dataframe: pl.DataFrame,
    *,
    text_column: str = "text",
    normalized_text_column: str = "text_normalized",
    tokens_column: str | None = None,
    keep_hashtag_word: bool = True,
    expand_slang: bool = True,
    apply_negation_marking: bool = True,
    apply_inclusion_filters: bool = True,
    minimum_characters: int = 5,
    minimum_words: int = 2,
    minimum_portuguese_ratio: float = 0.15,
    max_repeated_word_ratio: float = 0.5,
    drop_duplicate_text: bool = True,
) -> pl.DataFrame:
    """Executa o pipeline reprodutível de pré-processamento sobre um corpus de tweets.

    Adiciona ``normalized_text_column`` com o resultado de
    :func:`normalize_tweet_text` aplicado a cada linha e, opcionalmente,
    filtra o corpus pelos critérios de inclusão de
    ``src/preprocessing/filtering.py`` (conteúdo mínimo, idioma provável,
    ausência de spam e duplicatas) e adiciona uma coluna de tokens via
    :func:`preprocessing.tokenization.tokenize_and_normalize`.

    Parameters
    ----------
    dataframe : pl.DataFrame
        Corpus de entrada, contendo ao menos ``text_column``. Não vazio.
    text_column : str, optional
        Nome da coluna com o texto bruto, by default "text".
    normalized_text_column : str, optional
        Nome da coluna a ser criada com o texto normalizado, by default
        "text_normalized".
    tokens_column : str | None, optional
        Nome da coluna de tokens a ser criada, aplicada após os filtros de
        inclusão (quando habilitados). Se ``None``, a tokenização não é
        executada, by default None.
    keep_hashtag_word : bool, optional
        Repassado a :func:`normalize_tweet_text`, by default True.
    expand_slang : bool, optional
        Repassado a :func:`preprocessing.tokenization.tokenize_and_normalize`,
        usado apenas quando ``tokens_column`` é informado, by default True.
    apply_negation_marking : bool, optional
        Repassado a :func:`preprocessing.tokenization.tokenize_and_normalize`,
        usado apenas quando ``tokens_column`` é informado, by default True.
    apply_inclusion_filters : bool, optional
        Se ``True``, aplica
        :func:`preprocessing.filtering.filter_by_inclusion_criteria` sobre
        ``normalized_text_column`` após a normalização, by default True.
    minimum_characters : int, optional
        Repassado ao filtro de conteúdo mínimo, by default 5.
    minimum_words : int, optional
        Repassado ao filtro de conteúdo mínimo, by default 2.
    minimum_portuguese_ratio : float, optional
        Repassado ao filtro de idioma, by default 0.15.
    max_repeated_word_ratio : float, optional
        Repassado ao filtro de spam, by default 0.5.
    drop_duplicate_text : bool, optional
        Repassado ao filtro de duplicatas, by default True.

    Returns
    -------
    pl.DataFrame
        Corpus com a coluna de texto normalizado (e, quando solicitado, a
        coluna de tokens), filtrado pelos critérios de inclusão quando
        ``apply_inclusion_filters`` for ``True``.

    Raises
    ------
    EmptyDatasetError
        Se ``dataframe`` estiver vazio.
    PipelineStageError
        Se a normalização falhar para alguma linha do corpus.

    Examples
    --------
    >>> df = pl.DataFrame({"id": ["1", "2"], "text": ["RT @a: muito bom!! 😍", "RT @b: oi"]})
    >>> resultado = run_preprocessing_pipeline(df)
    >>> resultado["text_normalized"].to_list()
    ['muito bom!! [EMOJI_POSITIVO]']
    """
    validate_not_empty_collection(dataframe, collection_name="dataframe")

    normalized_texts = [
        _normalize_row_text(text, keep_hashtag_word=keep_hashtag_word)
        for text in dataframe[text_column].to_list()
    ]
    result = dataframe.with_columns(pl.Series(normalized_text_column, normalized_texts))

    if apply_inclusion_filters:
        result = filter_by_inclusion_criteria(
            result,
            text_column=normalized_text_column,
            minimum_characters=minimum_characters,
            minimum_words=minimum_words,
            minimum_portuguese_ratio=minimum_portuguese_ratio,
            max_repeated_word_ratio=max_repeated_word_ratio,
            drop_duplicate_text=drop_duplicate_text,
        )

    if tokens_column is not None:
        token_lists = [
            tokenize_and_normalize(
                text, expand_slang=expand_slang, apply_negation_marking=apply_negation_marking
            )
            for text in result[normalized_text_column].to_list()
        ]
        result = result.with_columns(pl.Series(tokens_column, token_lists, dtype=pl.List(pl.Utf8)))

    logger.info(
        "Pipeline de pré-processamento concluído: %d/%d linha(s) mantida(s)",
        result.height,
        dataframe.height,
    )
    return result
