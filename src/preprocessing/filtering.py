"""Filtragem de linhas de um corpus de tweets segundo critérios de inclusão/exclusão.

Aplica, em nível de DataFrame, os predicados definidos em
``src/preprocessing/cleaning.py`` (conteúdo mínimo, idioma provável,
indícios de spam) e a remoção de textos duplicados, registrando em log
quantas linhas cada critério remove. Critérios documentados na Fase 6 do
plano de elaboração (``PLANO-ELABORACAO.md``).
"""

import logging

import polars as pl

from preprocessing.cleaning import (
    is_minimum_length_content,
    is_probable_portuguese_text,
    is_spam_like,
)

logger = logging.getLogger(__name__)


def filter_by_minimum_length(
    dataframe: pl.DataFrame,
    *,
    text_column: str,
    minimum_characters: int = 5,
    minimum_words: int = 2,
) -> pl.DataFrame:
    """Remove linhas cujo texto não atende ao conteúdo mínimo.

    Parameters
    ----------
    dataframe : pl.DataFrame
        DataFrame de entrada, contendo ao menos ``text_column``.
    text_column : str
        Nome da coluna de texto usada na avaliação do critério.
    minimum_characters : int, optional
        Número mínimo de caracteres exigido, by default 5.
    minimum_words : int, optional
        Número mínimo de palavras exigido, by default 2.

    Returns
    -------
    pl.DataFrame
        Subconjunto do DataFrame original com apenas as linhas que atendem
        ao conteúdo mínimo.

    Examples
    --------
    >>> df = pl.DataFrame({"text": ["bom", "muito bom mesmo"]})
    >>> filter_by_minimum_length(df, text_column="text").height
    1
    """
    mask = [
        is_minimum_length_content(
            text, minimum_characters=minimum_characters, minimum_words=minimum_words
        )
        for text in dataframe[text_column].to_list()
    ]
    filtered = dataframe.filter(pl.Series(mask))
    logger.info(
        "Filtro de conteúdo mínimo: %d/%d linha(s) mantida(s)", filtered.height, dataframe.height
    )
    return filtered


def filter_by_portuguese_language(
    dataframe: pl.DataFrame, *, text_column: str, minimum_ratio: float = 0.15
) -> pl.DataFrame:
    """Remove linhas cujo texto provavelmente não está em português (heurística).

    Parameters
    ----------
    dataframe : pl.DataFrame
        DataFrame de entrada, contendo ao menos ``text_column``.
    text_column : str
        Nome da coluna de texto usada na avaliação do critério.
    minimum_ratio : float, optional
        Proporção mínima de stopwords em português exigida, by default 0.15.

    Returns
    -------
    pl.DataFrame
        Subconjunto do DataFrame original com apenas as linhas
        provavelmente escritas em português.

    Examples
    --------
    >>> df = pl.DataFrame({"text": ["o produto que eu comprei é muito bom", "great product"]})
    >>> filter_by_portuguese_language(df, text_column="text").height
    1
    """
    mask = [
        is_probable_portuguese_text(text, minimum_ratio=minimum_ratio)
        for text in dataframe[text_column].to_list()
    ]
    filtered = dataframe.filter(pl.Series(mask))
    logger.info(
        "Filtro de idioma (heurística pt-BR): %d/%d linha(s) mantida(s)",
        filtered.height,
        dataframe.height,
    )
    return filtered


def filter_spam_like_rows(
    dataframe: pl.DataFrame, *, text_column: str, max_repeated_word_ratio: float = 0.5
) -> pl.DataFrame:
    """Remove linhas com indícios de spam por repetição excessiva de uma palavra.

    Parameters
    ----------
    dataframe : pl.DataFrame
        DataFrame de entrada, contendo ao menos ``text_column``.
    text_column : str
        Nome da coluna de texto usada na avaliação do critério.
    max_repeated_word_ratio : float, optional
        Proporção máxima tolerada da palavra mais frequente, by default 0.5.

    Returns
    -------
    pl.DataFrame
        Subconjunto do DataFrame original sem as linhas identificadas como spam.

    Examples
    --------
    >>> df = pl.DataFrame({"text": ["compre compre compre compre agora", "muito bom o produto"]})
    >>> filter_spam_like_rows(df, text_column="text").height
    1
    """
    mask = [
        not is_spam_like(text, max_repeated_word_ratio=max_repeated_word_ratio)
        for text in dataframe[text_column].to_list()
    ]
    filtered = dataframe.filter(pl.Series(mask))
    logger.info("Filtro de spam: %d/%d linha(s) mantida(s)", filtered.height, dataframe.height)
    return filtered


def remove_duplicate_text_rows(dataframe: pl.DataFrame, *, text_column: str) -> pl.DataFrame:
    """Remove linhas com texto duplicado, mantendo a primeira ocorrência.

    Parameters
    ----------
    dataframe : pl.DataFrame
        DataFrame de entrada, contendo ao menos ``text_column``.
    text_column : str
        Nome da coluna de texto usada para identificar duplicatas.

    Returns
    -------
    pl.DataFrame
        Subconjunto do DataFrame original sem linhas de texto duplicado.

    Examples
    --------
    >>> df = pl.DataFrame({"text": ["ótimo", "ótimo", "péssimo"]})
    >>> remove_duplicate_text_rows(df, text_column="text").height
    2
    """
    deduplicated = dataframe.unique(subset=[text_column], keep="first", maintain_order=True)
    logger.info(
        "Remoção de duplicatas: %d/%d linha(s) mantida(s)", deduplicated.height, dataframe.height
    )
    return deduplicated


def filter_by_inclusion_criteria(
    dataframe: pl.DataFrame,
    *,
    text_column: str,
    minimum_characters: int = 5,
    minimum_words: int = 2,
    minimum_portuguese_ratio: float = 0.15,
    max_repeated_word_ratio: float = 0.5,
    drop_duplicate_text: bool = True,
) -> pl.DataFrame:
    """Aplica em sequência todos os critérios de inclusão/exclusão do corpus.

    Critérios aplicados, na ordem (documentados na Fase 6 do plano de
    elaboração): conteúdo mínimo, idioma provável em português, ausência de
    indícios de spam e, opcionalmente, remoção de texto duplicado.

    Parameters
    ----------
    dataframe : pl.DataFrame
        DataFrame de entrada, contendo ao menos ``text_column``.
    text_column : str
        Nome da coluna de texto usada na avaliação dos critérios.
    minimum_characters : int, optional
        Número mínimo de caracteres exigido, by default 5.
    minimum_words : int, optional
        Número mínimo de palavras exigido, by default 2.
    minimum_portuguese_ratio : float, optional
        Proporção mínima de stopwords em português exigida, by default 0.15.
    max_repeated_word_ratio : float, optional
        Proporção máxima tolerada da palavra mais frequente, by default 0.5.
    drop_duplicate_text : bool, optional
        Se ``True``, remove linhas de texto duplicado após os demais
        critérios, by default True.

    Returns
    -------
    pl.DataFrame
        Subconjunto do DataFrame original que atende a todos os critérios
        de inclusão selecionados.

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "text": [
    ...             "muito bom o produto",
    ...             "ok",
    ...             "compre compre compre compre agora",
    ...         ]
    ...     }
    ... )
    >>> filter_by_inclusion_criteria(df, text_column="text").height
    1
    """
    filtered = filter_by_minimum_length(
        dataframe,
        text_column=text_column,
        minimum_characters=minimum_characters,
        minimum_words=minimum_words,
    )
    filtered = filter_by_portuguese_language(
        filtered, text_column=text_column, minimum_ratio=minimum_portuguese_ratio
    )
    filtered = filter_spam_like_rows(
        filtered, text_column=text_column, max_repeated_word_ratio=max_repeated_word_ratio
    )
    if drop_duplicate_text:
        filtered = remove_duplicate_text_rows(filtered, text_column=text_column)
    logger.info(
        "Critérios de inclusão aplicados: %d/%d linha(s) mantida(s) no total",
        filtered.height,
        dataframe.height,
    )
    return filtered
