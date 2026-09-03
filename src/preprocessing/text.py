"""Normalização de elementos estruturais de tweets em português brasileiro.

Cobre a substituição de URLs, menções e hashtags por tokens semânticos (ver
``src/constants/tokens.py``), além da redução de repetições ortográficas
(ex.: "muitooo" -> "muito") e da normalização de sequências numéricas.
Difere de ``src/utils/text.py``: aqui as operações têm conhecimento do
domínio de tweets/redes sociais, em vez de serem utilidades genéricas de
manipulação de texto.
"""

import re

from constants.regex import (
    HASHTAG_PATTERN,
    MENTION_PATTERN,
    NUMBER_PATTERN,
    REPEATED_CHARACTERS_PATTERN,
    URL_PATTERN,
)
from constants.tokens import HASHTAG_TOKEN, MENTION_TOKEN, NUMBER_TOKEN, URL_TOKEN

_HASHTAG_WORD_PATTERN = re.compile(r"#(\w+)")


def normalize_urls(text: str) -> str:
    """Substitui URLs pelo token semântico ``[URL]``.

    Parameters
    ----------
    text : str
        Texto de entrada, possivelmente contendo URLs.

    Returns
    -------
    str
        Texto com toda URL substituída pelo token :data:`constants.tokens.URL_TOKEN`.

    Examples
    --------
    >>> normalize_urls("confira em https://exemplo.com/promo")
    'confira em [URL]'
    """
    return URL_PATTERN.sub(URL_TOKEN, text)


def normalize_mentions(text: str) -> str:
    """Substitui menções a usuários (ex.: ``@usuario``) pelo token ``[MENCAO]``.

    Parameters
    ----------
    text : str
        Texto de entrada, possivelmente contendo menções.

    Returns
    -------
    str
        Texto com toda menção substituída pelo token :data:`constants.tokens.MENTION_TOKEN`.

    Examples
    --------
    >>> normalize_mentions("bom dia @exemplo")
    'bom dia [MENCAO]'
    """
    return MENTION_PATTERN.sub(MENTION_TOKEN, text)


def normalize_hashtags(text: str, *, keep_word: bool = True) -> str:
    """Normaliza hashtags, preservando a palavra interna ou substituindo por token.

    Parameters
    ----------
    text : str
        Texto de entrada, possivelmente contendo hashtags.
    keep_word : bool, optional
        Se ``True``, remove apenas o caractere ``#`` e preserva a palavra
        (ex.: ``"#recomendo"`` -> ``"recomendo"``), mantendo o sinal
        semântico da hashtag no texto. Se ``False``, substitui a hashtag
        inteira pelo token :data:`constants.tokens.HASHTAG_TOKEN`, by
        default True.

    Returns
    -------
    str
        Texto com as hashtags normalizadas.

    Examples
    --------
    >>> normalize_hashtags("adorei o produto #recomendo")
    'adorei o produto recomendo'
    >>> normalize_hashtags("adorei o produto #recomendo", keep_word=False)
    'adorei o produto [HASHTAG]'
    """
    if keep_word:
        return _HASHTAG_WORD_PATTERN.sub(r"\1", text)
    return HASHTAG_PATTERN.sub(HASHTAG_TOKEN, text)


def normalize_numbers(text: str) -> str:
    """Substitui sequências de dígitos pelo token semântico ``[NUMERO]``.

    Não é aplicada por padrão no pipeline principal (ver
    ``src/preprocessing/pipeline.py``), pois números podem carregar sinal
    relevante para o sentimento (ex.: notas, quantidades); fica disponível
    para uso pontual em outras etapas (ex.: engenharia de features).

    Parameters
    ----------
    text : str
        Texto de entrada, possivelmente contendo números.

    Returns
    -------
    str
        Texto com toda sequência de dígitos substituída pelo token
        :data:`constants.tokens.NUMBER_TOKEN`.

    Examples
    --------
    >>> normalize_numbers("nota 10 para o produto")
    'nota [NUMERO] para o produto'
    """
    return NUMBER_PATTERN.sub(NUMBER_TOKEN, text)


def normalize_repeated_characters(text: str) -> str:
    """Reduz caracteres repetidos mais de duas vezes seguidas a uma única ocorrência.

    Preserva o sinal semântico do alongamento vocálico/consonantal comum em
    redes sociais (ex.: ênfase em "muitooo"), sem inflar o vocabulário com
    variações do mesmo termo.

    Parameters
    ----------
    text : str
        Texto de entrada, possivelmente com caracteres repetidos.

    Returns
    -------
    str
        Texto com as repetições reduzidas a um único caractere.

    Examples
    --------
    >>> normalize_repeated_characters("muitooo bom")
    'muito bom'
    """
    return REPEATED_CHARACTERS_PATTERN.sub(r"\1", text)
