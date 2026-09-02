"""Funções utilitárias genéricas de manipulação de texto.

Diferem de ``src/preprocessing/text.py``: este módulo oferece operações de
propósito geral (normalização de espaços, remoção de acentos, truncamento),
sem conhecimento do domínio de análise de sentimentos.
"""

import unicodedata

from constants.regex import MULTIPLE_WHITESPACE_PATTERN


def normalize_whitespace(text: str) -> str:
    """Reduz sequências de espaços em branco a um único espaço e remove bordas.

    Parameters
    ----------
    text : str
        Texto de entrada.

    Returns
    -------
    str
        Texto com espaçamento normalizado.

    Examples
    --------
    >>> normalize_whitespace("  ola   mundo\\n")
    'ola mundo'
    """
    return MULTIPLE_WHITESPACE_PATTERN.sub(" ", text).strip()


def remove_accents(text: str) -> str:
    """Remove acentos e diacríticos de um texto, preservando os caracteres-base.

    Parameters
    ----------
    text : str
        Texto de entrada, possivelmente acentuado.

    Returns
    -------
    str
        Texto sem acentos.

    Examples
    --------
    >>> remove_accents("análise de sentimentos")
    'analise de sentimentos'
    """
    texto_normalizado = unicodedata.normalize("NFKD", text)
    return "".join(
        caractere for caractere in texto_normalizado if not unicodedata.combining(caractere)
    )


def truncate_text(text: str, max_length: int, *, suffix: str = "...") -> str:
    """Trunca um texto para um comprimento máximo, anexando um sufixo indicativo.

    Parameters
    ----------
    text : str
        Texto de entrada.
    max_length : int
        Comprimento máximo permitido para o texto final (incluindo o sufixo).
    suffix : str, optional
        Sufixo anexado quando o texto é truncado, by default "...".

    Returns
    -------
    str
        Texto original, se já couber em ``max_length``, ou truncado com sufixo.

    Raises
    ------
    ValueError
        Se ``max_length`` for menor que o comprimento do sufixo.

    Examples
    --------
    >>> truncate_text("um texto muito longo", 10)
    'um text...'
    """
    if max_length < len(suffix):
        raise ValueError(
            f"max_length ({max_length}) deve ser >= ao comprimento do sufixo ({len(suffix)})"
        )
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix
