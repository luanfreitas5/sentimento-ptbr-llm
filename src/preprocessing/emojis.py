"""Tratamento de emojis em tweets: mapeamento semântico e remoção controlada.

Emojis carregam sinal de polaridade relevante em textos curtos de redes
sociais. Este módulo mapeia um conjunto curado de emojis comuns em
português brasileiro para tokens semânticos de polaridade
(positivo/negativo/neutro) e trata os demais emojis reconhecidos via o
token genérico ``[EMOJI]`` (ver ``src/constants/tokens.py``) ou remoção
completa, quando o texto normalizado não deve conter o caractere original.
"""

from constants.regex import EMOJI_PATTERN
from constants.tokens import EMOJI_TOKEN

POSITIVE_EMOJI_TOKEN = "[EMOJI_POSITIVO]"
NEGATIVE_EMOJI_TOKEN = "[EMOJI_NEGATIVO]"
NEUTRAL_EMOJI_TOKEN = "[EMOJI_NEUTRO]"

# Conjunto curado de emojis comuns em tweets em português brasileiro,
# associados à polaridade mais frequentemente expressa por eles. Não
# pretende ser exaustivo: emojis fora deste conjunto, mas reconhecidos por
# ``EMOJI_PATTERN``, recebem o token genérico ``[EMOJI]``.
POSITIVE_EMOJIS: frozenset[str] = frozenset("😀😃😄😁😆😊🙂😍🥰😘👍🎉❤💕✨🔥👏😂")
NEGATIVE_EMOJIS: frozenset[str] = frozenset("😢😭😡😠👎💔😞😔😒🤢😤")
NEUTRAL_EMOJIS: frozenset[str] = frozenset("😐😑🤔😶")

_SENTIMENT_TOKEN_BY_EMOJI: dict[str, str] = {
    **dict.fromkeys(POSITIVE_EMOJIS, POSITIVE_EMOJI_TOKEN),
    **dict.fromkeys(NEGATIVE_EMOJIS, NEGATIVE_EMOJI_TOKEN),
    **dict.fromkeys(NEUTRAL_EMOJIS, NEUTRAL_EMOJI_TOKEN),
}


def normalize_emojis(text: str) -> str:
    """Substitui emojis por tokens semânticos de polaridade ou pelo token genérico.

    Emojis do conjunto curado (:data:`POSITIVE_EMOJIS`, :data:`NEGATIVE_EMOJIS`,
    :data:`NEUTRAL_EMOJIS`) são substituídos pelo token de polaridade
    correspondente; os demais emojis reconhecidos por
    :data:`constants.regex.EMOJI_PATTERN` recebem o token genérico
    :data:`constants.tokens.EMOJI_TOKEN`.

    Parameters
    ----------
    text : str
        Texto de entrada, possivelmente contendo emojis.

    Returns
    -------
    str
        Texto com os emojis substituídos pelos tokens semânticos correspondentes.

    Examples
    --------
    >>> normalize_emojis("adorei o produto 😍")
    'adorei o produto [EMOJI_POSITIVO]'
    >>> normalize_emojis("que decepção 😢")
    'que decepção [EMOJI_NEGATIVO]'
    """
    characters_with_known_sentiment = (
        _SENTIMENT_TOKEN_BY_EMOJI.get(character, character) for character in text
    )
    normalized_text = "".join(characters_with_known_sentiment)
    return EMOJI_PATTERN.sub(EMOJI_TOKEN, normalized_text)


def remove_emojis(text: str) -> str:
    """Remove todos os emojis do texto, sem substituição por token.

    Uso indicado quando o sinal de polaridade dos emojis já foi extraído
    separadamente (ver :func:`calculate_emoji_sentiment_counts`) e não deve
    ser duplicado no texto normalizado.

    Parameters
    ----------
    text : str
        Texto de entrada, possivelmente contendo emojis.

    Returns
    -------
    str
        Texto sem emojis.

    Examples
    --------
    >>> remove_emojis("adorei o produto 😍")
    'adorei o produto '
    """
    return EMOJI_PATTERN.sub("", text)


def calculate_emoji_sentiment_counts(text: str) -> dict[str, int]:
    """Conta emojis positivos, negativos e neutros presentes no texto.

    Parameters
    ----------
    text : str
        Texto de entrada, possivelmente contendo emojis.

    Returns
    -------
    dict[str, int]
        Dicionário com as chaves ``"positivo"``, ``"negativo"`` e
        ``"neutro"``, cada uma contendo a contagem de emojis do conjunto
        curado correspondente encontrados no texto.

    Examples
    --------
    >>> calculate_emoji_sentiment_counts("bom 😊 mas chegou quebrado 😢😢")
    {'positivo': 1, 'negativo': 2, 'neutro': 0}
    """
    counts = {"positivo": 0, "negativo": 0, "neutro": 0}
    for character in text:
        if character in POSITIVE_EMOJIS:
            counts["positivo"] += 1
        elif character in NEGATIVE_EMOJIS:
            counts["negativo"] += 1
        elif character in NEUTRAL_EMOJIS:
            counts["neutro"] += 1
    return counts
