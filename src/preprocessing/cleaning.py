"""Limpeza de tweets: remoção de marcador de retweet e predicados de qualidade mínima.

Fornece as transformações e os predicados usados por
``src/preprocessing/filtering.py`` para decidir se um tweet deve
permanecer no corpus: presença de conteúdo mínimo, sinal heurístico de
idioma português e indícios de spam (repetição excessiva de uma palavra).
"""

import logging
import re

from constants.regex import RETWEET_PATTERN
from utils.text import normalize_whitespace

logger = logging.getLogger(__name__)

_WORD_PATTERN = re.compile(r"\w+")

# Lista curada das palavras funcionais (stopwords) mais frequentes em
# português brasileiro, usada como base determinística mesmo sem o nltk
# instalado (ver :func:`load_nltk_portuguese_stopwords`). Não pretende ser
# exaustiva nem substituir uma biblioteca de detecção de idioma: serve
# apenas como heurística leve para estimar se um texto é predominantemente
# escrito em português.
_CURATED_PORTUGUESE_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "o",
        "as",
        "os",
        "de",
        "do",
        "da",
        "dos",
        "das",
        "em",
        "no",
        "na",
        "nos",
        "nas",
        "um",
        "uma",
        "uns",
        "umas",
        "e",
        "ou",
        "que",
        "com",
        "para",
        "por",
        "se",
        "não",
        "mais",
        "mas",
        "como",
        "ao",
        "aos",
        "à",
        "às",
        "sua",
        "seu",
        "suas",
        "seus",
        "é",
        "foi",
        "está",
        "são",
        "eu",
        "você",
        "ele",
        "ela",
        "isso",
        "muito",
        "já",
        "só",
        "também",
        "porque",
    }
)


def load_nltk_portuguese_stopwords() -> frozenset[str]:
    """Carrega as stopwords em português do corpus ``stopwords`` do nltk.

    A biblioteca ``nltk`` é opcional (ver ``pyproject.toml``, extra "nlp"):
    o import ocorre de forma tardia, dentro desta função, para que o
    restante do módulo permaneça importável sem ela. Não dispara download
    em tempo de execução — o corpus deve ser obtido antecipadamente via
    ``make install-nlp``; sem ele, o conjunto retornado é vazio e
    :data:`PORTUGUESE_STOPWORDS` cai no fallback da lista curada
    (:data:`_CURATED_PORTUGUESE_STOPWORDS`), com aviso no log.

    Returns
    -------
    frozenset[str]
        Stopwords em português do nltk, ou conjunto vazio se a biblioteca
        não estiver instalada ou o corpus não tiver sido baixado.

    Examples
    --------
    >>> load_nltk_portuguese_stopwords() >= frozenset()
    True
    """
    try:
        from nltk.corpus import stopwords  # pyright: ignore[reportMissingImports]
    except ImportError:
        logger.debug("nltk não está instalado; usando apenas a lista curada de stopwords.")
        return frozenset()

    try:
        return frozenset(stopwords.words("portuguese"))
    except LookupError:
        logger.warning(
            "Corpus 'stopwords' do nltk não encontrado. Rode `make install-nlp` para "
            "baixá-lo; usando apenas a lista curada de stopwords enquanto isso."
        )
        return frozenset()


# União da lista curada com o corpus do nltk (quando disponível), calculada
# uma única vez na importação do módulo. Sem o nltk instalado/baixado,
# equivale exatamente à lista curada (ver :func:`load_nltk_portuguese_stopwords`).
PORTUGUESE_STOPWORDS: frozenset[str] = (
    _CURATED_PORTUGUESE_STOPWORDS | load_nltk_portuguese_stopwords()
)


def remove_retweet_marker(text: str) -> str:
    """Remove o marcador de retweet (ex.: ``"RT @usuario:"``) do início do texto.

    Parameters
    ----------
    text : str
        Texto de entrada, possivelmente iniciado por um marcador de retweet.

    Returns
    -------
    str
        Texto sem o marcador de retweet.

    Examples
    --------
    >>> remove_retweet_marker("RT @exemplo: ótimo produto")
    'ótimo produto'
    """
    return RETWEET_PATTERN.sub("", text)


def clean_tweet_text(text: str) -> str:
    """Remove o marcador de retweet e normaliza o espaçamento de um tweet.

    Combinação mínima de limpeza aplicada antes das demais etapas de
    normalização (ver ``src/preprocessing/pipeline.py``).

    Parameters
    ----------
    text : str
        Texto de entrada.

    Returns
    -------
    str
        Texto limpo, sem marcador de retweet e com espaçamento normalizado.

    Examples
    --------
    >>> clean_tweet_text("RT @exemplo:   ótimo   produto  ")
    'ótimo produto'
    """
    return normalize_whitespace(remove_retweet_marker(text))


def is_minimum_length_content(
    text: str, *, minimum_characters: int = 5, minimum_words: int = 2
) -> bool:
    """Verifica se o texto atende aos limiares mínimos de conteúdo.

    Parameters
    ----------
    text : str
        Texto de entrada, tipicamente já limpo por :func:`clean_tweet_text`.
    minimum_characters : int, optional
        Número mínimo de caracteres (após remoção de espaços nas bordas),
        by default 5.
    minimum_words : int, optional
        Número mínimo de palavras, by default 2.

    Returns
    -------
    bool
        ``True`` se o texto atender a ambos os limiares mínimos.

    Examples
    --------
    >>> is_minimum_length_content("bom")
    False
    >>> is_minimum_length_content("muito bom mesmo")
    True
    """
    stripped_text = text.strip()
    word_count = len(_WORD_PATTERN.findall(stripped_text))
    return len(stripped_text) >= minimum_characters and word_count >= minimum_words


def calculate_repeated_word_ratio(text: str) -> float:
    """Calcula a proporção que a palavra mais frequente representa no texto.

    Usado como sinal heurístico de spam: textos dominados pela repetição de
    uma única palavra (ex.: "promo promo promo compre agora") tendem a ter
    uma proporção alta.

    Parameters
    ----------
    text : str
        Texto de entrada.

    Returns
    -------
    float
        Proporção entre 0.0 e 1.0. Retorna 0.0 para texto sem palavras.

    Examples
    --------
    >>> calculate_repeated_word_ratio("promo promo promo compre agora")
    0.6
    """
    words = [word.lower() for word in _WORD_PATTERN.findall(text)]
    if not words:
        return 0.0
    most_common_count = max(words.count(word) for word in set(words))
    return most_common_count / len(words)


def is_spam_like(text: str, *, max_repeated_word_ratio: float = 0.5) -> bool:
    """Verifica se o texto tem indícios de spam pela repetição excessiva de uma palavra.

    Parameters
    ----------
    text : str
        Texto de entrada.
    max_repeated_word_ratio : float, optional
        Proporção máxima tolerada da palavra mais frequente sobre o total
        de palavras, by default 0.5.

    Returns
    -------
    bool
        ``True`` se a proporção da palavra mais repetida exceder o limiar.

    Examples
    --------
    >>> is_spam_like("compre compre compre compre agora")
    True
    >>> is_spam_like("muito bom o atendimento")
    False
    """
    return calculate_repeated_word_ratio(text) > max_repeated_word_ratio


def calculate_portuguese_stopword_ratio(text: str) -> float:
    """Calcula a proporção de palavras do texto que são stopwords em português.

    Heurística leve para estimar se um texto é predominantemente escrito em
    português brasileiro, sem depender de uma biblioteca de detecção de
    idioma (ver :data:`PORTUGUESE_STOPWORDS`). Não deve ser interpretada
    como uma detecção de idioma robusta.

    Parameters
    ----------
    text : str
        Texto de entrada.

    Returns
    -------
    float
        Proporção entre 0.0 e 1.0. Retorna 0.0 para texto sem palavras.

    Examples
    --------
    >>> calculate_portuguese_stopword_ratio("o produto que eu comprei é muito bom")
    0.625
    """
    words = [word.lower() for word in _WORD_PATTERN.findall(text)]
    if not words:
        return 0.0
    stopword_count = sum(1 for word in words if word in PORTUGUESE_STOPWORDS)
    return stopword_count / len(words)


def is_probable_portuguese_text(text: str, *, minimum_ratio: float = 0.15) -> bool:
    """Verifica se o texto é provavelmente escrito em português, por heurística.

    Parameters
    ----------
    text : str
        Texto de entrada.
    minimum_ratio : float, optional
        Proporção mínima de stopwords em português exigida, by default 0.15.

    Returns
    -------
    bool
        ``True`` se a proporção de stopwords em português atingir o limiar.

    Examples
    --------
    >>> is_probable_portuguese_text("o produto que eu comprei é muito bom")
    True
    >>> is_probable_portuguese_text("the product I bought is great")
    False
    """
    return calculate_portuguese_stopword_ratio(text) >= minimum_ratio
