"""Lematização de textos em português brasileiro via spaCy.

A biblioteca ``spacy`` e o modelo ``pt_core_news_sm`` são opcionais (ver
``pyproject.toml``, extra "nlp", e ``make install-nlp``/``make
spacy-model``): o import ocorre de forma tardia, dentro de
:func:`load_spacy_model`, para que o restante de ``src/preprocessing/``
permaneça importável sem eles. Sem a biblioteca ou o modelo instalados,
:func:`lemmatize_text` retorna o texto original inalterado, com aviso no
log — mesmo padrão de fallback usado em ``src/utils/seed.py`` para o
PyTorch.
"""

import functools
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import spacy  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)

_SPACY_MODEL_NAME = "pt_core_news_sm"


@functools.lru_cache(maxsize=1)
def load_spacy_model() -> "spacy.language.Language | None":
    """Carrega e armazena em cache o modelo do spaCy para português.

    O cache evita recarregar o modelo (custoso) a cada chamada de
    :func:`lemmatize_text` dentro do mesmo processo.

    Returns
    -------
    spacy.language.Language | None
        Pipeline carregado do spaCy, ou ``None`` se a biblioteca ``spacy``
        ou o modelo :data:`_SPACY_MODEL_NAME` não estiverem instalados.

    Examples
    --------
    >>> load_spacy_model() is None or load_spacy_model() is not None
    True
    """
    try:
        import spacy  # pyright: ignore[reportMissingImports]
    except ImportError:
        logger.warning(
            "A biblioteca 'spacy' não está instalada; lematização desabilitada "
            "(mantendo o texto original). Instale com `make install-nlp`."
        )
        return None

    try:
        return spacy.load(_SPACY_MODEL_NAME)
    except OSError:
        logger.warning(
            "Modelo '%s' do spaCy não encontrado; lematização desabilitada "
            "(mantendo o texto original). Baixe com `make spacy-model`.",
            _SPACY_MODEL_NAME,
        )
        return None


def lemmatize_text(text: str) -> str:
    """Lematiza um texto em português, reduzindo cada palavra à sua forma canônica.

    Usa o modelo :data:`_SPACY_MODEL_NAME` do spaCy (carregado por
    :func:`load_spacy_model`) para reduzir flexões de gênero, número e
    tempo verbal à forma de dicionário (ex.: "gostei" -> "gostar"),
    reduzindo a esparsidade do vocabulário em representações
    bag-of-words/TF-IDF. Se o spaCy ou o modelo não estiverem disponíveis,
    retorna o texto original sem alteração.

    Parameters
    ----------
    text : str
        Texto de entrada, tipicamente já normalizado por
        ``src/preprocessing/pipeline.py``.

    Returns
    -------
    str
        Texto lematizado, com os tokens separados por espaço simples; o
        texto original se o modelo do spaCy não estiver disponível.

    Examples
    --------
    >>> lemmatize_text("gostei muito dos produtos")  # doctest: +SKIP
    'gostar muito de o produto'
    """
    model = load_spacy_model()
    if model is None:
        return text
    document = model(text)
    return " ".join(token.lemma_ for token in document)
