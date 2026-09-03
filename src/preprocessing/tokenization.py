"""Tokenização de textos em português brasileiro para os classificadores clássicos.

Inclui um tokenizador simples baseado em expressões regulares, expansão de
contrações e gírias comuns em redes sociais e marcação do escopo de
negação (ex.: "não gostei nada" -> "não gostei_NEG nada_NEG"), técnica
usada para que classificadores de bag-of-words/TF-IDF capturem a inversão
de polaridade causada por palavras de negação (ver CLAUDE.md, Seção 4.2).
"""

import re

_TOKEN_PATTERN = re.compile(r"\[[A-ZÀ-Ú_]+\]|[A-Za-zÀ-ÿ]+(?:['-][A-Za-zÀ-ÿ]+)*|[.,!?;:]")

# Contrações e gírias comuns em textos informais de redes sociais em
# português brasileiro, mapeadas para a forma normalizada equivalente.
CONTRACTION_EXPANSIONS: dict[str, str] = {
    "vc": "você",
    "vcs": "vocês",
    "cê": "você",
    "tb": "também",
    "tbm": "também",
    "pq": "porque",
    "blz": "beleza",
    "q": "que",
    "n": "não",
    "naum": "não",
    "eh": "é",
    "mto": "muito",
    "mt": "muito",
    "obg": "obrigado",
    "vlw": "valeu",
    "flw": "falou",
    "dnv": "de novo",
}

# Palavras que iniciam o escopo de negação (ver :func:`mark_negation_scope`).
NEGATION_WORDS: frozenset[str] = frozenset({"não", "nunca", "jamais", "nem"})

# Tokens de pontuação que encerram o escopo de negação ao serem encontrados.
_SCOPE_BOUNDARY_TOKENS: frozenset[str] = frozenset({".", ",", "!", "?", ";", ":"})

_NEGATION_SUFFIX = "_NEG"


def tokenize_text(text: str) -> list[str]:
    """Tokeniza um texto em palavras, tokens especiais e pontuação.

    Reconhece três categorias de token: marcadores especiais entre colchetes
    (ex.: ``[URL]``, produzidos por ``src/preprocessing/text.py`` e
    ``src/preprocessing/emojis.py``), palavras (incluindo contrações com
    apóstrofo ou hífen) e sinais de pontuação isolados.

    Parameters
    ----------
    text : str
        Texto de entrada, tipicamente já normalizado por
        ``src/preprocessing/pipeline.py``.

    Returns
    -------
    list[str]
        Lista de tokens, na ordem em que aparecem no texto.

    Examples
    --------
    >>> tokenize_text("não gostei, [URL]")
    ['não', 'gostei', ',', '[URL]']
    """
    return _TOKEN_PATTERN.findall(text)


def expand_contractions(tokens: list[str]) -> list[str]:
    """Expande contrações e gírias comuns para a forma normalizada equivalente.

    A comparação é feita em minúsculas; tokens sem uma expansão conhecida em
    :data:`CONTRACTION_EXPANSIONS` são preservados sem alteração.

    Parameters
    ----------
    tokens : list[str]
        Tokens de entrada, tipicamente produzidos por :func:`tokenize_text`.

    Returns
    -------
    list[str]
        Tokens com as contrações/gírias conhecidas expandidas.

    Examples
    --------
    >>> expand_contractions(["vc", "eh", "mto", "gente"])
    ['você', 'é', 'muito', 'gente']
    """
    return [CONTRACTION_EXPANSIONS.get(token.lower(), token) for token in tokens]


def mark_negation_scope(tokens: list[str]) -> list[str]:
    """Marca o escopo de negação, anexando o sufixo ``_NEG`` aos tokens afetados.

    Ao encontrar uma palavra de negação (:data:`NEGATION_WORDS`), todos os
    tokens seguintes recebem o sufixo ``_NEG`` até o primeiro sinal de
    pontuação (:data:`_SCOPE_BOUNDARY_TOKENS`) ou o fim da lista — técnica
    clássica para que modelos bag-of-words capturem a inversão de
    polaridade causada pela negação, sem alterar a quantidade de tokens.

    Parameters
    ----------
    tokens : list[str]
        Tokens de entrada, tipicamente produzidos por :func:`tokenize_text`.

    Returns
    -------
    list[str]
        Tokens com o escopo de negação marcado, na mesma quantidade e
        ordem dos tokens de entrada.

    Examples
    --------
    >>> mark_negation_scope(["não", "gostei", "nada", ",", "mas", "voltaria"])
    ['não', 'gostei_NEG', 'nada_NEG', ',', 'mas', 'voltaria']
    """
    marked_tokens: list[str] = []
    negation_scope_active = False
    for token in tokens:
        lowered_token = token.lower()
        if lowered_token in _SCOPE_BOUNDARY_TOKENS:
            negation_scope_active = False
            marked_tokens.append(token)
        elif lowered_token in NEGATION_WORDS:
            negation_scope_active = True
            marked_tokens.append(token)
        elif negation_scope_active:
            marked_tokens.append(f"{token}{_NEGATION_SUFFIX}")
        else:
            marked_tokens.append(token)
    return marked_tokens


def tokenize_and_normalize(
    text: str, *, expand_slang: bool = True, apply_negation_marking: bool = True
) -> list[str]:
    """Tokeniza um texto e aplica, opcionalmente, expansão de gírias e marcação de negação.

    Composição de :func:`tokenize_text`, :func:`expand_contractions` e
    :func:`mark_negation_scope`, na ordem em que são citadas.

    Parameters
    ----------
    text : str
        Texto de entrada, tipicamente já normalizado por
        ``src/preprocessing/pipeline.py``.
    expand_slang : bool, optional
        Se ``True``, aplica :func:`expand_contractions` aos tokens, by
        default True.
    apply_negation_marking : bool, optional
        Se ``True``, aplica :func:`mark_negation_scope` aos tokens, by
        default True.

    Returns
    -------
    list[str]
        Tokens resultantes da composição das etapas selecionadas.

    Examples
    --------
    >>> tokenize_and_normalize("não gostei vc entendeu")
    ['não', 'gostei_NEG', 'você_NEG', 'entendeu_NEG']
    """
    tokens = tokenize_text(text)
    if expand_slang:
        tokens = expand_contractions(tokens)
    if apply_negation_marking:
        tokens = mark_negation_scope(tokens)
    return tokens
