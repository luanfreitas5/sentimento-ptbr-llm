"""Padrões de expressão regular usados na limpeza e normalização de tweets.

Os padrões são pré-compilados para reuso eficiente em ``src/preprocessing/``.
Cobrem os elementos típicos de tweets em português brasileiro: URLs, menções,
hashtags, retweets, emojis, números e espaçamento irregular.
"""

import re

# URLs (http, https ou encurtadores sem esquema explícito)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")

# Menções a outros usuários (ex.: @usuario)
MENTION_PATTERN = re.compile(r"@\w+")

# Hashtags (ex.: #sentimento)
HASHTAG_PATTERN = re.compile(r"#\w+")

# Marcador de retweet no início do texto (ex.: "RT @usuario:")
RETWEET_PATTERN = re.compile(r"^RT\s+@\w+:?\s*")

# Sequência de dígitos
NUMBER_PATTERN = re.compile(r"\d+")

# Emojis e outros símbolos pictográficos (faixas Unicode mais comuns)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001f300-\U0001f5ff"  # símbolos e pictogramas diversos
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f680-\U0001f6ff"  # transporte e mapas
    "\U0001f1e0-\U0001f1ff"  # bandeiras (pares de indicadores regionais)
    "\U00002700-\U000027bf"  # dingbats
    "\U0001f900-\U0001f9ff"  # símbolos suplementares
    "\U00002600-\U000026ff"  # símbolos diversos
    "]+",
    flags=re.UNICODE,
)

# Múltiplos espaços em branco consecutivos (inclui tabs e quebras de linha)
MULTIPLE_WHITESPACE_PATTERN = re.compile(r"\s+")

# Caracteres repetidos mais de duas vezes seguidas (ex.: "muitooo" -> "muito")
REPEATED_CHARACTERS_PATTERN = re.compile(r"(\w)\1{2,}")

# Marcadores de negação do português (palavra inteira, sem acento obrigatório)
NEGATION_PATTERN = re.compile(
    r"\b(?:n[aã]o|nunca|jamais|nem|nada|ningu[eé]m|nenhum[a]?|nao|sem|tampouco|n)\b",
    flags=re.IGNORECASE,
)

# Marcadores lexicais de ironia aparente: proxy, não detecta ironia de fato
IRONY_PATTERN = re.compile(
    r"s[oó] que n[aã]o|claro que|com certeza|ah t[aá]|ah sim|imagina|nossa,? que"
    r"|que (?:lindo|maravilha|[oó]timo|legal|bom)\b.*[!?]{2,}|parab[eé]ns,? hein"
    r"|obrigad[oa],? hein|[\"“”][^\"“”]{2,40}[\"“”]|#ironia|\b/s\b|\bvaleu hein\b",
    flags=re.IGNORECASE,
)

# Gíria e escrita informal de tweets pt-BR (léxico curto e auditável)
SLANG_PATTERN = re.compile(
    r"\b(?:mano|mina|vei|v[eê]i|cara|mlk|mlr|slc|krl|krlh|pqp|porra|caralho|pq|vc|vcs|tb|tbm"
    r"|blz|fds|fdp|bora|massa|daora|top|zuado|zuera|lacrou|lacrar|mds|pls|migu[eé]|bagulho"
    r"|trampo|rol[eê]|tnc|sdds|kkk+|rs+|haha+|aff+|ata|eita|oxe|uai|sô)\b",
    flags=re.IGNORECASE,
)
