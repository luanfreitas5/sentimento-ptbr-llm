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
