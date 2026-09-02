"""Tokens especiais usados na normalização textual e na tokenização.

Os tokens de substituição (``[URL]``, ``[MENCAO]`` etc.) preservam o sinal
semântico de um elemento removido do texto durante o pré-processamento
(ver ``src/preprocessing/text.py``), em vez de simplesmente descartá-lo.
"""

URL_TOKEN = "[URL]"
MENTION_TOKEN = "[MENCAO]"
HASHTAG_TOKEN = "[HASHTAG]"
NUMBER_TOKEN = "[NUMERO]"
EMOJI_TOKEN = "[EMOJI]"

# Tokens de controle usados por modelos de sequência (ex.: BiLSTM com
# vocabulário próprio, fora do tokenizador de um transformer pré-treinado).
PAD_TOKEN = "[PAD]"
UNKNOWN_TOKEN = "[DESCONHECIDO]"

SPECIAL_TOKENS: tuple[str, ...] = (
    URL_TOKEN,
    MENTION_TOKEN,
    HASHTAG_TOKEN,
    NUMBER_TOKEN,
    EMOJI_TOKEN,
    PAD_TOKEN,
    UNKNOWN_TOKEN,
)
