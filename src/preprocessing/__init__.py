"""Pré-processamento e NLP de tweets em português brasileiro.

Implementa a Fase 6 do plano de elaboração (``PLANO-ELABORACAO.md``):
normalização de elementos estruturais (URLs, menções, hashtags), tratamento
de emojis, tokenização com expansão de gírias e marcação de negação,
limpeza (retweets/spam/conteúdo mínimo) e filtragem por critérios de
inclusão, compostas em um pipeline reprodutível.

Modules
-------
text
    Normalização de URLs, menções, hashtags, números e repetições
    ortográficas.
emojis
    Mapeamento de emojis para tokens semânticos de polaridade ou remoção
    controlada.
tokenization
    Tokenização, expansão de contrações/gírias e marcação de escopo de
    negação.
cleaning
    Remoção de marcador de retweet e predicados de qualidade mínima
    (conteúdo, idioma, spam).
filtering
    Filtragem de um corpus (``pl.DataFrame``) pelos critérios de
    inclusão/exclusão definidos em ``cleaning``.
pipeline
    Composição de todas as etapas acima em um pipeline reprodutível.
"""

from preprocessing.cleaning import (
    PORTUGUESE_STOPWORDS,
    calculate_portuguese_stopword_ratio,
    calculate_repeated_word_ratio,
    clean_tweet_text,
    is_minimum_length_content,
    is_probable_portuguese_text,
    is_spam_like,
    remove_retweet_marker,
)
from preprocessing.emojis import (
    NEGATIVE_EMOJI_TOKEN,
    NEGATIVE_EMOJIS,
    NEUTRAL_EMOJI_TOKEN,
    NEUTRAL_EMOJIS,
    POSITIVE_EMOJI_TOKEN,
    POSITIVE_EMOJIS,
    calculate_emoji_sentiment_counts,
    normalize_emojis,
    remove_emojis,
)
from preprocessing.filtering import (
    filter_by_inclusion_criteria,
    filter_by_minimum_length,
    filter_by_portuguese_language,
    filter_spam_like_rows,
    remove_duplicate_text_rows,
)
from preprocessing.pipeline import normalize_tweet_text, run_preprocessing_pipeline
from preprocessing.text import (
    normalize_hashtags,
    normalize_mentions,
    normalize_numbers,
    normalize_repeated_characters,
    normalize_urls,
)
from preprocessing.tokenization import (
    CONTRACTION_EXPANSIONS,
    NEGATION_WORDS,
    expand_contractions,
    mark_negation_scope,
    tokenize_and_normalize,
    tokenize_text,
)

__all__: list[str] = [
    "CONTRACTION_EXPANSIONS",
    "NEGATION_WORDS",
    "NEGATIVE_EMOJIS",
    "NEGATIVE_EMOJI_TOKEN",
    "NEUTRAL_EMOJIS",
    "NEUTRAL_EMOJI_TOKEN",
    "PORTUGUESE_STOPWORDS",
    "POSITIVE_EMOJIS",
    "POSITIVE_EMOJI_TOKEN",
    "calculate_emoji_sentiment_counts",
    "calculate_portuguese_stopword_ratio",
    "calculate_repeated_word_ratio",
    "clean_tweet_text",
    "expand_contractions",
    "filter_by_inclusion_criteria",
    "filter_by_minimum_length",
    "filter_by_portuguese_language",
    "filter_spam_like_rows",
    "is_minimum_length_content",
    "is_probable_portuguese_text",
    "is_spam_like",
    "mark_negation_scope",
    "normalize_emojis",
    "normalize_hashtags",
    "normalize_mentions",
    "normalize_numbers",
    "normalize_repeated_characters",
    "normalize_tweet_text",
    "normalize_urls",
    "remove_duplicate_text_rows",
    "remove_emojis",
    "remove_retweet_marker",
    "run_preprocessing_pipeline",
    "tokenize_and_normalize",
    "tokenize_text",
]
