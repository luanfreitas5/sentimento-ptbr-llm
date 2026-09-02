"""Constantes globais do projeto ``sentimento-ptbr-llm``.

Centraliza valores fixos reutilizados por múltiplos módulos, evitando
strings e números mágicos espalhados pelo código.

Modules
-------
columns
    Nomes de colunas usados nos DataFrames do projeto.
labels
    Classes de sentimento e conversões entre rótulo e identificador numérico.
metrics
    Nomes de métricas de avaliação reconhecidas pelo projeto.
regex
    Padrões de expressão regular para limpeza de texto de tweets.
tokens
    Tokens especiais usados na normalização textual e tokenização.
defaults
    Valores padrão de hiperparâmetros e limiares do projeto.
"""

from constants.columns import (
    COLLECTION_DATE_COLUMN,
    CONFIDENCE_COLUMN,
    ID_COLUMN,
    LABELED_CORPUS_REQUIRED_COLUMNS,
    LABELER_COLUMN,
    LABELER_WEIGHT_COLUMN,
    PREDICTED_LABEL_COLUMN,
    RAW_CORPUS_REQUIRED_COLUMNS,
    SOURCE_COLUMN,
    SPLIT_COLUMN,
    TARGET_COLUMN,
    TEXT_COLUMN,
    TEXT_NORMALIZED_COLUMN,
)
from constants.defaults import (
    DEFAULT_BOOTSTRAP_ITERATIONS,
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_CROSS_VALIDATION_FOLDS,
    DEFAULT_F1_MACRO_MINIMUM,
    DEFAULT_MCC_MINIMUM,
    DEFAULT_MINIMUM_TEST_COVERAGE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SIGNIFICANCE_ALPHA,
    DEFAULT_TEST_SIZE,
    DEFAULT_VALIDATION_SIZE,
)
from constants.labels import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    NEGATIVE_LABEL,
    NEUTRAL_LABEL,
    POSITIVE_LABEL,
    SENTIMENT_CLASSES,
    transform_id_to_label,
    transform_label_to_id,
    validate_label,
)
from constants.metrics import (
    ALL_METRICS,
    OPERATIONAL_METRICS,
    PRIMARY_METRIC,
    RANKING_METRICS,
    SECONDARY_METRICS,
    validate_metric_name,
)
from constants.regex import (
    EMOJI_PATTERN,
    HASHTAG_PATTERN,
    MENTION_PATTERN,
    MULTIPLE_WHITESPACE_PATTERN,
    NUMBER_PATTERN,
    REPEATED_CHARACTERS_PATTERN,
    RETWEET_PATTERN,
    URL_PATTERN,
)
from constants.tokens import (
    EMOJI_TOKEN,
    HASHTAG_TOKEN,
    MENTION_TOKEN,
    NUMBER_TOKEN,
    PAD_TOKEN,
    SPECIAL_TOKENS,
    UNKNOWN_TOKEN,
    URL_TOKEN,
)

__all__: list[str] = [
    "ALL_METRICS",
    "COLLECTION_DATE_COLUMN",
    "CONFIDENCE_COLUMN",
    "DEFAULT_BOOTSTRAP_ITERATIONS",
    "DEFAULT_CONFIDENCE_LEVEL",
    "DEFAULT_CROSS_VALIDATION_FOLDS",
    "DEFAULT_F1_MACRO_MINIMUM",
    "DEFAULT_MCC_MINIMUM",
    "DEFAULT_MINIMUM_TEST_COVERAGE",
    "DEFAULT_RANDOM_SEED",
    "DEFAULT_SIGNIFICANCE_ALPHA",
    "DEFAULT_TEST_SIZE",
    "DEFAULT_VALIDATION_SIZE",
    "EMOJI_PATTERN",
    "EMOJI_TOKEN",
    "HASHTAG_PATTERN",
    "HASHTAG_TOKEN",
    "ID_COLUMN",
    "ID_TO_LABEL",
    "LABELED_CORPUS_REQUIRED_COLUMNS",
    "LABELER_COLUMN",
    "LABELER_WEIGHT_COLUMN",
    "LABEL_TO_ID",
    "MENTION_PATTERN",
    "MENTION_TOKEN",
    "MULTIPLE_WHITESPACE_PATTERN",
    "NEGATIVE_LABEL",
    "NEUTRAL_LABEL",
    "NUMBER_PATTERN",
    "NUMBER_TOKEN",
    "OPERATIONAL_METRICS",
    "PAD_TOKEN",
    "POSITIVE_LABEL",
    "PREDICTED_LABEL_COLUMN",
    "PRIMARY_METRIC",
    "RANKING_METRICS",
    "RAW_CORPUS_REQUIRED_COLUMNS",
    "REPEATED_CHARACTERS_PATTERN",
    "RETWEET_PATTERN",
    "SECONDARY_METRICS",
    "SENTIMENT_CLASSES",
    "SOURCE_COLUMN",
    "SPECIAL_TOKENS",
    "SPLIT_COLUMN",
    "TARGET_COLUMN",
    "TEXT_COLUMN",
    "TEXT_NORMALIZED_COLUMN",
    "UNKNOWN_TOKEN",
    "URL_PATTERN",
    "URL_TOKEN",
    "transform_id_to_label",
    "transform_label_to_id",
    "validate_label",
    "validate_metric_name",
]
