"""Nomes de colunas usados nos DataFrames do projeto.

Centraliza os nomes de colunas para evitar strings mágicas espalhadas pelo
código. Os valores refletem os nomes definidos em ``configs/config.yaml`` e
``configs/paths.yaml``.
"""

# Identificação e conteúdo textual
ID_COLUMN = "id"
TEXT_COLUMN = "texto"
TEXT_NORMALIZED_COLUMN = "texto_normalizado"

# Rotulagem e alvo de predição
TARGET_COLUMN = "sentimento"
PREDICTED_LABEL_COLUMN = "sentimento_predito"
CONFIDENCE_COLUMN = "confianca"
LABELER_COLUMN = "rotulador"
LABELER_WEIGHT_COLUMN = "peso"

# Origem e metadados de coleta
SOURCE_COLUMN = "fonte_dados"
COLLECTION_DATE_COLUMN = "data_coleta"

# Particionamento de dados
SPLIT_COLUMN = "split"

# Colunas obrigatórias no corpus bruto recém-coletado/importado
RAW_CORPUS_REQUIRED_COLUMNS: tuple[str, ...] = (
    ID_COLUMN,
    TEXT_COLUMN,
    SOURCE_COLUMN,
    COLLECTION_DATE_COLUMN,
)

# Colunas obrigatórias no corpus já rotulado, pronto para modelagem
LABELED_CORPUS_REQUIRED_COLUMNS: tuple[str, ...] = (ID_COLUMN, TEXT_COLUMN, TARGET_COLUMN)

# Colunas do lote bruto real coletado por usuário (data/raw/*.parquet),
# usadas por data.loader.load_raw_tweet_batch e
# preprocessing.filtering.filter_by_raw_metadata.
TWEET_ID_COLUMN = "tweet_id"
IS_RETWEET_COLUMN = "is_retweet"
LANGUAGE_COLUMN = "language"
