"""Contratos de dados (schemas) para o corpus de tweets, bruto e rotulado.

Os schemas usam ``pandera.polars`` como contrato de dados versionado (ver
CLAUDE.md, "Data Contracts"): validam tipos, nulidade, unicidade e valores
aceitos nas fronteiras entre as etapas ``raw -> interim -> processed``.
"""

from datetime import datetime

import pandera.polars as pa
import polars as pl
from pandera.api.polars.model_config import BaseConfig
from pandera.errors import SchemaError
from pandera.typing.polars import Series

from constants.labels import SENTIMENT_CLASSES
from exceptions.data import DataValidationError


class RawTweetSchema(pa.DataFrameModel):
    """Contrato de dados para tweets brutos coletados por usuário (``data/raw``).

    Reflete as colunas reais dos arquivos Parquet coletados via
    ``twscrape`` (um arquivo por usuário). ``source_query``/``source_group``
    são nulos quando a coleta foi feita por usuário, não por termo de
    busca. Validado sobre o lote já concatenado por
    :func:`data.loader.load_raw_tweet_batch`, que em seguida renomeia
    ``tweet_id`` para ``id`` (contrato usado pelo restante do pipeline).
    """

    tweet_id: Series[str] = pa.Field(unique=True)
    user_id: Series[str]
    text: Series[str]
    created_at: Series[datetime]
    language: Series[str]
    is_reply: Series[bool]
    is_retweet: Series[bool]
    like_count: Series[int] = pa.Field(ge=0)
    reply_count: Series[int] = pa.Field(ge=0)
    retweet_count: Series[int] = pa.Field(ge=0)
    quote_count: Series[int] = pa.Field(ge=0)
    source_query: Series[str] = pa.Field(nullable=True)
    source_group: Series[str] = pa.Field(nullable=True)

    class Config(BaseConfig):
        """Configuração do schema: rejeita colunas não declaradas."""

        strict = True


class LabeledCorpusSchema(pa.DataFrameModel):
    """Contrato de dados para o corpus rotulado, pronto para modelagem (``data/processed``)."""

    id: Series[str] = pa.Field(unique=True)
    text: Series[str]
    sentiment_label: Series[str] = pa.Field(isin=list(SENTIMENT_CLASSES))

    class Config(BaseConfig):
        """Configuração do schema: permite colunas extras (ex.: metadados de rotulagem)."""

        strict = False


def validate_raw_tweet_dataset(dataframe: pl.DataFrame) -> pl.DataFrame:
    """Valida um DataFrame de tweets brutos contra :class:`RawTweetSchema`.

    Parameters
    ----------
    dataframe : pl.DataFrame
        DataFrame a ser validado.

    Returns
    -------
    pl.DataFrame
        O mesmo DataFrame, quando válido.

    Raises
    ------
    DataValidationError
        Se o DataFrame violar o contrato de dados.

    Examples
    --------
    >>> from datetime import datetime
    >>> df = pl.DataFrame(
    ...     {
    ...         "tweet_id": ["1"],
    ...         "user_id": ["u1"],
    ...         "text": ["ótimo produto"],
    ...         "created_at": [datetime(2026, 1, 1)],
    ...         "language": ["pt"],
    ...         "is_reply": [False],
    ...         "is_retweet": [False],
    ...         "like_count": [0],
    ...         "reply_count": [0],
    ...         "retweet_count": [0],
    ...         "quote_count": [0],
    ...         "source_query": [None],
    ...         "source_group": [None],
    ...     }
    ... )
    >>> validate_raw_tweet_dataset(df).height
    1
    """
    try:
        return RawTweetSchema.validate(dataframe)
    except SchemaError as exception:
        raise DataValidationError(
            schema_name="RawTweetSchema", detail=str(exception)
        ) from exception


def validate_labeled_corpus(dataframe: pl.DataFrame) -> pl.DataFrame:
    """Valida um DataFrame de corpus rotulado contra :class:`LabeledCorpusSchema`.

    Parameters
    ----------
    dataframe : pl.DataFrame
        DataFrame a ser validado.

    Returns
    -------
    pl.DataFrame
        O mesmo DataFrame, quando válido.

    Raises
    ------
    DataValidationError
        Se o DataFrame violar o contrato de dados.

    Examples
    --------
    >>> df = pl.DataFrame({"id": ["1"], "text": ["ótimo produto"], "sentiment_label": ["positivo"]})
    >>> validate_labeled_corpus(df).height
    1
    """
    try:
        return LabeledCorpusSchema.validate(dataframe)
    except SchemaError as exception:
        raise DataValidationError(
            schema_name="LabeledCorpusSchema", detail=str(exception)
        ) from exception
