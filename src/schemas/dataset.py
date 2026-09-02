"""Contratos de dados (schemas) para o corpus de tweets, bruto e rotulado.

Os schemas usam ``pandera.polars`` como contrato de dados versionado (ver
CLAUDE.md, "Data Contracts"): validam tipos, nulidade, unicidade e valores
aceitos nas fronteiras entre as etapas ``raw -> interim -> processed``.
"""

import pandera.polars as pa
import polars as pl
from pandera.errors import SchemaError
from pandera.typing.polars import Series

from constants.labels import SENTIMENT_CLASSES
from exceptions.data import DataValidationError


class RawTweetSchema(pa.DataFrameModel):
    """Contrato de dados para tweets recém-coletados/importados (``data/raw``, ``data/external``)."""

    id: Series[str] = pa.Field(unique=True)
    texto: Series[str]
    fonte_dados: Series[str]
    data_coleta: Series[str]

    class Config:
        """Configuração do schema: rejeita colunas não declaradas."""

        strict = True


class LabeledCorpusSchema(pa.DataFrameModel):
    """Contrato de dados para o corpus rotulado, pronto para modelagem (``data/processed``)."""

    id: Series[str] = pa.Field(unique=True)
    texto: Series[str]
    sentimento: Series[str] = pa.Field(isin=list(SENTIMENT_CLASSES))

    class Config:
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
    >>> df = pl.DataFrame(
    ...     {
    ...         "id": ["1"],
    ...         "texto": ["ótimo produto"],
    ...         "fonte_dados": ["scraping"],
    ...         "data_coleta": ["2026-01-01"],
    ...     }
    ... )
    >>> validate_raw_tweet_dataset(df).height
    1
    """
    try:
        return RawTweetSchema.validate(dataframe)
    except SchemaError as excecao:
        raise DataValidationError(schema_name="RawTweetSchema", detail=str(excecao)) from excecao


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
    >>> df = pl.DataFrame({"id": ["1"], "texto": ["ótimo produto"], "sentimento": ["positivo"]})
    >>> validate_labeled_corpus(df).height
    1
    """
    try:
        return LabeledCorpusSchema.validate(dataframe)
    except SchemaError as excecao:
        raise DataValidationError(
            schema_name="LabeledCorpusSchema", detail=str(excecao)
        ) from excecao
