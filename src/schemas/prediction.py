"""Contrato de dados para as predições de sentimento geradas por um modelo.

Usado para validar a saída de ``src/inference/predictor.py`` antes de
persistir resultados ou expô-los via API/dashboard, evitando desvio entre
treino e serviço (train-serve skew).
"""

import pandera.polars as pa
import polars as pl
from pandera.api.polars.model_config import BaseConfig
from pandera.errors import SchemaError
from pandera.typing.polars import Series

from constants.labels import SENTIMENT_CLASSES
from exceptions.data import DataValidationError


class PredictionSchema(pa.DataFrameModel):
    """Contrato de dados para uma predição individual de sentimento."""

    id: Series[str]
    text: Series[str]
    sentiment_label: Series[str] = pa.Field(isin=list(SENTIMENT_CLASSES))
    confidence: Series[float] = pa.Field(ge=0.0, le=1.0)

    class Config(BaseConfig):
        """Configuração do schema: permite colunas extras (ex.: probabilidades por classe)."""

        strict = False


def validate_prediction(dataframe: pl.DataFrame) -> pl.DataFrame:
    """Valida um DataFrame de predições contra :class:`PredictionSchema`.

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
    ...         "text": ["ótimo produto"],
    ...         "sentiment_label": ["positivo"],
    ...         "confidence": [0.95],
    ...     }
    ... )
    >>> validate_prediction(df).height
    1
    """
    try:
        return PredictionSchema.validate(dataframe)
    except SchemaError as exception:
        raise DataValidationError(
            schema_name="PredictionSchema", detail=str(exception)
        ) from exception
