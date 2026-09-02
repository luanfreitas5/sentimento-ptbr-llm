"""Contrato de dados para os resultados da rotulagem semiautomática em cascata.

Reflete ``configs/labeling.yaml``: cada rotulador (heurística, LLM, modelo
de referência) produz um rótulo candidato com confiança e peso, agregados
posteriormente em ``src/labeling/consensus.py``.
"""

import pandera.polars as pa
import polars as pl
from pandera.api.polars.model_config import BaseConfig
from pandera.errors import SchemaError
from pandera.typing.polars import Series

from constants.labels import SENTIMENT_CLASSES
from exceptions.data import DataValidationError


class LabelingResultSchema(pa.DataFrameModel):
    """Contrato de dados para um resultado individual de rotulagem candidata."""

    id: Series[str]
    tagger: Series[str]
    sentiment_label: Series[str] = pa.Field(isin=list(SENTIMENT_CLASSES))
    confidence: Series[float] = pa.Field(ge=0.0, le=1.0)
    weight: Series[float] = pa.Field(gt=0.0)

    class Config(BaseConfig):
        """Configuração do schema: rejeita colunas não declaradas."""

        strict = True


def validate_labeling_result(dataframe: pl.DataFrame) -> pl.DataFrame:
    """Valida um DataFrame de resultados de rotulagem contra :class:`LabelingResultSchema`.

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
    ...         "tagger": ["heuristica_lexica"],
    ...         "sentiment_label": ["positivo"],
    ...         "confidence": [0.9],
    ...         "weight": [1.0],
    ...     }
    ... )
    >>> validate_labeling_result(df).height
    1
    """
    try:
        return LabelingResultSchema.validate(dataframe)
    except SchemaError as exception:
        raise DataValidationError(
            schema_name="LabelingResultSchema", detail=str(exception)
        ) from exception
