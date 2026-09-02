"""Contrato de dados para os exemplos de treino/validação/teste.

Valida o resultado do particionamento estratificado (``configs/config.yaml``
-> ``data_split``), garantindo que cada exemplo tenha um rótulo válido e
pertença a um dos três conjuntos esperados.
"""

import pandera.polars as pa
import polars as pl
from pandera.api.polars.model_config import BaseConfig
from pandera.errors import SchemaError
from pandera.typing.polars import Series

from constants.labels import SENTIMENT_CLASSES
from exceptions.data import DataValidationError

DATA_SPLITS: tuple[str, ...] = ("treino", "validacao", "teste")


class TrainingExampleSchema(pa.DataFrameModel):
    """Contrato de dados para um exemplo de treino/validação/teste."""

    id: Series[str] = pa.Field(unique=True)
    text: Series[str]
    sentiment_label: Series[str] = pa.Field(isin=list(SENTIMENT_CLASSES))
    split: Series[str] = pa.Field(isin=list(DATA_SPLITS))

    class Config(BaseConfig):
        """Configuração do schema: rejeita colunas não declaradas."""

        strict = True


def validate_training_example(dataframe: pl.DataFrame) -> pl.DataFrame:
    """Valida um DataFrame de exemplos de treino contra :class:`TrainingExampleSchema`.

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
    ...         "split": ["treino"],
    ...     }
    ... )
    >>> validate_training_example(df).height
    1
    """
    try:
        return TrainingExampleSchema.validate(dataframe)
    except SchemaError as exception:
        raise DataValidationError(
            schema_name="TrainingExampleSchema", detail=str(exception)
        ) from exception
