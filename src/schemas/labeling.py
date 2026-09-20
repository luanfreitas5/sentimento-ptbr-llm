"""Contratos de dados da rotulagem de sentimento.

* :class:`LabelingResultSchema`: resultados da rotulagem semiautomática em
  cascata (``src/labeling/automatic.py``), agregados em
  ``src/labeling/consensus.py``.
* :class:`LabeledSourceSchema`: base de tweets rotulada por uma única fonte
  (``tweets_data_huggingface`` / ``tweets_data_openai``), ver
  ``src/pipelines/labeling.py``.
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
    confidence_score: Series[float] = pa.Field(ge=0.0, le=1.0)
    weight: Series[float] = pa.Field(gt=0.0)

    class Config(BaseConfig):
        """Configuração do schema: rejeita colunas não declaradas."""

        strict = True


class LabeledSourceSchema(pa.DataFrameModel):
    """Contrato de uma base rotulada por uma única fonte (LLM do Hugging Face ou OpenAI).

    Colunas padronizadas do projeto: ``text`` é o texto original do tweet,
    ``text_normalized`` o texto após o pré-processamento (sem menções/URLs),
    ``sentiment_label`` a classe atribuída pelo modelo e ``confidence_score``
    a confiança da classificação. O modelo usado fica no arquivo de metadados
    ao lado da base, não em coluna.
    """

    id: Series[str] = pa.Field(unique=True)
    text: Series[str]
    text_normalized: Series[str]
    sentiment_label: Series[str] = pa.Field(isin=list(SENTIMENT_CLASSES))
    confidence_score: Series[float] = pa.Field(ge=0.0, le=1.0)

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
    ...         "confidence_score": [0.9],
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


def validate_labeled_source(dataframe: pl.DataFrame) -> pl.DataFrame:
    """Valida uma base rotulada por uma única fonte contra :class:`LabeledSourceSchema`.

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
    ...         "text": ["Adorei @fulano!"],
    ...         "text_normalized": ["adorei"],
    ...         "sentiment_label": ["positivo"],
    ...         "confidence_score": [0.9],
    ...     }
    ... )
    >>> validate_labeled_source(df).height
    1
    """
    try:
        return LabeledSourceSchema.validate(dataframe)
    except SchemaError as exception:
        raise DataValidationError(
            schema_name="LabeledSourceSchema", detail=str(exception)
        ) from exception
