"""Contratos de dados da camada de diagnóstico HypotheSAEs.

Definem a entrada (corpus com rótulos por modelo e ``agreement_score``) e a
saída (alvos binário/contínuo) de ``src/diagnostics/targets.py``. As colunas
de rótulo por modelo seguem o padrão ``lab_<modelo>`` e são validadas por
:func:`validate_diagnostic_corpus`, pois o seu número varia com o ensemble.
"""

import pandera.polars as pa
import polars as pl
from pandera.api.polars.model_config import BaseConfig
from pandera.errors import SchemaError
from pandera.typing.polars import Series

from constants.labels import SENTIMENT_CLASSES
from exceptions.data import DataValidationError

MODEL_LABEL_PREFIX = "lab_"
GOLD_LABEL_COLUMN = "gold_label"


class DiagnosticCorpusSchema(pa.DataFrameModel):
    """Colunas fixas do corpus de diagnóstico (as colunas ``lab_*`` são checadas à parte)."""

    id: Series[str] = pa.Field(unique=True)
    text_normalized: Series[str]
    agreement_score: Series[float] = pa.Field(ge=0.0, le=1.0)

    class Config(BaseConfig):
        """Configuração do schema: aceita colunas extras (``lab_*``, ``gold_label``)."""

        strict = False


class BinaryTargetSchema(pa.DataFrameModel):
    """Alvo binário (0/1) por tweet."""

    id: Series[str] = pa.Field(unique=True)
    target: Series[int] = pa.Field(isin=[0, 1])

    class Config(BaseConfig):
        """Configuração do schema: rejeita colunas não declaradas."""

        strict = True


class ContinuousTargetSchema(pa.DataFrameModel):
    """Alvo contínuo em ``[0, 1]`` por tweet (ex.: incerteza)."""

    id: Series[str] = pa.Field(unique=True)
    target: Series[float] = pa.Field(ge=0.0, le=1.0)

    class Config(BaseConfig):
        """Configuração do schema: rejeita colunas não declaradas."""

        strict = True


def list_model_label_columns(dataframe: pl.DataFrame) -> list[str]:
    """Lista as colunas de rótulo por modelo (prefixo ``lab_``) de um DataFrame.

    Parameters
    ----------
    dataframe : pl.DataFrame
        DataFrame no formato do contrato de diagnóstico.

    Returns
    -------
    list[str]
        Nomes das colunas ``lab_<modelo>``, na ordem em que aparecem.

    Examples
    --------
    >>> list_model_label_columns(pl.DataFrame({"id": ["1"], "lab_a": ["positivo"]}))
    ['lab_a']
    """
    return [name for name in dataframe.columns if name.startswith(MODEL_LABEL_PREFIX)]


def _validate_label_column_values(dataframe: pl.DataFrame, column: str) -> None:
    """Garante que os valores não nulos de ``column`` pertencem às classes de sentimento."""
    invalid = dataframe.filter(
        pl.col(column).is_not_null() & ~pl.col(column).is_in(list(SENTIMENT_CLASSES))
    )
    if invalid.height:
        raise DataValidationError(
            schema_name="DiagnosticCorpusSchema",
            detail=(
                f"coluna '{column}' com {invalid.height} rótulo(s) "
                f"fora de {list(SENTIMENT_CLASSES)}"
            ),
        )


def validate_diagnostic_corpus(dataframe: pl.DataFrame) -> pl.DataFrame:
    """Valida o corpus de diagnóstico: colunas fixas, ``lab_*`` e ``gold_label``.

    Parameters
    ----------
    dataframe : pl.DataFrame
        Corpus com ``id``, ``text_normalized``, ``agreement_score`` e ao menos
        uma coluna ``lab_<modelo>``. ``gold_label`` é opcional.

    Returns
    -------
    pl.DataFrame
        O mesmo DataFrame, quando válido.

    Raises
    ------
    DataValidationError
        Se faltar coluna obrigatória, não houver coluna ``lab_*`` ou algum
        rótulo estiver fora de :data:`constants.labels.SENTIMENT_CLASSES`.

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "id": ["1"],
    ...         "text_normalized": ["ótimo"],
    ...         "agreement_score": [0.9],
    ...         "lab_a": ["positivo"],
    ...     }
    ... )
    >>> validate_diagnostic_corpus(df).height
    1
    """
    try:
        DiagnosticCorpusSchema.validate(dataframe)
    except SchemaError as exception:
        raise DataValidationError(
            schema_name="DiagnosticCorpusSchema", detail=str(exception)
        ) from exception

    model_columns = list_model_label_columns(dataframe)
    if not model_columns:
        raise DataValidationError(
            schema_name="DiagnosticCorpusSchema",
            detail=(
                f"nenhuma coluna de rótulo por modelo (prefixo '{MODEL_LABEL_PREFIX}') encontrada"
            ),
        )
    optional_gold = [GOLD_LABEL_COLUMN] if GOLD_LABEL_COLUMN in dataframe.columns else []
    for column in [*model_columns, *optional_gold]:
        _validate_label_column_values(dataframe, column)
    return dataframe


def validate_binary_target(dataframe: pl.DataFrame) -> pl.DataFrame:
    """Valida um alvo binário contra :class:`BinaryTargetSchema`.

    Parameters
    ----------
    dataframe : pl.DataFrame
        DataFrame com colunas ``id`` e ``target`` (0/1).

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
    >>> validate_binary_target(pl.DataFrame({"id": ["1"], "target": [1]})).height
    1
    """
    try:
        return BinaryTargetSchema.validate(dataframe)
    except SchemaError as exception:
        raise DataValidationError(
            schema_name="BinaryTargetSchema", detail=str(exception)
        ) from exception


def validate_continuous_target(dataframe: pl.DataFrame) -> pl.DataFrame:
    """Valida um alvo contínuo em ``[0, 1]`` contra :class:`ContinuousTargetSchema`.

    Parameters
    ----------
    dataframe : pl.DataFrame
        DataFrame com colunas ``id`` e ``target`` (float em ``[0, 1]``).

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
    >>> validate_continuous_target(pl.DataFrame({"id": ["1"], "target": [0.3]})).height
    1
    """
    try:
        return ContinuousTargetSchema.validate(dataframe)
    except SchemaError as exception:
        raise DataValidationError(
            schema_name="ContinuousTargetSchema", detail=str(exception)
        ) from exception
