"""Contrato de dados para registros de métricas de execuções de experimento.

Usado para validar os registros exportados do MLflow Tracking
(``src/experiment/tracker.py``) antes de consolidá-los em relatórios
comparativos, garantindo rastreabilidade (SHA do Git + hash do dataset).
"""

import pandera.polars as pa
import polars as pl
from pandera.api.polars.model_config import BaseConfig
from pandera.errors import SchemaError
from pandera.typing.polars import Series

from constants.metrics import ALL_METRICS
from exceptions.data import DataValidationError


class ExperimentRunMetricSchema(pa.DataFrameModel):
    """Contrato de dados para uma métrica registrada de uma execução de experimento."""

    run_id: Series[str]
    model_name: Series[str]
    metric_name: Series[str] = pa.Field(isin=list(ALL_METRICS))
    metric_value: Series[float]
    git_sha: Series[str]
    dataset_hash: Series[str]

    class Config(BaseConfig):
        """Configuração do schema: rejeita colunas não declaradas."""

        strict = True


def validate_experiment_run_metric(dataframe: pl.DataFrame) -> pl.DataFrame:
    """Valida um DataFrame de métricas de execução contra :class:`ExperimentRunMetricSchema`.

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
    ...         "run_id": ["abc123"],
    ...         "model_name": ["logistic_regression"],
    ...         "metric_name": ["f1_macro"],
    ...         "metric_value": [0.82],
    ...         "git_sha": ["deadbeef"],
    ...         "dataset_hash": ["0f3123a4"],
    ...     }
    ... )
    >>> validate_experiment_run_metric(df).height
    1
    """
    try:
        return ExperimentRunMetricSchema.validate(dataframe)
    except SchemaError as exception:
        raise DataValidationError(
            schema_name="ExperimentRunMetricSchema", detail=str(exception)
        ) from exception
