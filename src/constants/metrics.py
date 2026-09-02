"""Nomes de métricas de avaliação usadas no projeto.

Reflete ``configs/evaluation.yaml``: a métrica principal é o F1-macro,
robusto ao desbalanceamento típico entre as três classes de sentimento.
"""

from exceptions.data import DataValidationError

PRIMARY_METRIC = "f1_macro"

SECONDARY_METRICS: tuple[str, ...] = (
    "mcc",
    "f1_weighted",
    "accuracy",
    "precision_macro",
    "recall_macro",
)

RANKING_METRICS: tuple[str, ...] = ("roc_auc_ovr", "pr_auc_ovr")

OPERATIONAL_METRICS: tuple[str, ...] = ("inference_time_ms", "computational_cost")

ALL_METRICS: tuple[str, ...] = (
    PRIMARY_METRIC,
    *SECONDARY_METRICS,
    *RANKING_METRICS,
    *OPERATIONAL_METRICS,
)


def validate_metric_name(metric_name: str) -> str:
    """Valida se um nome de métrica é reconhecido pelo projeto.

    Parameters
    ----------
    metric_name : str
        Nome da métrica a ser validado (ex.: ``"f1_macro"``).

    Returns
    -------
    str
        O próprio nome da métrica, quando válido.

    Raises
    ------
    DataValidationError
        Se o nome da métrica não pertencer a :data:`ALL_METRICS`.

    Examples
    --------
    >>> validate_metric_name("f1_macro")
    'f1_macro'
    """
    if metric_name not in ALL_METRICS:
        raise DataValidationError(
            schema_name="metrics",
            detail=f"métrica '{metric_name}' não reconhecida. Métricas disponíveis: {ALL_METRICS}",
        )
    return metric_name
