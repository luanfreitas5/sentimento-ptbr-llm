"""Análise de *ablation study*: impacto de cada componente no desempenho do modelo.

Implementa ``configs/evaluation.yaml`` -> ``ablation``: compara a métrica
principal de uma configuração completa (baseline) contra versões com um
componente removido de cada vez (ex.: ``sem_embeddings_contextuais``),
quantificando a contribuição de cada peça do pipeline.
"""

import logging

import polars as pl

from constants.metrics import PRIMARY_METRIC
from exceptions.data import EmptyDatasetError

logger = logging.getLogger(__name__)


def calculate_ablation_impact(
    baseline_metrics: dict[str, float],
    ablated_metrics: dict[str, dict[str, float]],
    *,
    metric_name: str = PRIMARY_METRIC,
) -> pl.DataFrame:
    """Calcula o impacto da remoção de cada componente sobre a métrica principal.

    Parameters
    ----------
    baseline_metrics : dict[str, float]
        Métricas da configuração completa (todos os componentes ativos),
        no formato de :func:`metrics.classification.calculate_classification_metrics`.
    ablated_metrics : dict[str, dict[str, float]]
        Métricas de cada configuração com um componente removido, indexadas
        pelo nome do componente (ver ``configs/evaluation.yaml`` ->
        ``ablation.components``).
    metric_name : str, optional
        Métrica usada para comparação, by default
        :data:`constants.metrics.PRIMARY_METRIC`.

    Returns
    -------
    pl.DataFrame
        Uma linha por componente removido, com ``component``,
        ``baseline_value``, ``ablated_value`` e ``impact`` (queda de
        desempenho ao remover o componente; quanto maior, mais importante o
        componente), ordenada do componente mais para o menos importante.

    Raises
    ------
    EmptyDatasetError
        Se ``ablated_metrics`` estiver vazio.
    ValueError
        Se ``metric_name`` não existir em ``baseline_metrics`` ou em algum
        item de ``ablated_metrics``.

    Examples
    --------
    >>> baseline = {"f1_macro": 0.80}
    >>> ablado = {
    ...     "sem_embeddings_contextuais": {"f1_macro": 0.65},
    ...     "sem_autoencoder": {"f1_macro": 0.78},
    ... }
    >>> resultado = calculate_ablation_impact(baseline, ablado)
    >>> resultado["component"].to_list()
    ['sem_embeddings_contextuais', 'sem_autoencoder']
    """
    if len(ablated_metrics) == 0:
        raise EmptyDatasetError("ablated_metrics")
    if metric_name not in baseline_metrics:
        raise ValueError(f"metric_name '{metric_name}' não encontrado em baseline_metrics")

    baseline_value = baseline_metrics[metric_name]
    rows: list[dict[str, str | float]] = []
    for component_name, component_metrics in ablated_metrics.items():
        if metric_name not in component_metrics:
            raise ValueError(
                f"metric_name '{metric_name}' não encontrado nas métricas do componente "
                f"'{component_name}'"
            )
        ablated_value = component_metrics[metric_name]
        rows.append(
            {
                "component": component_name,
                "baseline_value": baseline_value,
                "ablated_value": ablated_value,
                "impact": baseline_value - ablated_value,
            }
        )

    result = pl.DataFrame(rows).sort("impact", descending=True)
    logger.info("Impacto de ablation calculado para %d componente(s).", result.height)
    return result


def identify_most_impactful_component(ablation_impact: pl.DataFrame) -> dict[str, str | float]:
    """Identifica o componente cuja remoção mais degrada a métrica principal.

    Parameters
    ----------
    ablation_impact : pl.DataFrame
        Saída de :func:`calculate_ablation_impact`, já ordenada por
        ``impact`` decrescente.

    Returns
    -------
    dict[str, str | float]
        Primeira linha de ``ablation_impact`` (maior impacto), como
        dicionário.

    Raises
    ------
    EmptyDatasetError
        Se ``ablation_impact`` estiver vazio.

    Examples
    --------
    >>> import polars as pl
    >>> impacto = pl.DataFrame(
    ...     {
    ...         "component": ["a", "b"],
    ...         "baseline_value": [0.8, 0.8],
    ...         "ablated_value": [0.5, 0.7],
    ...         "impact": [0.3, 0.1],
    ...     }
    ... )
    >>> identify_most_impactful_component(impacto)["component"]
    'a'
    """
    if ablation_impact.height == 0:
        raise EmptyDatasetError("ablation_impact")
    return ablation_impact.row(0, named=True)
