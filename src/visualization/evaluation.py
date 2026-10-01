"""Gráficos da avaliação no teste: comparação entre modelos e ablação.

Desenham as tabelas produzidas pela etapa ``evaluate`` com a paleta do projeto
(:mod:`visualization.theme`). Cada função devolve uma
:class:`matplotlib.figure.Figure` pronta para
:func:`visualization.theme.save_figure` (PNG 300 dpi + SVG).
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from constants.metrics import PRIMARY_METRIC
from exceptions.data import EmptyDatasetError
from visualization.theme import CATEGORY_COLOR_PALETTE

_CATEGORY_DISPLAY_NAMES: dict[str, str] = {
    "classical": "Baseline + ML tradicional",
    "deep_learning": "Deep Learning",
    "transformer": "Transformer",
    "llm": "LLM open-source",
}


def plot_model_comparison(metrics_table: pl.DataFrame, *, metric: str = PRIMARY_METRIC) -> Figure:
    """Barras horizontais da métrica principal por modelo, com o IC bootstrap como barra de erro.

    Parameters
    ----------
    metrics_table : pl.DataFrame
        Tabela de ``evaluate`` (``model``, ``category``, ``<metric>``, ``<metric>_ci_low`` e
        ``<metric>_ci_high``).
    metric : str, optional
        Métrica exibida, by default ``constants.metrics.PRIMARY_METRIC``.

    Returns
    -------
    matplotlib.figure.Figure
        Figura com um modelo por barra, ordenados do melhor para o pior e coloridos por categoria.

    Raises
    ------
    EmptyDatasetError
        Se ``metrics_table`` estiver vazia.

    Examples
    --------
    >>> tabela = pl.DataFrame(
    ...     {
    ...         "model": ["svm"],
    ...         "category": ["classical"],
    ...         "f1_macro": [0.7],
    ...         "f1_macro_ci_low": [0.65],
    ...         "f1_macro_ci_high": [0.75],
    ...     }
    ... )
    >>> plot_model_comparison(tabela).axes[0].get_xlabel()
    'F1-macro (IC 95%)'
    """
    if metrics_table.is_empty():
        raise EmptyDatasetError("tabela de métricas")
    data = metrics_table.sort(metric)
    values = data[metric].to_numpy()
    errors = np.vstack(
        [
            values - data[f"{metric}_ci_low"].to_numpy(),
            data[f"{metric}_ci_high"].to_numpy() - values,
        ]
    )
    colors = [CATEGORY_COLOR_PALETTE[category] for category in data["category"]]

    figure, axis = plt.subplots(figsize=(9, max(3.0, 0.5 * data.height + 1.5)))
    axis.barh(data["model"].to_list(), values, xerr=errors, color=colors, capsize=3)
    axis.set_title("Desempenho no conjunto de teste por modelo")
    axis.set_xlabel("F1-macro (IC 95%)" if metric == PRIMARY_METRIC else f"{metric} (IC 95%)")
    axis.set_ylabel("Modelo")
    axis.set_xlim(0, 1)
    present = [
        category for category in _CATEGORY_DISPLAY_NAMES if category in set(data["category"])
    ]
    axis.legend(
        handles=[
            Patch(color=CATEGORY_COLOR_PALETTE[category], label=_CATEGORY_DISPLAY_NAMES[category])
            for category in present
        ],
        title="Categoria",
        loc="lower right",
    )
    figure.tight_layout()
    return figure


def plot_ablation_impact(ablation_table: pl.DataFrame) -> Figure:
    """Barras do impacto (queda de F1-macro) de cada componente removido, com IC pareado.

    Parameters
    ----------
    ablation_table : pl.DataFrame
        Saída de :func:`evaluation.ablation.run_pipeline_ablation` (``component``, ``impact``,
        ``impact_ci_low`` e ``impact_ci_high``).

    Returns
    -------
    matplotlib.figure.Figure
        Figura com um componente por barra, do mais para o menos impactante.

    Raises
    ------
    EmptyDatasetError
        Se ``ablation_table`` estiver vazia.

    Examples
    --------
    >>> tabela = pl.DataFrame(
    ...     {
    ...         "component": ["sem_bigramas"],
    ...         "impact": [0.02],
    ...         "impact_ci_low": [0.0],
    ...         "impact_ci_high": [0.04],
    ...     }
    ... )
    >>> plot_ablation_impact(tabela).axes[0].get_ylabel()
    'Componente removido'
    """
    if ablation_table.is_empty():
        raise EmptyDatasetError("tabela de ablação")
    data = ablation_table.sort("impact")
    values = data["impact"].to_numpy()
    errors = np.vstack(
        [values - data["impact_ci_low"].to_numpy(), data["impact_ci_high"].to_numpy() - values]
    )
    figure, axis = plt.subplots(figsize=(8, max(3.0, 0.6 * data.height + 1.5)))
    axis.barh(data["component"].to_list(), values, xerr=errors, color="#0072B2", capsize=3)
    axis.axvline(0, color="#333333", linewidth=0.8)
    axis.set_title("Ablação do pipeline clássico (conjunto de validação)")
    axis.set_xlabel("Queda de F1-macro ao remover o componente (IC 95% pareado)")
    axis.set_ylabel("Componente removido")
    figure.tight_layout()
    return figure
