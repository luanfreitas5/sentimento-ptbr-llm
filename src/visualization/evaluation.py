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


def plot_reliability_with_intervals(reliability_table: pl.DataFrame, *, model: str) -> Figure:
    """Curva de confiabilidade de um modelo, antes e depois do *temperature scaling*.

    Cada ponto traz o IC de Clopper-Pearson da acurácia da faixa e o ``n`` anotado, para que
    faixas com poucas amostras não sejam lidas como calibradas (ou descalibradas).

    Parameters
    ----------
    reliability_table : pl.DataFrame
        ``calibracao_confiabilidade.csv`` (``model``, ``variant``, ``n``, ``mean_confidence``,
        ``accuracy``, ``ci_low``, ``ci_high``).
    model : str
        Modelo a desenhar.

    Returns
    -------
    matplotlib.figure.Figure
        Figura com a diagonal de calibração perfeita e uma curva por variante.

    Raises
    ------
    EmptyDatasetError
        Se não houver linhas para ``model``.

    Examples
    --------
    >>> tabela = pl.DataFrame(
    ...     {
    ...         "model": ["m"],
    ...         "variant": ["original"],
    ...         "n": [10],
    ...         "mean_confidence": [0.8],
    ...         "accuracy": [0.7],
    ...         "ci_low": [0.4],
    ...         "ci_high": [0.9],
    ...     }
    ... )
    >>> plot_reliability_with_intervals(tabela, model="m").axes[0].get_xlabel()
    'Confiança média predita'
    """
    data = reliability_table.filter(pl.col("model") == model)
    if data.is_empty():
        raise EmptyDatasetError(f"tabela de confiabilidade de {model}")
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="calibração perfeita")
    styles = {"original": ("#D55E00", "original"), "temperature_scaling": ("#0072B2", "calibrado")}
    for variant, (color, label) in styles.items():
        part = data.filter(pl.col("variant") == variant).sort("mean_confidence")
        if part.is_empty():
            continue
        accuracy = part["accuracy"].to_numpy()
        errors = np.vstack(
            [accuracy - part["ci_low"].to_numpy(), part["ci_high"].to_numpy() - accuracy]
        )
        axis.errorbar(
            part["mean_confidence"].to_numpy(),
            accuracy,
            yerr=errors,
            marker="o",
            capsize=3,
            color=color,
            label=label,
        )
        for x_value, y_value, count in zip(
            part["mean_confidence"], accuracy, part["n"], strict=True
        ):
            axis.annotate(
                f"n={count}",
                (x_value, y_value),
                fontsize=6,
                xytext=(3, -9),
                textcoords="offset points",
                color=color,
            )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Confiança média predita")
    axis.set_ylabel("Acurácia observada (IC 95%)")
    axis.set_title(f"Curva de confiabilidade (teste) — {model}")
    axis.legend(loc="upper left")
    figure.tight_layout()
    return figure


def plot_risk_coverage_curves(risk_coverage_table: pl.DataFrame) -> Figure:
    """Macro-F1 contra a fração de tweets respondidos, uma linha por modelo, com IC.

    Parameters
    ----------
    risk_coverage_table : pl.DataFrame
        ``risco_cobertura.csv`` (``model``, ``coverage``, ``f1_macro`` e seu IC).

    Returns
    -------
    matplotlib.figure.Figure
        Figura com uma linha por modelo e faixa de IC.

    Raises
    ------
    EmptyDatasetError
        Se ``risk_coverage_table`` estiver vazia.

    Examples
    --------
    >>> tabela = pl.DataFrame(
    ...     {
    ...         "model": ["m", "m"],
    ...         "coverage": [0.5, 1.0],
    ...         "f1_macro": [0.8, 0.7],
    ...         "f1_macro_ci_low": [0.75, 0.68],
    ...         "f1_macro_ci_high": [0.85, 0.72],
    ...     }
    ... )
    >>> plot_risk_coverage_curves(tabela).axes[0].get_xlabel()
    'Cobertura (fração de tweets respondidos)'
    """
    if risk_coverage_table.is_empty():
        raise EmptyDatasetError("tabela risco-cobertura")
    figure, axis = plt.subplots(figsize=(8, 5))
    for model in risk_coverage_table["model"].unique(maintain_order=True):
        part = risk_coverage_table.filter(pl.col("model") == model).sort("coverage")
        coverage = part["coverage"].to_numpy()
        axis.plot(coverage, part["f1_macro"].to_numpy(), marker="o", label=model)
        axis.fill_between(
            coverage,
            part["f1_macro_ci_low"].to_numpy(),
            part["f1_macro_ci_high"].to_numpy(),
            alpha=0.15,
        )
    axis.set_xlabel("Cobertura (fração de tweets respondidos)")
    axis.set_ylabel("F1-macro nos respondidos (IC 95%)")
    axis.set_title("Predição com abstenção: F1-macro contra cobertura (teste)")
    axis.legend(title="Modelo", fontsize=8)
    figure.tight_layout()
    return figure
