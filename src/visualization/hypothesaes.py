"""Visualização das hipóteses de inconsistência geradas pelo HypotheSAEs.

Complementa ``evaluation.hypothesaes_report``: plota o poder preditivo de
cada hipótese como um gráfico de barras horizontais divergente (hipóteses
associadas a baixa confiança de um lado, alta confiança do outro), com a
opacidade de cada barra proporcional à fidelidade da interpretação
(``f1_fidelity_score``, quando disponível).
"""

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from exceptions.data import EmptyDatasetError

logger = logging.getLogger(__name__)

DIVERGING_PALETTE = "vlag"
_MINIMUM_BAR_OPACITY = 0.35
_LABEL_MAX_CHARS = 58
_FIDELITY_COLUMN = "f1_fidelity_score"
_INTERPRETATION_COLUMN = "interpretation"


def _apply_fidelity_styling(axis: Any, bars: Any, fidelity_scores: np.ndarray) -> None:
    """Ajusta a opacidade das barras e anota o F1 de fidelidade de cada uma.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Eixo onde as barras foram desenhadas.
    bars : matplotlib.container.BarContainer
        Barras retornadas por ``axis.barh``, mesma ordem de ``fidelity_scores``.
    fidelity_scores : np.ndarray
        Fidelidade da interpretação (``f1_fidelity_score``) de cada barra.
    """
    for bar, fidelity_score in zip(bars, fidelity_scores, strict=True):
        bar.set_alpha(
            _MINIMUM_BAR_OPACITY + (1 - _MINIMUM_BAR_OPACITY) * float(np.clip(fidelity_score, 0, 1))
        )
    for bar, fidelity_score in zip(bars, fidelity_scores, strict=True):
        bar_width = bar.get_width()
        axis.annotate(
            f"F1={fidelity_score:.2f}",
            (bar_width, bar.get_y() + bar.get_height() / 2),
            xytext=(4 if bar_width >= 0 else -4, 0),
            textcoords="offset points",
            va="center",
            ha="left" if bar_width >= 0 else "right",
            fontsize=8,
            color="#5a5a5a",
        )


def plot_hypotheses_bars(
    hypotheses_table: pl.DataFrame,
    *,
    target_column: str,
    title: str = "Hipóteses de inconsistência x baixa confiança",
) -> Figure:
    """Plota o poder preditivo de cada hipótese como barras horizontais divergentes.

    Parameters
    ----------
    hypotheses_table : pl.DataFrame
        Saída de
        :func:`evaluation.hypothesaes_report.build_top_hypotheses_table`.
    target_column : str
        Nome da coluna de poder preditivo (eixo x do gráfico).
    title : str, optional
        Título do gráfico, by default
        "Hipóteses de inconsistência x baixa confiança".

    Returns
    -------
    matplotlib.figure.Figure
        Figura pronta para ser salva com
        :func:`visualization.theme.save_figure`.

    Raises
    ------
    EmptyDatasetError
        Se ``hypotheses_table`` estiver vazio.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"interpretation": ["usa ironia"], "target_separation_score": [0.3]})
    >>> figura = plot_hypotheses_bars(df, target_column="target_separation_score")
    >>> figura.axes[0].get_title()
    'Hipóteses de inconsistência x baixa confiança'
    """
    if hypotheses_table.height == 0:
        raise EmptyDatasetError("hypotheses_table")

    labels = [
        text[:_LABEL_MAX_CHARS] for text in hypotheses_table[_INTERPRETATION_COLUMN].to_list()
    ]
    values = hypotheses_table[target_column].to_numpy()
    has_fidelity = _FIDELITY_COLUMN in hypotheses_table.columns

    figure, axis = plt.subplots(figsize=(12, max(3, 0.4 * len(labels))))
    normalizer = Normalize(-np.abs(values).max(), np.abs(values).max())
    colors = sns.color_palette(DIVERGING_PALETTE, as_cmap=True)(normalizer(values))
    bars = axis.barh(labels, values, color=colors)

    if has_fidelity:
        fidelity_scores = hypotheses_table[_FIDELITY_COLUMN].to_numpy()
        _apply_fidelity_styling(axis, bars, fidelity_scores)

    axis.axvline(0, color="black", linewidth=0.8)
    axis.invert_yaxis()
    axis.set(title=title, xlabel=target_column, ylabel="")
    figure.tight_layout()
    return figure
