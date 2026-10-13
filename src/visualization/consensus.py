"""Gráficos do consenso de rotulagem entre os rotuladores (modelos do Hugging Face e gold sets).

Desenham as tabelas de :mod:`evaluation.label_consensus` com a paleta do projeto
(:mod:`visualization.theme`). Cada função devolve uma :class:`matplotlib.figure.Figure` pronta
para :func:`visualization.theme.save_figure` (PNG 300 dpi + SVG).
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.figure import Figure

from constants.labels import SENTIMENT_CLASSES
from evaluation.label_consensus import CONSENSUS_LEVELS, NO_CONSENSUS_LABEL
from exceptions.data import EmptyDatasetError
from visualization.theme import SENTIMENT_COLOR_PALETTE

_NO_CONSENSUS_COLOR = "#CC79A7"
_LEVEL_ALPHAS: dict[str, float] = {
    "unanimidade": 1.0,
    "maioria": 0.75,
    "pluralidade": 0.5,
    "empate": 0.3,
}


def plot_pairwise_kappa_heatmap(pairwise: pl.DataFrame, source_names: list[str]) -> Figure:
    """Mapa de calor do Kappa de Cohen entre cada par de rotuladores.

    Parameters
    ----------
    pairwise : pl.DataFrame
        Saída de :func:`evaluation.label_consensus.calculate_pairwise_agreement`.
    source_names : list[str]
        Rotuladores, na ordem das linhas/colunas.

    Returns
    -------
    matplotlib.figure.Figure
        Matriz simétrica de Kappa (diagonal = 1).

    Raises
    ------
    EmptyDatasetError
        Se ``pairwise`` estiver vazio.

    Examples
    --------
    >>> plot_pairwise_kappa_heatmap(pairwise, ["a", "b"])  # doctest: +SKIP
    """
    if pairwise.is_empty():
        raise EmptyDatasetError("concordância par a par")
    index = {name: position for position, name in enumerate(source_names)}
    matrix = np.eye(len(source_names))
    for row in pairwise.iter_rows(named=True):
        kappa = np.nan if row["cohen_kappa"] is None else row["cohen_kappa"]
        matrix[index[row["source_a"]], index[row["source_b"]]] = kappa
        matrix[index[row["source_b"]], index[row["source_a"]]] = kappa
    figure, axis = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        vmin=0.0,
        vmax=1.0,
        cmap="Blues",
        xticklabels=source_names,
        yticklabels=source_names,
        cbar_kws={"label": "Kappa de Cohen"},
        ax=axis,
    )
    axis.set_title("Concordância par a par entre os rotuladores (Kappa de Cohen)")
    axis.set_xlabel("Rotulador")
    axis.set_ylabel("Rotulador")
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    return figure


def plot_consensus_by_class(by_class: pl.DataFrame) -> Figure:
    """Barras empilhadas: nível de consenso dentro de cada classe de consenso.

    Parameters
    ----------
    by_class : pl.DataFrame
        Saída de :func:`evaluation.label_consensus.summarize_consensus_by_class`.

    Returns
    -------
    matplotlib.figure.Figure
        Número de tweets por classe de consenso, empilhado por nível de consenso.

    Raises
    ------
    EmptyDatasetError
        Se ``by_class`` estiver vazio.

    Examples
    --------
    >>> plot_consensus_by_class(by_class)  # doctest: +SKIP
    """
    if by_class.is_empty():
        raise EmptyDatasetError("consenso por classe")
    labels = [*SENTIMENT_CLASSES, NO_CONSENSUS_LABEL]
    colors = {**SENTIMENT_COLOR_PALETTE, NO_CONSENSUS_LABEL: _NO_CONSENSUS_COLOR}
    data = by_class.to_dict(as_series=False)
    row_of = {label: data["consensus_label"].index(label) for label in labels}
    figure, axis = plt.subplots(figsize=(9, 5))
    bottoms = np.zeros(len(labels))
    for level in CONSENSUS_LEVELS:
        values = np.array([data[f"n_{level}"][row_of[label]] for label in labels])
        axis.bar(
            labels,
            values,
            bottom=bottoms,
            color=[colors[label] for label in labels],
            alpha=_LEVEL_ALPHAS[level],
            edgecolor="white",
            label=level,
        )
        bottoms += values
    axis.set_title("Nível de consenso por classe de consenso")
    axis.set_xlabel("Classe de consenso")
    axis.set_ylabel("Tweets (n)")
    axis.legend(title="Nível (transparência)")
    figure.tight_layout()
    return figure
