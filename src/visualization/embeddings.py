"""Visualização de embeddings reduzidos a duas dimensões.

Consome coordenadas 2D já produzidas por uma técnica de redução de
dimensionalidade (ex.: o autoencoder de ``src/features/reduction.py`` ou
UMAP/t-SNE em uma etapa exploratória), colorindo cada ponto pela classe de
sentimento para inspecionar visualmente a separabilidade das
representações.
"""

import logging
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from constants.labels import SENTIMENT_CLASSES
from exceptions.data import EmptyDatasetError
from visualization.theme import SENTIMENT_COLOR_PALETTE

logger = logging.getLogger(__name__)


def plot_embedding_scatter(
    coordinates_2d: np.ndarray,
    labels: Sequence[str],
    *,
    class_order: Sequence[str] = SENTIMENT_CLASSES,
    title: str = "Projeção 2D dos Embeddings por Sentimento",
) -> Figure:
    """Plota um gráfico de dispersão de embeddings reduzidos a 2D, coloridos por classe.

    Parameters
    ----------
    coordinates_2d : np.ndarray
        Matriz ``(n_amostras, 2)`` de coordenadas já reduzidas a duas
        dimensões.
    labels : Sequence[str]
        Classe de sentimento de cada amostra, mesmo tamanho de
        ``coordinates_2d``.
    class_order : Sequence[str], optional
        Ordem das classes na legenda, by default
        :data:`constants.labels.SENTIMENT_CLASSES`.
    title : str, optional
        Título do gráfico, by default "Projeção 2D dos Embeddings por Sentimento".

    Returns
    -------
    matplotlib.figure.Figure
        Figura pronta para ser salva com
        :func:`visualization.theme.save_figure`.

    Raises
    ------
    EmptyDatasetError
        Se ``coordinates_2d`` estiver vazia.
    ValueError
        Se ``coordinates_2d`` não tiver exatamente duas colunas, ou se seu
        número de linhas diferir do tamanho de ``labels``.

    Examples
    --------
    >>> import numpy as np
    >>> coordenadas = np.array([[0.1, 0.2], [0.9, 0.8]])
    >>> figura = plot_embedding_scatter(coordenadas, ["negativo", "positivo"])
    >>> figura.axes[0].get_title()
    'Projeção 2D dos Embeddings por Sentimento'
    """
    if coordinates_2d.shape[0] == 0:
        raise EmptyDatasetError("coordinates_2d")
    if coordinates_2d.shape[1] != 2:
        raise ValueError(
            f"coordinates_2d deve ter exatamente 2 colunas, recebido: {coordinates_2d.shape[1]}"
        )
    if coordinates_2d.shape[0] != len(labels):
        raise ValueError(
            "coordinates_2d e labels devem ter o mesmo número de amostras, recebido "
            f"{coordinates_2d.shape[0]} e {len(labels)}"
        )

    labels_array = np.asarray(labels)
    figure, axis = plt.subplots(figsize=(7, 6))
    for class_label in class_order:
        class_mask = labels_array == class_label
        if not np.any(class_mask):
            continue
        axis.scatter(
            coordinates_2d[class_mask, 0],
            coordinates_2d[class_mask, 1],
            label=class_label,
            color=SENTIMENT_COLOR_PALETTE.get(class_label, "#666666"),
            alpha=0.7,
            s=20,
        )
    axis.set_xlabel("Dimensão 1")
    axis.set_ylabel("Dimensão 2")
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    return figure
