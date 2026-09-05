"""Gráficos de distribuição de classes e de comprimento de texto.

Apoiam a análise exploratória inicial (``notebooks/01_eda.ipynb``) e a
identificação de desbalanceamento entre classes de sentimento e de
diferenças de comprimento de texto entre fontes de dados.
"""

import logging
from collections import Counter
from collections.abc import Sequence

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

from constants.labels import SENTIMENT_CLASSES
from exceptions.data import EmptyDatasetError
from visualization.theme import SENTIMENT_COLOR_PALETTE

logger = logging.getLogger(__name__)


def plot_class_distribution(
    labels: Sequence[str],
    *,
    class_order: Sequence[str] = SENTIMENT_CLASSES,
    title: str = "Distribuição das Classes de Sentimento",
) -> Figure:
    """Plota a contagem de amostras por classe de sentimento.

    Parameters
    ----------
    labels : Sequence[str]
        Rótulos de sentimento de cada amostra.
    class_order : Sequence[str], optional
        Ordem das classes no eixo, by default
        :data:`constants.labels.SENTIMENT_CLASSES`.
    title : str, optional
        Título do gráfico, by default "Distribuição das Classes de Sentimento".

    Returns
    -------
    matplotlib.figure.Figure
        Figura pronta para ser salva com
        :func:`visualization.theme.save_figure`.

    Raises
    ------
    EmptyDatasetError
        Se ``labels`` estiver vazio.

    Examples
    --------
    >>> figura = plot_class_distribution(["positivo", "positivo", "negativo"])
    >>> figura.axes[0].get_title()
    'Distribuição das Classes de Sentimento'
    """
    if len(labels) == 0:
        raise EmptyDatasetError("labels")

    label_counts = Counter(labels)
    counts = [label_counts.get(label, 0) for label in class_order]
    palette = [SENTIMENT_COLOR_PALETTE.get(label, "#666666") for label in class_order]

    figure, axis = plt.subplots(figsize=(6, 5))
    sns.barplot(
        x=list(class_order), y=counts, hue=list(class_order), palette=palette, legend=False, ax=axis
    )
    axis.set_xlabel("Classe de sentimento")
    axis.set_ylabel("Número de amostras")
    axis.set_title(title)
    figure.tight_layout()
    return figure


def plot_text_length_distribution(
    text_lengths: Sequence[int],
    *,
    class_labels: Sequence[str] | None = None,
    title: str = "Distribuição do Comprimento de Texto",
) -> Figure:
    """Plota a distribuição do comprimento de texto (em tokens ou caracteres).

    Parameters
    ----------
    text_lengths : Sequence[int]
        Comprimento de cada texto.
    class_labels : Sequence[str] | None, optional
        Classe de sentimento de cada amostra, mesmo tamanho de
        ``text_lengths``; quando informado, sobrepõe uma distribuição por
        classe, by default None.
    title : str, optional
        Título do gráfico, by default "Distribuição do Comprimento de Texto".

    Returns
    -------
    matplotlib.figure.Figure
        Figura pronta para ser salva com
        :func:`visualization.theme.save_figure`.

    Raises
    ------
    EmptyDatasetError
        Se ``text_lengths`` estiver vazio.

    Examples
    --------
    >>> figura = plot_text_length_distribution([10, 20, 15, 30])
    >>> figura.axes[0].get_title()
    'Distribuição do Comprimento de Texto'
    """
    if len(text_lengths) == 0:
        raise EmptyDatasetError("text_lengths")

    figure, axis = plt.subplots(figsize=(6, 5))
    sns.histplot(x=list(text_lengths), hue=list(class_labels) if class_labels else None, ax=axis)
    axis.set_xlabel("Comprimento do texto")
    axis.set_ylabel("Número de amostras")
    axis.set_title(title)
    figure.tight_layout()
    return figure
