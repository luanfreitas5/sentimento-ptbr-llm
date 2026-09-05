"""Visualização das curvas ROC e Precisão-Revocação (esquema *one-vs-rest*).

Complementa ``src/metrics/ranking.py``: representa graficamente, por
classe, as curvas usadas para calcular ROC-AUC e PR-AUC no esquema
*one-vs-rest*, apropriado ao cenário multiclasse de sentimento.
"""

import logging
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize

from constants.labels import SENTIMENT_CLASSES
from exceptions.data import EmptyDatasetError
from visualization.theme import SENTIMENT_COLOR_PALETTE

logger = logging.getLogger(__name__)


def plot_roc_curves_one_vs_rest(
    y_true: Sequence[str],
    y_score: np.ndarray,
    *,
    labels: Sequence[str] = SENTIMENT_CLASSES,
    title: str = "Curvas ROC (one-vs-rest)",
) -> Figure:
    """Plota uma curva ROC por classe, no esquema *one-vs-rest*.

    Parameters
    ----------
    y_true : Sequence[str]
        Rótulos de sentimento verdadeiros.
    y_score : np.ndarray
        Matriz ``(n_amostras, n_classes)`` de probabilidades preditas, na
        ordem de ``labels``.
    labels : Sequence[str], optional
        Classes do problema, by default
        :data:`constants.labels.SENTIMENT_CLASSES`.
    title : str, optional
        Título do gráfico, by default "Curvas ROC (one-vs-rest)".

    Returns
    -------
    matplotlib.figure.Figure
        Figura pronta para ser salva com
        :func:`visualization.theme.save_figure`.

    Raises
    ------
    EmptyDatasetError
        Se ``y_true`` estiver vazio.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = ["negativo", "positivo", "negativo", "positivo"]
    >>> y_score = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]])
    >>> figura = plot_roc_curves_one_vs_rest(y_true, y_score, labels=["negativo", "positivo"])
    >>> len(figura.axes[0].lines) >= 2
    True
    """
    if len(y_true) == 0:
        raise EmptyDatasetError("y_true")

    y_true_binarized = label_binarize(y_true, classes=list(labels))
    figure, axis = plt.subplots(figsize=(6, 5))
    for class_index, class_label in enumerate(labels):
        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_true_binarized[:, class_index], y_score[:, class_index]
        )
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=class_label,
            color=SENTIMENT_COLOR_PALETTE.get(class_label),
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="acaso")
    axis.set_xlabel("Taxa de falsos positivos")
    axis.set_ylabel("Taxa de verdadeiros positivos")
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    return figure


def plot_precision_recall_curves_one_vs_rest(
    y_true: Sequence[str],
    y_score: np.ndarray,
    *,
    labels: Sequence[str] = SENTIMENT_CLASSES,
    title: str = "Curvas Precisão-Revocação (one-vs-rest)",
) -> Figure:
    """Plota uma curva Precisão-Revocação por classe, no esquema *one-vs-rest*.

    Mais informativa que a curva ROC sob desbalanceamento de classes.

    Parameters
    ----------
    y_true : Sequence[str]
        Rótulos de sentimento verdadeiros.
    y_score : np.ndarray
        Matriz ``(n_amostras, n_classes)`` de probabilidades preditas, na
        ordem de ``labels``.
    labels : Sequence[str], optional
        Classes do problema, by default
        :data:`constants.labels.SENTIMENT_CLASSES`.
    title : str, optional
        Título do gráfico, by default "Curvas Precisão-Revocação (one-vs-rest)".

    Returns
    -------
    matplotlib.figure.Figure
        Figura pronta para ser salva com
        :func:`visualization.theme.save_figure`.

    Raises
    ------
    EmptyDatasetError
        Se ``y_true`` estiver vazio.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = ["negativo", "positivo", "negativo", "positivo"]
    >>> y_score = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]])
    >>> figura = plot_precision_recall_curves_one_vs_rest(
    ...     y_true, y_score, labels=["negativo", "positivo"]
    ... )
    >>> len(figura.axes[0].lines) >= 2
    True
    """
    if len(y_true) == 0:
        raise EmptyDatasetError("y_true")

    y_true_binarized = label_binarize(y_true, classes=list(labels))
    figure, axis = plt.subplots(figsize=(6, 5))
    for class_index, class_label in enumerate(labels):
        precision, recall, _ = precision_recall_curve(
            y_true_binarized[:, class_index], y_score[:, class_index]
        )
        axis.plot(
            recall, precision, label=class_label, color=SENTIMENT_COLOR_PALETTE.get(class_label)
        )
    axis.set_xlabel("Revocação")
    axis.set_ylabel("Precisão")
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    return figure
