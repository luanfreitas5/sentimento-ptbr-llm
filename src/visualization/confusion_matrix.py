"""Visualização da matriz de confusão.

Complementa ``src/metrics/classification.py``: representa graficamente a
matriz calculada por
:func:`metrics.classification.calculate_confusion_matrix`, em contagens
absolutas ou normalizada por linha (equivalente ao recall por classe).
"""

import logging
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

from constants.labels import SENTIMENT_CLASSES
from exceptions.data import EmptyDatasetError

logger = logging.getLogger(__name__)


def plot_confusion_matrix_heatmap(
    confusion_matrix: np.ndarray,
    *,
    labels: Sequence[str] = SENTIMENT_CLASSES,
    normalize: bool = False,
    title: str = "Matriz de Confusão",
) -> Figure:
    """Plota a matriz de confusão como um mapa de calor anotado.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        Matriz ``(n_classes, n_classes)``, ver
        :func:`metrics.classification.calculate_confusion_matrix`.
    labels : Sequence[str], optional
        Rótulos das classes, na ordem das linhas/colunas, by default
        :data:`constants.labels.SENTIMENT_CLASSES`.
    normalize : bool, optional
        Se ``True``, normaliza cada linha pelo total (equivalente ao
        recall por classe), by default False.
    title : str, optional
        Título do gráfico, by default "Matriz de Confusão".

    Returns
    -------
    matplotlib.figure.Figure
        Figura pronta para ser salva com
        :func:`visualization.theme.save_figure`.

    Raises
    ------
    EmptyDatasetError
        Se ``confusion_matrix`` estiver vazia.

    Examples
    --------
    >>> import numpy as np
    >>> matriz = np.array([[5, 1], [2, 8]])
    >>> figura = plot_confusion_matrix_heatmap(matriz, labels=["negativo", "positivo"])
    >>> figura.axes[0].get_title()
    'Matriz de Confusão'
    """
    if confusion_matrix.size == 0:
        raise EmptyDatasetError("confusion_matrix")

    if normalize:
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        matrix_to_plot = np.divide(
            confusion_matrix.astype(float),
            row_sums,
            out=np.zeros(confusion_matrix.shape, dtype=float),
            where=row_sums != 0,
        )
        value_format = ".2f"
    else:
        matrix_to_plot = confusion_matrix
        value_format = "d"

    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix_to_plot,
        annot=True,
        fmt=value_format,
        cmap="Blues",
        xticklabels=list(labels),
        yticklabels=list(labels),
        ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel("Classe predita")
    axis.set_ylabel("Classe verdadeira")
    figure.tight_layout()
    return figure
