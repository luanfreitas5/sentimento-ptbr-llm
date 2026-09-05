"""Gráficos de diagnóstico de calibração e de erros do modelo.

Complementa ``src/evaluation/calibration.py``: representa graficamente a
curva de confiabilidade e a distribuição de confiança das predições
corretas vs. incorretas, apoiando a identificação de excesso ou falta de
confiança do modelo.
"""

import logging
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

from exceptions.data import EmptyDatasetError

logger = logging.getLogger(__name__)


def plot_calibration_curve(
    reliability_curve: dict[str, np.ndarray], *, title: str = "Curva de Confiabilidade"
) -> Figure:
    """Plota a curva de confiabilidade (confiança predita vs. acurácia observada).

    Parameters
    ----------
    reliability_curve : dict[str, np.ndarray]
        Saída de :func:`evaluation.calibration.calculate_reliability_curve`,
        com ``bin_confidence_means``, ``bin_accuracy`` e ``bin_counts``.
    title : str, optional
        Título do gráfico, by default "Curva de Confiabilidade".

    Returns
    -------
    matplotlib.figure.Figure
        Figura pronta para ser salva com
        :func:`visualization.theme.save_figure`.

    Raises
    ------
    EmptyDatasetError
        Se todas as faixas de confiança estiverem vazias.

    Examples
    --------
    >>> import numpy as np
    >>> curva = {
    ...     "bin_confidence_means": np.array([0.6, 0.9]),
    ...     "bin_accuracy": np.array([0.5, 0.85]),
    ...     "bin_counts": np.array([2, 3]),
    ... }
    >>> figura = plot_calibration_curve(curva)
    >>> figura.axes[0].get_title()
    'Curva de Confiabilidade'
    """
    non_empty_mask = reliability_curve["bin_counts"] > 0
    if not np.any(non_empty_mask):
        raise EmptyDatasetError("reliability_curve")

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="calibração perfeita")
    axis.plot(
        reliability_curve["bin_confidence_means"][non_empty_mask],
        reliability_curve["bin_accuracy"][non_empty_mask],
        marker="o",
        label="modelo",
    )
    axis.set_xlabel("Confiança média predita")
    axis.set_ylabel("Acurácia observada")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    return figure


def plot_confidence_distribution_by_correctness(
    confidences: np.ndarray,
    correctness: Sequence[bool],
    *,
    title: str = "Distribuição de Confiança por Acerto",
) -> Figure:
    """Plota a distribuição de confiança separada entre predições corretas e incorretas.

    Um modelo bem calibrado concentra baixa confiança nos erros e alta
    confiança nos acertos; sobreposição das distribuições indica confiança
    pouco informativa.

    Parameters
    ----------
    confidences : np.ndarray
        Confiança (probabilidade da classe predita) por amostra.
    correctness : Sequence[bool]
        Indicador de acerto por amostra, mesmo tamanho de ``confidences``.
    title : str, optional
        Título do gráfico, by default "Distribuição de Confiança por Acerto".

    Returns
    -------
    matplotlib.figure.Figure
        Figura pronta para ser salva com
        :func:`visualization.theme.save_figure`.

    Raises
    ------
    EmptyDatasetError
        Se ``confidences`` estiver vazio.

    Examples
    --------
    >>> import numpy as np
    >>> figura = plot_confidence_distribution_by_correctness(
    ...     np.array([0.9, 0.6, 0.95, 0.55]), [True, False, True, False]
    ... )
    >>> figura.axes[0].get_title()
    'Distribuição de Confiança por Acerto'
    """
    if len(confidences) == 0:
        raise EmptyDatasetError("confidences")

    status_labels = ["correto" if is_correct else "incorreto" for is_correct in correctness]
    figure, axis = plt.subplots(figsize=(6, 5))
    sns.histplot(
        x=confidences,
        hue=status_labels,
        bins=10,
        kde=False,
        element="step",
        ax=axis,
        stat="density",
    )
    axis.set_xlabel("Confiança da predição")
    axis.set_ylabel("Densidade")
    axis.set_title(title)
    figure.tight_layout()
    return figure
