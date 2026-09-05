"""Visualização de interpretabilidade de modelos: SHAP e importância de features.

Complementa a etapa de interpretabilidade do pipeline (``shap`` já é
dependência do projeto), permitindo explicar tanto modelos baseados em
árvore quanto lineares/clássicos que exponham importância ou coeficientes
por feature.
"""

import logging
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from exceptions.data import EmptyDatasetError

logger = logging.getLogger(__name__)


def plot_shap_summary(
    shap_values: np.ndarray,
    feature_matrix: np.ndarray,
    feature_names: Sequence[str],
    *,
    title: str = "Resumo de Importância SHAP",
) -> Figure:
    """Plota o resumo de valores SHAP (importância e efeito por feature).

    Parameters
    ----------
    shap_values : np.ndarray
        Matriz ``(n_amostras, n_features)`` de valores SHAP, calculada
        externamente (ex.: ``shap.Explainer``).
    feature_matrix : np.ndarray
        Matriz ``(n_amostras, n_features)`` de valores de entrada
        correspondentes a ``shap_values``.
    feature_names : Sequence[str]
        Nome de cada feature, na ordem das colunas de ``shap_values``.
    title : str, optional
        Título do gráfico, by default "Resumo de Importância SHAP".

    Returns
    -------
    matplotlib.figure.Figure
        Figura pronta para ser salva com
        :func:`visualization.theme.save_figure`.

    Raises
    ------
    EmptyDatasetError
        Se ``shap_values`` estiver vazio.

    Examples
    --------
    >>> plot_shap_summary(shap_values, feature_matrix, ["f1", "f2"])  # doctest: +SKIP
    """
    if shap_values.shape[0] == 0:
        raise EmptyDatasetError("shap_values")

    import shap

    figure = plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_values, feature_matrix, feature_names=list(feature_names), show=False, plot_size=None
    )
    figure.gca().set_title(title)
    figure.tight_layout()
    return figure


def plot_top_feature_importances(
    importances: np.ndarray,
    feature_names: Sequence[str],
    *,
    top_n: int = 20,
    title: str = "Importância das Features",
) -> Figure:
    """Plota as ``top_n`` features de maior importância (ou coeficiente absoluto).

    Parameters
    ----------
    importances : np.ndarray
        Importância (ou coeficiente) de cada feature, mesmo tamanho de
        ``feature_names`` (ex.: ``model.feature_importances_`` ou
        ``model.coef_``).
    feature_names : Sequence[str]
        Nome de cada feature, mesmo tamanho de ``importances``.
    top_n : int, optional
        Número de features mais importantes a exibir, by default 20.
    title : str, optional
        Título do gráfico, by default "Importância das Features".

    Returns
    -------
    matplotlib.figure.Figure
        Figura pronta para ser salva com
        :func:`visualization.theme.save_figure`.

    Raises
    ------
    EmptyDatasetError
        Se ``importances`` estiver vazio.
    ValueError
        Se ``importances`` e ``feature_names`` tiverem tamanhos diferentes,
        ou se ``top_n`` for menor que 1.

    Examples
    --------
    >>> import numpy as np
    >>> figura = plot_top_feature_importances(
    ...     np.array([0.5, -0.8, 0.1]), ["preco", "qualidade", "cor"], top_n=2
    ... )
    >>> figura.axes[0].get_title()
    'Importância das Features'
    """
    if len(importances) == 0:
        raise EmptyDatasetError("importances")
    if len(importances) != len(feature_names):
        raise ValueError(
            "importances e feature_names devem ter o mesmo tamanho, recebido "
            f"{len(importances)} e {len(feature_names)}"
        )
    if top_n < 1:
        raise ValueError(f"top_n deve ser >= 1, recebido: {top_n}")

    order = np.argsort(-np.abs(importances))[:top_n]
    top_names = [feature_names[index] for index in order][::-1]
    top_values = [importances[index] for index in order][::-1]

    figure, axis = plt.subplots(figsize=(7, max(3, 0.4 * len(top_names))))
    axis.barh(top_names, top_values, color="#0072B2")
    axis.set_xlabel("Importância")
    axis.set_ylabel("Feature")
    axis.set_title(title)
    figure.tight_layout()
    return figure
