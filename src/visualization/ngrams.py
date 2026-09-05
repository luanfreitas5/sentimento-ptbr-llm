"""Visualização de frequência de n-gramas.

Complementa ``src/features/lexical.py``: representa graficamente os
n-gramas mais frequentes de um corpus (ou de uma classe de sentimento
específica), útil para inspeção qualitativa do vocabulário discriminativo.
"""

import logging
from collections.abc import Mapping

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from exceptions.data import EmptyDatasetError

logger = logging.getLogger(__name__)


def plot_top_ngrams_bar(
    ngram_frequencies: Mapping[str, int],
    *,
    top_n: int = 15,
    title: str = "N-gramas Mais Frequentes",
) -> Figure:
    """Plota um gráfico de barras horizontais com os n-gramas mais frequentes.

    Parameters
    ----------
    ngram_frequencies : Mapping[str, int]
        Frequência de cada n-grama (ex.: saída de
        ``collections.Counter`` sobre :func:`features.lexical.extract_ngrams`).
    top_n : int, optional
        Número de n-gramas mais frequentes a exibir, by default 15.
    title : str, optional
        Título do gráfico, by default "N-gramas Mais Frequentes".

    Returns
    -------
    matplotlib.figure.Figure
        Figura pronta para ser salva com
        :func:`visualization.theme.save_figure`.

    Raises
    ------
    EmptyDatasetError
        Se ``ngram_frequencies`` estiver vazio.
    ValueError
        Se ``top_n`` for menor que 1.

    Examples
    --------
    >>> figura = plot_top_ngrams_bar({"bom_dia": 10, "não_gostei": 8, "ótimo": 5}, top_n=2)
    >>> figura.axes[0].get_title()
    'N-gramas Mais Frequentes'
    """
    if len(ngram_frequencies) == 0:
        raise EmptyDatasetError("ngram_frequencies")
    if top_n < 1:
        raise ValueError(f"top_n deve ser >= 1, recebido: {top_n}")

    top_ngrams = sorted(ngram_frequencies.items(), key=lambda item: item[1], reverse=True)[:top_n]
    ngram_labels = [ngram for ngram, _ in top_ngrams][::-1]
    ngram_counts = [count for _, count in top_ngrams][::-1]

    figure, axis = plt.subplots(figsize=(7, max(3, 0.4 * len(ngram_labels))))
    axis.barh(ngram_labels, ngram_counts, color="#0072B2")
    axis.set_xlabel("Frequência")
    axis.set_ylabel("N-grama")
    axis.set_title(title)
    figure.tight_layout()
    return figure
