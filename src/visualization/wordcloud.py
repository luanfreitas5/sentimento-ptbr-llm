"""Geração de nuvem de palavras por classe de sentimento.

A biblioteca ``wordcloud`` é opcional (não faz parte das dependências
centrais do projeto, ver ``pyproject.toml`` -> "Core Stack"): o import
ocorre de forma tardia, dentro da função, para que o restante de
``src/visualization/`` permaneça importável sem ela.
"""

import logging
from collections.abc import Mapping

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from exceptions.data import EmptyDatasetError

logger = logging.getLogger(__name__)

_WORDCLOUD_INSTALL_MESSAGE = (
    "A biblioteca 'wordcloud' não está instalada. Instale com `uv add wordcloud` "
    "para gerar nuvens de palavras."
)


def generate_sentiment_wordcloud(
    word_frequencies: Mapping[str, int],
    *,
    title: str = "Nuvem de Palavras",
    background_color: str = "white",
    max_words: int = 100,
) -> Figure:
    """Gera uma nuvem de palavras a partir de frequências de termos.

    Parameters
    ----------
    word_frequencies : Mapping[str, int]
        Frequência de cada termo (ex.: saída de
        ``collections.Counter`` sobre os tokens de uma classe de
        sentimento).
    title : str, optional
        Título do gráfico, by default "Nuvem de Palavras".
    background_color : str, optional
        Cor de fundo da nuvem, by default "white".
    max_words : int, optional
        Número máximo de palavras exibidas, by default 100.

    Returns
    -------
    matplotlib.figure.Figure
        Figura pronta para ser salva com
        :func:`visualization.theme.save_figure`.

    Raises
    ------
    EmptyDatasetError
        Se ``word_frequencies`` estiver vazio.
    ImportError
        Se a biblioteca ``wordcloud`` não estiver instalada.

    Examples
    --------
    >>> generate_sentiment_wordcloud({"ótimo": 10, "produto": 8})  # doctest: +SKIP
    """
    if len(word_frequencies) == 0:
        raise EmptyDatasetError("word_frequencies")

    try:
        from wordcloud import WordCloud
    except ImportError as exception:
        raise ImportError(_WORDCLOUD_INSTALL_MESSAGE) from exception

    cloud = WordCloud(
        width=800, height=400, background_color=background_color, max_words=max_words
    ).generate_from_frequencies(dict(word_frequencies))

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.imshow(cloud, interpolation="bilinear")
    axis.axis("off")
    axis.set_title(title)
    figure.tight_layout()
    return figure
