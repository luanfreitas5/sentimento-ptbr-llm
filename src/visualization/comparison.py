"""Gráficos da comparação entre o modelo primário do Hugging Face e uma base de referência.

A referência é um gold set (LLM via API OpenAI-compatível) ou o segundo modelo do Hugging Face;
os nomes exibidos nos gráficos (os dos modelos) são parametrizados por ``primary_label`` e
``reference_label``.

Desenham as tabelas de :mod:`evaluation.llm_comparison` com a paleta do
projeto (:mod:`visualization.theme`). Cada função devolve uma
:class:`matplotlib.figure.Figure` pronta para
:func:`visualization.theme.save_figure` (PNG 300 dpi + SVG).
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.figure import Figure

from constants.labels import SENTIMENT_CLASSES
from evaluation.llm_comparison import MODEL_NAMES
from exceptions.data import EmptyDatasetError
from visualization.theme import MODEL_COLOR_PALETTE, SENTIMENT_COLOR_PALETTE

_HF, _OA = MODEL_NAMES
DEFAULT_REFERENCE_LABEL = "Gold set (API)"
DEFAULT_PRIMARY_LABEL = "LLM Hugging Face"


def _display_names(primary_label: str, reference_label: str) -> dict[str, str]:
    """Nome exibido de cada modelo; o segundo é a base de referência."""
    return {_HF: primary_label, _OA: reference_label}


def plot_class_distribution_by_model(
    distribution: pl.DataFrame,
    *,
    primary_label: str = DEFAULT_PRIMARY_LABEL,
    reference_label: str = DEFAULT_REFERENCE_LABEL,
) -> Figure:
    """Barras agrupadas com a proporção de cada classe de sentimento, por modelo.

    Parameters
    ----------
    distribution : pl.DataFrame
        Saída de :func:`evaluation.llm_comparison.calculate_class_distribution`.
    primary_label : str, optional
        Nome exibido do modelo primário, by default "LLM Hugging Face".
    reference_label : str, optional
        Nome exibido da base de referência, by default "Gold set (API)".

    Returns
    -------
    matplotlib.figure.Figure
        Figura com a proporção (%) de tweets por classe e modelo.

    Raises
    ------
    EmptyDatasetError
        Se ``distribution`` estiver vazio.

    Examples
    --------
    >>> distribution = pl.DataFrame(
    ...     {
    ...         "model": ["huggingface", "openai"],
    ...         "sentiment_label": ["positivo", "positivo"],
    ...         "count": [1, 2],
    ...         "proportion": [0.5, 1.0],
    ...     }
    ... )
    >>> plot_class_distribution_by_model(distribution).axes[0].get_ylabel()
    'Tweets (%)'
    """
    if distribution.is_empty():
        raise EmptyDatasetError("distribuição de classes")
    display_names = _display_names(primary_label, reference_label)
    data = distribution.with_columns(
        (pl.col("proportion") * 100).alias("percentage"),
        pl.col("model").replace(display_names).alias("model_name"),
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=data.to_dict(as_series=False),
        x="sentiment_label",
        y="percentage",
        hue="model_name",
        order=list(SENTIMENT_CLASSES),
        palette={display_names[m]: color for m, color in MODEL_COLOR_PALETTE.items()},
        ax=axis,
    )
    axis.set_title("Distribuição das classes de sentimento por modelo")
    axis.set_xlabel("Classe de sentimento")
    axis.set_ylabel("Tweets (%)")
    axis.legend(title="Modelo")
    figure.tight_layout()
    return figure


def plot_confidence_comparison(
    frame: pl.DataFrame,
    *,
    primary_label: str = DEFAULT_PRIMARY_LABEL,
    reference_label: str = DEFAULT_REFERENCE_LABEL,
) -> Figure:
    """Distribuição da confiança de cada modelo e dispersão das duas confianças por tweet.

    Parameters
    ----------
    frame : pl.DataFrame
        Saída de :func:`evaluation.llm_comparison.build_comparison_frame`.
    primary_label : str, optional
        Nome exibido do modelo primário, by default "LLM Hugging Face".
    reference_label : str, optional
        Nome exibido da base de referência, by default "Gold set (API)".

    Returns
    -------
    matplotlib.figure.Figure
        Figura com dois painéis: histogramas sobrepostos e mapa de densidade
        (Hugging Face no eixo x, referência no eixo y; a diagonal é a concordância de confiança).

    Raises
    ------
    EmptyDatasetError
        Se ``frame`` estiver vazio.

    Examples
    --------
    >>> frame = pl.DataFrame(
    ...     {"confidence_huggingface": [0.9, 0.8], "confidence_openai": [0.7, 0.95]}
    ... )
    >>> len(plot_confidence_comparison(frame).axes)
    3
    """
    if frame.is_empty():
        raise EmptyDatasetError("frame de comparação")
    display_names = _display_names(primary_label, reference_label)
    figure, (hist_axis, density_axis) = plt.subplots(1, 2, figsize=(12, 5))
    for model in MODEL_NAMES:
        sns.histplot(
            frame[f"confidence_{model}"].to_numpy(),
            bins=20,
            binrange=(0, 1),
            stat="proportion",
            alpha=0.55,
            color=MODEL_COLOR_PALETTE[model],
            label=display_names[model],
            ax=hist_axis,
        )
    hist_axis.set_title("Distribuição da confiança por modelo")
    hist_axis.set_xlabel("Confiança (0 a 1)")
    hist_axis.set_ylabel("Proporção de tweets")
    hist_axis.legend(title="Modelo")

    hexbin = density_axis.hexbin(
        frame[f"confidence_{_HF}"].to_numpy(),
        frame[f"confidence_{_OA}"].to_numpy(),
        gridsize=20,
        extent=(0, 1, 0, 1),
        cmap="Blues",
        mincnt=1,
    )
    density_axis.plot([0, 1], [0, 1], linestyle="--", color="#555555", linewidth=1)
    density_axis.set_title(f"Confiança por tweet: {primary_label} vs. {reference_label}")
    density_axis.set_xlabel(f"Confiança — {primary_label}")
    density_axis.set_ylabel(f"Confiança — {reference_label}")
    figure.colorbar(hexbin, ax=density_axis, label="Tweets")
    figure.tight_layout()
    return figure


def plot_agreement_by_text_length(length_analysis: pl.DataFrame) -> Figure:
    """Taxa de concordância (com IC de Wilson) por faixa de tamanho do texto.

    Parameters
    ----------
    length_analysis : pl.DataFrame
        Saída de :func:`evaluation.llm_comparison.analyze_by_text_length`.

    Returns
    -------
    matplotlib.figure.Figure
        Figura com a concordância por faixa de palavras.

    Raises
    ------
    EmptyDatasetError
        Se ``length_analysis`` estiver vazio.

    Examples
    --------
    >>> table = pl.DataFrame(
    ...     {
    ...         "length_bin": ["1-3 palavras"],
    ...         "agreement_rate": [0.7],
    ...         "agreement_rate_ci_lower": [0.6],
    ...         "agreement_rate_ci_upper": [0.8],
    ...     }
    ... )
    >>> plot_agreement_by_text_length(table).axes[0].get_ylabel()
    'Concordância (%)'
    """
    if length_analysis.is_empty():
        raise EmptyDatasetError("análise por tamanho do texto")
    rates = length_analysis["agreement_rate"].to_numpy() * 100
    # clip: o IC pode divergir do ponto por ruído de ponto flutuante (yerr não aceita negativos)
    lower = np.clip(rates - length_analysis["agreement_rate_ci_lower"].to_numpy() * 100, 0.0, None)
    upper = np.clip(length_analysis["agreement_rate_ci_upper"].to_numpy() * 100 - rates, 0.0, None)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.errorbar(
        length_analysis["length_bin"].to_list(),
        rates,
        yerr=[lower, upper],
        marker="o",
        capsize=4,
        color=SENTIMENT_COLOR_PALETTE["positivo"],
    )
    axis.set_title("Concordância entre os modelos por tamanho do texto (IC de Wilson)")
    axis.set_xlabel("Faixa de tamanho do texto normalizado")
    axis.set_ylabel("Concordância (%)")
    axis.set_ylim(0, 100)
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    return figure


def plot_divergence_transitions(
    transitions: pl.DataFrame,
    *,
    top_n: int = 6,
    primary_label: str = DEFAULT_PRIMARY_LABEL,
    reference_label: str = DEFAULT_REFERENCE_LABEL,
) -> Figure:
    """Barras horizontais com as principais migrações de classe (modelo primário → referência).

    Parameters
    ----------
    transitions : pl.DataFrame
        Primeiro elemento de
        :func:`evaluation.llm_comparison.summarize_classification_differences`.
    top_n : int, optional
        Quantidade de pares exibidos, by default 6 (todos os pares possíveis entre 3 classes).
    primary_label : str, optional
        Nome exibido do modelo primário, by default "LLM Hugging Face".
    reference_label : str, optional
        Nome exibido da base de referência, by default "Gold set (API)".

    Returns
    -------
    matplotlib.figure.Figure
        Figura com o número de tweets por par de classes divergentes.

    Raises
    ------
    EmptyDatasetError
        Se não houver divergências (``transitions`` vazio).

    Examples
    --------
    >>> transitions = pl.DataFrame(
    ...     {"label_huggingface": ["positivo"], "label_openai": ["neutro"], "n_tweets": [3]}
    ... )
    >>> plot_divergence_transitions(transitions).axes[0].get_xlabel()
    'Tweets divergentes'
    """
    if transitions.is_empty():
        raise EmptyDatasetError("divergências entre os modelos")
    top = transitions.head(top_n)
    names = [
        f"{hf} → {oa}"
        for hf, oa in zip(top[f"label_{_HF}"].to_list(), top[f"label_{_OA}"].to_list(), strict=True)
    ]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.barh(names[::-1], top["n_tweets"].to_list()[::-1], color=MODEL_COLOR_PALETTE[_HF])
    axis.set_title(f"Principais diferenças de classificação ({primary_label} → {reference_label})")
    axis.set_xlabel("Tweets divergentes")
    axis.set_ylabel(f"Classe atribuída ({primary_label} → {reference_label})")
    figure.tight_layout()
    return figure
