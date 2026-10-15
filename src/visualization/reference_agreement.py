"""Gráficos da concordância com a referência por classe e por faixa de confiança.

Desenham as tabelas de :mod:`evaluation.reference_agreement` com a paleta do projeto
(:mod:`visualization.theme`). Cada função devolve uma :class:`matplotlib.figure.Figure` pronta
para :func:`visualization.theme.save_figure` (PNG 300 dpi + SVG).
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.figure import Figure

from evaluation.llm_comparison import MODEL_NAMES
from evaluation.reference_agreement import MACRO_LABEL
from exceptions.data import EmptyDatasetError
from visualization.comparison import DEFAULT_PRIMARY_LABEL, DEFAULT_REFERENCE_LABEL
from visualization.theme import MODEL_COLOR_PALETTE

_HF, _OA = MODEL_NAMES
_METRIC_COLORS: dict[str, str] = {"precision": "#56B4E9", "recall": "#E69F00", "f1": "#009E73"}
_METRIC_LABELS: dict[str, str] = {"precision": "Precisão", "recall": "Revocação", "f1": "F1"}
DEFAULT_MINIMUM_CELL_TWEETS = 30


def _display_names(primary_label: str, reference_label: str) -> dict[str, str]:
    """Nome exibido de cada modelo; o segundo é a base de referência."""
    return {_HF: primary_label, _OA: reference_label}


def plot_per_class_metrics(
    metrics: pl.DataFrame,
    *,
    primary_label: str = DEFAULT_PRIMARY_LABEL,
    reference_label: str = DEFAULT_REFERENCE_LABEL,
) -> Figure:
    """Barras agrupadas de precisão, revocação e F1 por classe (F1 com IC por bootstrap).

    Parameters
    ----------
    metrics : pl.DataFrame
        Saída de :func:`evaluation.reference_agreement.calculate_per_class_metrics`.
    primary_label : str, optional
        Nome exibido do modelo primário, by default "LLM Hugging Face".
    reference_label : str, optional
        Nome exibido da base de referência, by default "Gold set (API)".

    Returns
    -------
    matplotlib.figure.Figure
        Figura com uma barra por métrica em cada classe e na média macro.

    Raises
    ------
    EmptyDatasetError
        Se ``metrics`` estiver vazio.

    Examples
    --------
    >>> table = pl.DataFrame(
    ...     {
    ...         "label": ["positivo", "macro"],
    ...         "precision": [0.8, 0.8],
    ...         "recall": [0.7, 0.7],
    ...         "f1": [0.75, 0.75],
    ...         "f1_ci_lower": [0.7, 0.7],
    ...         "f1_ci_upper": [0.8, 0.8],
    ...     }
    ... )
    >>> plot_per_class_metrics(table).axes[0].get_ylabel()
    'Valor da métrica'
    """
    if metrics.is_empty():
        raise EmptyDatasetError("métricas por classe")
    labels = metrics["label"].to_list()
    positions = np.arange(len(labels))
    width = 0.26
    figure, axis = plt.subplots(figsize=(9, 5))
    for offset, metric in zip((-width, 0.0, width), _METRIC_COLORS, strict=True):
        values = metrics[metric].to_numpy()
        error = None
        if metric == "f1":
            # clip: o IC pode divergir do ponto por ruído de ponto flutuante
            # (yerr não aceita negativos)
            error = np.clip(
                [
                    values - metrics["f1_ci_lower"].to_numpy(),
                    metrics["f1_ci_upper"].to_numpy() - values,
                ],
                0.0,
                None,
            )
        bars = axis.bar(
            positions + offset,
            values,
            width,
            yerr=error,
            capsize=3,
            color=_METRIC_COLORS[metric],
            label=_METRIC_LABELS[metric],
        )
        axis.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    axis.axvline(labels.index(MACRO_LABEL) - 0.5, color="#555555", linewidth=0.8, linestyle=":")
    axis.set_xticks(positions, labels)
    axis.set_ylim(0, 1.1)
    axis.set_title(f"{primary_label} vs. {reference_label}: métricas por classe (F1 com IC)")
    axis.set_xlabel(f"Classe atribuída por {reference_label}")
    axis.set_ylabel("Valor da métrica")
    axis.legend(
        title="Métrica", loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False
    )
    figure.tight_layout()
    return figure


def plot_agreement_by_confidence(
    agreement_by_confidence: pl.DataFrame,
    *,
    primary_label: str = DEFAULT_PRIMARY_LABEL,
    reference_label: str = DEFAULT_REFERENCE_LABEL,
) -> Figure:
    """Concordância (IC de Wilson) por faixa da confiança de cada modelo.

    Parameters
    ----------
    agreement_by_confidence : pl.DataFrame
        Saída de :func:`evaluation.reference_agreement.analyze_agreement_by_confidence`.
    primary_label : str, optional
        Nome exibido do modelo primário, by default "LLM Hugging Face".
    reference_label : str, optional
        Nome exibido da base de referência, by default "Gold set (API)".

    Returns
    -------
    matplotlib.figure.Figure
        Uma linha por modelo: concordância (%) conforme a confiança do próprio modelo.

    Raises
    ------
    EmptyDatasetError
        Se ``agreement_by_confidence`` estiver vazio.

    Examples
    --------
    >>> table = pl.DataFrame(
    ...     {
    ...         "model": ["huggingface", "openai"],
    ...         "confidence_band": ["0.00–0.80", "0.00–0.80"],
    ...         "band_min": [0.0, 0.0],
    ...         "agreement_rate": [0.5, 0.6],
    ...         "agreement_rate_ci_lower": [0.4, 0.5],
    ...         "agreement_rate_ci_upper": [0.6, 0.7],
    ...     }
    ... )
    >>> plot_agreement_by_confidence(table).axes[0].get_ylabel()
    'Concordância (%)'
    """
    if agreement_by_confidence.is_empty():
        raise EmptyDatasetError("concordância por faixa de confiança")
    display_names = _display_names(primary_label, reference_label)
    bands = (
        agreement_by_confidence.unique("confidence_band")
        .sort("band_min")["confidence_band"]
        .to_list()
    )
    figure, axis = plt.subplots(figsize=(9, 5))
    for model in MODEL_NAMES:
        rows = agreement_by_confidence.filter(pl.col("model") == model)
        rates = rows["agreement_rate"].to_numpy() * 100
        axis.errorbar(
            [bands.index(band) for band in rows["confidence_band"]],
            rates,
            yerr=np.clip(
                [
                    rates - rows["agreement_rate_ci_lower"].to_numpy() * 100,
                    rows["agreement_rate_ci_upper"].to_numpy() * 100 - rates,
                ],
                0.0,
                None,
            ),
            marker="o",
            capsize=4,
            color=MODEL_COLOR_PALETTE[model],
            label=display_names[model],
        )
    axis.set_xticks(range(len(bands)), bands)
    axis.set_ylim(0, 100)
    axis.set_title("Concordância entre os modelos por faixa de confiança (IC de Wilson)")
    axis.set_xlabel("Faixa de confiança do próprio modelo")
    axis.set_ylabel("Concordância (%)")
    axis.legend(title="Confiança de")
    figure.tight_layout()
    return figure


def plot_agreement_confidence_heatmap(
    agreement_pair: pl.DataFrame,
    *,
    minimum_tweets: int = DEFAULT_MINIMUM_CELL_TWEETS,
    primary_label: str = DEFAULT_PRIMARY_LABEL,
    reference_label: str = DEFAULT_REFERENCE_LABEL,
) -> Figure:
    """Mapa de calor da concordância no cruzamento das faixas de confiança dos dois modelos.

    Parameters
    ----------
    agreement_pair : pl.DataFrame
        Saída de :func:`evaluation.reference_agreement.analyze_agreement_by_confidence_pair`.
    minimum_tweets : int, optional
        Células com menos tweets ficam em branco (taxa instável), by default 30.
    primary_label : str, optional
        Nome exibido do modelo primário, by default "LLM Hugging Face".
    reference_label : str, optional
        Nome exibido da base de referência, by default "Gold set (API)".

    Returns
    -------
    matplotlib.figure.Figure
        Concordância (%) com a confiança do modelo primário nas linhas e a da referência nas
        colunas.

    Raises
    ------
    EmptyDatasetError
        Se ``agreement_pair`` estiver vazio.

    Examples
    --------
    >>> table = pl.DataFrame(
    ...     {
    ...         "confidence_band_huggingface": ["0.00–0.80"],
    ...         "confidence_band_openai": ["0.80–1.00"],
    ...         "n_tweets": [50],
    ...         "agreement_rate": [0.6],
    ...     }
    ... )
    >>> plot_agreement_confidence_heatmap(table).axes[0].get_xlabel()
    'Confiança — Gold set (API)'
    """
    if agreement_pair.is_empty():
        raise EmptyDatasetError("concordância por cruzamento de confiança")
    row_column, column_column = f"confidence_band_{_HF}", f"confidence_band_{_OA}"
    matrix = (
        agreement_pair.with_columns(
            pl.when(pl.col("n_tweets") >= minimum_tweets)
            .then(pl.col("agreement_rate") * 100)
            .otherwise(None)
            .alias("agreement_percent")
        )
        .pivot(on=column_column, index=row_column, values="agreement_percent", sort_columns=True)
        .sort(row_column)
    )
    figure, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix.drop(row_column).to_numpy().astype(float),
        annot=True,
        fmt=".0f",
        vmin=0,
        vmax=100,
        cmap="RdYlGn",
        xticklabels=[c for c in matrix.columns if c != row_column],
        yticklabels=matrix[row_column].to_list(),
        cbar_kws={"label": "Concordância (%)"},
        ax=axis,
    )
    axis.grid(False)
    axis.set_title(f"Concordância por combinação de confianças (células com n ≥ {minimum_tweets})")
    axis.set_xlabel(f"Confiança — {reference_label}")
    axis.set_ylabel(f"Confiança — {primary_label}")
    figure.tight_layout()
    return figure
