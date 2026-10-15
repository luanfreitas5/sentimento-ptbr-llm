"""Métricas de concordância com a referência, por classe e por faixa de confiança.

Complementa :mod:`evaluation.llm_comparison` (concordância global, Kappa, tamanho do
texto) com dois recortes que respondem a perguntas diferentes:

* **por classe**: precisão, revocação e F1 do modelo primário tomando a base de referência como
  rótulo de comparação, com IC por bootstrap — mostra *quais* classes o modelo do Hugging Face
  reproduz pior (um Kappa global esconde isso);
* **por confiança**: a concordância cresce quando os modelos estão confiantes? — a resposta
  sustenta (ou desmente) o uso da baixa confiança como alvo de incerteza nas hipóteses.

Como o gold set é rotulado por LLM (não por humanos), "precisão/revocação/F1" aqui medem a
**concordância** com o rotulador de referência, nunca o acerto absoluto. Funções puras sobre o
quadro de :func:`evaluation.llm_comparison.build_comparison_frame`, sem E/S nem gráficos.
"""

from itertools import pairwise
from typing import Any

import numpy as np
import polars as pl
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

from constants.defaults import DEFAULT_BOOTSTRAP_ITERATIONS, DEFAULT_CONFIDENCE_LEVEL
from constants.labels import LABEL_TO_ID, SENTIMENT_CLASSES
from evaluation.llm_comparison import MODEL_NAMES
from exceptions.data import EmptyDatasetError

_HF, _OA = MODEL_NAMES
_DEFAULT_SEED = 42
MACRO_LABEL = "macro"
DEFAULT_CONFIDENCE_BAND_EDGES: tuple[float, ...] = (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0)


def _calculate_class_scores(
    reference_ids: np.ndarray, predicted_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precisão, revocação e F1 de cada classe a partir dos identificadores numéricos."""
    n_classes = len(SENTIMENT_CLASSES)
    counts = (
        np.bincount(reference_ids * n_classes + predicted_ids, minlength=n_classes**2)
        .reshape(n_classes, n_classes)
        .astype(float)
    )
    true_positives = np.diag(counts)
    predicted_totals = counts.sum(axis=0)
    reference_totals = counts.sum(axis=1)
    precision = np.divide(
        true_positives, predicted_totals, out=np.zeros(n_classes), where=predicted_totals > 0
    )
    recall = np.divide(
        true_positives, reference_totals, out=np.zeros(n_classes), where=reference_totals > 0
    )
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall, denominator, out=np.zeros(n_classes), where=denominator > 0
    )
    return precision, recall, f1


def calculate_per_class_metrics(
    frame: pl.DataFrame,
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    random_seed: int = _DEFAULT_SEED,
) -> pl.DataFrame:
    """Precisão, revocação e F1 do modelo primário por classe, tomando a referência como rótulo.

    Parameters
    ----------
    frame : pl.DataFrame
        Saída de :func:`evaluation.llm_comparison.build_comparison_frame`.
    n_bootstrap : int, optional
        Reamostragens (de tweets, com reposição) do IC do F1, by default
        :data:`constants.defaults.DEFAULT_BOOTSTRAP_ITERATIONS`.
    confidence_level : float, optional
        Nível de confiança do IC, by default :data:`constants.defaults.DEFAULT_CONFIDENCE_LEVEL`.
    random_seed : int, optional
        Semente do bootstrap, by default 42.

    Returns
    -------
    pl.DataFrame
        Uma linha por classe de :data:`constants.labels.SENTIMENT_CLASSES` mais uma linha
        ``macro`` (média simples das classes): ``label``, ``support`` (tweets da classe na
        referência), ``precision``, ``recall``, ``f1``, ``f1_ci_lower`` e ``f1_ci_upper``.
        Classe sem tweets na referência tem revocação e F1 iguais a ``0``.

    Raises
    ------
    EmptyDatasetError
        Se ``frame`` estiver vazio.

    Examples
    --------
    >>> frame = pl.DataFrame(
    ...     {
    ...         "label_huggingface": ["positivo", "negativo", "neutro", "positivo"],
    ...         "label_openai": ["positivo", "negativo", "positivo", "positivo"],
    ...     }
    ... )
    >>> table = calculate_per_class_metrics(frame, n_bootstrap=20)
    >>> table["label"].to_list()
    ['negativo', 'neutro', 'positivo', 'macro']
    """
    if frame.is_empty():
        raise EmptyDatasetError("frame de comparação")

    reference_ids = frame[f"label_{_OA}"].replace_strict(LABEL_TO_ID, return_dtype=pl.Int64)
    predicted_ids = frame[f"label_{_HF}"].replace_strict(LABEL_TO_ID, return_dtype=pl.Int64)
    reference, predicted = reference_ids.to_numpy(), predicted_ids.to_numpy()
    n_tweets = len(reference)
    n_classes = len(SENTIMENT_CLASSES)

    precision, recall, f1 = _calculate_class_scores(reference, predicted)
    generator = np.random.default_rng(random_seed)
    f1_samples = np.empty((n_bootstrap, n_classes + 1))
    for iteration in range(n_bootstrap):
        indices = generator.integers(0, n_tweets, size=n_tweets)
        _, _, sample_f1 = _calculate_class_scores(reference[indices], predicted[indices])
        f1_samples[iteration] = [*sample_f1, sample_f1.mean()]
    tail = 100 * (1 - confidence_level) / 2
    lower, upper = np.percentile(f1_samples, [tail, 100 - tail], axis=0)

    supports = np.bincount(reference, minlength=n_classes)
    rows: list[dict[str, Any]] = [
        {
            "label": label,
            "support": int(supports[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "f1_ci_lower": float(lower[index]),
            "f1_ci_upper": float(upper[index]),
        }
        for index, label in enumerate(SENTIMENT_CLASSES)
    ]
    rows.append(
        {
            "label": MACRO_LABEL,
            "support": n_tweets,
            "precision": float(precision.mean()),
            "recall": float(recall.mean()),
            "f1": float(f1.mean()),
            "f1_ci_lower": float(lower[n_classes]),
            "f1_ci_upper": float(upper[n_classes]),
        }
    )
    return pl.DataFrame(rows)


def _assign_confidence_bands(
    values: np.ndarray, edges: tuple[float, ...]
) -> tuple[np.ndarray, list[str]]:
    """Índice da faixa de cada confiança (``[a, b)``, última faixa fechada) e seus rótulos."""
    indices = np.digitize(values, edges[1:-1], right=False)
    labels = [f"{low:.2f}–{high:.2f}" for low, high in pairwise(edges)]
    return indices, labels


def analyze_agreement_by_confidence(
    frame: pl.DataFrame,
    *,
    band_edges: tuple[float, ...] = DEFAULT_CONFIDENCE_BAND_EDGES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> pl.DataFrame:
    """Taxa de concordância por faixa da confiança de cada modelo (IC de Wilson).

    Parameters
    ----------
    frame : pl.DataFrame
        Saída de :func:`evaluation.llm_comparison.build_comparison_frame`.
    band_edges : tuple[float, ...], optional
        Limites crescentes das faixas (``[a, b)``; a última inclui o limite superior), by default
        :data:`DEFAULT_CONFIDENCE_BAND_EDGES`.
    confidence_level : float, optional
        Nível do intervalo de Wilson, by default
        :data:`constants.defaults.DEFAULT_CONFIDENCE_LEVEL`.

    Returns
    -------
    pl.DataFrame
        Uma linha por modelo e faixa não vazia: ``model``, ``confidence_band``, ``band_min``,
        ``band_max``, ``n_tweets``, ``agreement_rate``, ``agreement_rate_ci_lower`` e
        ``agreement_rate_ci_upper``.

    Raises
    ------
    EmptyDatasetError
        Se ``frame`` estiver vazio.

    Examples
    --------
    >>> frame = pl.DataFrame(
    ...     {
    ...         "confidence_huggingface": [0.6, 0.95],
    ...         "confidence_openai": [0.9, 0.99],
    ...         "agree": [False, True],
    ...     }
    ... )
    >>> analyze_agreement_by_confidence(frame, band_edges=(0.0, 0.8, 1.0))["n_tweets"].to_list()
    [1, 1, 2]
    """
    if frame.is_empty():
        raise EmptyDatasetError("frame de comparação")

    agree = frame["agree"].to_numpy()
    rows: list[dict[str, Any]] = []
    for model in MODEL_NAMES:
        indices, labels = _assign_confidence_bands(
            frame[f"confidence_{model}"].to_numpy(), band_edges
        )
        for band, label in enumerate(labels):
            in_band = indices == band
            n_tweets = int(in_band.sum())
            if n_tweets == 0:
                continue
            n_agree = int(agree[in_band].sum())
            interval: Any = proportion_confint(
                n_agree, n_tweets, alpha=1 - confidence_level, method="wilson"
            )
            lower, upper = interval
            rows.append(
                {
                    "model": model,
                    "confidence_band": label,
                    "band_min": band_edges[band],
                    "band_max": band_edges[band + 1],
                    "n_tweets": n_tweets,
                    "agreement_rate": n_agree / n_tweets,
                    "agreement_rate_ci_lower": float(lower),
                    "agreement_rate_ci_upper": float(upper),
                }
            )
    return pl.DataFrame(rows)


def analyze_agreement_by_confidence_pair(
    frame: pl.DataFrame,
    *,
    band_edges: tuple[float, ...] = DEFAULT_CONFIDENCE_BAND_EDGES,
) -> pl.DataFrame:
    """Concordância no cruzamento das faixas de confiança dos dois modelos.

    Parameters
    ----------
    frame : pl.DataFrame
        Saída de :func:`evaluation.llm_comparison.build_comparison_frame`.
    band_edges : tuple[float, ...], optional
        Limites crescentes das faixas, by default :data:`DEFAULT_CONFIDENCE_BAND_EDGES`.

    Returns
    -------
    pl.DataFrame
        Uma linha por par de faixas não vazio: ``confidence_band_huggingface``,
        ``confidence_band_openai``, ``n_tweets`` e ``agreement_rate``, ordenada pelas faixas.

    Raises
    ------
    EmptyDatasetError
        Se ``frame`` estiver vazio.

    Examples
    --------
    >>> frame = pl.DataFrame(
    ...     {
    ...         "confidence_huggingface": [0.6, 0.95],
    ...         "confidence_openai": [0.9, 0.99],
    ...         "agree": [False, True],
    ...     }
    ... )
    >>> analyze_agreement_by_confidence_pair(frame, band_edges=(0.0, 0.8, 1.0)).height
    2
    """
    if frame.is_empty():
        raise EmptyDatasetError("frame de comparação")

    hf_indices, labels = _assign_confidence_bands(frame[f"confidence_{_HF}"].to_numpy(), band_edges)
    oa_indices, _ = _assign_confidence_bands(frame[f"confidence_{_OA}"].to_numpy(), band_edges)
    return (
        frame.with_columns(
            pl.Series("_hf_band", hf_indices),
            pl.Series("_oa_band", oa_indices),
        )
        .group_by("_hf_band", "_oa_band")
        .agg(
            pl.len().alias("n_tweets"),
            pl.col("agree").mean().alias("agreement_rate"),
        )
        .sort("_hf_band", "_oa_band")
        .select(
            pl.col("_hf_band")
            .replace_strict(dict(enumerate(labels)))
            .alias(f"confidence_band_{_HF}"),
            pl.col("_oa_band")
            .replace_strict(dict(enumerate(labels)))
            .alias(f"confidence_band_{_OA}"),
            "n_tweets",
            "agreement_rate",
        )
    )


def compare_confidence_by_agreement(frame: pl.DataFrame) -> dict[str, dict[str, Any]]:
    """Compara a confiança de cada modelo nos tweets concordantes e nos divergentes.

    Parameters
    ----------
    frame : pl.DataFrame
        Saída de :func:`evaluation.llm_comparison.build_comparison_frame`.

    Returns
    -------
    dict[str, dict[str, Any]]
        Por modelo: ``median_when_agree``, ``median_when_diverge``, ``mannwhitney_p_value`` e
        ``rank_biserial`` (tamanho de efeito em [-1, 1]; positivo = mais confiante quando
        concorda). Os testes ficam ``None`` se não houver concordâncias ou divergências.

    Raises
    ------
    EmptyDatasetError
        Se ``frame`` estiver vazio.

    Examples
    --------
    >>> frame = pl.DataFrame(
    ...     {
    ...         "confidence_huggingface": [0.9, 0.6, 0.8],
    ...         "confidence_openai": [0.8, 0.7, 0.9],
    ...         "agree": [True, False, True],
    ...     }
    ... )
    >>> compare_confidence_by_agreement(frame)["huggingface"]["median_when_agree"]
    0.85
    """
    if frame.is_empty():
        raise EmptyDatasetError("frame de comparação")

    agree = frame["agree"].to_numpy()
    summary: dict[str, dict[str, Any]] = {}
    for model in MODEL_NAMES:
        values = frame[f"confidence_{model}"].to_numpy()
        when_agree, when_diverge = values[agree], values[~agree]
        entry: dict[str, Any] = {
            "median_when_agree": float(np.median(when_agree)) if when_agree.size else None,
            "median_when_diverge": float(np.median(when_diverge)) if when_diverge.size else None,
            "mannwhitney_p_value": None,
            "rank_biserial": None,
        }
        if when_agree.size and when_diverge.size:
            result: Any = stats.mannwhitneyu(when_agree, when_diverge, alternative="two-sided")
            entry["mannwhitney_p_value"] = float(result.pvalue)
            entry["rank_biserial"] = float(
                2 * result.statistic / (when_agree.size * when_diverge.size) - 1
            )
        summary[model] = entry
    return summary
