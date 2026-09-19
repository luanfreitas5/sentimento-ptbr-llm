"""MCC e macro-F1 por conceito e por modelo, com IC por bootstrap.

Depois que a amostra de ``para_rotular.csv`` é rotulada por humanos, junta os
rótulos às predições dos modelos (via a chave ``sample_id -> id``) e calcula,
para cada conceito (e para o total) e cada modelo, MCC e macro-F1 com intervalo
de confiança por bootstrap. Amostras pequenas (~20 por conceito) geram ICs
largos: reporte-os sempre junto do ponto.
"""

import logging
import math
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from constants.defaults import DEFAULT_BOOTSTRAP_ITERATIONS, DEFAULT_RANDOM_SEED
from constants.labels import SENTIMENT_CLASSES
from diagnostics.sampling import HUMAN_LABEL_COLUMN
from evaluation.evaluator import calculate_bootstrap_confidence_intervals
from exceptions.data import DataValidationError, EmptyDatasetError
from metrics.classification import calculate_classification_metrics

logger = logging.getLogger(__name__)

OVERALL_GROUP = "__todos__"
_REPORTED_METRICS: tuple[str, ...] = ("mcc", "f1_macro")


def load_human_labels(csv_path: Path) -> pl.DataFrame:
    """Lê o CSV rotulado, descarta linhas em branco e valida as classes.

    Parameters
    ----------
    csv_path : Path
        ``para_rotular.csv`` preenchido (coluna ``rotulo_humano``).

    Returns
    -------
    pl.DataFrame
        Colunas ``sample_id``, ``concept`` e ``rotulo_humano`` (só linhas rotuladas).

    Raises
    ------
    DataValidationError
        Se faltarem colunas ou houver rótulo fora de ``SENTIMENT_CLASSES``.
    EmptyDatasetError
        Se nenhuma linha estiver rotulada.
    """
    raw = pl.read_csv(csv_path, infer_schema_length=0)
    missing = [c for c in ("sample_id", "concept", HUMAN_LABEL_COLUMN) if c not in raw.columns]
    if missing:
        raise DataValidationError(schema_name="HumanLabels", detail=f"colunas ausentes: {missing}")
    labeled = raw.select(
        "sample_id", "concept", pl.col(HUMAN_LABEL_COLUMN).str.strip_chars().str.to_lowercase()
    ).filter(pl.col(HUMAN_LABEL_COLUMN).is_not_null() & (pl.col(HUMAN_LABEL_COLUMN) != ""))
    if labeled.is_empty():
        raise EmptyDatasetError(str(csv_path))
    invalid = labeled.filter(~pl.col(HUMAN_LABEL_COLUMN).is_in(list(SENTIMENT_CLASSES)))
    if invalid.height:
        raise DataValidationError(
            schema_name="HumanLabels",
            detail=f"{invalid.height} rótulo(s) fora de {list(SENTIMENT_CLASSES)}",
        )
    return labeled


def build_labeled_evaluation_frame(
    human_labels: pl.DataFrame, key: pl.DataFrame, corpus: pl.DataFrame
) -> pl.DataFrame:
    """Junta rótulos humanos à chave e às predições dos modelos (colunas ``lab_*``).

    Parameters
    ----------
    human_labels : pl.DataFrame
        Saída de :func:`load_human_labels`.
    key : pl.DataFrame
        Chave ``sample_id``/``id`` (arquivo mantido fora do git).
    corpus : pl.DataFrame
        Corpus no contrato de diagnóstico (com ``lab_*``).

    Returns
    -------
    pl.DataFrame
        ``sample_id``, ``concept``, ``rotulo_humano`` e uma coluna por modelo.
    """
    model_columns = [name for name in corpus.columns if name.startswith("lab_")]
    return (
        human_labels.join(key.select("sample_id", "id"), on="sample_id", how="inner")
        .join(corpus.select("id", *model_columns), on="id", how="inner")
        .drop("id")
    )


def calculate_scores_with_ci(
    y_true: list[str], y_pred: list[str], *, n_bootstrap: int, confidence_level: float, seed: int
) -> dict[str, float]:
    """Calcula MCC e macro-F1 com IC por bootstrap; ``nan`` se houver menos de 2 amostras.

    Parameters
    ----------
    y_true, y_pred : list[str]
        Rótulos de referência e previstos.
    n_bootstrap : int
        Reamostragens.
    confidence_level : float
        Nível do IC.
    seed : int
        Semente do bootstrap.

    Returns
    -------
    dict[str, float]
        ``mcc``, ``mcc_ci_low``, ``mcc_ci_high``, ``f1_macro``, ``f1_macro_ci_low``,
        ``f1_macro_ci_high``.

    Examples
    --------
    >>> calculate_scores_with_ci(
    ...     ["positivo", "negativo"],
    ...     ["positivo", "negativo"],
    ...     n_bootstrap=5,
    ...     confidence_level=0.9,
    ...     seed=0,
    ... )["mcc"]
    1.0
    """
    empty = {f"{m}{s}": math.nan for m in _REPORTED_METRICS for s in ("", "_ci_low", "_ci_high")}
    if len(y_true) < 2:
        return empty
    point = calculate_classification_metrics(y_true, y_pred)
    intervals = calculate_bootstrap_confidence_intervals(
        y_true,
        y_pred,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=seed,
    )
    scores: dict[str, float] = {}
    for metric in _REPORTED_METRICS:
        low, high = intervals[metric]
        scores[metric] = float(point[metric])
        scores[f"{metric}_ci_low"] = float(low)
        scores[f"{metric}_ci_high"] = float(high)
    return scores


def evaluate_models_by_concept(
    frame: pl.DataFrame,
    *,
    model_columns: Sequence[str],
    n_bootstrap: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    confidence_level: float = 0.95,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pl.DataFrame:
    """Calcula MCC e macro-F1 (com IC bootstrap) por conceito e por modelo.

    Parameters
    ----------
    frame : pl.DataFrame
        Saída de :func:`build_labeled_evaluation_frame`.
    model_columns : Sequence[str]
        Colunas ``lab_<modelo>`` a avaliar contra ``rotulo_humano``.
    n_bootstrap : int, optional
        Reamostragens, by default :data:`constants.defaults.DEFAULT_BOOTSTRAP_ITERATIONS`.
    confidence_level : float, optional
        Nível do IC, by default 0.95.
    random_seed : int, optional
        Semente do bootstrap, by default :data:`constants.defaults.DEFAULT_RANDOM_SEED`.

    Returns
    -------
    pl.DataFrame
        Uma linha por (conceito, modelo) mais o total (``__todos__``): ``n``,
        ``mcc``, ``mcc_ci_low``, ``mcc_ci_high``, ``f1_macro`` e ICs.

    Examples
    --------
    >>> evaluate_models_by_concept(frame, model_columns=["lab_a"])  # doctest: +SKIP
    """
    groups: list[tuple[str, pl.DataFrame]] = [(OVERALL_GROUP, frame)] + [
        (str(name[0]), part) for name, part in frame.group_by("concept", maintain_order=True)
    ]
    rows: list[dict[str, object]] = []
    for concept, part in groups:
        for model in model_columns:
            valid = part.drop_nulls(subset=[model, HUMAN_LABEL_COLUMN])
            scores = calculate_scores_with_ci(
                valid[HUMAN_LABEL_COLUMN].to_list(),
                valid[model].to_list(),
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                seed=random_seed,
            )
            row: dict[str, object] = {
                "concept": concept,
                "model": model,
                "n": valid.height,
                **scores,
            }
            rows.append(row)
    return pl.DataFrame(rows)
