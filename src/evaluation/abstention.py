"""Predição com abstenção: curva risco-cobertura e encaminhamento para revisão humana.

Um classificador não precisa responder a todo tweet. Ao abster-se nos de menor
confiança, a qualidade nos tweets respondidos sobe, ao custo da cobertura. Este módulo
mede esse custo-benefício (macro-F1 contra a fração respondida, com IC bootstrap) e
define o limiar de confiança **no conjunto de validação**, para aplicá-lo ao teste sem
que o teste oriente a escolha.
"""

import logging
from collections.abc import Sequence

import numpy as np
import polars as pl

from constants.defaults import (
    DEFAULT_BOOTSTRAP_ITERATIONS,
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_RANDOM_SEED,
)
from constants.labels import SENTIMENT_CLASSES
from exceptions.data import EmptyDatasetError

logger = logging.getLogger(__name__)

DEFAULT_COVERAGES: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def _encode_labels(values: Sequence[str], labels: Sequence[str]) -> np.ndarray:
    """Converte rótulos em códigos inteiros, na ordem de ``labels``."""
    label_to_code = {label: code for code, label in enumerate(labels)}
    unknown = set(values) - set(label_to_code)
    if unknown:
        raise ValueError(f"Rótulos fora de `labels`: {sorted(unknown)}")
    return np.array([label_to_code[value] for value in values])


def _calculate_macro_f1(true_codes: np.ndarray, pred_codes: np.ndarray, n_classes: int) -> float:
    """Macro-F1 vetorizada via matriz de confusão (classes ausentes contam como 0)."""
    confusion = np.bincount(true_codes * n_classes + pred_codes, minlength=n_classes**2)
    confusion = confusion.reshape(n_classes, n_classes)
    true_positives = np.diag(confusion).astype(float)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1_per_class = np.divide(
        2 * true_positives, denominator, out=np.zeros(n_classes), where=denominator > 0
    )
    return float(np.mean(f1_per_class))


def _bootstrap_macro_f1(
    true_codes: np.ndarray,
    pred_codes: np.ndarray,
    n_classes: int,
    *,
    n_bootstrap: int,
    confidence_level: float,
    random_generator: np.random.Generator,
) -> tuple[float, float]:
    """IC percentil da macro-F1 por reamostragem com reposição."""
    n_samples = len(true_codes)
    scores = np.empty(n_bootstrap)
    for iteration in range(n_bootstrap):
        indices = random_generator.integers(0, n_samples, size=n_samples)
        scores[iteration] = _calculate_macro_f1(true_codes[indices], pred_codes[indices], n_classes)
    alpha = 1 - confidence_level
    return (
        float(np.percentile(scores, 100 * alpha / 2)),
        float(np.percentile(scores, 100 * (1 - alpha / 2))),
    )


def select_confidence_threshold(confidences: np.ndarray, coverage: float) -> float:
    """Escolhe o limiar de confiança que mantém a fração ``coverage`` das predições.

    Parameters
    ----------
    confidences : np.ndarray
        Confiança da classe predita, preferencialmente de um conjunto de validação.
    coverage : float
        Fração desejada de tweets respondidos, em ``(0, 1]``.

    Returns
    -------
    float
        Limiar: responde-se quando ``confiança >= limiar``.

    Raises
    ------
    EmptyDatasetError
        Se ``confidences`` estiver vazio.
    ValueError
        Se ``coverage`` não estiver em ``(0, 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> select_confidence_threshold(np.array([0.5, 0.6, 0.7, 0.8, 0.9]), coverage=0.6)
    0.7
    """
    if confidences.size == 0:
        raise EmptyDatasetError("confidences")
    if not 0 < coverage <= 1:
        raise ValueError(f"coverage deve estar em (0, 1], recebido: {coverage}")
    return float(np.quantile(confidences, 1 - coverage, method="higher"))


def calculate_risk_coverage_curve(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    confidences: np.ndarray,
    *,
    coverages: Sequence[float] = DEFAULT_COVERAGES,
    labels: Sequence[str] = SENTIMENT_CLASSES,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> pl.DataFrame:
    """Calcula macro-F1, acurácia e risco ao responder só os tweets mais confiantes.

    Para cada cobertura, mantém a fração de predições de maior confiança e mede a
    qualidade nelas. O IC bootstrap trata o conjunto mantido como fixo (a incerteza da
    escolha do limiar não entra), por isso é indicativo, não uma garantia.

    Parameters
    ----------
    y_true : Sequence[str]
        Rótulos verdadeiros.
    y_pred : Sequence[str]
        Rótulos preditos.
    confidences : np.ndarray
        Confiança da classe predita por amostra.
    coverages : Sequence[float], optional
        Frações respondidas, em ``(0, 1]``, by default :data:`DEFAULT_COVERAGES`.
    labels : Sequence[str], optional
        Classes para a macro-F1, by default :data:`constants.labels.SENTIMENT_CLASSES`.
    n_bootstrap : int, optional
        Reamostragens do IC, by default 1000.
    confidence_level : float, optional
        Nível do IC, by default 0.95.
    random_state : int, optional
        Semente, by default 42.

    Returns
    -------
    pl.DataFrame
        Uma linha por cobertura: ``coverage``, ``threshold``, ``n_answered``,
        ``accuracy``, ``risk`` (1 - acurácia), ``f1_macro`` e seu IC.

    Raises
    ------
    EmptyDatasetError
        Se ``y_true`` estiver vazio.
    ValueError
        Se alguma cobertura estiver fora de ``(0, 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> y = ["positivo", "negativo", "positivo", "negativo"]
    >>> p = ["positivo", "negativo", "negativo", "negativo"]
    >>> c = np.array([0.95, 0.9, 0.55, 0.6])
    >>> curva = calculate_risk_coverage_curve(y, p, c, coverages=[0.5, 1.0], n_bootstrap=10)
    >>> curva["accuracy"].to_list()
    [1.0, 0.75]
    """
    if not y_true:
        raise EmptyDatasetError("y_true")
    n_classes = len(labels)
    true_codes = _encode_labels(y_true, labels)
    pred_codes = _encode_labels(y_pred, labels)
    random_generator = np.random.default_rng(random_state)
    order = np.argsort(-np.asarray(confidences), kind="stable")

    rows = []
    for coverage in coverages:
        if not 0 < coverage <= 1:
            raise ValueError(f"coverage deve estar em (0, 1], recebido: {coverage}")
        n_kept = max(1, int(np.ceil(coverage * len(true_codes))))
        kept = order[:n_kept]
        low, high = _bootstrap_macro_f1(
            true_codes[kept],
            pred_codes[kept],
            n_classes,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_generator=random_generator,
        )
        accuracy = float(np.mean(true_codes[kept] == pred_codes[kept]))
        rows.append(
            {
                "coverage": float(coverage),
                "threshold": float(confidences[kept[-1]]),
                "n_answered": n_kept,
                "accuracy": accuracy,
                "risk": 1.0 - accuracy,
                "f1_macro": _calculate_macro_f1(true_codes[kept], pred_codes[kept], n_classes),
                "f1_macro_ci_low": low,
                "f1_macro_ci_high": high,
            }
        )
    return pl.DataFrame(rows)


def evaluate_abstention_at_threshold(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    confidences: np.ndarray,
    threshold: float,
    *,
    labels: Sequence[str] = SENTIMENT_CLASSES,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> dict[str, float | int]:
    """Avalia, no teste, a abstenção com um limiar escolhido na validação.

    Parameters
    ----------
    y_true : Sequence[str]
        Rótulos verdadeiros do teste.
    y_pred : Sequence[str]
        Rótulos preditos.
    confidences : np.ndarray
        Confiança por amostra.
    threshold : float
        Limiar fixado na validação: responde-se quando ``confiança >= limiar``.
    labels : Sequence[str], optional
        Classes para a macro-F1, by default :data:`constants.labels.SENTIMENT_CLASSES`.
    n_bootstrap : int, optional
        Reamostragens do IC, by default 1000.
    confidence_level : float, optional
        Nível do IC, by default 0.95.
    random_state : int, optional
        Semente, by default 42.

    Returns
    -------
    dict[str, float | int]
        ``threshold``, ``coverage``, ``n_answered``, ``n_abstained``, ``f1_macro_all``,
        ``f1_macro_answered`` (com IC) e ``f1_macro_abstained`` (``nan`` se ninguém se
        absteve ou se a abstenção for total).

    Raises
    ------
    EmptyDatasetError
        Se ``y_true`` estiver vazio.

    Examples
    --------
    >>> import numpy as np
    >>> r = evaluate_abstention_at_threshold(
    ...     ["positivo", "negativo"],
    ...     ["positivo", "positivo"],
    ...     np.array([0.9, 0.4]),
    ...     0.5,
    ...     n_bootstrap=5,
    ... )
    >>> (r["n_answered"], r["n_abstained"])
    (1, 1)
    """
    if not y_true:
        raise EmptyDatasetError("y_true")
    n_classes = len(labels)
    true_codes = _encode_labels(y_true, labels)
    pred_codes = _encode_labels(y_pred, labels)
    answered = np.asarray(confidences) >= threshold
    n_answered = int(answered.sum())

    result: dict[str, float | int] = {
        "threshold": float(threshold),
        "coverage": n_answered / len(true_codes),
        "n_answered": n_answered,
        "n_abstained": len(true_codes) - n_answered,
        "f1_macro_all": _calculate_macro_f1(true_codes, pred_codes, n_classes),
        "f1_macro_answered": float("nan"),
        "f1_macro_answered_ci_low": float("nan"),
        "f1_macro_answered_ci_high": float("nan"),
        "f1_macro_abstained": float("nan"),
    }
    if n_answered:
        result["f1_macro_answered"] = _calculate_macro_f1(
            true_codes[answered], pred_codes[answered], n_classes
        )
        result["f1_macro_answered_ci_low"], result["f1_macro_answered_ci_high"] = (
            _bootstrap_macro_f1(
                true_codes[answered],
                pred_codes[answered],
                n_classes,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                random_generator=np.random.default_rng(random_state),
            )
        )
    if n_answered < len(true_codes):
        result["f1_macro_abstained"] = _calculate_macro_f1(
            true_codes[~answered], pred_codes[~answered], n_classes
        )
    return result
