"""Avaliação orquestrada de classificadores de sentimento.

Implementa a Seção 4.8 do documento mestre e a diretriz "Rigorous
evaluation" do CLAUDE.md: nunca reportar uma métrica pontual isolada,
sempre acompanhada de intervalo de confiança (bootstrap, ver
``configs/evaluation.yaml`` -> ``uncertainty``) e de um detalhamento por
classe, evitando que médias agregadas escondam falhas pontuais.
"""

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from constants.defaults import (
    DEFAULT_BOOTSTRAP_ITERATIONS,
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_RANDOM_SEED,
)
from constants.labels import SENTIMENT_CLASSES
from constants.metrics import PRIMARY_METRIC
from exceptions.data import EmptyDatasetError
from metrics.classification import (
    calculate_classification_metrics,
    calculate_confusion_matrix,
    calculate_per_class_report,
)
from metrics.ranking import calculate_ranking_metrics

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Resultado consolidado da avaliação de um classificador.

    Parameters
    ----------
    point_metrics : dict[str, float]
        Métricas de classificação (e de ranqueamento, quando ``y_score`` é
        informado) calculadas sobre o conjunto de teste completo.
    confidence_intervals : dict[str, tuple[float, float]]
        Intervalo de confiança bootstrap de cada métrica de classificação
        em ``point_metrics``.
    per_class_report : dict[str, dict[str, float]]
        Precisão, revocação, F1 e suporte detalhados por classe.
    confusion_matrix : np.ndarray
        Matriz de confusão do conjunto de teste completo.
    """

    point_metrics: dict[str, float]
    confidence_intervals: dict[str, tuple[float, float]]
    per_class_report: dict[str, dict[str, float]]
    confusion_matrix: np.ndarray


def calculate_bootstrap_confidence_intervals(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> dict[str, tuple[float, float]]:
    """Calcula intervalos de confiança bootstrap para as métricas de classificação.

    Reamostra o conjunto de teste com reposição ``n_bootstrap`` vezes,
    recalculando :func:`metrics.classification.calculate_classification_metrics`
    a cada reamostragem, e reporta o intervalo percentil de cada métrica —
    uma métrica pontual isolada não expressa a incerteza da estimativa
    (ver CLAUDE.md, "Rigorous evaluation").

    Parameters
    ----------
    y_true : Sequence[str]
        Rótulos de sentimento verdadeiros.
    y_pred : Sequence[str]
        Rótulos de sentimento preditos, mesmo tamanho de ``y_true``.
    n_bootstrap : int, optional
        Número de reamostragens, by default
        :data:`constants.defaults.DEFAULT_BOOTSTRAP_ITERATIONS`.
    confidence_level : float, optional
        Nível de confiança do intervalo, by default
        :data:`constants.defaults.DEFAULT_CONFIDENCE_LEVEL`.
    random_state : int, optional
        Semente aleatória, by default
        :data:`constants.defaults.DEFAULT_RANDOM_SEED`.

    Returns
    -------
    dict[str, tuple[float, float]]
        Uma entrada por métrica de
        :func:`metrics.classification.calculate_classification_metrics`,
        com o par ``(limite_inferior, limite_superior)``.

    Raises
    ------
    EmptyDatasetError
        Se ``y_true`` estiver vazio.

    Examples
    --------
    >>> y_true = ["positivo", "negativo", "positivo", "negativo", "neutro"]
    >>> y_pred = ["positivo", "negativo", "negativo", "negativo", "neutro"]
    >>> intervalos = calculate_bootstrap_confidence_intervals(y_true, y_pred, n_bootstrap=20)
    >>> "f1_macro" in intervalos
    True
    """
    if len(y_true) == 0:
        raise EmptyDatasetError("y_true")

    random_generator = np.random.default_rng(random_state)
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    n_samples = len(y_true_array)

    bootstrap_scores: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_bootstrap):
        resample_indices = random_generator.integers(0, n_samples, size=n_samples)
        resample_metrics = calculate_classification_metrics(
            y_true_array[resample_indices].tolist(), y_pred_array[resample_indices].tolist()
        )
        for metric_name, metric_value in resample_metrics.items():
            bootstrap_scores[metric_name].append(metric_value)

    alpha = 1 - confidence_level
    return {
        metric_name: (
            float(np.percentile(values, 100 * alpha / 2)),
            float(np.percentile(values, 100 * (1 - alpha / 2))),
        )
        for metric_name, values in bootstrap_scores.items()
    }


def evaluate_classifier(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    y_score: np.ndarray | None = None,
    labels: Sequence[str] = SENTIMENT_CLASSES,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> EvaluationResult:
    """Avalia um classificador de sentimento de forma completa e rigorosa.

    Combina métricas pontuais de classificação (e de ranqueamento, quando
    ``y_score`` é informado), intervalos de confiança bootstrap, relatório
    detalhado por classe e matriz de confusão em um único resultado,
    consumido por ``src/evaluation/reports.py``.

    Parameters
    ----------
    y_true : Sequence[str]
        Rótulos de sentimento verdadeiros.
    y_pred : Sequence[str]
        Rótulos de sentimento preditos, mesmo tamanho de ``y_true``.
    y_score : np.ndarray | None, optional
        Matriz ``(n_amostras, n_classes)`` de probabilidades preditas, na
        ordem de ``labels``; quando informada, inclui métricas de
        ranqueamento no resultado, by default None.
    labels : Sequence[str], optional
        Classes do problema, by default
        :data:`constants.labels.SENTIMENT_CLASSES`.
    n_bootstrap : int, optional
        Número de reamostragens do bootstrap, by default
        :data:`constants.defaults.DEFAULT_BOOTSTRAP_ITERATIONS`.
    confidence_level : float, optional
        Nível de confiança dos intervalos, by default
        :data:`constants.defaults.DEFAULT_CONFIDENCE_LEVEL`.
    random_state : int, optional
        Semente aleatória do bootstrap, by default
        :data:`constants.defaults.DEFAULT_RANDOM_SEED`.

    Returns
    -------
    EvaluationResult
        Resultado consolidado da avaliação.

    Raises
    ------
    EmptyDatasetError
        Se ``y_true`` estiver vazio.

    Examples
    --------
    >>> y_true = ["positivo", "negativo", "positivo", "negativo", "neutro"]
    >>> y_pred = ["positivo", "negativo", "negativo", "negativo", "neutro"]
    >>> resultado = evaluate_classifier(y_true, y_pred, n_bootstrap=20)
    >>> "f1_macro" in resultado.point_metrics
    True
    """
    if len(y_true) == 0:
        raise EmptyDatasetError("y_true")

    point_metrics = calculate_classification_metrics(y_true, y_pred)
    if y_score is not None:
        point_metrics.update(calculate_ranking_metrics(y_true, y_score, labels=labels))

    confidence_intervals = calculate_bootstrap_confidence_intervals(
        y_true,
        y_pred,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=random_state,
    )

    result = EvaluationResult(
        point_metrics=point_metrics,
        confidence_intervals=confidence_intervals,
        per_class_report=calculate_per_class_report(y_true, y_pred, labels=labels),
        confusion_matrix=calculate_confusion_matrix(y_true, y_pred, labels=labels),
    )
    lower_bound, upper_bound = confidence_intervals[PRIMARY_METRIC]
    logger.info(
        "Avaliação concluída: %s=%.4f (IC %.0f%%: [%.4f, %.4f]).",
        PRIMARY_METRIC,
        point_metrics[PRIMARY_METRIC],
        confidence_level * 100,
        lower_bound,
        upper_bound,
    )
    return result
