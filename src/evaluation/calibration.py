"""Avaliação de calibração das probabilidades preditas.

Implementa ``configs/evaluation.yaml`` -> ``calibration``: verifica se a
confiança do modelo reflete sua real probabilidade de acerto — essencial
quando as probabilidades são usadas para decisões (ex.: classificação
seletiva em :func:`metrics.confidence.calculate_selective_prediction_accuracy`).
"""

import logging
from collections.abc import Sequence

import numpy as np

from constants.labels import SENTIMENT_CLASSES
from exceptions.data import EmptyDatasetError
from metrics.confidence import calculate_multiclass_brier_score

logger = logging.getLogger(__name__)

DEFAULT_N_CALIBRATION_BINS = 10


def calculate_reliability_curve(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    confidences: np.ndarray,
    *,
    n_bins: int = DEFAULT_N_CALIBRATION_BINS,
) -> dict[str, np.ndarray]:
    """Calcula a curva de confiabilidade (confiança predita vs. acurácia observada).

    Agrupa as predições em ``n_bins`` faixas de confiança e, em cada faixa,
    compara a confiança média com a fração de acertos observada. Uma
    calibração perfeita produz pontos sobre a diagonal.

    Parameters
    ----------
    y_true : Sequence[str]
        Rótulos de sentimento verdadeiros.
    y_pred : Sequence[str]
        Rótulos de sentimento preditos, mesmo tamanho de ``y_true``.
    confidences : np.ndarray
        Confiança (probabilidade da classe predita) por amostra, mesmo
        tamanho de ``y_true``, com valores em ``[0, 1]``.
    n_bins : int, optional
        Número de faixas de confiança de largura igual, by default
        :data:`DEFAULT_N_CALIBRATION_BINS`.

    Returns
    -------
    dict[str, np.ndarray]
        Dicionário com ``bin_confidence_means``, ``bin_accuracy`` (``nan``
        em faixas vazias) e ``bin_counts``, um valor por faixa.

    Raises
    ------
    EmptyDatasetError
        Se ``y_true`` estiver vazio.
    ValueError
        Se ``n_bins`` for menor que 1.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = ["positivo", "positivo", "negativo", "negativo"]
    >>> y_pred = ["positivo", "positivo", "negativo", "positivo"]
    >>> confidences = np.array([0.9, 0.95, 0.85, 0.3])
    >>> curva = calculate_reliability_curve(y_true, y_pred, confidences, n_bins=2)
    >>> curva["bin_counts"].tolist()
    [1, 3]
    """
    if len(y_true) == 0:
        raise EmptyDatasetError("y_true")
    if n_bins < 1:
        raise ValueError(f"n_bins deve ser >= 1, recebido: {n_bins}")

    correctness = np.array(
        [true_label == predicted_label for true_label, predicted_label in zip(y_true, y_pred)],
        dtype=float,
    )
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(confidences, bin_edges[1:-1], right=True), 0, n_bins - 1)

    bin_confidence_means = np.full(n_bins, np.nan)
    bin_accuracy = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=int)

    for bin_index in range(n_bins):
        mask = bin_indices == bin_index
        count = int(np.sum(mask))
        bin_counts[bin_index] = count
        if count > 0:
            bin_confidence_means[bin_index] = float(np.mean(confidences[mask]))
            bin_accuracy[bin_index] = float(np.mean(correctness[mask]))

    return {
        "bin_confidence_means": bin_confidence_means,
        "bin_accuracy": bin_accuracy,
        "bin_counts": bin_counts,
    }


def calculate_expected_calibration_error(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    confidences: np.ndarray,
    *,
    n_bins: int = DEFAULT_N_CALIBRATION_BINS,
) -> float:
    """Calcula o Erro de Calibração Esperado (ECE).

    Média ponderada, por faixa de confiança, da diferença absoluta entre
    confiança e acurácia observada (ver :func:`calculate_reliability_curve`).

    Parameters
    ----------
    y_true : Sequence[str]
        Rótulos de sentimento verdadeiros.
    y_pred : Sequence[str]
        Rótulos de sentimento preditos, mesmo tamanho de ``y_true``.
    confidences : np.ndarray
        Confiança (probabilidade da classe predita) por amostra, mesmo
        tamanho de ``y_true``.
    n_bins : int, optional
        Número de faixas de confiança, by default
        :data:`DEFAULT_N_CALIBRATION_BINS`.

    Returns
    -------
    float
        ECE, entre 0.0 (calibração perfeita) e 1.0.

    Raises
    ------
    EmptyDatasetError
        Se ``y_true`` estiver vazio.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = ["positivo", "positivo", "negativo", "negativo"]
    >>> y_pred = ["positivo", "positivo", "negativo", "positivo"]
    >>> confidences = np.array([0.9, 0.9, 0.9, 0.9])
    >>> round(calculate_expected_calibration_error(y_true, y_pred, confidences, n_bins=1), 4)
    0.15
    """
    curve = calculate_reliability_curve(y_true, y_pred, confidences, n_bins=n_bins)
    total_samples = len(y_true)
    expected_calibration_error = 0.0
    for count, confidence_mean, accuracy in zip(
        curve["bin_counts"], curve["bin_confidence_means"], curve["bin_accuracy"]
    ):
        if count > 0:
            expected_calibration_error += (count / total_samples) * abs(accuracy - confidence_mean)
    return float(expected_calibration_error)


def calculate_calibration_metrics(
    y_true: Sequence[str],
    y_score: np.ndarray,
    *,
    labels: Sequence[str] = SENTIMENT_CLASSES,
    n_bins: int = DEFAULT_N_CALIBRATION_BINS,
) -> dict[str, float]:
    """Calcula o conjunto de métricas de calibração usado no projeto.

    Combina o Brier score multiclasse (:func:`metrics.confidence.calculate_multiclass_brier_score`)
    com o ECE calculado sobre a confiança da classe predita (ver
    ``configs/evaluation.yaml`` -> ``calibration``).

    Parameters
    ----------
    y_true : Sequence[str]
        Rótulos de sentimento verdadeiros.
    y_score : np.ndarray
        Matriz ``(n_amostras, n_classes)`` de probabilidades preditas, na
        ordem de ``labels``.
    labels : Sequence[str], optional
        Ordem das classes nas colunas de ``y_score``, by default
        :data:`constants.labels.SENTIMENT_CLASSES`.
    n_bins : int, optional
        Número de faixas de confiança usadas no ECE, by default
        :data:`DEFAULT_N_CALIBRATION_BINS`.

    Returns
    -------
    dict[str, float]
        Dicionário com as chaves ``brier_score`` e ``expected_calibration_error``.

    Raises
    ------
    EmptyDatasetError
        Se ``y_true`` estiver vazio.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = ["positivo", "negativo"]
    >>> y_score = np.array([[0.1, 0.9], [0.8, 0.2]])
    >>> resultado = calculate_calibration_metrics(y_true, y_score, labels=["negativo", "positivo"])
    >>> sorted(resultado.keys())
    ['brier_score', 'expected_calibration_error']
    """
    labels_list = list(labels)
    y_pred = [labels_list[index] for index in np.argmax(y_score, axis=1)]
    confidences = np.max(y_score, axis=1)

    metrics = {
        "brier_score": calculate_multiclass_brier_score(y_true, y_score, labels=labels_list),
        "expected_calibration_error": calculate_expected_calibration_error(
            y_true, y_pred, confidences, n_bins=n_bins
        ),
    }
    logger.info(
        "Métricas de calibração: Brier score=%.4f, ECE=%.4f.",
        metrics["brier_score"],
        metrics["expected_calibration_error"],
    )
    return metrics
