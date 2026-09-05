"""Métricas baseadas na confiança (probabilidade) das predições.

Complementam as métricas de classificação e ranqueamento ao avaliar a
qualidade das probabilidades emitidas pelo modelo — insumo para a análise
de calibração (``src/evaluation/calibration.py``) e para diagnósticos de
predições de baixa confiança (``src/visualization/diagnostics.py``).
"""

import logging
from collections.abc import Sequence

import numpy as np
from scipy.stats import pointbiserialr
from sklearn.preprocessing import label_binarize

from constants.labels import SENTIMENT_CLASSES
from exceptions.data import EmptyDatasetError

logger = logging.getLogger(__name__)


def calculate_average_confidence(y_score: np.ndarray) -> float:
    """Calcula a confiança média das predições (probabilidade da classe vencedora).

    Parameters
    ----------
    y_score : np.ndarray
        Matriz ``(n_amostras, n_classes)`` de probabilidades preditas.

    Returns
    -------
    float
        Média, sobre as amostras, da maior probabilidade por linha.

    Raises
    ------
    EmptyDatasetError
        Se ``y_score`` estiver vazio.

    Examples
    --------
    >>> import numpy as np
    >>> calculate_average_confidence(np.array([[0.9, 0.1], [0.6, 0.4]]))
    0.75
    """
    if y_score.shape[0] == 0:
        raise EmptyDatasetError("y_score")
    return float(np.mean(np.max(y_score, axis=1)))


def calculate_confidence_accuracy_correlation(
    y_true: Sequence[str], y_pred: Sequence[str], confidences: np.ndarray
) -> float:
    """Calcula a correlação ponto-bisserial entre a confiança e o acerto da predição.

    Um modelo bem calibrado deve ser mais confiante justamente quando
    acerta: correlação próxima de 1.0 indica esse comportamento desejável;
    próxima de 0 indica que a confiança não discrimina acertos de erros.

    Parameters
    ----------
    y_true : Sequence[str]
        Rótulos de sentimento verdadeiros.
    y_pred : Sequence[str]
        Rótulos de sentimento preditos, mesmo tamanho de ``y_true``.
    confidences : np.ndarray
        Confiança (probabilidade da classe predita) por amostra, mesmo
        tamanho de ``y_true``.

    Returns
    -------
    float
        Coeficiente de correlação ponto-bisserial, entre -1.0 e 1.0. Pode
        ser ``nan`` se todas as predições forem corretas ou todas
        incorretas (variância nula da variável binária).

    Raises
    ------
    EmptyDatasetError
        Se ``y_true`` estiver vazio.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = ["positivo", "negativo", "positivo", "negativo"]
    >>> y_pred = ["positivo", "negativo", "negativo", "negativo"]
    >>> confidences = np.array([0.95, 0.90, 0.55, 0.60])
    >>> round(calculate_confidence_accuracy_correlation(y_true, y_pred, confidences), 4)
    0.6532
    """
    if len(y_true) == 0:
        raise EmptyDatasetError("y_true")
    correctness = np.array(
        [true_label == predicted_label for true_label, predicted_label in zip(y_true, y_pred)],
        dtype=int,
    )
    correlation, _ = pointbiserialr(correctness, confidences)
    return float(correlation)


def calculate_multiclass_brier_score(
    y_true: Sequence[str], y_score: np.ndarray, *, labels: Sequence[str] = SENTIMENT_CLASSES
) -> float:
    """Calcula o Brier score multiclasse (erro quadrático médio das probabilidades).

    Métrica de calibração: para cada amostra, soma o erro quadrático entre
    a probabilidade predita e o vetor *one-hot* do rótulo verdadeiro; 0.0
    indica calibração perfeita.

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

    Returns
    -------
    float
        Brier score multiclasse, entre 0.0 e 2.0 (quanto menor, melhor).

    Raises
    ------
    EmptyDatasetError
        Se ``y_true`` estiver vazio.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = ["positivo", "negativo"]
    >>> y_score = np.array([[0.0, 1.0], [1.0, 0.0]])
    >>> calculate_multiclass_brier_score(y_true, y_score, labels=["negativo", "positivo"])
    0.0
    """
    if len(y_true) == 0:
        raise EmptyDatasetError("y_true")
    y_true_one_hot = label_binarize(y_true, classes=list(labels))
    squared_errors = np.sum((y_true_one_hot - y_score) ** 2, axis=1)
    return float(np.mean(squared_errors))


def calculate_selective_prediction_accuracy(
    y_true: Sequence[str], y_pred: Sequence[str], confidences: np.ndarray, *, coverage: float
) -> float:
    """Calcula a acurácia restrita às predições mais confiantes (classificação seletiva).

    Simula a rejeição das amostras menos confiantes (ex.: para revisão
    humana), mantendo apenas a fração ``coverage`` com maior confiança.

    Parameters
    ----------
    y_true : Sequence[str]
        Rótulos de sentimento verdadeiros.
    y_pred : Sequence[str]
        Rótulos de sentimento preditos, mesmo tamanho de ``y_true``.
    confidences : np.ndarray
        Confiança (probabilidade da classe predita) por amostra, mesmo
        tamanho de ``y_true``.
    coverage : float
        Fração de amostras mais confiantes a manter, entre 0 (exclusivo) e
        1 (inclusivo).

    Returns
    -------
    float
        Acurácia calculada apenas sobre as amostras mantidas.

    Raises
    ------
    EmptyDatasetError
        Se ``y_true`` estiver vazio.
    ValueError
        Se ``coverage`` não estiver no intervalo ``(0, 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = ["positivo", "negativo", "positivo", "negativo"]
    >>> y_pred = ["positivo", "negativo", "negativo", "negativo"]
    >>> confidences = np.array([0.95, 0.90, 0.55, 0.60])
    >>> calculate_selective_prediction_accuracy(y_true, y_pred, confidences, coverage=0.5)
    1.0
    """
    if len(y_true) == 0:
        raise EmptyDatasetError("y_true")
    if not (0 < coverage <= 1):
        raise ValueError(f"coverage deve estar em (0, 1], recebido: {coverage}")

    n_samples_to_keep = max(1, int(np.ceil(coverage * len(y_true))))
    most_confident_indices = np.argsort(-confidences)[:n_samples_to_keep]

    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    kept_correctness = y_true_array[most_confident_indices] == y_pred_array[most_confident_indices]
    return float(np.mean(kept_correctness))
