"""Métricas de classificação para o problema de sentimento pt-BR.

Implementa a Seção 4.8 do documento mestre
(``projeto-mestrado-analise-sentimentos-ptbr.md``): métricas robustas ao
desbalanceamento típico entre as três classes de sentimento
(``negativo``/``neutro``/``positivo``), com F1-macro como métrica principal
e MCC como métrica secundária robusta (ver ``configs/evaluation.yaml``).
Consumido por ``src/evaluation/evaluator.py``.
"""

import logging
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

from constants.labels import SENTIMENT_CLASSES
from exceptions.data import EmptyDatasetError

logger = logging.getLogger(__name__)


def _validate_prediction_inputs(y_true: Sequence[str], y_pred: Sequence[str]) -> None:
    """Valida que os vetores de rótulos verdadeiros e preditos sejam consistentes.

    Parameters
    ----------
    y_true : Sequence[str]
        Rótulos verdadeiros.
    y_pred : Sequence[str]
        Rótulos preditos.

    Raises
    ------
    EmptyDatasetError
        Se ``y_true`` estiver vazio.
    ValueError
        Se ``y_true`` e ``y_pred`` tiverem tamanhos diferentes.
    """
    if len(y_true) == 0:
        raise EmptyDatasetError("y_true")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true e y_pred devem ter o mesmo tamanho, recebido {len(y_true)} e {len(y_pred)}"
        )


def calculate_confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str], *, labels: Sequence[str] = SENTIMENT_CLASSES
) -> np.ndarray:
    """Calcula a matriz de confusão entre rótulos verdadeiros e preditos.

    Parameters
    ----------
    y_true : Sequence[str]
        Rótulos de sentimento verdadeiros.
    y_pred : Sequence[str]
        Rótulos de sentimento preditos, mesmo tamanho de ``y_true``.
    labels : Sequence[str], optional
        Ordem das classes nas linhas/colunas da matriz, by default
        :data:`constants.labels.SENTIMENT_CLASSES`.

    Returns
    -------
    np.ndarray
        Matriz ``(n_classes, n_classes)``: linha = classe verdadeira,
        coluna = classe predita.

    Raises
    ------
    EmptyDatasetError
        Se ``y_true`` estiver vazio.
    ValueError
        Se ``y_true`` e ``y_pred`` tiverem tamanhos diferentes.

    Examples
    --------
    >>> matriz = calculate_confusion_matrix(
    ...     ["positivo", "negativo"], ["positivo", "positivo"], labels=["negativo", "positivo"]
    ... )
    >>> matriz.tolist()
    [[0, 1], [0, 1]]
    """
    _validate_prediction_inputs(y_true, y_pred)
    return confusion_matrix(y_true, y_pred, labels=list(labels))


def calculate_accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Calcula a acurácia (fração de acertos) entre rótulos verdadeiros e preditos.

    Parameters
    ----------
    y_true : Sequence[str]
        Rótulos de sentimento verdadeiros.
    y_pred : Sequence[str]
        Rótulos de sentimento preditos, mesmo tamanho de ``y_true``.

    Returns
    -------
    float
        Acurácia, entre 0.0 e 1.0.

    Raises
    ------
    EmptyDatasetError
        Se ``y_true`` estiver vazio.
    ValueError
        Se ``y_true`` e ``y_pred`` tiverem tamanhos diferentes.

    Examples
    --------
    >>> calculate_accuracy(["positivo", "negativo"], ["positivo", "positivo"])
    0.5
    """
    _validate_prediction_inputs(y_true, y_pred)
    return float(accuracy_score(y_true, y_pred))


def calculate_precision_recall_f1(
    y_true: Sequence[str], y_pred: Sequence[str], *, average: str = "macro"
) -> dict[str, float]:
    """Calcula precisão, revocação e F1 agregados segundo uma estratégia de média.

    Parameters
    ----------
    y_true : Sequence[str]
        Rótulos de sentimento verdadeiros.
    y_pred : Sequence[str]
        Rótulos de sentimento preditos, mesmo tamanho de ``y_true``.
    average : str, optional
        Estratégia de agregação entre classes (``"macro"`` ou
        ``"weighted"``), by default "macro".

    Returns
    -------
    dict[str, float]
        Dicionário com as chaves ``precision``, ``recall`` e ``f1``.

    Raises
    ------
    EmptyDatasetError
        Se ``y_true`` estiver vazio.
    ValueError
        Se ``y_true`` e ``y_pred`` tiverem tamanhos diferentes.

    Examples
    --------
    >>> resultado = calculate_precision_recall_f1(
    ...     ["positivo", "negativo"], ["positivo", "positivo"], average="macro"
    ... )
    >>> round(resultado["f1"], 4)
    0.3333
    """
    _validate_prediction_inputs(y_true, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1_score)}


def calculate_per_class_report(
    y_true: Sequence[str], y_pred: Sequence[str], *, labels: Sequence[str] = SENTIMENT_CLASSES
) -> dict[str, dict[str, float]]:
    """Calcula precisão, revocação, F1 e suporte individualmente por classe.

    Essencial para a avaliação por fatia (ver
    ``src/evaluation/slice_evaluation.py``): métricas agregadas escondem
    falhas em classes específicas.

    Parameters
    ----------
    y_true : Sequence[str]
        Rótulos de sentimento verdadeiros.
    y_pred : Sequence[str]
        Rótulos de sentimento preditos, mesmo tamanho de ``y_true``.
    labels : Sequence[str], optional
        Classes a incluir no relatório, by default
        :data:`constants.labels.SENTIMENT_CLASSES`.

    Returns
    -------
    dict[str, dict[str, float]]
        Uma entrada por classe, cada uma com ``precision``, ``recall``,
        ``f1`` e ``support`` (número de amostras verdadeiras da classe).

    Raises
    ------
    EmptyDatasetError
        Se ``y_true`` estiver vazio.
    ValueError
        Se ``y_true`` e ``y_pred`` tiverem tamanhos diferentes.

    Examples
    --------
    >>> relatorio = calculate_per_class_report(
    ...     ["positivo", "negativo"], ["positivo", "positivo"], labels=["negativo", "positivo"]
    ... )
    >>> relatorio["positivo"]["recall"]
    1.0
    """
    _validate_prediction_inputs(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(labels),
        target_names=list(labels),
        output_dict=True,
        zero_division=0,
    )
    return {
        label: {
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "f1": float(report[label]["f1-score"]),
            "support": float(report[label]["support"]),
        }
        for label in labels
    }


def calculate_matthews_correlation_coefficient(
    y_true: Sequence[str], y_pred: Sequence[str]
) -> float:
    """Calcula o Coeficiente de Correlação de Matthews (MCC).

    Métrica secundária robusta ao desbalanceamento entre classes (ver
    ``configs/evaluation.yaml`` -> ``metrics.secondary``), complementar ao
    F1-macro por considerar todas as células da matriz de confusão.

    Parameters
    ----------
    y_true : Sequence[str]
        Rótulos de sentimento verdadeiros.
    y_pred : Sequence[str]
        Rótulos de sentimento preditos, mesmo tamanho de ``y_true``.

    Returns
    -------
    float
        MCC, entre -1.0 (discordância total) e 1.0 (concordância perfeita).

    Raises
    ------
    EmptyDatasetError
        Se ``y_true`` estiver vazio.
    ValueError
        Se ``y_true`` e ``y_pred`` tiverem tamanhos diferentes.

    Examples
    --------
    >>> calculate_matthews_correlation_coefficient(
    ...     ["positivo", "negativo", "positivo", "negativo"],
    ...     ["positivo", "negativo", "positivo", "negativo"],
    ... )
    1.0
    """
    _validate_prediction_inputs(y_true, y_pred)
    return float(matthews_corrcoef(y_true, y_pred))


def calculate_classification_metrics(
    y_true: Sequence[str], y_pred: Sequence[str]
) -> dict[str, float]:
    """Calcula o conjunto completo de métricas de classificação usado no projeto.

    Combina a métrica principal e as métricas secundárias definidas em
    ``configs/evaluation.yaml`` (:data:`constants.metrics.PRIMARY_METRIC` e
    :data:`constants.metrics.SECONDARY_METRICS`) em um único dicionário,
    consumido por ``src/evaluation/evaluator.py``.

    Parameters
    ----------
    y_true : Sequence[str]
        Rótulos de sentimento verdadeiros.
    y_pred : Sequence[str]
        Rótulos de sentimento preditos, mesmo tamanho de ``y_true``.

    Returns
    -------
    dict[str, float]
        Dicionário com as chaves ``f1_macro``, ``mcc``, ``f1_weighted``,
        ``accuracy``, ``precision_macro`` e ``recall_macro``.

    Raises
    ------
    EmptyDatasetError
        Se ``y_true`` estiver vazio.
    ValueError
        Se ``y_true`` e ``y_pred`` tiverem tamanhos diferentes.

    Examples
    --------
    >>> metricas = calculate_classification_metrics(
    ...     ["positivo", "negativo"], ["positivo", "positivo"]
    ... )
    >>> sorted(metricas.keys())
    ['accuracy', 'f1_macro', 'f1_weighted', 'mcc', 'precision_macro', 'recall_macro']
    """
    _validate_prediction_inputs(y_true, y_pred)
    macro_scores = calculate_precision_recall_f1(y_true, y_pred, average="macro")
    weighted_scores = calculate_precision_recall_f1(y_true, y_pred, average="weighted")
    metrics = {
        "f1_macro": macro_scores["f1"],
        "precision_macro": macro_scores["precision"],
        "recall_macro": macro_scores["recall"],
        "f1_weighted": weighted_scores["f1"],
        "accuracy": calculate_accuracy(y_true, y_pred),
        "mcc": calculate_matthews_correlation_coefficient(y_true, y_pred),
    }
    logger.info(
        "Métricas de classificação: F1-macro=%.4f, MCC=%.4f, acurácia=%.4f.",
        metrics["f1_macro"],
        metrics["mcc"],
        metrics["accuracy"],
    )
    return metrics
