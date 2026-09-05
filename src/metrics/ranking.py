"""Métricas de ranqueamento (baseadas em probabilidade) para o problema multiclasse.

Implementa a Seção 4.8 do documento mestre: ROC-AUC e PR-AUC calculados no
esquema *one-vs-rest* (OvR), apropriado ao cenário multiclasse de três
classes de sentimento (ver ``configs/evaluation.yaml`` -> ``metrics.ranking``).
"""

import logging
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import label_binarize

from constants.labels import SENTIMENT_CLASSES
from exceptions.data import EmptyDatasetError

logger = logging.getLogger(__name__)


def _validate_ranking_inputs(
    y_true: Sequence[str], y_score: np.ndarray, labels: Sequence[str]
) -> None:
    """Valida os vetores de entrada das métricas de ranqueamento.

    Parameters
    ----------
    y_true : Sequence[str]
        Rótulos verdadeiros.
    y_score : np.ndarray
        Matriz de probabilidades preditas.
    labels : Sequence[str]
        Classes esperadas, na ordem das colunas de ``y_score``.

    Raises
    ------
    EmptyDatasetError
        Se ``y_true`` estiver vazio.
    ValueError
        Se ``y_score`` não tiver uma coluna por classe em ``labels``.
    """
    if len(y_true) == 0:
        raise EmptyDatasetError("y_true")
    if y_score.shape[0] != len(y_true):
        raise ValueError(
            f"y_true e y_score devem ter o mesmo número de amostras, "
            f"recebido {len(y_true)} e {y_score.shape[0]}"
        )
    if y_score.shape[1] != len(labels):
        raise ValueError(
            f"y_score deve ter uma coluna por classe em 'labels' ({len(labels)}), "
            f"recebido {y_score.shape[1]}"
        )


def calculate_roc_auc_ovr(
    y_true: Sequence[str], y_score: np.ndarray, *, labels: Sequence[str] = SENTIMENT_CLASSES
) -> float:
    """Calcula a área sob a curva ROC no esquema *one-vs-rest*, média macro.

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
        ROC-AUC OvR macro-médio, entre 0.0 e 1.0.

    Raises
    ------
    EmptyDatasetError
        Se ``y_true`` estiver vazio.
    ValueError
        Se as dimensões de ``y_true``/``y_score``/``labels`` forem
        inconsistentes, ou se menos de duas classes estiverem presentes em
        ``y_true``.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = ["negativo", "positivo", "negativo", "positivo"]
    >>> y_score = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]])
    >>> calculate_roc_auc_ovr(y_true, y_score, labels=["negativo", "positivo"])
    1.0
    """
    _validate_ranking_inputs(y_true, y_score, labels)
    return float(
        roc_auc_score(y_true, y_score, multi_class="ovr", average="macro", labels=list(labels))
    )


def calculate_pr_auc_ovr(
    y_true: Sequence[str], y_score: np.ndarray, *, labels: Sequence[str] = SENTIMENT_CLASSES
) -> float:
    """Calcula a área sob a curva Precisão-Revocação no esquema *one-vs-rest*, média macro.

    Mais informativa que o ROC-AUC sob desbalanceamento de classes, por não
    considerar os verdadeiros negativos.

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
        PR-AUC OvR macro-médio, entre 0.0 e 1.0.

    Raises
    ------
    EmptyDatasetError
        Se ``y_true`` estiver vazio.
    ValueError
        Se as dimensões de ``y_true``/``y_score``/``labels`` forem
        inconsistentes.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = ["negativo", "positivo", "negativo", "positivo"]
    >>> y_score = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]])
    >>> calculate_pr_auc_ovr(y_true, y_score, labels=["negativo", "positivo"])
    1.0
    """
    _validate_ranking_inputs(y_true, y_score, labels)
    y_true_binarized = label_binarize(y_true, classes=list(labels))
    per_class_scores = [
        average_precision_score(y_true_binarized[:, class_index], y_score[:, class_index])
        for class_index in range(len(labels))
    ]
    return float(np.mean(per_class_scores))


def calculate_ranking_metrics(
    y_true: Sequence[str], y_score: np.ndarray, *, labels: Sequence[str] = SENTIMENT_CLASSES
) -> dict[str, float]:
    """Calcula o conjunto de métricas de ranqueamento usado no projeto.

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
    dict[str, float]
        Dicionário com as chaves ``roc_auc_ovr`` e ``pr_auc_ovr`` (ver
        ``configs/evaluation.yaml`` -> ``metrics.ranking``).

    Raises
    ------
    EmptyDatasetError
        Se ``y_true`` estiver vazio.
    ValueError
        Se as dimensões de ``y_true``/``y_score``/``labels`` forem
        inconsistentes.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = ["negativo", "positivo", "negativo", "positivo"]
    >>> y_score = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]])
    >>> sorted(calculate_ranking_metrics(y_true, y_score, labels=["negativo", "positivo"]).keys())
    ['pr_auc_ovr', 'roc_auc_ovr']
    """
    metrics = {
        "roc_auc_ovr": calculate_roc_auc_ovr(y_true, y_score, labels=labels),
        "pr_auc_ovr": calculate_pr_auc_ovr(y_true, y_score, labels=labels),
    }
    logger.info(
        "Métricas de ranqueamento: ROC-AUC OvR=%.4f, PR-AUC OvR=%.4f.",
        metrics["roc_auc_ovr"],
        metrics["pr_auc_ovr"],
    )
    return metrics
