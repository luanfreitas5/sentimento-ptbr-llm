"""Classificador Gradient Boosting (XGBoost) para classificação de sentimento.

Implementa a Fase 9 (Seção 4.4 do documento mestre): ensemble de árvores por
boosting sobre embeddings densos (``src/features/``), consumido por
``src/models/factory.py``.
"""

import logging

from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def build_gradient_boosting_classifier(
    *,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    random_state: int = 42,
) -> XGBClassifier:
    """Constrói um classificador Gradient Boosting (XGBoost).

    Parameters
    ----------
    n_estimators : int, optional
        Número de árvores (rodadas de boosting), by default 300.
    max_depth : int, optional
        Profundidade máxima de cada árvore, by default 6.
    learning_rate : float, optional
        Taxa de aprendizado (encolhimento), by default 0.05.
    subsample : float, optional
        Fração de amostras usada no treino de cada árvore, by default 0.8.
    random_state : int, optional
        Semente aleatória, by default 42.

    Returns
    -------
    XGBClassifier
        Classificador XGBoost não treinado. Espera rótulos ``y`` já
        codificados como inteiros (ver
        :func:`constants.labels.transform_label_to_id`), conforme exigido
        pela API do XGBoost.

    Examples
    --------
    >>> build_gradient_boosting_classifier(n_estimators=50).n_estimators
    50
    """
    logger.info(
        "Construindo classificador Gradient Boosting (n_estimators=%d, max_depth=%d).",
        n_estimators,
        max_depth,
    )
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        random_state=random_state,
        objective="multi:softprob",
        eval_metric="mlogloss",
    )
