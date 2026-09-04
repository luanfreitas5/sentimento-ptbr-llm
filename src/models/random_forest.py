"""Classificador Random Forest para classificação de sentimento.

Implementa a Fase 9 (Seção 4.4 do documento mestre): ensemble de árvores
sobre embeddings densos (``src/features/``), consumido por
``src/models/factory.py``.
"""

import logging

from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


def build_random_forest_classifier(
    *,
    n_estimators: int = 300,
    max_depth: int | None = 20,
    min_samples_leaf: int = 2,
    class_weight: str | dict[str, float] | None = "balanced",
    n_jobs: int = -1,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Constrói um classificador Random Forest.

    Parameters
    ----------
    n_estimators : int, optional
        Número de árvores na floresta, by default 300.
    max_depth : int | None, optional
        Profundidade máxima de cada árvore, by default 20.
    min_samples_leaf : int, optional
        Número mínimo de amostras exigido em um nó folha, by default 2.
    class_weight : str | dict[str, float] | None, optional
        Estratégia de ponderação de classes, by default "balanced".
    n_jobs : int, optional
        Número de processos paralelos (``-1`` usa todos os núcleos), by
        default -1.
    random_state : int, optional
        Semente aleatória, by default 42.

    Returns
    -------
    RandomForestClassifier
        Classificador scikit-learn não treinado.

    Examples
    --------
    >>> build_random_forest_classifier(n_estimators=100).n_estimators
    100
    """
    logger.info("Construindo classificador Random Forest (n_estimators=%d).", n_estimators)
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        n_jobs=n_jobs,
        random_state=random_state,
    )
