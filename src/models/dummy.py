"""Classificador de referência (baseline) para classificação de sentimento.

Estabelece o piso de desempenho do estudo: sem um benchmark trivial não há
como afirmar que a complexidade adicional dos demais modelos trouxe ganho
real. Consumido por ``src/models/factory.py``.
"""

import logging

from sklearn.dummy import DummyClassifier

logger = logging.getLogger(__name__)


def build_dummy_classifier(
    *, strategy: str = "stratified", random_state: int = 42
) -> DummyClassifier:
    """Constrói o classificador de referência que ignora as features.

    Parameters
    ----------
    strategy : str, optional
        Estratégia do ``DummyClassifier`` (``"stratified"`` sorteia a classe
        respeitando a distribuição do treino; ``"most_frequent"`` prediz sempre
        a classe majoritária), by default "stratified".
    random_state : int, optional
        Semente do sorteio, by default 42.

    Returns
    -------
    DummyClassifier
        Classificador scikit-learn não treinado.

    Examples
    --------
    >>> build_dummy_classifier(strategy="most_frequent").strategy
    'most_frequent'
    """
    logger.info("Construindo classificador baseline (strategy=%s).", strategy)
    return DummyClassifier(strategy=strategy, random_state=random_state)
