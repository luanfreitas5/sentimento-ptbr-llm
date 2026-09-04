"""Classificador Naive Bayes multinomial para classificação de sentimento.

Implementa a Fase 9 (Seção 4.4 do documento mestre): baseline clássico sobre
a representação TF-IDF (``src/features/lexical.py``), consumido por
``src/models/factory.py``.
"""

import logging

from sklearn.naive_bayes import MultinomialNB

logger = logging.getLogger(__name__)


def build_naive_bayes_classifier(*, alpha: float = 1.0) -> MultinomialNB:
    """Constrói um classificador Naive Bayes multinomial.

    Parameters
    ----------
    alpha : float, optional
        Parâmetro de suavização aditiva (Laplace/Lidstone), by default 1.0
        (``configs/model_params.yaml -> classical.naive_bayes.alpha``).

    Returns
    -------
    MultinomialNB
        Classificador scikit-learn não treinado, pronto para ``fit`` sobre
        uma matriz TF-IDF densa/esparsa não negativa (ver
        :func:`features.lexical.pivot_tfidf_features_to_wide`).

    Examples
    --------
    >>> build_naive_bayes_classifier(alpha=0.5).alpha
    0.5
    """
    logger.info("Construindo classificador Naive Bayes multinomial (alpha=%.3f).", alpha)
    return MultinomialNB(alpha=alpha)
