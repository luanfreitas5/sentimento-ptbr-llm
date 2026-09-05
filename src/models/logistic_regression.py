"""Classificador de Regressão Logística para classificação de sentimento.

Implementa a Fase 9 (Seção 4.4 do documento mestre): baseline clássico linear
sobre representações TF-IDF ou embeddings (``src/features/``), consumido por
``src/models/factory.py``.
"""

import logging

from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


def build_logistic_regression_classifier(
    *,
    C: float = 1.0,  # noqa: N803
    penalty: str = "l2",
    solver: str = "lbfgs",
    max_iter: int = 1000,
    class_weight: str | dict[str, float] | None = "balanced",
    random_state: int = 42,
) -> LogisticRegression:
    """Constrói um classificador de Regressão Logística multiclasse.

    Parameters
    ----------
    C : float, optional
        Inverso da força de regularização, by default 1.0.
    penalty : str, optional
        Tipo de penalização, by default "l2".
    solver : str, optional
        Algoritmo de otimização, by default "lbfgs".
    max_iter : int, optional
        Número máximo de iterações até a convergência, by default 1000.
    class_weight : str | dict[str, float] | None, optional
        Estratégia de ponderação de classes, by default "balanced" (ver
        ``configs/model_params.yaml -> classical.logistic_regression``,
        compensando o desbalanceamento entre as classes de sentimento).
    random_state : int, optional
        Semente aleatória, by default 42.

    Returns
    -------
    LogisticRegression
        Classificador scikit-learn não treinado.

    Examples
    --------
    >>> build_logistic_regression_classifier(C=0.5).C
    0.5
    """
    logger.info("Construindo classificador de Regressão Logística (C=%.3f).", C)
    return LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
    )
