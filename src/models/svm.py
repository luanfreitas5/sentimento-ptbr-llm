"""Classificador SVM (Support Vector Machine) para classificação de sentimento.

Implementa a Fase 9 (Seção 4.4 do documento mestre): SVM linear sobre TF-IDF
(``src/features/``), consumido por ``src/models/factory.py``.

Usa :class:`sklearn.svm.LinearSVC` (liblinear), de custo aproximadamente linear
no número de amostras, em vez de :class:`sklearn.svm.SVC` (libsvm), de custo
entre quadrático e cúbico e proibitivo sobre ~45 mil textos com 20 mil features.
"""

import logging

from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)


def build_svm_classifier(
    *,
    C: float = 1.0,  # noqa: N803
    class_weight: str | dict[str, float] | None = "balanced",
    probability: bool = True,
    calibration_folds: int = 3,
    max_iter: int = 2000,
    random_state: int = 42,
) -> LinearSVC | CalibratedClassifierCV:
    """Constrói um classificador SVM linear multiclasse (um-contra-todos).

    Parameters
    ----------
    C : float, optional
        Parâmetro de regularização, by default 1.0.
    class_weight : str | dict[str, float] | None, optional
        Estratégia de ponderação de classes, by default "balanced".
    probability : bool, optional
        Se ``True``, envolve o SVM em :class:`CalibratedClassifierCV`
        (calibração sigmoide), exigida por ``predict_proba``, by default True.
    calibration_folds : int, optional
        Número de folds da calibração (só usado se ``probability=True``), by
        default 3.
    max_iter : int, optional
        Máximo de iterações do solver, by default 2000.
    random_state : int, optional
        Semente aleatória, by default 42.

    Returns
    -------
    LinearSVC | CalibratedClassifierCV
        Classificador scikit-learn não treinado: ``CalibratedClassifierCV``
        quando ``probability=True``, ``LinearSVC`` puro caso contrário.

    Examples
    --------
    >>> build_svm_classifier(probability=False).C
    1.0
    """
    logger.info("Construindo classificador SVM linear (C=%.3f, probability=%s).", C, probability)
    svm = LinearSVC(
        C=C,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
    )
    if not probability:
        return svm
    return CalibratedClassifierCV(svm, method="sigmoid", cv=calibration_folds)
