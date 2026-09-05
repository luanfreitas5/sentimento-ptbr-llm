"""Classificador SVM (Support Vector Machine) para classificação de sentimento.

Implementa a Fase 9 (Seção 4.4 do documento mestre): SVM linear sobre TF-IDF
ou SVM com outros kernels sobre embeddings (``src/features/``), consumido
por ``src/models/factory.py``.
"""

import logging

from sklearn.svm import SVC

logger = logging.getLogger(__name__)


def build_svm_classifier(
    *,
    kernel: str = "linear",
    C: float = 1.0,  # noqa: N803
    class_weight: str | dict[str, float] | None = "balanced",
    probability: bool = True,
    random_state: int = 42,
) -> SVC:
    """Constrói um classificador SVM multiclasse.

    Parameters
    ----------
    kernel : str, optional
        Função de kernel, by default "linear" (ver
        ``configs/model_params.yaml -> classical.svm.kernel``; "linear"
        para TF-IDF, "rbf" tipicamente sobre embeddings densos).
    C : float, optional
        Parâmetro de regularização, by default 1.0.
    class_weight : str | dict[str, float] | None, optional
        Estratégia de ponderação de classes, by default "balanced".
    probability : bool, optional
        Se ``True``, habilita a estimativa de probabilidade via calibração
        interna (Platt scaling), exigida por :meth:`predict_proba`, by
        default True.
    random_state : int, optional
        Semente aleatória (usada na calibração de probabilidade), by
        default 42.

    Returns
    -------
    SVC
        Classificador scikit-learn não treinado.

    Examples
    --------
    >>> build_svm_classifier(kernel="rbf").kernel
    'rbf'
    """
    logger.info("Construindo classificador SVM (kernel=%s, C=%.3f).", kernel, C)
    return SVC(
        kernel=kernel,
        C=C,
        class_weight=class_weight,
        probability=probability,
        random_state=random_state,
    )
