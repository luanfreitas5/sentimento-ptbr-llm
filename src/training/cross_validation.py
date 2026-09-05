"""Validação cruzada estratificada com sementes fixas e intervalo de confiança.

Implementa a prática de avaliação rigorosa exigida por ``CLAUDE.md``
("Rigorous evaluation"): nunca reportar uma métrica pontual isolada, mas sim
a média entre dobras acompanhada de um intervalo de confiança. Opera sobre
qualquer classificador que satisfaça
:class:`models.base.SentimentClassifier`, reconstruído do zero a cada dobra
via ``model_builder`` (tipicamente ``functools.partial(create_classifier,
"logistic_regression")``, de ``src/models/factory.py``).
"""

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

from constants.defaults import DEFAULT_CROSS_VALIDATION_FOLDS, DEFAULT_RANDOM_SEED
from models.base import SentimentClassifier
from utils.seed import seed_everything
from utils.validation import validate_not_empty_collection

logger = logging.getLogger(__name__)

_CLASSIFICATION_SCORERS: dict[str, Callable[[Sequence[str], Sequence[str]], float]] = {
    "accuracy": accuracy_score,
    "f1_macro": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"),
    "mcc": matthews_corrcoef,
}


def compute_classification_score(
    y_true: Sequence[str], y_pred: Sequence[str], *, scoring: str = "f1_macro"
) -> float:
    """Calcula uma métrica de classificação nomeada, a partir dos rótulos verdadeiro/predito.

    Parameters
    ----------
    y_true : Sequence[str]
        Rótulos de sentimento verdadeiros.
    y_pred : Sequence[str]
        Rótulos de sentimento preditos, mesmo tamanho de ``y_true``.
    scoring : str, optional
        Nome da métrica, uma das chaves de ``{"accuracy", "f1_macro",
        "mcc"}`` (métrica principal e robusta a desbalanceamento definidas
        em ``CLAUDE.md`` -> "Project-Specific Overrides"), by default
        "f1_macro".

    Returns
    -------
    float
        Valor da métrica calculada.

    Raises
    ------
    ValueError
        Se ``scoring`` não for uma métrica suportada.

    Examples
    --------
    >>> compute_classification_score(["positivo", "negativo"], ["positivo", "negativo"])
    1.0
    """
    scorer = _CLASSIFICATION_SCORERS.get(scoring)
    if scorer is None:
        raise ValueError(
            f"Métrica '{scoring}' não suportada. Métricas disponíveis: "
            f"{sorted(_CLASSIFICATION_SCORERS)}"
        )
    return float(scorer(y_true, y_pred))


def _select_rows(
    data: Sequence[Any] | np.ndarray, indices: np.ndarray
) -> Sequence[Any] | np.ndarray:
    """Seleciona um subconjunto de linhas de ``data`` pelos índices informados.

    Parameters
    ----------
    data : Sequence[Any] | np.ndarray
        Coleção de entrada: uma sequência de textos/documentos ou uma
        matriz de features NumPy.
    indices : np.ndarray
        Índices inteiros das linhas a selecionar.

    Returns
    -------
    Sequence[Any] | np.ndarray
        Subconjunto de ``data`` correspondente a ``indices``, preservando o
        tipo (lista para sequências genéricas, indexação vetorizada para
        ``np.ndarray``).
    """
    if isinstance(data, np.ndarray):
        return data[indices]
    return [data[index] for index in indices]


@dataclass
class CrossValidationResult:
    """Resultado agregado de uma validação cruzada estratificada.

    Parameters
    ----------
    scoring : str
        Nome da métrica avaliada em cada dobra.
    fold_scores : list[float]
        Pontuação obtida em cada dobra executada (pode ser menor que
        ``cv`` se a validação cruzada tiver sido interrompida
        antecipadamente via ``on_fold_end``).
    """

    scoring: str
    fold_scores: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        """Média das pontuações entre dobras.

        Returns
        -------
        float
            Média aritmética de :attr:`fold_scores`, ou ``0.0`` se vazio.
        """
        return float(np.mean(self.fold_scores)) if self.fold_scores else 0.0

    @property
    def std(self) -> float:
        """Desvio padrão das pontuações entre dobras.

        Returns
        -------
        float
            Desvio padrão (amostral, ``ddof=0``) de :attr:`fold_scores`, ou
            ``0.0`` se houver menos de duas dobras.
        """
        return float(np.std(self.fold_scores)) if len(self.fold_scores) > 1 else 0.0

    @property
    def confidence_interval_95(self) -> float:
        """Semi-amplitude do intervalo de confiança de 95% em torno de :attr:`mean`.

        Segue a convenção documentada em ``pyproject.toml`` (seção
        "Rigorous evaluation"): ``1.96 * desvio_padrao / sqrt(n_dobras)``.
        O valor real da métrica é esperado no intervalo
        ``[mean - confidence_interval_95, mean + confidence_interval_95]``.

        Returns
        -------
        float
            Semi-amplitude do intervalo de confiança de 95%, ou ``0.0`` se
            houver menos de duas dobras.
        """
        if len(self.fold_scores) < 2:
            return 0.0
        return 1.96 * self.std / math.sqrt(len(self.fold_scores))


def run_stratified_cross_validation(
    model_builder: Callable[[], SentimentClassifier],
    X: Sequence[Any] | np.ndarray,
    y: Sequence[str],
    *,
    cv: int = DEFAULT_CROSS_VALIDATION_FOLDS,
    scoring: str = "f1_macro",
    random_state: int = DEFAULT_RANDOM_SEED,
    on_fold_end: Callable[[int, float, SentimentClassifier], bool] | None = None,
) -> CrossValidationResult:
    """Executa validação cruzada estratificada com semente fixa, dobra a dobra.

    Reconstrói um modelo do zero a cada dobra (via ``model_builder``),
    evitando vazamento de estado entre dobras. Fixa toda a aleatoriedade do
    processo antes de iniciar (:func:`utils.seed.seed_everything`).

    Parameters
    ----------
    model_builder : Callable[[], SentimentClassifier]
        Função que constrói uma nova instância de modelo não treinada a
        cada chamada (ex.: ``functools.partial(create_classifier,
        "logistic_regression")``).
    X : Sequence[Any] | np.ndarray
        Amostras de entrada (textos ou matriz de features). Não vazio.
    y : Sequence[str]
        Rótulos de sentimento, mesmo tamanho de ``X``. Usado também para a
        estratificação das dobras.
    cv : int, optional
        Número de dobras, by default
        :data:`constants.defaults.DEFAULT_CROSS_VALIDATION_FOLDS`.
    scoring : str, optional
        Métrica avaliada por dobra, uma das chaves aceitas por
        :func:`compute_classification_score`, by default "f1_macro".
    random_state : int, optional
        Semente aleatória do particionamento e de todo o processo, by
        default :data:`constants.defaults.DEFAULT_RANDOM_SEED`.
    on_fold_end : Callable[[int, float, SentimentClassifier], bool] | None, optional
        Função de retorno de chamada opcional, invocada ao final de cada
        dobra com ``(índice_da_dobra, pontuação, modelo_da_dobra)``; se
        retornar ``True``, interrompe a validação cruzada antes de
        completar as ``cv`` dobras, by default None.

    Returns
    -------
    CrossValidationResult
        Pontuações por dobra e estatísticas agregadas (média, desvio padrão
        e intervalo de confiança de 95%).

    Raises
    ------
    EmptyDatasetError
        Se ``X`` ou ``y`` estiverem vazios.

    Examples
    --------
    >>> from functools import partial
    >>> from models.factory import create_classifier
    >>> X = ["ótimo", "péssimo", "razoável", "horrível", "excelente", "ruim"]
    >>> y = ["positivo", "negativo", "neutro", "negativo", "positivo", "negativo"]
    >>> resultado = run_stratified_cross_validation(
    ...     partial(create_classifier, "naive_bayes"), X, y, cv=2
    ... )  # doctest: +SKIP
    >>> len(resultado.fold_scores)  # doctest: +SKIP
    2
    """
    validate_not_empty_collection(X, collection_name="X")
    validate_not_empty_collection(y, collection_name="y")
    seed_everything(random_state)

    y_array = np.asarray(y)
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    result = CrossValidationResult(scoring=scoring)

    dummy_features = np.zeros((len(y_array), 1))
    for fold_index, (train_indices, val_indices) in enumerate(
        splitter.split(dummy_features, y_array)
    ):
        X_train = _select_rows(X, train_indices)
        y_train_fold = _select_rows(y_array, train_indices)
        X_val = _select_rows(X, val_indices)
        y_val_fold = _select_rows(y_array, val_indices)

        fold_model = model_builder()
        fold_model.fit(X_train, y_train_fold)
        y_pred = fold_model.predict(X_val)
        fold_score = compute_classification_score(y_val_fold, y_pred, scoring=scoring)
        result.fold_scores.append(fold_score)

        logger.info("Dobra %d/%d: %s=%.4f.", fold_index + 1, cv, scoring, fold_score)
        if on_fold_end is not None and on_fold_end(fold_index, fold_score, fold_model):
            logger.info("Validação cruzada interrompida antecipadamente na dobra %d.", fold_index)
            break

    logger.info(
        "Validação cruzada concluída: %s=%.4f +/- %.4f (IC 95%%) em %d dobra(s).",
        scoring,
        result.mean,
        result.confidence_interval_95,
        len(result.fold_scores),
    )
    return result
