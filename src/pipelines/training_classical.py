"""Treino dos classificadores clássicos de sentimento.

Implementa o estágio ``training_classical`` de ``configs/config.yaml ->
stages``: treina cada modelo clássico configurado
(``configs/model_params.yaml -> classical``) via
:class:`training.trainer.Trainer`, sobre a matriz de features de
treino/validação, e persiste os modelos treinados
(``src/models/persistence.py``).
"""

import logging
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from constants.labels import transform_label_to_id
from models.factory import create_classifier
from models.persistence import save_classifier
from training.trainer import Trainer, TrainingResult

logger = logging.getLogger(__name__)

DEFAULT_CLASSICAL_MODEL_NAMES: tuple[str, ...] = (
    "naive_bayes",
    "logistic_regression",
    "svm",
    "random_forest",
    "gradient_boosting",
)

# XGBoost (``gradient_boosting``) exige rótulos codificados como inteiro
# (ver ``constants.labels.transform_label_to_id``); os demais modelos
# clássicos aceitam o rótulo textual diretamente.
_INTEGER_LABEL_MODEL_NAMES: frozenset[str] = frozenset({"gradient_boosting"})


def run_training_classical_stage(
    X_train: np.ndarray,
    y_train: Sequence[str],
    X_val: np.ndarray | None,
    y_val: Sequence[str] | None,
    *,
    model_names: Sequence[str] = DEFAULT_CLASSICAL_MODEL_NAMES,
    model_params: Mapping[str, Mapping[str, Any]] | None = None,
    checkpoints_dir: Path,
    track_with_mlflow: bool = False,
) -> dict[str, TrainingResult]:
    """Treina cada classificador clássico configurado, salvando os checkpoints resultantes.

    Parameters
    ----------
    X_train : np.ndarray
        Matriz de features de treino (ex.: TF-IDF denso, de
        ``src/pipelines/features.py``).
    y_train : Sequence[str]
        Rótulos de sentimento de treino, mesmo tamanho de ``X_train``.
    X_val : np.ndarray | None
        Matriz de features de validação, usada para calcular métricas ao
        final do treino de cada modelo.
    y_val : Sequence[str] | None
        Rótulos de sentimento de validação, mesmo tamanho de ``X_val``.
    model_names : Sequence[str], optional
        Nomes dos modelos a treinar, uma das chaves de
        :func:`models.factory.create_classifier`, by default
        :data:`DEFAULT_CLASSICAL_MODEL_NAMES`.
    model_params : Mapping[str, Mapping[str, Any]] | None, optional
        Hiperparâmetros por modelo (``configs/model_params.yaml ->
        classical``), indexados pelo nome do modelo, by default None
        (hiperparâmetros padrão de cada modelo).
    checkpoints_dir : Path
        Diretório de destino dos checkpoints (``paths.models_checkpoints_dir``).
    track_with_mlflow : bool, optional
        Repassado a :class:`training.trainer.Trainer`, by default False.

    Returns
    -------
    dict[str, TrainingResult]
        Resultado de treino de cada modelo, indexado pelo nome do modelo.

    Examples
    --------
    >>> run_training_classical_stage(
    ...     X_train,
    ...     y_train,
    ...     X_val,
    ...     y_val,
    ...     model_names=("naive_bayes",),
    ...     checkpoints_dir=Path("models/checkpoints"),
    ... )  # doctest: +SKIP
    """
    resolved_model_params = model_params or {}
    results: dict[str, TrainingResult] = {}

    for model_name in model_names:
        overrides = resolved_model_params.get(model_name, {})
        model_builder = partial(create_classifier, model_name, **overrides)
        trainer = Trainer(model_builder, track_with_mlflow=track_with_mlflow)

        if model_name in _INTEGER_LABEL_MODEL_NAMES:
            fold_y_train = [transform_label_to_id(label) for label in y_train]
            fold_y_val = (
                [transform_label_to_id(label) for label in y_val] if y_val is not None else None
            )
        else:
            fold_y_train = y_train
            fold_y_val = y_val

        result = trainer.fit(X_train, fold_y_train, X_val, fold_y_val)
        save_classifier(result.model, checkpoints_dir / f"{model_name}.joblib")
        results[model_name] = result
        logger.info(
            "Modelo clássico '%s' treinado em %.2fs (métricas=%s).",
            model_name,
            result.elapsed_seconds,
            result.metrics,
        )

    return results
