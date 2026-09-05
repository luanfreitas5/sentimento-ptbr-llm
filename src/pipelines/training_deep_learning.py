"""Treino dos classificadores de deep learning de sentimento.

Implementa o estágio ``training_deep_learning`` de ``configs/config.yaml ->
stages``: treina cada modelo de deep learning/Transformer configurado
(``configs/model_params.yaml -> deep_learning``/``transformers``) via
:class:`training.trainer.Trainer`, com parada antecipada e checkpoint por
passo (``src/training/callbacks.py``), e persiste os modelos treinados em
formato PyTorch (``src/models/persistence.py``).
"""

import logging
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

from models.factory import create_classifier
from models.persistence import save_classifier
from training.callbacks import EarlyStoppingCallback, ModelCheckpointCallback
from training.trainer import Trainer, TrainingResult

logger = logging.getLogger(__name__)

DEFAULT_DEEP_LEARNING_MODEL_NAMES: tuple[str, ...] = (
    "lstm",
    "cnn",
    "bertimbau",
    "roberta",
    "distilbert",
)


def run_training_deep_learning_stage(
    X_train: Sequence[Any],
    y_train: Sequence[str],
    X_val: Sequence[Any] | None,
    y_val: Sequence[str] | None,
    *,
    model_names: Sequence[str] = DEFAULT_DEEP_LEARNING_MODEL_NAMES,
    model_params: Mapping[str, Mapping[str, Any]] | None = None,
    checkpoints_dir: Path,
    early_stopping_monitor: str = "f1_macro",
    early_stopping_patience: int = 5,
    track_with_mlflow: bool = False,
) -> dict[str, TrainingResult]:
    """Treina cada modelo de deep learning/Transformer configurado, com parada antecipada.

    Parameters
    ----------
    X_train : Sequence[Any]
        Amostras de treino (documentos tokenizados para ``lstm``/``cnn``,
        textos crus para os Transformers).
    y_train : Sequence[str]
        Rótulos de sentimento de treino, mesmo tamanho de ``X_train``.
    X_val : Sequence[Any] | None
        Amostras de validação, monitoradas pela parada antecipada e pelo
        checkpoint de melhor modelo.
    y_val : Sequence[str] | None
        Rótulos de sentimento de validação, mesmo tamanho de ``X_val``.
    model_names : Sequence[str], optional
        Nomes dos modelos a treinar, uma das chaves de
        :func:`models.factory.create_classifier`, by default
        :data:`DEFAULT_DEEP_LEARNING_MODEL_NAMES`.
    model_params : Mapping[str, Mapping[str, Any]] | None, optional
        Hiperparâmetros por modelo (``configs/model_params.yaml ->
        deep_learning``/``transformers``), indexados pelo nome do modelo,
        by default None (hiperparâmetros padrão de cada modelo).
    checkpoints_dir : Path
        Diretório-raiz dos checkpoints (``paths.models_checkpoints_dir``);
        cada modelo grava seus checkpoints em um subdiretório próprio.
    early_stopping_monitor : str, optional
        Métrica monitorada pela parada antecipada e pelo checkpoint, by
        default "f1_macro".
    early_stopping_patience : int, optional
        Repassado a :class:`training.callbacks.EarlyStoppingCallback`, by
        default 5.
    track_with_mlflow : bool, optional
        Repassado a :class:`training.trainer.Trainer`, by default False.

    Returns
    -------
    dict[str, TrainingResult]
        Resultado de treino de cada modelo, indexado pelo nome do modelo.

    Examples
    --------
    >>> run_training_deep_learning_stage(
    ...     X_train,
    ...     y_train,
    ...     X_val,
    ...     y_val,
    ...     model_names=("lstm",),
    ...     checkpoints_dir=Path("models/checkpoints"),
    ... )  # doctest: +SKIP
    """
    resolved_model_params = model_params or {}
    results: dict[str, TrainingResult] = {}

    for model_name in model_names:
        overrides = resolved_model_params.get(model_name, {})
        model_builder = partial(create_classifier, model_name, **overrides)
        callbacks = [
            EarlyStoppingCallback(early_stopping_monitor, patience=early_stopping_patience),
            ModelCheckpointCallback(checkpoints_dir / model_name, monitor=early_stopping_monitor),
        ]
        trainer = Trainer(model_builder, callbacks=callbacks, track_with_mlflow=track_with_mlflow)
        result = trainer.fit(X_train, y_train, X_val, y_val)
        save_classifier(result.model, checkpoints_dir / f"{model_name}.pt", backend="torch")
        results[model_name] = result
        logger.info(
            "Modelo de deep learning '%s' treinado em %.2fs (métricas=%s).",
            model_name,
            result.elapsed_seconds,
            result.metrics,
        )

    return results
