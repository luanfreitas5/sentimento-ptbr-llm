"""Treino dos classificadores Transformer (BERTimbau, RoBERTa, DistilBERT) de sentimento.

Implementa o estágio ``training_transformer`` de ``configs/config.yaml ->
stages``: faz o fine-tuning de cada Transformer configurado
(``configs/model_params.yaml -> transformers``) com parada antecipada e
checkpoint por passo, reutilizando
:func:`pipelines.training_deep_learning.train_neural_models`. Os modelos são
persistidos em formato PyTorch em ``models/checkpoints/<modelo>.pt``.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pipelines.training_deep_learning import train_neural_models
from training.trainer import TrainingResult

DEFAULT_TRANSFORMER_MODEL_NAMES: tuple[str, ...] = ("bertimbau", "roberta", "distilbert")


def run_training_transformer_stage(
    X_train: Sequence[Any],  # noqa: N803
    y_train: Sequence[str],
    X_val: Sequence[Any] | None,  # noqa: N803
    y_val: Sequence[str] | None,
    *,
    model_names: Sequence[str] = DEFAULT_TRANSFORMER_MODEL_NAMES,
    model_params: Mapping[str, Mapping[str, Any]] | None = None,
    checkpoints_dir: Path,
    early_stopping_monitor: str = "f1_macro",
    early_stopping_patience: int = 2,
    track_with_mlflow: bool = False,
) -> dict[str, TrainingResult]:
    """Faz o fine-tuning de cada Transformer configurado, com parada antecipada.

    Parameters
    ----------
    X_train : Sequence[Any]
        Textos de treino (cada Transformer tokeniza com o próprio tokenizador).
    y_train : Sequence[str]
        Rótulos de sentimento de treino, mesmo tamanho de ``X_train``.
    X_val : Sequence[Any] | None
        Textos de validação, monitorados pela parada antecipada.
    y_val : Sequence[str] | None
        Rótulos de validação, mesmo tamanho de ``X_val``.
    model_names : Sequence[str], optional
        Modelos a treinar, by default :data:`DEFAULT_TRANSFORMER_MODEL_NAMES`.
    model_params : Mapping[str, Mapping[str, Any]] | None, optional
        Hiperparâmetros por modelo (``configs/model_params.yaml -> transformers``),
        by default None.
    checkpoints_dir : Path
        Diretório-raiz dos checkpoints (``paths.models_checkpoints_dir``).
    early_stopping_monitor : str, optional
        Métrica monitorada pela parada antecipada, by default "f1_macro".
    early_stopping_patience : int, optional
        Paciência da parada antecipada, by default 2.
    track_with_mlflow : bool, optional
        Repassado a :class:`training.trainer.Trainer`, by default False.

    Returns
    -------
    dict[str, TrainingResult]
        Resultado de treino de cada modelo, indexado pelo nome do modelo.

    Examples
    --------
    >>> run_training_transformer_stage(
    ...     X_train, y_train, X_val, y_val, checkpoints_dir=Path("models/checkpoints")
    ... )  # doctest: +SKIP
    """
    return train_neural_models(
        X_train,
        y_train,
        X_val,
        y_val,
        model_names=model_names,
        model_params=model_params,
        checkpoints_dir=checkpoints_dir,
        early_stopping_monitor=early_stopping_monitor,
        early_stopping_patience=early_stopping_patience,
        track_with_mlflow=track_with_mlflow,
    )
