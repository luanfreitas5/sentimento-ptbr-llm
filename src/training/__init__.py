"""Treino de classificadores de sentimento: loop genérico, callbacks e validação cruzada.

Implementa a Fase 10 do plano de elaboração (``PLANO-ELABORACAO.md``):
suporta o treino de ML clássico, deep learning e fine-tuning de Transformer
sobre a interface comum ``fit``/``predict``/``predict_proba``
(:class:`models.base.SentimentClassifier`), sem conhecer a implementação
concreta de cada modelo.

Modules
-------
early_stopping
    Critério de parada antecipada, agnóstico ao modelo/framework.
scheduler
    Funções puras de agendamento de taxa de aprendizado.
checkpoint
    Checkpointing de modelos a partir de uma métrica monitorada.
cross_validation
    Validação cruzada estratificada com sementes fixas e intervalo de
    confiança de 95%.
resume
    Salvamento/retomada do estado completo de um treino interrompido.
callbacks
    Interface comum de callback e adaptadores para parada antecipada e
    checkpoint.
trainer
    Orquestrador genérico de treino, com logging, callbacks e MLflow
    opcional.
"""

from training.callbacks import (
    Callback,
    CallbackList,
    EarlyStoppingCallback,
    LoggingCallback,
    ModelCheckpointCallback,
)
from training.checkpoint import ModelCheckpoint
from training.cross_validation import (
    CrossValidationResult,
    compute_classification_score,
    run_stratified_cross_validation,
)
from training.early_stopping import EarlyStopping
from training.resume import TrainingCheckpointState, resume_training_state, save_training_state
from training.scheduler import constant_with_warmup, cosine_warmup_decay, linear_warmup_decay
from training.trainer import Trainer, TrainingResult

__all__: list[str] = [
    "Callback",
    "CallbackList",
    "CrossValidationResult",
    "EarlyStopping",
    "EarlyStoppingCallback",
    "LoggingCallback",
    "ModelCheckpoint",
    "ModelCheckpointCallback",
    "Trainer",
    "TrainingCheckpointState",
    "TrainingResult",
    "compute_classification_score",
    "constant_with_warmup",
    "cosine_warmup_decay",
    "linear_warmup_decay",
    "resume_training_state",
    "run_stratified_cross_validation",
    "save_training_state",
]
