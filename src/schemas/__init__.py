"""Contratos de dados (schemas ``pandera.polars``) do projeto.

Validam DataFrames nas fronteiras entre etapas do pipeline (``raw ->
interim -> processed`` e entrada/saída de modelo), falhando cedo com uma
:class:`exceptions.data.DataValidationError` em vez de propagar dados
corrompidos silenciosamente (ver CLAUDE.md, "Data Contracts").

Modules
-------
dataset
    Schemas do corpus de tweets, bruto (:class:`RawTweetSchema`) e rotulado
    (:class:`LabeledCorpusSchema`).
experiment
    Schema de métricas de execuções de experimento (:class:`ExperimentRunMetricSchema`).
labeling
    Schema dos resultados de rotulagem em cascata (:class:`LabelingResultSchema`).
prediction
    Schema das predições de sentimento (:class:`PredictionSchema`).
training
    Schema dos exemplos de treino/validação/teste (:class:`TrainingExampleSchema`).
"""

from schemas.dataset import (
    LabeledCorpusSchema,
    RawTweetSchema,
    validate_labeled_corpus,
    validate_raw_tweet_dataset,
)
from schemas.experiment import ExperimentRunMetricSchema, validate_experiment_run_metric
from schemas.labeling import LabelingResultSchema, validate_labeling_result
from schemas.prediction import PredictionSchema, validate_prediction
from schemas.training import DATA_SPLITS, TrainingExampleSchema, validate_training_example

__all__: list[str] = [
    "DATA_SPLITS",
    "ExperimentRunMetricSchema",
    "LabeledCorpusSchema",
    "LabelingResultSchema",
    "PredictionSchema",
    "RawTweetSchema",
    "TrainingExampleSchema",
    "validate_experiment_run_metric",
    "validate_labeled_corpus",
    "validate_labeling_result",
    "validate_prediction",
    "validate_raw_tweet_dataset",
    "validate_training_example",
]
