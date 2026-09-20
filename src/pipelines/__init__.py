"""Pipelines de orquestração ponta a ponta do projeto de análise de sentimentos pt-BR.

Cada módulo implementa um estágio de ``configs/config.yaml -> stages``,
compondo os demais pacotes de ``src/`` (dados, pré-processamento, rotulagem,
features, modelos, treino, inferência, avaliação e experimentos) em uma
única função de entrada por etapa. ``src/pipelines/workflow.py`` registra
todos os estágios e orquestra a execução individual ou completa do
pipeline.

Modules
-------
ingestion
    Coleta de tweets e datasets externos, com catalogação de rastreabilidade.
preprocessing
    Normalização e limpeza do corpus bruto de tweets.
labeling
    Rotulagem de todos os tweets por dois LLMs independentes (Hugging Face e
    OpenAI), gerando as bases ``tweets_data_huggingface``/``tweets_data_openai``.
features
    Split estratificado e extração de features do corpus rotulado.
training_classical
    Treino dos classificadores clássicos de sentimento.
training_deep_learning
    Treino dos classificadores de deep learning de sentimento.
comparative_evaluation
    Avaliação comparativa (concordância, confiança, divergências e
    hipóteses do HypotheSAEs) entre as duas bases rotuladas.
diagnostics_analysis
    Estágio opt-in ``diagnostics``: camada de diagnóstico HypotheSAEs
    (hipóteses, validação no holdout e comparação de prompts v1 vs v2).
workflow
    Orquestração das etapas do pipeline por nome.
"""

from pipelines.comparative_evaluation import (
    ComparativeEvaluationResult,
    run_comparative_evaluation_stage,
)
from pipelines.diagnostics_analysis import run_diagnostics_stage
from pipelines.features import FeatureArtifacts, run_features_stage
from pipelines.ingestion import run_ingestion_stage
from pipelines.labeling import run_labeling_stage
from pipelines.preprocessing import run_preprocessing_stage
from pipelines.training_classical import (
    DEFAULT_CLASSICAL_MODEL_NAMES,
    run_training_classical_stage,
)
from pipelines.training_deep_learning import (
    DEFAULT_DEEP_LEARNING_MODEL_NAMES,
    run_training_deep_learning_stage,
)
from pipelines.workflow import STAGE_REGISTRY, run_full_workflow, run_pipeline_stage

__all__: list[str] = [
    "DEFAULT_CLASSICAL_MODEL_NAMES",
    "DEFAULT_DEEP_LEARNING_MODEL_NAMES",
    "STAGE_REGISTRY",
    "ComparativeEvaluationResult",
    "FeatureArtifacts",
    "run_comparative_evaluation_stage",
    "run_diagnostics_stage",
    "run_features_stage",
    "run_full_workflow",
    "run_ingestion_stage",
    "run_labeling_stage",
    "run_pipeline_stage",
    "run_preprocessing_stage",
    "run_training_classical_stage",
    "run_training_deep_learning_stage",
]
