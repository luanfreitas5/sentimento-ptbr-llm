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
    Rotulagem semiautomática em cascata do corpus normalizado.
features
    Split estratificado e extração de features do corpus rotulado.
training_classical
    Treino dos classificadores clássicos de sentimento.
training_deep_learning
    Treino dos classificadores de deep learning de sentimento.
llm_evaluation
    Classificação e avaliação via LLM local (Ollama/Hugging Face).
comparative_evaluation
    Avaliação comparativa entre múltiplos classificadores de sentimento.
workflow
    Orquestração das etapas do pipeline por nome.
"""

from pipelines.comparative_evaluation import (
    ComparativeEvaluationResult,
    run_comparative_evaluation_stage,
)
from pipelines.features import FeatureArtifacts, run_features_stage
from pipelines.ingestion import run_ingestion_stage
from pipelines.labeling import run_labeling_stage
from pipelines.llm_evaluation import run_llm_evaluation_stage
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
    "run_features_stage",
    "run_full_workflow",
    "run_ingestion_stage",
    "run_labeling_stage",
    "run_llm_evaluation_stage",
    "run_pipeline_stage",
    "run_preprocessing_stage",
    "run_training_classical_stage",
    "run_training_deep_learning_stage",
]
