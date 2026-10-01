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
    Treino dos classificadores de deep learning (LSTM e CNN).
training_transformer
    Fine-tuning dos Transformers (BERTimbau, RoBERTa, DistilBERT).
training_llm
    Preparo (few-shot) e validação dos LLMs open-source via Ollama.
comparative_evaluation
    Avaliação comparativa (concordância, confiança e divergências) entre as
    duas bases rotuladas.
hypotheses
    Geração de hipóteses com o HypotheSAEs (discordância/incerteza entre as
    bases, padrões de baixa confiança e camada de diagnóstico).
evaluate
    Avaliação no teste (métricas com IC, McNemar) e ablação do pipeline.
report
    Figuras, tabelas, Model Cards e Datasheet a partir de ``evaluate``.
diagnostics_analysis, hypothesaes_analysis
    Implementações usadas pelos modos ``diagnostics`` e ``patterns`` do
    estágio ``hypotheses``.
workflow
    Orquestração das etapas do pipeline por nome.
"""

from pipelines.comparative_evaluation import (
    ComparativeEvaluationResult,
    run_comparative_evaluation_stage,
)
from pipelines.evaluate import EvaluateResult, run_evaluate_stage
from pipelines.features import FeatureArtifacts, run_features_stage
from pipelines.hypotheses import run_hypotheses_stage
from pipelines.ingestion import run_ingestion_stage
from pipelines.labeling import run_labeling_stage
from pipelines.preprocessing import run_preprocessing_stage
from pipelines.report import ReportResult, run_report_stage
from pipelines.training_classical import (
    DEFAULT_CLASSICAL_MODEL_NAMES,
    run_training_classical_stage,
)
from pipelines.training_deep_learning import (
    DEFAULT_DEEP_LEARNING_MODEL_NAMES,
    run_training_deep_learning_stage,
)
from pipelines.training_llm import DEFAULT_LLM_MODEL_NAMES, run_training_llm_stage
from pipelines.training_transformer import (
    DEFAULT_TRANSFORMER_MODEL_NAMES,
    run_training_transformer_stage,
)
from pipelines.workflow import STAGE_REGISTRY, run_full_workflow, run_pipeline_stage

__all__: list[str] = [
    "DEFAULT_CLASSICAL_MODEL_NAMES",
    "DEFAULT_DEEP_LEARNING_MODEL_NAMES",
    "DEFAULT_LLM_MODEL_NAMES",
    "DEFAULT_TRANSFORMER_MODEL_NAMES",
    "STAGE_REGISTRY",
    "ComparativeEvaluationResult",
    "EvaluateResult",
    "FeatureArtifacts",
    "ReportResult",
    "run_comparative_evaluation_stage",
    "run_evaluate_stage",
    "run_features_stage",
    "run_full_workflow",
    "run_hypotheses_stage",
    "run_ingestion_stage",
    "run_labeling_stage",
    "run_pipeline_stage",
    "run_preprocessing_stage",
    "run_report_stage",
    "run_training_classical_stage",
    "run_training_deep_learning_stage",
    "run_training_llm_stage",
    "run_training_transformer_stage",
]
