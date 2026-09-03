"""Representações e engenharia de features de tweets em português brasileiro.

Implementa a Fase 8 do plano de elaboração (``PLANO-ELABORACAO.md``) e a
Seção 4.4 do documento mestre (``projeto-mestrado-analise-sentimentos-ptbr.md``):
representações lexicais (TF-IDF), embeddings estáticos (FastText) e
contextuais (BERTimbau/Sentence-BERT), redução de dimensionalidade via
autoencoder, seleção de features para os classificadores clássicos e
estatísticas descritivas das representações produzidas — insumo comum às
etapas de modelagem clássica e de deep learning/LLM do pipeline.

Modules
-------
lexical
    Representação TF-IDF/bag-of-words, calculada em ``polars``/``numpy``
    sem dependência de ``scikit-learn``.
static_embeddings
    Embeddings estáticos por documento (média dos vetores de palavra
    FastText pré-treinados/treinados em pt-BR).
contextual_embeddings
    Sentence embeddings via mean pooling sobre um encoder contextual
    pré-treinado em português (BERTimbau/Sentence-BERT).
reduction
    Redução de dimensionalidade dos embeddings contextuais via autoencoder
    (PyTorch), com o erro de reconstrução como sinal diagnóstico.
selection
    Seleção de features por variância, redundância e correlação com o
    alvo, e suporte ao *ablation study* por grupo de features.
statistics
    Estatísticas descritivas (por feature e por documento) das
    representações produzidas pelos demais módulos.
"""

from features.contextual_embeddings import (
    ContextualEncoder,
    extract_contextual_embeddings,
    load_contextual_encoder,
)
from features.lexical import (
    build_document_frequencies,
    build_vocabulary,
    calculate_term_frequency,
    compute_tfidf_features,
    extract_ngrams,
    pivot_tfidf_features_to_wide,
)
from features.reduction import (
    AutoencoderArtifacts,
    compute_reconstruction_error,
    encode_with_autoencoder,
    train_autoencoder,
)
from features.selection import (
    build_feature_group_mask,
    calculate_feature_correlation_matrix,
    calculate_feature_variance,
    select_features_by_redundancy,
    select_features_by_variance_threshold,
    select_k_best_features_by_target_correlation,
)
from features.static_embeddings import (
    StaticEmbeddingModel,
    compute_document_embedding,
    extract_static_embeddings,
    load_fasttext_model,
)
from features.statistics import (
    calculate_descriptive_statistics,
    calculate_embedding_norms,
    calculate_feature_sparsity_ratio,
    summarize_feature_matrix,
)

__all__: list[str] = [
    "AutoencoderArtifacts",
    "ContextualEncoder",
    "StaticEmbeddingModel",
    "build_document_frequencies",
    "build_feature_group_mask",
    "build_vocabulary",
    "calculate_descriptive_statistics",
    "calculate_embedding_norms",
    "calculate_feature_correlation_matrix",
    "calculate_feature_sparsity_ratio",
    "calculate_feature_variance",
    "calculate_term_frequency",
    "compute_document_embedding",
    "compute_reconstruction_error",
    "compute_tfidf_features",
    "encode_with_autoencoder",
    "extract_contextual_embeddings",
    "extract_ngrams",
    "extract_static_embeddings",
    "load_contextual_encoder",
    "load_fasttext_model",
    "pivot_tfidf_features_to_wide",
    "select_features_by_redundancy",
    "select_features_by_variance_threshold",
    "select_k_best_features_by_target_correlation",
    "summarize_feature_matrix",
    "train_autoencoder",
]
