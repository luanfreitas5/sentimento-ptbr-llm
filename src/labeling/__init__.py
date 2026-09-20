"""Rotulagem de sentimento de tweets em português brasileiro.

Implementa a Fase 7 do plano de elaboração (``PLANO-ELABORACAO.md``) e a
Seção 4.3 do documento mestre: classificação de todo o corpus por duas fontes
independentes (classificador do Hugging Face e LLM via API OpenAI), com checkpoint e retomada,
sinalização de baixa confiança, amostragem e incorporação de validação
humana e validação contra gold sets de referência (TweetSentBR/RePro).

Modules
-------
huggingface
    Rotulagem via classificador de sentimento local do Hugging Face Hub (base
    ``tweets_data_huggingface``, ver ``configs/labeling.yaml -> huggingface``).
openai_labeler
    Rotulagem via API OpenAI-compatível, com pausa contra HTTP 429, timeout e
    retentativa (base ``tweets_data_openai``, ver ``configs/labeling.yaml -> openai``).
incremental
    Execução incremental e retomável da rotulagem (comum às duas fontes).
checkpoint
    Checkpoint JSON Lines que evita reprocessar tweets já rotulados.
llm_response
    Construção do prompt e interpretação da resposta JSON dos LLMs.
automatic
    Interface comum dos rotuladores (:class:`SentimentLabeler``) e
    rotulador heurístico baseado em léxico de sentimento e emojis —
    infraestrutura de cascata legada, não usada pelo estágio ``labeling``
    por padrão (ver ``src/labeling/huggingface.py``), mantida para reuso
    futuro em combinações de múltiplos rotuladores.
confidence
    Sinalização de amostras de baixa confiança
    (:func:`flag_low_confidence_predictions`, pipeline único) e, para a
    cascata legada, pontuação ponderada por rótulo, razão de concordância e
    discordância entre rotuladores.
consensus
    Agregação dos candidatos da cascata em um rótulo de consenso por
    votação majoritária ponderada e mesclagem ao corpus original.
manual
    Amostragem estratificada por confiança para validação humana,
    incorporação dos rótulos revisados e estimativa da taxa de erro da
    rotulagem automática.
validation
    Validação dos rótulos contra gold sets de referência via Kappa de
    Cohen e Alpha de Krippendorff.
"""

from labeling.automatic import (
    NEGATIVE_WORDS,
    POSITIVE_WORDS,
    LexicalHeuristicLabeler,
    SentimentLabeler,
    calculate_lexicon_sentiment_counts,
    classify_by_lexical_heuristic,
    run_cascade_labeling,
)
from labeling.confidence import (
    calculate_agreement_ratio,
    calculate_discordance_score,
    calculate_weighted_label_scores,
    flag_low_confidence_predictions,
    flag_low_confidence_samples,
)
from labeling.consensus import aggregate_by_weighted_majority_vote, merge_consensus_into_corpus
from labeling.huggingface import (
    DEFAULT_HUGGINGFACE_MODEL,
    HuggingFaceModel,
    create_huggingface_batch_classifier,
    load_huggingface_model,
    open_huggingface_classifier,
    unload_huggingface_model,
)
from labeling.incremental import BatchClassifier, run_incremental_labeling
from labeling.llm_response import build_labeling_prompt, parse_llm_label_response
from labeling.manual import (
    apply_human_validation_labels,
    calculate_labeling_error_rate,
    select_samples_for_human_validation,
)
from labeling.openai_labeler import create_openai_batch_classifier
from labeling.validation import (
    GoldSetValidationResult,
    calculate_cohen_kappa,
    calculate_krippendorff_alpha,
    evaluate_against_gold_set,
)

__all__: list[str] = [
    "DEFAULT_HUGGINGFACE_MODEL",
    "NEGATIVE_WORDS",
    "POSITIVE_WORDS",
    "BatchClassifier",
    "GoldSetValidationResult",
    "HuggingFaceModel",
    "LexicalHeuristicLabeler",
    "SentimentLabeler",
    "aggregate_by_weighted_majority_vote",
    "apply_human_validation_labels",
    "build_labeling_prompt",
    "calculate_agreement_ratio",
    "calculate_cohen_kappa",
    "calculate_discordance_score",
    "calculate_krippendorff_alpha",
    "calculate_labeling_error_rate",
    "calculate_lexicon_sentiment_counts",
    "calculate_weighted_label_scores",
    "classify_by_lexical_heuristic",
    "create_huggingface_batch_classifier",
    "create_openai_batch_classifier",
    "evaluate_against_gold_set",
    "flag_low_confidence_predictions",
    "flag_low_confidence_samples",
    "load_huggingface_model",
    "merge_consensus_into_corpus",
    "open_huggingface_classifier",
    "parse_llm_label_response",
    "run_cascade_labeling",
    "run_incremental_labeling",
    "select_samples_for_human_validation",
    "unload_huggingface_model",
]
