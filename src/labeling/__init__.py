"""Rotulagem de sentimento de tweets em português brasileiro.

Implementa a Fase 7 do plano de elaboração (``PLANO-ELABORACAO.md``) e a
Seção 4.3 do documento mestre: classificação do corpus via pipeline do
Hugging Face, sinalização de baixa confiança, amostragem e incorporação de
validação humana, validação contra gold sets de referência
(TweetSentBR/RePro) e re-rotulagem via LLM das amostras remanescentes de
baixa confiança.

Modules
-------
huggingface
    Pipeline de classificação de sentimento via ``transformers.pipeline``
    (fonte padrão de ``sentiment_label``/``confidence_score``, ver
    ``configs/labeling.yaml -> huggingface``).
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
    votação majoritária ponderada e mesclagem ao corpus original — a
    mesclagem também é reutilizada pelo pipeline Hugging Face.
manual
    Amostragem estratificada por confiança para validação humana,
    incorporação dos rótulos revisados e estimativa da taxa de erro da
    rotulagem automática.
validation
    Validação dos rótulos contra gold sets de referência via Kappa de
    Cohen e Alpha de Krippendorff.
llm_relabeling
    Re-rotulagem via LLM (``UnB-Llama-3.3-70B-Instruct``, ver
    ``configs/labeling.yaml -> llm_relabeling``) das amostras com
    ``confidence_score`` abaixo de um limiar configurável.
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
    DEFAULT_LABEL_MAPPING,
    SentimentPipeline,
    label_corpus_with_huggingface_pipeline,
    load_huggingface_sentiment_pipeline,
)
from labeling.llm_relabeling import (
    DEFAULT_RELABEL_MODEL,
    parse_relabel_response,
    relabel_low_confidence_samples,
)
from labeling.manual import (
    apply_human_validation_labels,
    calculate_labeling_error_rate,
    select_samples_for_human_validation,
)
from labeling.validation import (
    GoldSetValidationResult,
    calculate_cohen_kappa,
    calculate_krippendorff_alpha,
    evaluate_against_gold_set,
)

__all__: list[str] = [
    "DEFAULT_HUGGINGFACE_MODEL",
    "DEFAULT_LABEL_MAPPING",
    "DEFAULT_RELABEL_MODEL",
    "NEGATIVE_WORDS",
    "POSITIVE_WORDS",
    "GoldSetValidationResult",
    "LexicalHeuristicLabeler",
    "SentimentLabeler",
    "SentimentPipeline",
    "aggregate_by_weighted_majority_vote",
    "apply_human_validation_labels",
    "calculate_agreement_ratio",
    "calculate_cohen_kappa",
    "calculate_discordance_score",
    "calculate_krippendorff_alpha",
    "calculate_labeling_error_rate",
    "calculate_lexicon_sentiment_counts",
    "calculate_weighted_label_scores",
    "classify_by_lexical_heuristic",
    "evaluate_against_gold_set",
    "flag_low_confidence_predictions",
    "flag_low_confidence_samples",
    "label_corpus_with_huggingface_pipeline",
    "load_huggingface_sentiment_pipeline",
    "merge_consensus_into_corpus",
    "parse_relabel_response",
    "relabel_low_confidence_samples",
    "run_cascade_labeling",
    "select_samples_for_human_validation",
]
