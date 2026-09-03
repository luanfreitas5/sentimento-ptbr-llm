"""Rotulagem semiautomática em cascata de tweets em português brasileiro.

Implementa a Fase 7 do plano de elaboração (``PLANO-ELABORACAO.md``) e a
Seção 4.3 do documento mestre: rotuladores automáticos/heurísticos em
cascata, pontuação de confiança/discordância por amostra, agregação por
concordância entre rotuladores, amostragem e incorporação de validação
humana, e validação contra gold sets de referência (TweetSentBR/RePro).

Modules
-------
automatic
    Interface comum dos rotuladores (:class:`SentimentLabeler`), rotulador
    heurístico baseado em léxico de sentimento e emojis, e execução da
    cascata completa sobre um corpus.
confidence
    Pontuação ponderada por rótulo, razão de concordância, discordância e
    sinalização de amostras de baixa confiança.
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
    flag_low_confidence_samples,
)
from labeling.consensus import aggregate_by_weighted_majority_vote, merge_consensus_into_corpus
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
    "NEGATIVE_WORDS",
    "POSITIVE_WORDS",
    "GoldSetValidationResult",
    "LexicalHeuristicLabeler",
    "SentimentLabeler",
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
    "flag_low_confidence_samples",
    "merge_consensus_into_corpus",
    "run_cascade_labeling",
    "select_samples_for_human_validation",
]
