"""Camada de diagnóstico HypotheSAEs pós-rotulagem (opt-in).

Explica e ajuda a melhorar a rotulagem por LLMs de tweets pt-BR não
rotulados. O HypotheSAEs não classifica: treina um SAE sobre embeddings e
gera hipóteses em linguagem natural associadas a um alvo derivado. Hipóteses
sobre pseudo-rótulo descrevem o comportamento do modelo, nunca a verdade;
somente o gold set mede acerto.

Modules
-------
targets
    Alvos de diagnóstico (discordância, incerteza, pseudo-rótulo, erro vs
    gold): :func:`build_target`, :func:`adapt_labeled_corpus`.
sanity
    Gate de sanidade com Ridge nos embeddings:
    :func:`evaluate_sanity_gate`, :func:`assert_sanity_gate_passed`.
"""

from diagnostics.sanity import SanityGateResult, assert_sanity_gate_passed, evaluate_sanity_gate
from diagnostics.targets import (
    TARGET_NAMES,
    TargetName,
    adapt_labeled_corpus,
    build_disagreement_target,
    build_gold_error_target,
    build_pseudo_label_target,
    build_target,
    build_uncertainty_target,
)

__all__: list[str] = [
    "TARGET_NAMES",
    "SanityGateResult",
    "TargetName",
    "adapt_labeled_corpus",
    "assert_sanity_gate_passed",
    "build_disagreement_target",
    "build_gold_error_target",
    "build_pseudo_label_target",
    "build_target",
    "build_uncertainty_target",
    "evaluate_sanity_gate",
]
