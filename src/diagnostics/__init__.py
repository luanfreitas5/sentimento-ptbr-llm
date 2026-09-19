"""Camada de diagnóstico HypotheSAEs pós-rotulagem (opt-in).

Explica e ajuda a melhorar a rotulagem por LLMs de tweets pt-BR não
rotulados. O HypotheSAEs não classifica: treina um SAE sobre embeddings e
gera hipóteses em linguagem natural associadas a um alvo derivado. Hipóteses
sobre pseudo-rótulo descrevem o comportamento do modelo, nunca a verdade;
somente o gold set mede acerto.

Os módulos abaixo, exceto ``targets`` e ``sanity``, não são importados aqui
de propósito: dependem de ``torch``/``openai`` (extra ``hypothesaes``) e devem
ser importados sob demanda (ex.: ``python -m diagnostics.hypotheses``).

Modules
-------
settings
    Configuração validada de ``configs/diagnostics.yaml``:
    :func:`settings.load_diagnostics_settings`.
targets
    Alvos de diagnóstico (discordância, incerteza, pseudo-rótulo, erro vs
    gold): :func:`build_target`, :func:`adapt_labeled_corpus`.
sanity
    Gate de sanidade com Ridge nos embeddings:
    :func:`evaluate_sanity_gate`, :func:`assert_sanity_gate_passed`.
llm_client
    Cliente LLM único, assíncrono, OpenAI-compatível, com cache em disco,
    semáforo e modo dry-run: :class:`llm_client.AsyncLLMClient`.
cost
    Estimativa de chamadas, tokens e custo (``--dry-run``).
sae_runner
    Partições disjuntas, embeddings e SAE compartilhado entre alvos:
    :func:`sae_runner.prepare_discovery_data`.
hypotheses
    Hipóteses por alvo + CLI (``python -m diagnostics.hypotheses``).
annotation
    Anotação assíncrona N tweets x C conceitos (com trava de orçamento).
validation
    Validação no holdout com Bonferroni e amostra para rotulagem humana.
sampling
    Amostra estratificada por conceito, sem identificadores.
gold_eval
    MCC e macro-F1 por conceito e modelo, com IC bootstrap.
prompt_synthesis
    Geração de ``prompts/v2.md`` a partir das hipóteses validadas.
comparison
    Comparação v1 vs v2 (McNemar, Wilcoxon, IC do ganho, discordância).
tracking
    Registro dos experimentos no MLflow.
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
