"""Estimativa de nº de chamadas, tokens e custo (base do ``--dry-run``).

A estimativa é analítica (não faz rede nem carrega modelos) e depende de
constantes aproximadas do prompt de anotação do HypotheSAEs; use-a para
ordem de grandeza e valide com um ``--dry-run`` do cliente
(:class:`diagnostics.llm_client.AsyncLLMClient`) quando o cache já existir.
Preços vêm de ``configs/diagnostics.yaml -> pricing`` (Ollama local = 0).
"""

import logging
from dataclasses import dataclass

from diagnostics.llm_client import estimate_tokens
from diagnostics.settings import DiagnosticsSettings, PricingSettings

logger = logging.getLogger(__name__)

# ``prompts/annotate.txt`` tem ~7 exemplos few-shot: ~650 tokens de cabeçalho por chamada.
ANNOTATION_PROMPT_OVERHEAD_TOKENS = 650
INTERPRETATION_PROMPT_OVERHEAD_TOKENS = 400
TOKENS_PER_WORD = 1.6  # pt-BR (heurística)
ANNOTATION_OUTPUT_TOKENS = 40
DEFAULT_INTERPRETATION_OUTPUT_TOKENS = 150


@dataclass
class CostEstimate:
    """Estimativa de uma etapa.

    Attributes
    ----------
    n_calls : int
        Nº de chamadas ao LLM.
    input_tokens, output_tokens : int
        Tokens estimados.
    cost_usd : float
        Custo estimado em USD (0 para modelos locais).
    """

    n_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: "CostEstimate") -> "CostEstimate":
        """Soma duas estimativas."""
        return CostEstimate(
            self.n_calls + other.n_calls,
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cost_usd + other.cost_usd,
        )


def calculate_cost_usd(
    pricing: PricingSettings, model: str, input_tokens: int, output_tokens: int
) -> float:
    """Converte tokens em USD usando o preço do modelo (ou o padrão).

    Parameters
    ----------
    pricing : PricingSettings
        Tabela de preços (USD por 1M de tokens).
    model : str
        Nome do modelo.
    input_tokens, output_tokens : int
        Tokens de entrada e saída.

    Returns
    -------
    float
        Custo em USD.

    Examples
    --------
    >>> calculate_cost_usd(PricingSettings(), "qualquer", 1_000_000, 0)
    0.0
    """
    price = pricing.models.get(model, pricing.default)
    return (input_tokens * price.input + output_tokens * price.output) / 1_000_000


def estimate_annotation_cost(
    settings: DiagnosticsSettings, *, n_tweets: int, n_concepts: int, avg_words: int | None = None
) -> CostEstimate:
    """Estima N tweets × C conceitos anotados pelo modelo anotador.

    Parameters
    ----------
    settings : DiagnosticsSettings
        Configuração validada.
    n_tweets, n_concepts : int
        Tamanho do produto cartesiano (sempre subamostre).
    avg_words : int | None, optional
        Palavras médias por tweet; padrão ``hypotheses.max_words_per_example``.

    Returns
    -------
    CostEstimate
        Chamadas, tokens e custo.

    Examples
    --------
    >>> estimate_annotation_cost(settings, n_tweets=10, n_concepts=2).n_calls  # doctest: +SKIP
    20
    """
    words = avg_words or settings.hypotheses.max_words_per_example
    n_calls = n_tweets * n_concepts
    input_tokens = n_calls * (ANNOTATION_PROMPT_OVERHEAD_TOKENS + round(words * TOKENS_PER_WORD))
    output_tokens = n_calls * ANNOTATION_OUTPUT_TOKENS
    model = settings.llm.annotator_model
    return CostEstimate(
        n_calls,
        input_tokens,
        output_tokens,
        calculate_cost_usd(settings.pricing, model, input_tokens, output_tokens),
    )


def estimate_hypothesis_generation_cost(
    settings: DiagnosticsSettings, *, n_discovery_tweets: int
) -> CostEstimate:
    """Estima ``generate_hypotheses`` para UM alvo (interpretação + pontuação).

    Parameters
    ----------
    settings : DiagnosticsSettings
        Configuração validada.
    n_discovery_tweets : int
        Tweets na partição de descoberta (limita ``n_scoring_examples``).

    Returns
    -------
    CostEstimate
        Chamadas, tokens e custo (intérprete + anotador).

    Examples
    --------
    >>> estimate_hypothesis_generation_cost(settings, n_discovery_tweets=3000)  # doctest: +SKIP
    """
    hyp = settings.hypotheses
    n_interpretations = hyp.n_selected_neurons * hyp.n_candidate_interpretations
    example_tokens = round(hyp.max_words_per_example * TOKENS_PER_WORD)
    interpretation_input = n_interpretations * (
        INTERPRETATION_PROMPT_OVERHEAD_TOKENS + hyp.n_examples_for_interpretation * example_tokens
    )
    interpretation_output = n_interpretations * (
        hyp.max_interpretation_tokens or DEFAULT_INTERPRETATION_OUTPUT_TOKENS
    )
    interpretation = CostEstimate(
        n_interpretations,
        interpretation_input,
        interpretation_output,
        calculate_cost_usd(
            settings.pricing,
            settings.llm.interpreter_model,
            interpretation_input,
            interpretation_output,
        ),
    )
    n_scoring = min(hyp.n_scoring_examples, n_discovery_tweets)
    scoring = estimate_annotation_cost(settings, n_tweets=n_scoring, n_concepts=n_interpretations)
    return interpretation + scoring


def estimate_classification_cost(
    settings: DiagnosticsSettings, *, n_tweets: int, n_prompts: int, prompt_tokens: int
) -> CostEstimate:
    """Estima a reclassificação de tweets por uma ou mais versões de prompt.

    Parameters
    ----------
    settings : DiagnosticsSettings
        Configuração validada.
    n_tweets : int
        Tweets a classificar por versão.
    n_prompts : int
        Nº de (modelo × versão de prompt) a executar.
    prompt_tokens : int
        Tokens do template do prompt de rotulagem.

    Returns
    -------
    CostEstimate
        Chamadas, tokens e custo.

    Examples
    --------
    >>> estimate_classification_cost(
    ...     settings, n_tweets=1, n_prompts=1, prompt_tokens=10
    ... )  # doctest: +SKIP
    """
    n_calls = n_tweets * n_prompts
    tweet_tokens = round(settings.hypotheses.max_words_per_example * TOKENS_PER_WORD)
    input_tokens = n_calls * (prompt_tokens + tweet_tokens)
    output_tokens = n_calls * 60
    return CostEstimate(
        n_calls,
        input_tokens,
        output_tokens,
        calculate_cost_usd(
            settings.pricing, settings.llm.annotator_model, input_tokens, output_tokens
        ),
    )


def estimate_prompt_tokens(prompt_text: str) -> int:
    """Estima os tokens de um template de prompt (reexporta a heurística do cliente)."""
    return estimate_tokens(prompt_text)


def format_cost_report(estimates: dict[str, CostEstimate], *, provider: str) -> str:
    """Formata um relatório textual (uma linha por etapa + total).

    Parameters
    ----------
    estimates : dict[str, CostEstimate]
        Estimativa por etapa.
    provider : str
        Provedor ativo (``ollama`` = custo 0 em dinheiro, mas há custo de tempo).

    Returns
    -------
    str
        Relatório multilinha em pt-BR.

    Examples
    --------
    >>> print(format_cost_report({"x": CostEstimate(2, 10, 4, 0.0)}, provider="ollama"))
    ... # doctest: +SKIP
    """
    lines = [f"Estimativa de custo (provedor: {provider}) — valores aproximados:"]
    total = CostEstimate()
    for name, estimate in estimates.items():
        total = total + estimate
        lines.append(
            f"  {name:<28} {estimate.n_calls:>9,} chamadas | "
            f"{estimate.input_tokens:>12,} tok entrada | "
            f"{estimate.output_tokens:>10,} tok saída | US$ {estimate.cost_usd:>8.2f}"
        )
    lines.append(
        f"  {'TOTAL':<28} {total.n_calls:>9,} chamadas | {total.input_tokens:>12,} tok entrada | "
        f"{total.output_tokens:>10,} tok saída | US$ {total.cost_usd:>8.2f}"
    )
    if provider == "ollama":
        lines.append("  (Ollama local: sem custo em dinheiro; o tempo depende do hardware.)")
    return "\n".join(lines)
