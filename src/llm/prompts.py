"""Templates de prompt versionados para classificação de sentimento via LLM.

Implementa a Fase 11 (``configs/llm.yaml -> prompting``): três estratégias de
engenharia de prompt — ``zero_shot``, ``few_shot`` e ``chain_of_thought`` —
sob templates versionados (``orchestration.prompt_template_version``), para
que uma futura mudança de template não quebre silenciosamente experimentos
já registrados no MLflow (o texto do prompt usado deve poder ser
rastreado até a versão do template que o gerou). Independente da
estratégia, o LLM é instruído a responder em JSON estruturado
(``sentimento``/``confianca``/``justificativa``), consumido por
``src/llm/parsers.py``.

Este módulo é puro Python (sem dependências pesadas/opcionais): pode ser
testado e usado independentemente de ``langchain``/``ollama`` estarem
instalados.
"""

from collections.abc import Callable, Sequence
from typing import Literal

from constants.labels import SENTIMENT_CLASSES

PromptStrategy = Literal["zero_shot", "few_shot", "chain_of_thought"]

DEFAULT_PROMPT_TEMPLATE_VERSION = "v1"
PROMPT_STRATEGIES: tuple[PromptStrategy, ...] = ("zero_shot", "few_shot", "chain_of_thought")


def _build_output_format_instructions(allowed_labels: Sequence[str]) -> str:
    """Monta a instrução de formato de saída (JSON estruturado), comum a todas as estratégias.

    Parameters
    ----------
    allowed_labels : Sequence[str]
        Classes de sentimento aceitas.

    Returns
    -------
    str
        Trecho de instrução, listando as classes aceitas e as chaves JSON
        exigidas (``configs/llm.yaml -> prompting.output_format``).
    """
    labels_text = ", ".join(f'"{label}"' for label in allowed_labels)
    return (
        "Classifique o sentimento do texto em português brasileiro abaixo em "
        f"uma das classes {labels_text}. Responda apenas com um objeto JSON "
        'contendo as chaves "sentimento", "confianca" (entre 0.0 e 1.0) e '
        '"justificativa".'
    )


def _build_few_shot_examples_block(few_shot_examples: Sequence[tuple[str, str]]) -> list[str]:
    """Monta os blocos de exemplo few-shot, um por par ``(texto, rótulo)``.

    Parameters
    ----------
    few_shot_examples : Sequence[tuple[str, str]]
        Pares ``(texto, rótulo)`` de exemplo (ver
        ``models.llm.select_balanced_few_shot_examples``).

    Returns
    -------
    list[str]
        Um bloco de texto ``"Texto: ...\\nResposta: {...}"`` por exemplo.
    """
    return [
        f'Texto: "{example_text}"\nResposta: '
        f'{{"sentimento": "{example_label}", "confianca": 1.0, "justificativa": "exemplo"}}'
        for example_text, example_label in few_shot_examples
    ]


def _build_zero_shot_prompt(
    text: str, *, few_shot_examples: Sequence[tuple[str, str]], allowed_labels: Sequence[str]
) -> str:
    """Monta um prompt zero-shot (sem exemplos), ignorando ``few_shot_examples``.

    Parameters
    ----------
    text : str
        Texto a ser classificado.
    few_shot_examples : Sequence[tuple[str, str]]
        Ignorado nesta estratégia; aceito apenas para manter assinatura
        comum entre as estratégias.
    allowed_labels : Sequence[str]
        Classes de sentimento aceitas.

    Returns
    -------
    str
        Prompt completo.
    """
    del few_shot_examples
    instructions = _build_output_format_instructions(allowed_labels)
    return "\n\n".join([instructions, f'Texto: "{text}"\nResposta:'])


def _build_few_shot_prompt(
    text: str, *, few_shot_examples: Sequence[tuple[str, str]], allowed_labels: Sequence[str]
) -> str:
    """Monta um prompt few-shot, precedido de exemplos balanceados por classe.

    Parameters
    ----------
    text : str
        Texto a ser classificado.
    few_shot_examples : Sequence[tuple[str, str]]
        Pares ``(texto, rótulo)`` de exemplo.
    allowed_labels : Sequence[str]
        Classes de sentimento aceitas.

    Returns
    -------
    str
        Prompt completo.
    """
    instructions = _build_output_format_instructions(allowed_labels)
    example_blocks = _build_few_shot_examples_block(few_shot_examples)
    prompt_sections = [instructions, *example_blocks, f'Texto: "{text}"\nResposta:']
    return "\n\n".join(prompt_sections)


def _build_chain_of_thought_prompt(
    text: str, *, few_shot_examples: Sequence[tuple[str, str]], allowed_labels: Sequence[str]
) -> str:
    """Monta um prompt de cadeia de raciocínio (Chain-of-Thought).

    Instrui o LLM a raciocinar brevemente antes de responder, mantendo o
    mesmo contrato de saída JSON das demais estratégias — o raciocínio é
    apenas um passo intermediário do texto gerado, nunca substitui o
    formato estruturado esperado por ``src/llm/parsers.py``
    (``configs/llm.yaml -> prompting.chain_of_thought.max_reasoning_tokens``
    limita o tamanho desse raciocínio na geração).

    Parameters
    ----------
    text : str
        Texto a ser classificado.
    few_shot_examples : Sequence[tuple[str, str]]
        Pares ``(texto, rótulo)`` de exemplo, opcionalmente incluídos antes
        da instrução de raciocínio.
    allowed_labels : Sequence[str]
        Classes de sentimento aceitas.

    Returns
    -------
    str
        Prompt completo.
    """
    instructions = _build_output_format_instructions(allowed_labels)
    reasoning_instruction = (
        "Antes de responder, raciocine brevemente sobre as evidências textuais "
        "(palavras, expressões, emojis) que indicam o sentimento, em uma linha "
        'iniciada por "Raciocínio:". Em seguida, na linha seguinte, responda '
        'apenas com o objeto JSON, iniciado por "Resposta:".'
    )
    example_blocks = _build_few_shot_examples_block(few_shot_examples)
    prompt_sections = [
        instructions,
        reasoning_instruction,
        *example_blocks,
        f'Texto: "{text}"\nRaciocínio:',
    ]
    return "\n\n".join(prompt_sections)


_PROMPT_BUILDERS_V1: dict[PromptStrategy, Callable[..., str]] = {
    "zero_shot": _build_zero_shot_prompt,
    "few_shot": _build_few_shot_prompt,
    "chain_of_thought": _build_chain_of_thought_prompt,
}

_PROMPT_TEMPLATE_VERSIONS: dict[str, dict[PromptStrategy, Callable[..., str]]] = {
    "v1": _PROMPT_BUILDERS_V1,
}


def build_sentiment_prompt(
    text: str,
    *,
    strategy: PromptStrategy = "few_shot",
    few_shot_examples: Sequence[tuple[str, str]] = (),
    allowed_labels: Sequence[str] = SENTIMENT_CLASSES,
    version: str = DEFAULT_PROMPT_TEMPLATE_VERSION,
) -> str:
    """Monta o prompt de classificação de sentimento para a estratégia e versão informadas.

    Parameters
    ----------
    text : str
        Texto a ser classificado.
    strategy : {"zero_shot", "few_shot", "chain_of_thought"}, optional
        Estratégia de engenharia de prompt (``configs/llm.yaml ->
        prompting.strategies``), by default "few_shot"
        (``prompting.default_strategy``).
    few_shot_examples : Sequence[tuple[str, str]], optional
        Pares ``(texto, rótulo)`` de exemplo, usados pelas estratégias
        ``"few_shot"`` e ``"chain_of_thought"`` e ignorados por
        ``"zero_shot"``, by default ().
    allowed_labels : Sequence[str], optional
        Classes de sentimento aceitas, by default
        :data:`constants.labels.SENTIMENT_CLASSES`.
    version : str, optional
        Versão do template a usar, by default
        :data:`DEFAULT_PROMPT_TEMPLATE_VERSION`
        (``configs/llm.yaml -> orchestration.prompt_template_version``).

    Returns
    -------
    str
        Prompt completo, pronto para um backend LLM
        (``src/llm/backends.py``).

    Raises
    ------
    ValueError
        Se ``strategy`` ou ``version`` não forem suportados.

    Examples
    --------
    >>> prompt = build_sentiment_prompt(
    ...     "ótimo produto", strategy="zero_shot", allowed_labels=("positivo", "negativo")
    ... )
    >>> prompt.endswith("Resposta:")
    True
    >>> prompt_cot = build_sentiment_prompt("ótimo produto", strategy="chain_of_thought")
    >>> prompt_cot.endswith("Raciocínio:")
    True
    """
    version_builders = _PROMPT_TEMPLATE_VERSIONS.get(version)
    if version_builders is None:
        raise ValueError(
            f"Versão de template '{version}' não suportada. "
            f"Versões disponíveis: {sorted(_PROMPT_TEMPLATE_VERSIONS)}"
        )
    builder = version_builders.get(strategy)
    if builder is None:
        raise ValueError(
            f"Estratégia de prompt '{strategy}' não suportada. "
            f"Estratégias disponíveis: {sorted(version_builders)}"
        )
    return builder(text, few_shot_examples=few_shot_examples, allowed_labels=allowed_labels)
