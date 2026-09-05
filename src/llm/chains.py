"""Encadeamento LangChain (LCEL) para classificação de sentimento com justificativa.

Implementa a Fase 11 (``configs/llm.yaml -> orchestration.framework:
"langchain"``): compõe prompt -> geração -> parsing estruturado como uma
cadeia LangChain (``Runnable``, via o operador ``|`` do LCEL -
LangChain Expression Language), mantendo a etapa de parsing sobre a
implementação robusta e testável de ``src/llm/parsers.py`` (extração de
JSON por regex + validação Pydantic) em vez do
``PydanticOutputParser.parse`` nativo do LangChain, cujo comportamento
exato varia entre versões e modelos. O resultado da cadeia já inclui a
justificativa textual gerada pelo LLM
(:attr:`llm.parsers.SentimentLLMOutput.justificativa`).

``langchain-core`` é uma dependência pesada e opcional (extra ``llm`` de
``pyproject.toml``): o import ocorre de forma tardia, em
:func:`build_sentiment_classification_chain`, para que o restante do
projeto permaneça importável sem ela.
"""

import logging
from collections.abc import Sequence
from typing import Any

from constants.labels import NEUTRAL_LABEL, SENTIMENT_CLASSES
from exceptions.model import ModelError
from llm.backends import LLMBackend
from llm.parsers import SentimentLLMOutput, parse_structured_llm_output
from llm.prompts import DEFAULT_PROMPT_TEMPLATE_VERSION, PromptStrategy, build_sentiment_prompt

logger = logging.getLogger(__name__)


def build_sentiment_classification_chain(
    backend: LLMBackend,
    *,
    strategy: PromptStrategy = "few_shot",
    few_shot_examples: Sequence[tuple[str, str]] = (),
    allowed_labels: Sequence[str] = SENTIMENT_CLASSES,
    version: str = DEFAULT_PROMPT_TEMPLATE_VERSION,
) -> Any:
    """Monta uma cadeia LangChain (LCEL) prompt -> geração -> parsing estruturado.

    Parameters
    ----------
    backend : LLMBackend
        Backend de geração de texto (``src/llm/backends.py`` ou um dublê de
        teste que implemente ``LLMBackend``).
    strategy : {"zero_shot", "few_shot", "chain_of_thought"}, optional
        Estratégia de prompt, repassada a
        :func:`llm.prompts.build_sentiment_prompt`, by default "few_shot".
    few_shot_examples : Sequence[tuple[str, str]], optional
        Pares ``(texto, rótulo)`` de exemplo, by default ().
    allowed_labels : Sequence[str], optional
        Classes de sentimento aceitas, by default
        :data:`constants.labels.SENTIMENT_CLASSES`.
    version : str, optional
        Versão do template de prompt, by default
        :data:`llm.prompts.DEFAULT_PROMPT_TEMPLATE_VERSION`.

    Returns
    -------
    Runnable
        Cadeia LangChain (``RunnableSequence``) que recebe um texto e
        retorna um :class:`llm.parsers.SentimentLLMOutput` (ou ``None``,
        quando a resposta do LLM não é interpretável — ver
        :func:`run_chain_with_retry` para uma versão com retentativas e
        fallback).

    Raises
    ------
    ModelError
        Se a biblioteca ``langchain-core`` não estiver instalada.

    Examples
    --------
    >>> class _Backend:
    ...     def generate(self, prompt: str) -> str:
    ...         return '{"sentimento": "positivo", "confianca": 0.9}'
    >>> chain = build_sentiment_classification_chain(_Backend())  # doctest: +SKIP
    >>> chain.invoke("ótimo produto").sentimento  # doctest: +SKIP
    'positivo'
    """
    try:
        from langchain_core.runnables import RunnableLambda
    except ImportError as exception:
        raise ModelError(
            "A biblioteca 'langchain-core' não está instalada. Instale com "
            "`uv add langchain-core` (ou `uv sync --extra llm`) para montar cadeias LangChain."
        ) from exception

    prompt_step = RunnableLambda(
        lambda text: build_sentiment_prompt(
            text,
            strategy=strategy,
            few_shot_examples=few_shot_examples,
            allowed_labels=allowed_labels,
            version=version,
        )
    )
    generation_step = RunnableLambda(backend.generate)
    parsing_step = RunnableLambda(
        lambda raw_output: parse_structured_llm_output(raw_output, allowed_labels=allowed_labels)
    )
    return prompt_step | generation_step | parsing_step


def run_chain_with_retry(
    chain: Any,
    text: str,
    *,
    max_retries: int = 3,
    fallback_label: str = NEUTRAL_LABEL,
) -> SentimentLLMOutput:
    """Invoca uma cadeia de classificação, retentando enquanto a saída for não interpretável.

    Parameters
    ----------
    chain : Runnable
        Cadeia construída por :func:`build_sentiment_classification_chain`.
    text : str
        Texto a ser classificado.
    max_retries : int, optional
        Número de invocações da cadeia toleradas antes de desistir, by
        default 3 (``configs/llm.yaml -> parsing.max_retries``).
    fallback_label : str, optional
        Rótulo usado quando todas as tentativas retornam ``None``, by
        default :data:`constants.labels.NEUTRAL_LABEL`.

    Returns
    -------
    SentimentLLMOutput
        Resultado interpretado, ou um fallback com ``sentimento=fallback_label``
        e ``confianca=0.0`` se todas as tentativas falharem.

    Raises
    ------
    ValueError
        Se ``max_retries`` for menor que 1.

    Examples
    --------
    >>> class _Backend:
    ...     def generate(self, prompt: str) -> str:
    ...         return "resposta sem json"
    >>> chain = build_sentiment_classification_chain(_Backend())  # doctest: +SKIP
    >>> run_chain_with_retry(chain, "texto", max_retries=1).sentimento  # doctest: +SKIP
    'neutro'
    """
    if max_retries < 1:
        raise ValueError(f"max_retries deve ser >= 1, recebido: {max_retries}")

    for attempt in range(1, max_retries + 1):
        result = chain.invoke(text)
        if result is not None:
            return result
        logger.warning(
            "Tentativa %d/%d sem resposta interpretável na cadeia; retentando.",
            attempt,
            max_retries,
        )

    logger.warning(
        "Todas as %d tentativa(s) falharam; usando rótulo de fallback '%s'.",
        max_retries,
        fallback_label,
    )
    return SentimentLLMOutput(sentimento=fallback_label, confianca=0.0, justificativa="")
