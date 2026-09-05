"""Output parsers estruturados (Pydantic) para respostas de LLM, com retry.

Implementa a Fase 11 (``configs/llm.yaml -> parsing``): valida a resposta
bruta de um backend LLM (``src/llm/backends.py``) contra
:class:`SentimentLLMOutput` — o mesmo contrato JSON instruído em
``src/llm/prompts.py`` (``sentimento``/``confianca``/``justificativa``) — e
retenta a geração quando a resposta não é interpretável
(``parsing.retry_on_invalid_json``), até ``parsing.max_retries`` tentativas,
retornando um rótulo de fallback (``parsing.fallback_label``) caso todas
falhem.

Depende apenas de ``pydantic`` (dependência obrigatória do projeto): a
validação estruturada funciona mesmo sem ``langchain``/``ollama``
instalados. ``src/llm/chains.py`` oferece, como alternativa, um parser
equivalente construído com ``langchain_core.output_parsers.PydanticOutputParser``
para uso dentro de uma cadeia LangChain (LCEL).
"""

import json
import logging
import re
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, Field, ValidationError, field_validator

from constants.labels import NEUTRAL_LABEL, SENTIMENT_CLASSES

logger = logging.getLogger(__name__)

_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


class SentimentLLMOutput(BaseModel):
    """Contrato de saída estruturada de um LLM para classificação de sentimento.

    Parameters
    ----------
    sentimento : str
        Rótulo de sentimento predito, normalizado (minúsculas, sem espaços
        nas bordas).
    confianca : float
        Confiança relatada pelo LLM, restrita a ``[0.0, 1.0]``.
    justificativa : str
        Justificativa textual, opcional.
    """

    sentimento: str
    confianca: float = Field(ge=0.0, le=1.0, default=0.0)
    justificativa: str = ""

    @field_validator("sentimento")
    @classmethod
    def normalize_sentiment_label(cls, value: str) -> str:
        """Normaliza o rótulo de sentimento para minúsculas, sem espaços nas bordas."""
        return value.strip().lower()


class LLMOutputParser(Protocol):
    """Interface mínima de um parser de saída de LLM.

    Permite que :func:`generate_and_parse_with_retry` opere tanto sobre
    :func:`parse_structured_llm_output` quanto sobre um parser LangChain
    equivalente (``src/llm/chains.py``), sem acoplamento à implementação
    concreta.
    """

    def __call__(
        self, raw_output: str, *, allowed_labels: Sequence[str]
    ) -> SentimentLLMOutput | None:
        """Interpreta a resposta bruta do LLM, retornando ``None`` se inválida."""
        ...


class LLMBackendProtocol(Protocol):
    """Interface mínima de um backend de geração de texto por LLM (ver ``src/llm/backends.py``)."""

    def generate(self, prompt: str) -> str:
        """Gera a resposta do modelo para um prompt."""
        ...


def extract_json_object(raw_output: str) -> str | None:
    """Extrai o primeiro objeto JSON encontrado em um texto bruto.

    Parameters
    ----------
    raw_output : str
        Texto bruto gerado por um LLM, possivelmente contendo texto
        adicional (ex.: raciocínio de cadeia de pensamento) ao redor do
        objeto JSON.

    Returns
    -------
    str | None
        O trecho JSON encontrado, ou ``None`` se nenhum objeto ``{...}``
        estiver presente.

    Examples
    --------
    >>> extract_json_object('Raciocínio: ok\\nResposta: {"sentimento": "positivo"}')
    '{"sentimento": "positivo"}'
    >>> extract_json_object("sem json aqui") is None
    True
    """
    match = _JSON_OBJECT_PATTERN.search(raw_output)
    return match.group(0) if match is not None else None


def parse_structured_llm_output(
    raw_output: str, *, allowed_labels: Sequence[str] = SENTIMENT_CLASSES
) -> SentimentLLMOutput | None:
    """Interpreta a resposta bruta de um LLM como :class:`SentimentLLMOutput`.

    Parameters
    ----------
    raw_output : str
        Texto bruto gerado pelo LLM, esperado no formato instruído por
        ``src/llm/prompts.py``.
    allowed_labels : Sequence[str], optional
        Classes de sentimento aceitas, by default
        :data:`constants.labels.SENTIMENT_CLASSES`.

    Returns
    -------
    SentimentLLMOutput | None
        A saída estruturada, quando válida, ou ``None`` quando a resposta
        não contém um objeto JSON reconhecível, viola o contrato de
        :class:`SentimentLLMOutput` ou o rótulo extraído não pertence a
        ``allowed_labels``.

    Examples
    --------
    >>> resultado = parse_structured_llm_output('{"sentimento": "positivo", "confianca": 0.9}')
    >>> resultado.sentimento, resultado.confianca
    ('positivo', 0.9)
    >>> parse_structured_llm_output("resposta sem json") is None
    True
    """
    json_text = extract_json_object(raw_output)
    if json_text is None:
        logger.warning("Resposta do LLM sem objeto JSON reconhecível: %r", raw_output)
        return None

    try:
        payload = json.loads(json_text)
        parsed_output = SentimentLLMOutput.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, TypeError) as exception:
        logger.warning("Falha ao validar resposta estruturada do LLM: %s", exception)
        return None

    if parsed_output.sentimento not in allowed_labels:
        logger.warning(
            "Rótulo '%s' fora das classes conhecidas %s.", parsed_output.sentimento, allowed_labels
        )
        return None
    return parsed_output


def generate_and_parse_with_retry(
    backend: LLMBackendProtocol,
    prompt: str,
    *,
    allowed_labels: Sequence[str] = SENTIMENT_CLASSES,
    max_retries: int = 3,
    fallback_label: str = NEUTRAL_LABEL,
    parser: LLMOutputParser = parse_structured_llm_output,
) -> SentimentLLMOutput:
    """Gera e interpreta a resposta de um backend LLM, retentando em resposta inválida.

    Parameters
    ----------
    backend : LLMBackendProtocol
        Backend de geração de texto (``src/llm/backends.py``).
    prompt : str
        Prompt de entrada, via ``src/llm/prompts.py``.
    allowed_labels : Sequence[str], optional
        Classes de sentimento aceitas, by default
        :data:`constants.labels.SENTIMENT_CLASSES`.
    max_retries : int, optional
        Número de tentativas de geração ao receber uma resposta não
        interpretável, by default 3 (``configs/llm.yaml ->
        parsing.max_retries``).
    fallback_label : str, optional
        Rótulo usado quando todas as tentativas falham, by default
        :data:`constants.labels.NEUTRAL_LABEL` (``configs/llm.yaml ->
        parsing.fallback_label``).
    parser : LLMOutputParser, optional
        Função de interpretação da resposta bruta, by default
        :func:`parse_structured_llm_output`.

    Returns
    -------
    SentimentLLMOutput
        A saída estruturada interpretada, ou um fallback com
        ``sentimento=fallback_label`` e ``confianca=0.0`` se todas as
        tentativas falharem.

    Raises
    ------
    ValueError
        Se ``max_retries`` for menor que 1.

    Examples
    --------
    >>> class _Backend:
    ...     def generate(self, prompt: str) -> str:
    ...         return '{"sentimento": "positivo", "confianca": 0.8}'
    >>> generate_and_parse_with_retry(_Backend(), "texto qualquer").sentimento
    'positivo'
    """
    if max_retries < 1:
        raise ValueError(f"max_retries deve ser >= 1, recebido: {max_retries}")

    for attempt in range(1, max_retries + 1):
        raw_output = backend.generate(prompt)
        parsed_output = parser(raw_output, allowed_labels=allowed_labels)
        if parsed_output is not None:
            return parsed_output
        logger.warning(
            "Tentativa %d/%d sem resposta interpretável; retentando.", attempt, max_retries
        )

    logger.warning(
        "Todas as %d tentativa(s) falharam; usando rótulo de fallback '%s'.",
        max_retries,
        fallback_label,
    )
    return SentimentLLMOutput(sentimento=fallback_label, confianca=0.0, justificativa="")
