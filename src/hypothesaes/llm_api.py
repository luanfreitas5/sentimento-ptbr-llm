"""Cliente OpenAI-compatível para geração de completions (Responses API).

Camada fina sobre a Responses API da OpenAI (ou qualquer servidor
compatível, ex.: vLLM local), usada por ``interpret_neurons`` (interpretação
de neurônios) e ``annotate`` (checagem de conceitos em texto). Isola o
projeto do SDK ``openai`` (dependência pesada e opcional) e centraliza
autenticação, retomada com backoff exponencial e normalização de parâmetros
específicos de modelos ``gpt-5.x`` (``reasoning_effort``, ``verbosity``).

``openai`` não está listado nas dependências base do projeto: o import
ocorre de forma tardia, dentro das funções deste módulo, para que o restante
de ``hypothesaes`` permaneça importável sem ela. Instale com
``uv add openai tiktoken`` antes de usar geração de hipóteses via LLM.
"""

import logging
import time
from typing import Any
from urllib.parse import urlparse

from exceptions.configuration import MissingEnvironmentVariableError
from exceptions.model import ModelError

logger = logging.getLogger(__name__)

# Suprime logs de nível INFO do SDK da OpenAI/httpx, mantendo a saída limpa.
for _logger_name in ("openai", "openai._client", "openai._base_client", "httpx", "httpcore"):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)
    logging.getLogger(_logger_name).propagate = False

_CLIENT_CACHE: dict[tuple[str, str], Any] = {}

# Os IDs de modelo abaixo apontam para as versões mais recentes; versões
# específicas só são fixadas quando necessário.
MODEL_ABBREVIATION_TO_ID: dict[str, str] = {
    "gpt4o": "gpt-4o",
    "gpt-4o": "gpt-4o",
    "gpt4o-mini": "gpt-4o-mini",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt4.1": "gpt-4.1",
    "gpt-4.1": "gpt-4.1",
    "gpt4.1-mini": "gpt-4.1-mini",
    "gpt-4.1-mini": "gpt-4.1-mini",
    "gpt4.1-nano": "gpt-4.1-nano",
    "gpt-4.1-nano": "gpt-4.1-nano",
    "gpt5.2": "gpt-5.2",
    "gpt-5.2": "gpt-5.2",
    "gpt5-mini": "gpt-5-mini",
    "gpt-5-mini": "gpt-5-mini",
    "gpt5-nano": "gpt-5-nano",
    "gpt-5-nano": "gpt-5-nano",
    "gpt5": "gpt-5",
    "gpt-5": "gpt-5",
}

DEFAULT_MODEL = "gpt-5-mini"
LOCAL_OPENAI_API_KEY_PLACEHOLDER = "local-no-auth"


def _import_openai() -> Any:
    """Importa o SDK ``openai``, levantando ``ModelError`` se ausente.

    Returns
    -------
    Any
        Módulo ``openai`` importado.

    Raises
    ------
    ModelError
        Se a biblioteca ``openai`` não estiver instalada.
    """
    try:
        import openai  # type: ignore[reportMissingImports]
    except ImportError as exception:
        raise ModelError(
            "A biblioteca 'openai' não está instalada. Instale com `uv add openai` "
            "para usar geração de completions via LLM no HypotheSAEs."
        ) from exception
    return openai


def _extract_field(item: Any, key: str) -> Any:
    """Extrai um campo de um item de resposta, aceitando ``dict`` ou objeto com atributos."""
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _resolve_output_items(response: Any) -> list[Any]:
    """Resolve a lista de itens de output de uma resposta (objeto ou dict)."""
    output = getattr(response, "output", None)
    if output:
        return output
    if isinstance(response, dict):
        return response.get("output") or []
    return []


def _extract_text_from_message_content(item: Any) -> str | None:
    """Extrai o primeiro texto de saída dentro do conteúdo de um item do tipo 'message'."""
    for content in _extract_field(item, "content") or []:
        if _extract_field(content, "type") != "output_text":
            continue
        text = _extract_field(content, "text")
        if text:
            return text
    return None


def _extract_text_from_item(item: Any) -> str | None:
    """Extrai o texto de um item de output do tipo 'output_text' ou 'message'."""
    item_type = _extract_field(item, "type")
    if item_type == "output_text":
        return _extract_field(item, "text") or None
    if item_type == "message":
        return _extract_text_from_message_content(item)
    return None


def _extract_output_text(response: Any) -> str:
    """Extrai o texto gerado pelo assistente de uma resposta da Responses API.

    Parameters
    ----------
    response : Any
        Objeto de resposta retornado por ``client.responses.create(...)``.

    Returns
    -------
    str
        Texto gerado, ou string vazia se nenhum texto for encontrado.
    """
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    for item in _resolve_output_items(response):
        text = _extract_text_from_item(item)
        if text:
            return text

    return ""


def _apply_default(resolved: dict[str, Any], key: str, value: Any, *absent_keys: str) -> None:
    """Define ``resolved[key] = value`` se ``value`` não for ``None``
    e nenhuma de ``absent_keys`` já estiver definida.
    """
    if value is None:
        return
    if any(absent_key in resolved for absent_key in absent_keys):
        return
    resolved[key] = value


def normalize_llm_kwargs(
    llm_kwargs: dict[str, Any] | None = None,
    *,
    default_verbosity: str | None = None,
    default_reasoning_effort: str | None = None,
    default_timeout: float | None = None,
    default_max_output_tokens: int | None = None,
) -> dict[str, Any]:
    """Aplica valores padrão a ``llm_kwargs`` sem sobrescrever configurações explícitas.

    Parameters
    ----------
    llm_kwargs : dict[str, Any] | None, optional
        Argumentos já informados pelo usuário, by default None.
    default_verbosity : str | None, optional
        Valor padrão de verbosidade (``text.verbosity``), by default None.
    default_reasoning_effort : str | None, optional
        Valor padrão de esforço de raciocínio (``reasoning.effort``), by
        default None.
    default_timeout : float | None, optional
        Timeout padrão da requisição, em segundos, by default None.
    default_max_output_tokens : int | None, optional
        Limite padrão de tokens de saída, by default None.

    Returns
    -------
    dict[str, Any]
        Cópia de ``llm_kwargs`` com os padrões aplicados apenas onde ausentes.

    Examples
    --------
    >>> normalize_llm_kwargs({"temperature": 0.5}, default_timeout=30.0)
    {'temperature': 0.5, 'timeout': 30.0}
    """
    resolved = dict(llm_kwargs or {})
    _apply_default(resolved, "verbosity", default_verbosity, "verbosity", "text")
    _apply_default(
        resolved, "reasoning_effort", default_reasoning_effort, "reasoning", "reasoning_effort"
    )
    _apply_default(resolved, "timeout", default_timeout, "timeout")
    _apply_default(resolved, "max_output_tokens", default_max_output_tokens, "max_output_tokens")
    return resolved


def _uses_openai_hosted_auth(base_url: str | None) -> bool:
    """Indica se a requisição é destinada à API oficial (hospedada) da OpenAI."""
    if not base_url:
        return True
    hostname = (urlparse(base_url).hostname or "").lower()
    return hostname == "openai.com" or hostname.endswith(".openai.com")


def _resolve_api_key(base_url: str | None) -> str:
    """Resolve a chave de API a partir de ``OPENAI_KEY_SAE``.

    Parameters
    ----------
    base_url : str | None
        URL base configurada (endpoint local/compatível), se houver.

    Returns
    -------
    str
        Chave de API válida, ou o placeholder local quando ``base_url``
        aponta para um servidor não-OpenAI sem autenticação.

    Raises
    ------
    MissingEnvironmentVariableError
        Se a requisição for destinada à API oficial da OpenAI e
        ``OPENAI_KEY_SAE`` não estiver definida.
    """
    import os

    api_key = os.environ.get("OPENAI_KEY_SAE")
    if api_key and "..." not in api_key:
        return api_key
    if _uses_openai_hosted_auth(base_url):
        raise MissingEnvironmentVariableError("OPENAI_KEY_SAE")
    return LOCAL_OPENAI_API_KEY_PLACEHOLDER


def create_client() -> Any:
    """Cria (ou reaproveita do cache) um cliente OpenAI-compatível.

    Lê ``OPENAI_KEY_SAE`` (obrigatória para requisições hospedadas pela
    OpenAI) e ``OPENAI_BASE_URL`` (opcional, para apontar a um servidor
    local/compatível, ex.: vLLM). Clientes são cacheados por
    ``(api_key, base_url)`` para evitar reconexões desnecessárias.

    Returns
    -------
    Any
        Instância de ``openai.OpenAI``.

    Raises
    ------
    ModelError
        Se a biblioteca ``openai`` não estiver instalada.
    MissingEnvironmentVariableError
        Se ``OPENAI_KEY_SAE`` for necessária e não estiver definida.
    """
    import os

    openai = _import_openai()

    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = _resolve_api_key(base_url)
    cache_key = (api_key, base_url or "__openai_default__")
    if cache_key in _CLIENT_CACHE:
        return _CLIENT_CACHE[cache_key]

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    _CLIENT_CACHE[cache_key] = openai.OpenAI(**client_kwargs)
    return _CLIENT_CACHE[cache_key]


def _pop_max_output_tokens(kwargs: dict[str, Any]) -> Any:
    """Extrai o limite de tokens de saída de ``kwargs``, aceitando aliases legados."""
    max_output_tokens = kwargs.pop("max_output_tokens", None)
    if max_output_tokens is None:
        max_output_tokens = kwargs.pop("max_completion_tokens", None)
    if max_output_tokens is None:
        max_output_tokens = kwargs.pop("max_tokens", None)
    return max_output_tokens


def _apply_reasoning_and_verbosity(kwargs: dict[str, Any]) -> None:
    """Normaliza ``verbosity``/``reasoning_effort`` para os payloads aninhados da Responses API."""
    verbosity = kwargs.pop("verbosity", None)
    if verbosity is not None:
        text_payload = dict(kwargs.pop("text", {}) or {})
        text_payload["verbosity"] = verbosity
        kwargs["text"] = text_payload

    reasoning_effort = kwargs.pop("reasoning_effort", None)
    if reasoning_effort is not None:
        reasoning_payload = dict(kwargs.pop("reasoning", {}) or {})
        reasoning_payload["effort"] = reasoning_effort
        kwargs["reasoning"] = reasoning_payload


def _messages_contain_system_role(messages: list[dict[str, Any]]) -> bool:
    """Indica se ``messages`` já contém uma mensagem de sistema."""
    return any(message.get("role") == "system" for message in messages)


def _build_request_input(
    prompt: str | None,
    messages: list[dict[str, Any]] | None,
    system_prompt: str | None,
) -> Any:
    """Monta o ``input`` da Responses API a partir de prompt/mensagens/mensagem de sistema.

    Raises
    ------
    ValueError
        Se ``system_prompt`` for informado e ``messages`` já contiver uma
        mensagem de sistema.
    """
    if messages is not None:
        if system_prompt is None:
            return messages
        if _messages_contain_system_role(messages):
            raise ValueError(
                "system_prompt foi informado, mas 'messages' já contém uma mensagem de sistema"
            )
        return [{"role": "system", "content": system_prompt}, *messages]

    if system_prompt is not None:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    return prompt


def _build_attempt_kwargs(
    kwargs: dict[str, Any], max_output_tokens: Any, timeout: float | None
) -> dict[str, Any]:
    """Monta os kwargs de uma tentativa de requisição, incluindo
    ``max_output_tokens``/``timeout``.
    """
    request_kwargs = kwargs.copy()
    if max_output_tokens is not None:
        request_kwargs["max_output_tokens"] = max_output_tokens
    if timeout is not None:
        request_kwargs["timeout"] = timeout
    return request_kwargs


def _wait_before_retry(
    base_wait: float, backoff_factor: float, attempt: int, max_retries: int, exception: Exception
) -> None:
    """Aguarda com backoff exponencial antes de uma nova tentativa, registrando um aviso."""
    wait_time = base_wait * (backoff_factor**attempt)
    if attempt > 0:
        logger.warning(
            "Erro na API (%s); nova tentativa em %.1fs... (%d/%d)",
            exception,
            wait_time,
            attempt + 1,
            max_retries,
        )
    time.sleep(wait_time)


def generate_completion(
    prompt: str | None = None,
    *,
    messages: list[dict[str, Any]] | None = None,
    system_prompt: str | None = None,
    model: str = DEFAULT_MODEL,
    timeout: float | None = None,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    **kwargs: Any,
) -> str:
    """Gera uma completion via Responses API, com retomada e backoff exponencial.

    Parameters
    ----------
    prompt : str | None, optional
        Prompt de texto puro (ignorado se ``messages`` for informado), by
        default None.
    messages : list[dict[str, Any]] | None, optional
        Lista opcional de mensagens de chat, by default None.
    system_prompt : str | None, optional
        Mensagem de sistema opcional, by default None.
    model : str, optional
        Modelo (ou abreviação, ver :data:`MODEL_ABBREVIATION_TO_ID`), by
        default :data:`DEFAULT_MODEL`.
    timeout : float | None, optional
        Timeout da requisição, em segundos, by default None.
    max_retries : int, optional
        Número máximo de tentativas em caso de rate limit/timeout, by
        default 3.
    backoff_factor : float, optional
        Fator multiplicador do tempo de espera entre tentativas, by
        default 2.0.
    **kwargs : Any
        Argumentos adicionais repassados à Responses API (``max_output_tokens``,
        ``reasoning_effort``, ``verbosity``, ``temperature`` etc.).

    Returns
    -------
    str
        Texto gerado pelo modelo.

    Raises
    ------
    ValueError
        Se nem ``prompt`` nem ``messages`` forem informados.
    ModelError
        Se a biblioteca ``openai`` não estiver instalada.

    Examples
    --------
    >>> generate_completion(prompt="Olá!")  # doctest: +SKIP
    """
    if prompt is None is messages is None:
        raise ValueError("É necessário informar 'prompt' ou 'messages' para generate_completion()")

    openai = _import_openai()
    client = create_client()
    model_id = MODEL_ABBREVIATION_TO_ID.get(model, model)

    max_output_tokens = _pop_max_output_tokens(kwargs)
    _apply_reasoning_and_verbosity(kwargs)
    request_input = _build_request_input(prompt, messages, system_prompt)

    base_wait = timeout if timeout is not None else 1.0
    for attempt in range(max_retries):
        try:
            request_kwargs = _build_attempt_kwargs(kwargs, max_output_tokens, timeout)
            response = client.responses.create(
                model=model_id, input=request_input, **request_kwargs
            )
            return _extract_output_text(response)

        except (openai.RateLimitError, openai.APITimeoutError) as exception:
            if attempt == max_retries - 1:
                raise
            _wait_before_retry(base_wait, backoff_factor, attempt, max_retries, exception)

    return ""
