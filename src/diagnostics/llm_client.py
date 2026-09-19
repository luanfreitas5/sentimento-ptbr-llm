"""Cliente LLM único, OpenAI-compatível (Ollama ou OpenAI), assíncrono.

Usa a API de *Chat Completions* (suportada por Ollama e OpenAI) com:

* limite de concorrência por semáforo (``llm.max_concurrency``);
* cache de respostas em disco (uma entrada JSON por chave SHA-256, fora do git);
* retentativas com espera exponencial em erros transitórios;
* modo ``dry_run``: não faz rede, apenas registra o nº de chamadas e tokens
  estimados que **seriam** feitas (respeitando o cache), para estimar custo.

Chaves de API só vêm de variáveis de ambiente (``OPENAI_KEY``/``OPENAI_API_KEY``),
nunca do YAML. Loops ``asyncio`` diferentes não compartilham clientes: crie um
:class:`AsyncLLMClient` dentro de cada ``asyncio.run``.
"""

import asyncio
import hashlib
import json
import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from diagnostics.settings import LLMSettings
from exceptions.configuration import MissingEnvironmentVariableError
from exceptions.pipeline import PipelineStageError

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 3.5  # heurística para pt-BR; só usada na estimativa de custo (dry-run)
_OLLAMA_API_KEY = "ollama"
_LOCAL_PLACEHOLDER_KEY = "local-no-auth"


def estimate_tokens(text: str) -> int:
    """Estima o nº de tokens de um texto (heurística de caracteres por token).

    Parameters
    ----------
    text : str
        Texto a estimar.

    Returns
    -------
    int
        Estimativa (mínimo 1).

    Examples
    --------
    >>> estimate_tokens("")
    1
    >>> estimate_tokens("a" * 35)
    10
    """
    return max(1, round(len(text) / CHARS_PER_TOKEN))


@dataclass
class CallStats:
    """Contadores de uma sessão do cliente.

    Attributes
    ----------
    n_requests : int
        Requisições efetivamente enviadas (ou, em dry-run, que seriam enviadas).
    n_cache_hits : int
        Respostas servidas do cache em disco.
    n_failures : int
        Requisições que falharam após todas as retentativas.
    input_tokens : int
        Tokens de entrada (estimados) das requisições não cacheadas.
    output_tokens : int
        Tokens de saída (estimados; em dry-run, o teto ``max_tokens``).
    """

    n_requests: int = 0
    n_cache_hits: int = 0
    n_failures: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class DiskCompletionCache:
    """Cache de completions em disco: um arquivo JSON por chave SHA-256.

    Parameters
    ----------
    directory : Path
        Pasta do cache (criada sob demanda). Deve ficar fora do git.

    Examples
    --------
    >>> cache = DiskCompletionCache(Path("cache"))  # doctest: +SKIP
    """

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    @staticmethod
    def build_key(
        *,
        model: str,
        prompt: str,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int | None,
        namespace: str,
    ) -> str:
        """Gera a chave do cache a partir do conteúdo COMPLETO da requisição.

        Parameters
        ----------
        model : str
            Modelo usado.
        prompt : str
            Prompt do usuário (texto completo, sem truncar).
        system_prompt : str | None
            Prompt de sistema, se houver.
        temperature : float
            Temperatura da geração.
        max_tokens : int | None
            Limite de tokens de saída.
        namespace : str
            Rótulo livre (ex.: versão do prompt) para separar entradas.

        Returns
        -------
        str
            Hash SHA-256 hexadecimal.
        """
        payload = json.dumps(
            [model, prompt, system_prompt, temperature, max_tokens, namespace],
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self._directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> str | None:
        """Lê uma resposta do cache; ``None`` se ausente ou ilegível."""
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            return str(json.loads(path.read_text(encoding="utf-8"))["completion"])
        except (json.JSONDecodeError, KeyError, OSError):
            logger.warning("Entrada de cache ilegível ignorada: %s", path.name)
            return None

    def put(self, key: str, completion: str) -> None:
        """Grava uma resposta no cache (escrita atômica por arquivo temporário)."""
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"completion": completion}, ensure_ascii=False), "utf-8")
        temporary.replace(path)


def _resolve_endpoint(settings: LLMSettings) -> tuple[str | None, str]:
    """Resolve ``(base_url, api_key)`` do provedor; a chave vem só do ambiente."""
    if settings.provider == "ollama":
        return f"{settings.ollama_base_url.rstrip('/')}/v1", _OLLAMA_API_KEY
    base_url = settings.openai_base_url or os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_KEY") or os.environ.get("OPENAI_API_KEY")
    if api_key:
        return base_url, api_key
    if base_url:  # endpoint compatível local/privado costuma dispensar chave
        return base_url, _LOCAL_PLACEHOLDER_KEY
    raise MissingEnvironmentVariableError("OPENAI_KEY")


def _default_client_factory(settings: LLMSettings) -> Any:
    """Cria o ``openai.AsyncOpenAI`` (import tardio: ``openai`` é dependência opcional)."""
    import openai

    base_url, api_key = _resolve_endpoint(settings)
    return openai.AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=settings.request_timeout_seconds,
        max_retries=0,  # as retentativas são feitas aqui, com log
    )


def _transient_exceptions() -> tuple[type[Exception], ...]:
    """Erros da API considerados transitórios (import tardio)."""
    import openai

    return (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.InternalServerError,
    )


class AsyncLLMClient:
    """Cliente assíncrono com cache, semáforo, retentativas e modo dry-run.

    Parameters
    ----------
    settings : LLMSettings
        Configuração do provedor (``configs/diagnostics.yaml -> llm``).
    cache : DiskCompletionCache | None, optional
        Cache em disco; ``None`` desativa, by default None.
    dry_run : bool, optional
        Se verdadeiro, não faz rede: contabiliza as chamadas não cacheadas em
        :attr:`stats` e devolve string vazia, by default False.
    client_factory : Callable[[LLMSettings], Any] | None, optional
        Fábrica do cliente ``AsyncOpenAI`` (injetável em testes), by default None.

    Examples
    --------
    >>> client = AsyncLLMClient(settings, dry_run=True)  # doctest: +SKIP
    """

    def __init__(
        self,
        settings: LLMSettings,
        *,
        cache: DiskCompletionCache | None = None,
        dry_run: bool = False,
        client_factory: Callable[[LLMSettings], Any] | None = None,
    ) -> None:
        self._settings = settings
        self._cache = cache
        self._dry_run = dry_run
        self._client_factory = client_factory or _default_client_factory
        self._client: Any = None
        self._semaphore: asyncio.Semaphore | None = None
        self.stats = CallStats()

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = self._client_factory(self._settings)
        return self._client

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._settings.max_concurrency)
        return self._semaphore

    async def complete(
        self,
        prompt: str,
        *,
        model: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        namespace: str = "",
    ) -> str:
        """Obtém uma completion (do cache ou do provedor).

        Parameters
        ----------
        prompt : str
            Prompt do usuário.
        model : str
            Nome do modelo no provedor.
        system_prompt : str | None, optional
            Prompt de sistema, by default None.
        max_tokens : int | None, optional
            Limite de tokens de saída, by default None.
        temperature : float | None, optional
            Sobrescreve ``settings.temperature``, by default None.
        namespace : str, optional
            Separador de cache (ex.: versão do prompt), by default "".

        Returns
        -------
        str
            Texto gerado (vazio em dry-run).

        Raises
        ------
        PipelineStageError
            Se a requisição falhar após todas as retentativas.
        """
        used_temperature = self._settings.temperature if temperature is None else temperature
        key = DiskCompletionCache.build_key(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=used_temperature,
            max_tokens=max_tokens,
            namespace=namespace,
        )
        cached = self._cache.get(key) if self._cache else None
        if cached is not None:
            self.stats.n_cache_hits += 1
            return cached

        self.stats.n_requests += 1
        self.stats.input_tokens += estimate_tokens(prompt) + estimate_tokens(system_prompt or "")
        if self._dry_run:
            self.stats.output_tokens += max_tokens or 100
            return ""

        completion = await self._request_with_retries(
            prompt, model, system_prompt, max_tokens, used_temperature
        )
        self.stats.output_tokens += estimate_tokens(completion)
        if self._cache:
            self._cache.put(key, completion)
        return completion

    async def _request_with_retries(
        self,
        prompt: str,
        model: str,
        system_prompt: str | None,
        max_tokens: int | None,
        temperature: float,
    ) -> str:
        """Envia a requisição com semáforo e espera exponencial em erros transitórios."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        transient = _transient_exceptions()
        last_error: Exception | None = None
        for attempt in range(self._settings.max_retries + 1):
            try:
                async with self._get_semaphore():
                    response = await self._get_client().chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                return str(response.choices[0].message.content or "")
            except transient as error:
                last_error = error
                wait_seconds = 2.0**attempt
                logger.warning(
                    "Erro transitório do LLM (%s); nova tentativa em %.0fs (%d/%d).",
                    type(error).__name__,
                    wait_seconds,
                    attempt + 1,
                    self._settings.max_retries + 1,
                )
                await asyncio.sleep(wait_seconds)
        self.stats.n_failures += 1
        raise PipelineStageError("llm_client", f"falha após retentativas: {last_error}")

    async def complete_many(self, prompts: Sequence[str], **kwargs: Any) -> list[str | None]:
        """Executa vários prompts concorrentemente, na ordem de entrada.

        Parameters
        ----------
        prompts : Sequence[str]
            Prompts a enviar.
        **kwargs : Any
            Argumentos repassados a :meth:`complete` (``model`` obrigatório).

        Returns
        -------
        list[str | None]
            Completions na mesma ordem; ``None`` onde a requisição falhou.
        """

        async def _safe(prompt: str) -> str | None:
            try:
                return await self.complete(prompt, **kwargs)
            except PipelineStageError:
                return None

        return list(await asyncio.gather(*(_safe(prompt) for prompt in prompts)))
