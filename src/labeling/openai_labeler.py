"""Rotulagem de sentimento via API OpenAI-compatível (base ``tweets_data_openai``).

Implementa a seção ``openai`` de ``configs/labeling.yaml``: cada tweet é
classificado por um LLM acessado pelo cliente único do projeto
(:func:`hypothesaes.llm_api.generate_chat_completion`, autenticado via
``OPENAI_BASE_URL``/``OPENAI_KEY`` do ``.env``), com o prompt versionado em
``prompts/``. A chamada usa Chat Completions, aceita por qualquer endpoint
compatível (a Responses API não é implementada por servidores como o da UnB e
devolvia respostas vazias, sempre "fora do formato").

Robustez frente à API:

* limitador de taxa compartilhado entre as threads
  (:class:`labeling.rate_limiter.RateLimiter`): intervalo mínimo por
  ``requests_per_minute``, pausa global no HTTP 429 (respeitando ``Retry-After``)
  e adaptação do ritmo;
* HTTP 429 tem orçamento próprio de retentativas (``max_rate_limit_retries``);
  demais erros de API usam backoff exponencial com jitter e ``max_retries``;
* falhas de configuração (chave ausente, biblioteca não instalada) não são
  retentadas: interrompem a execução, e o checkpoint preserva o progresso;
* resposta do LLM inutilizável (texto livre, JSON inválido, rótulo desconhecido)
  ou texto vazio não derruba a execução: o tweet recebe o rótulo ``indefinido``
  com score ``0.0``;
* um tweet que esgota as retentativas de API devolve ``None`` (ver
  :mod:`labeling.incremental`): não é gravado no checkpoint e é retentado na
  próxima execução, sem derrubar os demais do lote.
"""

import concurrent.futures
import logging
import threading
import time
from collections.abc import Callable, Sequence
from typing import TypeVar

from constants.labels import SENTIMENT_CLASSES, UNDEFINED_LABEL, UNDEFINED_SCORE
from exceptions.base import ProjectError
from hypothesaes.llm_api import (
    generate_chat_completion,
    generate_chat_completion_first_token_logprobs,
)
from labeling.incremental import BatchClassifier, LabelPrediction
from labeling.llm_batching import (
    build_label_only_prompt,
    build_multi_tweet_prompt,
    parse_first_token_label,
    parse_multi_tweet_response,
)
from labeling.llm_response import build_labeling_prompt, parse_llm_label_response
from labeling.rate_limiter import (
    MAX_COOLDOWN_SECONDS,
    RateLimiter,
    calculate_backoff_seconds,
    get_retry_after_seconds,
    is_rate_limit_error,
)

logger = logging.getLogger(__name__)

# A resposta esperada é um JSON curto (rótulo + probabilidades ou score).
DEFAULT_MAX_TOKENS = 100
# Modos de resposta: JSON completo, ou só o rótulo com confiança via logprobs do 1º token.
RESPONSE_MODE_JSON = "json"
RESPONSE_MODE_LABEL_LOGPROBS = "label_logprobs"
RESPONSE_MODES = (RESPONSE_MODE_JSON, RESPONSE_MODE_LABEL_LOGPROBS)
_LABEL_ONLY_MAX_TOKENS = 4  # o rótulo em português ocupa até ~3 tokens
_TOKENS_PER_MULTI_ITEM = 24  # `{"id": 12, "label": "positivo", "confidence": 0.93},`

_T = TypeVar("_T")


def _handle_rate_limit_error(
    exception: Exception,
    rate_limit_attempt: int,
    max_rate_limit_retries: int,
    rate_limiter: RateLimiter,
) -> bool:
    """Trata um HTTP 429: pausa global no limitador ou desiste.

    Parameters
    ----------
    exception : Exception
        Erro 429 levantado pela API.
    rate_limit_attempt : int
        Número da retentativa atual após 429 (base 1).
    max_rate_limit_retries : int
        Orçamento de retentativas após 429.
    rate_limiter : RateLimiter
        Limitador compartilhado que recebe a pausa global.

    Returns
    -------
    bool
        ``True`` se deve retentar; ``False`` se o orçamento esgotou.
    """
    if rate_limit_attempt > max_rate_limit_retries:
        logger.error(
            "Limite de requisições (429) persistiu após %d retentativas: %s",
            max_rate_limit_retries,
            exception,
        )
        return False
    retry_after = get_retry_after_seconds(exception)
    wait_seconds = (
        retry_after if retry_after is not None else calculate_backoff_seconds(rate_limit_attempt)
    )
    logger.warning(
        "Limite de requisições (429) atingido (retentativa %d/%d). "
        "Pausando todas as requisições por %.1fs.",
        rate_limit_attempt,
        max_rate_limit_retries,
        min(wait_seconds, MAX_COOLDOWN_SECONDS),
    )
    rate_limiter.report_rate_limited(wait_seconds)
    return True


def _calculate_api_error_wait(exception: Exception, attempt: int, max_retries: int) -> float | None:
    """Calcula a espera após erro de API (exceto 429) ou sinaliza desistência.

    Parameters
    ----------
    exception : Exception
        Erro levantado pela API.
    attempt : int
        Número da tentativa que falhou (base 1).
    max_retries : int
        Máximo de tentativas por tweet.

    Returns
    -------
    float | None
        Segundos de backoff antes da próxima tentativa; ``None`` se esgotou.
    """
    if attempt >= max_retries:
        logger.error(
            "Não foi possível classificar o tweet após %d tentativas: %s",
            max_retries,
            exception,
        )
        return None
    wait_seconds = calculate_backoff_seconds(attempt)
    logger.warning(
        "Falha na chamada à API (tentativa %d/%d): %s. Aguardando %.1fs.",
        attempt,
        max_retries,
        exception,
        wait_seconds,
    )
    return wait_seconds


def _request_with_retries(
    send: Callable[[], _T],
    *,
    max_retries: int,
    max_rate_limit_retries: int,
    rate_limiter: RateLimiter,
) -> _T | None:
    """Envia uma requisição com limitador de taxa e retentativas.

    Erros 429 têm orçamento próprio de retentativas: a espera respeita o
    ``Retry-After`` do servidor e vale para todas as threads (via
    ``rate_limiter``). Demais erros usam backoff exponencial com jitter e
    ``max_retries``.

    Parameters
    ----------
    send : Callable[[], _T]
        Executa uma requisição e devolve seu resultado (nunca ``None``).
    max_retries : int
        Máximo de tentativas em falhas de API (exceto 429).
    max_rate_limit_retries : int
        Máximo de retentativas após respostas 429.
    rate_limiter : RateLimiter
        Limitador de taxa compartilhado entre as threads.

    Returns
    -------
    _T | None
        Resultado de ``send``; ``None`` se a API falhar em todas as tentativas.

    Raises
    ------
    ProjectError
        Falhas de configuração (ex.: ``OPENAI_KEY`` ausente) são propagadas sem retentativa.
    """
    attempt = 0
    rate_limit_attempt = 0

    while True:
        rate_limiter.acquire()
        try:
            result = send()
        except ProjectError:
            raise
        except Exception as exception:
            if is_rate_limit_error(exception):
                rate_limit_attempt += 1
                if not _handle_rate_limit_error(
                    exception, rate_limit_attempt, max_rate_limit_retries, rate_limiter
                ):
                    return None
                continue

            attempt += 1
            wait_seconds = _calculate_api_error_wait(exception, attempt, max_retries)
            if wait_seconds is None:
                return None
            time.sleep(wait_seconds)
            continue

        rate_limiter.report_success()
        return result


def _classify_single_text(
    text: str,
    prompt_template: str,
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    max_rate_limit_retries: int,
    rate_limiter: RateLimiter,
    request_timeout_seconds: float,
    allowed_labels: Sequence[str],
    response_mode: str = RESPONSE_MODE_JSON,
) -> LabelPrediction | None:
    """Classifica um tweet, com retentativas em caso de erro da API.

    Parameters
    ----------
    text : str
        Texto sanitizado do tweet.
    prompt_template : str
        Template do prompt (marcador ``{{TEXTO}}``).
    model : str
        Modelo da API OpenAI-compatível.
    temperature : float
        Temperatura de amostragem (0.0 = determinístico); ignorada no modo
        ``label_logprobs`` (sempre 0.0).
    max_tokens : int
        Máximo de tokens gerados na resposta; ignorado no modo ``label_logprobs``.
    max_retries : int
        Máximo de tentativas por tweet em falhas de API (exceto 429).
    max_rate_limit_retries : int
        Máximo de retentativas após respostas 429.
    rate_limiter : RateLimiter
        Limitador de taxa compartilhado entre as threads.
    request_timeout_seconds : float
        Timeout de cada requisição, em segundos.
    allowed_labels : Sequence[str]
        Classes de sentimento aceitas.
    response_mode : str, optional
        ``json`` (resposta em JSON, padrão) ou ``label_logprobs`` (só o rótulo, com a
        confiança vinda dos logprobs do primeiro token).

    Returns
    -------
    tuple[str, float] | None
        ``(rótulo, confiança)``; ``("indefinido", 0.0)`` se o texto for vazio ou a
        resposta do LLM for inutilizável; ``None`` se a API falhar em todas as tentativas.

    Raises
    ------
    ProjectError
        Falhas de configuração (ex.: ``OPENAI_KEY`` ausente) são propagadas sem retentativa.
    """
    if not text.strip():
        return UNDEFINED_LABEL, UNDEFINED_SCORE

    send: Callable[[], str | tuple[str, dict[str, float]]]
    if response_mode == RESPONSE_MODE_LABEL_LOGPROBS:
        label_only_messages = [
            {
                "role": "user",
                "content": build_label_only_prompt(
                    prompt_template, text, allowed_labels=allowed_labels
                ),
            }
        ]

        send = lambda: generate_chat_completion_first_token_logprobs(  # noqa: E731
            label_only_messages,
            model=model,
            max_tokens=_LABEL_ONLY_MAX_TOKENS,
            timeout=request_timeout_seconds,
        )

    else:
        json_messages = [{"role": "user", "content": build_labeling_prompt(prompt_template, text)}]
        send = lambda: generate_chat_completion(  # noqa: E731
            json_messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=request_timeout_seconds,
        )

    response = _request_with_retries(
        send,
        max_retries=max_retries,
        max_rate_limit_retries=max_rate_limit_retries,
        rate_limiter=rate_limiter,
    )
    if response is None:
        return None

    if isinstance(response, tuple):
        parsed = parse_first_token_label(*response, allowed_labels=allowed_labels)
    else:
        parsed = parse_llm_label_response(response, allowed_labels=allowed_labels)
    if parsed is None:
        logger.warning(
            "Resposta fora do formato; classificando como '%s' com score %.1f.",
            UNDEFINED_LABEL,
            UNDEFINED_SCORE,
        )
        return UNDEFINED_LABEL, UNDEFINED_SCORE
    return parsed


def _classify_tweet_group(
    texts: Sequence[str],
    prompt_template: str,
    *,
    model: str,
    temperature: float,
    max_retries: int,
    max_rate_limit_retries: int,
    rate_limiter: RateLimiter,
    request_timeout_seconds: float,
    allowed_labels: Sequence[str],
    fallback: Callable[[str], LabelPrediction | None],
) -> list[LabelPrediction | None]:
    """Classifica vários tweets em uma única requisição (modo multi-tweet).

    Tweets ausentes, duplicados ou com rótulo inválido na resposta são reclassificados
    um a um por ``fallback``. Se a requisição falhar em todas as tentativas, todos os
    tweets do grupo voltam ``None`` (retentados na próxima execução).

    Parameters
    ----------
    texts : Sequence[str]
        Textos sanitizados, não vazios.
    prompt_template : str
        Template do prompt (marcador ``{{TEXTO}}``).
    model : str
        Modelo da API OpenAI-compatível.
    temperature : float
        Temperatura de amostragem.
    max_retries : int
        Máximo de tentativas em falhas de API (exceto 429).
    max_rate_limit_retries : int
        Máximo de retentativas após respostas 429.
    rate_limiter : RateLimiter
        Limitador de taxa compartilhado entre as threads.
    request_timeout_seconds : float
        Timeout da requisição, em segundos.
    allowed_labels : Sequence[str]
        Classes de sentimento aceitas.
    fallback : Callable[[str], tuple[str, float] | None]
        Classificador de um tweet, usado nos itens que a resposta não trouxe.

    Returns
    -------
    list[tuple[str, float] | None]
        Uma posição por texto, na mesma ordem.

    Raises
    ------
    ProjectError
        Falhas de configuração (ex.: ``OPENAI_KEY`` ausente) são propagadas sem retentativa.
    """
    messages = [
        {
            "role": "user",
            "content": build_multi_tweet_prompt(
                prompt_template, texts, allowed_labels=allowed_labels
            ),
        }
    ]

    def send() -> str:
        return generate_chat_completion(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=_TOKENS_PER_MULTI_ITEM * len(texts) + 16,
            timeout=request_timeout_seconds,
        )

    raw_response = _request_with_retries(
        send,
        max_retries=max_retries,
        max_rate_limit_retries=max_rate_limit_retries,
        rate_limiter=rate_limiter,
    )
    if raw_response is None:
        return [None] * len(texts)

    results = parse_multi_tweet_response(raw_response, len(texts), allowed_labels=allowed_labels)
    n_missing = sum(result is None for result in results)
    if n_missing:
        logger.warning(
            "Resposta multi-tweet sem %d de %d item(ns); reclassificando individualmente.",
            n_missing,
            len(texts),
        )
    return [
        result if result is not None else fallback(text)
        for text, result in zip(texts, results, strict=True)
    ]


def create_openai_batch_classifier(
    prompt_template: str,
    *,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_retries: int = 3,
    max_rate_limit_retries: int = 8,
    n_workers: int = 4,
    requests_per_minute: float | None = 120.0,
    request_timeout_seconds: float = 60.0,
    allowed_labels: Sequence[str] = SENTIMENT_CLASSES,
    response_mode: str = RESPONSE_MODE_JSON,
    tweets_per_request: int = 1,
) -> BatchClassifier:
    """Cria o classificador de lotes da base ``tweets_data_openai``.

    Textos repetidos (dentro do lote e entre lotes) são enviados à API uma única vez:
    o resultado fica em cache na vida do classificador.

    Parameters
    ----------
    prompt_template : str
        Template do prompt (marcador ``{{TEXTO}}``).
    model : str
        Modelo da API OpenAI-compatível (``configs/labeling.yaml -> openai.model``).
    temperature : float, optional
        Temperatura de amostragem, by default 0.0.
    max_tokens : int, optional
        Máximo de tokens gerados por classificação, by default :data:`DEFAULT_MAX_TOKENS`.
    max_retries : int, optional
        Máximo de tentativas por tweet em falhas de API (exceto 429), by default 3.
    max_rate_limit_retries : int, optional
        Máximo de retentativas após respostas 429, by default 8.
    n_workers : int, optional
        Chamadas simultâneas dentro de um lote, by default 4.
    requests_per_minute : float | None, optional
        Taxa máxima de requisições por minuto, somada entre todas as threads; se a
        API responder 429, o ritmo cai sozinho e volta ao normal após uma sequência
        de sucessos. ``None`` desativa o limite base, by default 120.0.
    request_timeout_seconds : float, optional
        Timeout por requisição, by default 60.0.
    allowed_labels : Sequence[str], optional
        Classes aceitas, by default :data:`constants.labels.SENTIMENT_CLASSES`.
    response_mode : str, optional
        ``json`` (JSON completo com probabilidades) ou ``label_logprobs`` (só o rótulo;
        confiança dos logprobs do primeiro token; exige endpoint com ``logprobs``, como
        vLLM/OpenAI), by default ``json``. Ignorado quando ``tweets_per_request > 1``.
    tweets_per_request : int, optional
        Tweets por requisição. Com valor maior que 1, cada requisição classifica um grupo
        de tweets numerados e devolve uma lista JSON; itens ausentes são reclassificados
        um a um, by default 1.

    Returns
    -------
    BatchClassifier
        Função ``textos -> [(rótulo, confiança) | None]`` para
        :func:`labeling.incremental.run_incremental_labeling`. Segura para chamadas
        concorrentes (o cache é protegido por lock).

    Raises
    ------
    ValueError
        Se ``requests_per_minute`` não for ``None`` e for menor ou igual a 0, se
        ``response_mode`` for desconhecido ou se ``tweets_per_request`` for menor que 1.

    Examples
    --------
    >>> classifier = create_openai_batch_classifier(
    ...     'Tweet: "{{TEXTO}}"', model="gpt-5-mini"
    ... )  # doctest: +SKIP
    """
    if response_mode not in RESPONSE_MODES:
        raise ValueError(
            f"'response_mode' deve ser um de {RESPONSE_MODES}, recebido {response_mode!r}."
        )
    if tweets_per_request < 1:
        raise ValueError(f"'tweets_per_request' deve ser >= 1, recebido {tweets_per_request}.")
    rate_limiter = RateLimiter(requests_per_minute)
    cache: dict[str, LabelPrediction] = {}
    cache_lock = threading.Lock()

    def classify_one(text: str) -> LabelPrediction | None:
        return _classify_single_text(
            text,
            prompt_template,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            max_rate_limit_retries=max_rate_limit_retries,
            rate_limiter=rate_limiter,
            request_timeout_seconds=request_timeout_seconds,
            allowed_labels=allowed_labels,
            response_mode=response_mode,
        )

    def classify_group(group: Sequence[str]) -> list[LabelPrediction | None]:
        if len(group) == 1:
            return [classify_one(group[0])]
        return _classify_tweet_group(
            group,
            prompt_template,
            model=model,
            temperature=temperature,
            max_retries=max_retries,
            max_rate_limit_retries=max_rate_limit_retries,
            rate_limiter=rate_limiter,
            request_timeout_seconds=request_timeout_seconds,
            allowed_labels=allowed_labels,
            fallback=classify_one,
        )

    def classify_batch(texts: Sequence[str]) -> list[LabelPrediction | None]:
        with cache_lock:
            pending = [text for text in dict.fromkeys(texts) if text not in cache]
        blank = [text for text in pending if not text.strip()]
        to_send = [text for text in pending if text.strip()]
        groups = [
            to_send[start : start + tweets_per_request]
            for start in range(0, len(to_send), tweets_per_request)
        ]

        new_results: dict[str, LabelPrediction | None] = dict.fromkeys(
            blank, (UNDEFINED_LABEL, UNDEFINED_SCORE)
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            for group, group_results in zip(
                groups, executor.map(classify_group, groups), strict=True
            ):
                new_results.update(zip(group, group_results, strict=True))

        with cache_lock:
            cache.update(
                {text: result for text, result in new_results.items() if result is not None}
            )
            return [cache.get(text) for text in texts]

    return classify_batch
