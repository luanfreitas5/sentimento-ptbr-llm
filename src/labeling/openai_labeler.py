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
import time
from collections.abc import Sequence

from constants.labels import SENTIMENT_CLASSES, UNDEFINED_LABEL, UNDEFINED_SCORE
from exceptions.base import ProjectError
from hypothesaes.llm_api import generate_chat_completion
from labeling.incremental import BatchClassifier, LabelPrediction
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
) -> LabelPrediction | None:
    """Classifica um tweet, com retentativas em caso de erro da API.

    Erros 429 têm orçamento próprio de retentativas: a espera respeita o
    ``Retry-After`` do servidor e vale para todas as threads (via
    ``rate_limiter``). Demais erros usam backoff exponencial com jitter e
    ``max_retries``.

    Parameters
    ----------
    text : str
        Texto sanitizado do tweet.
    prompt_template : str
        Template do prompt (marcador ``{{TEXTO}}``).
    model : str
        Modelo da API OpenAI-compatível.
    temperature : float
        Temperatura de amostragem (0.0 = determinístico).
    max_tokens : int
        Máximo de tokens gerados na resposta.
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

    messages = [{"role": "user", "content": build_labeling_prompt(prompt_template, text)}]
    attempt = 0
    rate_limit_attempt = 0

    while True:
        rate_limiter.acquire()
        try:
            raw_response = generate_chat_completion(
                messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=request_timeout_seconds,
            )
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
        parsed = parse_llm_label_response(raw_response, allowed_labels=allowed_labels)
        if parsed is None:
            logger.warning(
                "Resposta fora do formato; classificando como '%s' com score %.1f.",
                UNDEFINED_LABEL,
                UNDEFINED_SCORE,
            )
            return UNDEFINED_LABEL, UNDEFINED_SCORE
        return parsed


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
) -> BatchClassifier:
    """Cria o classificador de lotes da base ``tweets_data_openai``.

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

    Returns
    -------
    BatchClassifier
        Função ``textos -> [(rótulo, confiança) | None]`` para
        :func:`labeling.incremental.run_incremental_labeling`.

    Raises
    ------
    ValueError
        Se ``requests_per_minute`` não for ``None`` e for menor ou igual a 0.

    Examples
    --------
    >>> classifier = create_openai_batch_classifier(
    ...     'Tweet: "{{TEXTO}}"', model="gpt-5-mini"
    ... )  # doctest: +SKIP
    """
    rate_limiter = RateLimiter(requests_per_minute)

    def classify_batch(texts: Sequence[str]) -> list[LabelPrediction | None]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    _classify_single_text,
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
                )
                for text in texts
            ]
            return [future.result() for future in futures]

    return classify_batch
