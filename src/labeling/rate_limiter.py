"""Controle de taxa de requisições compartilhado entre threads (evita HTTP 429).

Usado pela rotulagem via API (:mod:`labeling.openai_labeler`). Combina três
mecanismos:

1. **Intervalo mínimo** entre requisições (``requests_per_minute``).
2. **Pausa global** (*cooldown*): ao receber um 429, todas as threads esperam o
   tempo pedido pelo servidor (``Retry-After``).
3. **Adaptação (AIMD)**: cada 429 dobra o intervalo entre requisições e uma
   sequência de sucessos o reduz gradualmente até o valor base.
"""

import random
import threading
import time
from typing import Any

# Limites do controle de taxa (HTTP 429)
_MAX_BACKOFF_SECONDS = 60.0  # teto do backoff exponencial sem cabeçalho Retry-After
MAX_COOLDOWN_SECONDS = 300.0  # teto da pausa global, mesmo que o servidor peça mais
_MIN_THROTTLED_INTERVAL = 0.25  # intervalo inicial entre requisições ao detectar o 1º 429
_MAX_THROTTLED_INTERVAL = 30.0  # intervalo máximo entre requisições após 429 repetidos
_SUCCESSES_TO_SPEED_UP = 20  # sucessos seguidos necessários para acelerar de novo
_SPEED_UP_FACTOR = 0.8


class RateLimiter:
    """Controla a taxa de requisições compartilhada entre todas as threads.

    Parameters
    ----------
    requests_per_minute : float | None
        Taxa base máxima de requisições por minuto. ``None`` desativa o
        intervalo base (a pausa global e a adaptação continuam ativas).

    Raises
    ------
    ValueError
        Se ``requests_per_minute`` não for ``None`` e for menor ou igual a 0.

    Examples
    --------
    >>> limiter = RateLimiter(requests_per_minute=None)
    >>> limiter.acquire()
    """

    def __init__(self, requests_per_minute: float | None) -> None:
        if requests_per_minute is not None and requests_per_minute <= 0:
            raise ValueError("'requests_per_minute' precisa ser maior que 0 ou None.")

        self._base_interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self._interval = self._base_interval
        self._next_slot = 0.0
        self._cooldown_until = 0.0
        self._successes = 0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Bloqueia até a thread poder enviar a próxima requisição."""
        while True:
            with self._lock:
                now = time.monotonic()
                ready_at = max(self._next_slot, self._cooldown_until)
                if ready_at <= now:
                    self._next_slot = now + self._interval
                    return
                wait_seconds = ready_at - now
            time.sleep(wait_seconds)

    def report_success(self) -> None:
        """Registra um sucesso e, após uma sequência deles, acelera o ritmo."""
        with self._lock:
            self._successes += 1
            if self._successes >= _SUCCESSES_TO_SPEED_UP and self._interval > self._base_interval:
                self._interval = max(self._base_interval, self._interval * _SPEED_UP_FACTOR)
                self._successes = 0

    def report_rate_limited(self, wait_seconds: float) -> None:
        """Registra um 429: pausa todas as threads e desacelera o ritmo.

        Vários 429 simultâneos (rajada) só estendem a pausa e contam como uma
        única penalidade no intervalo.

        Parameters
        ----------
        wait_seconds : float
            Tempo de espera pedido pelo servidor ou calculado por backoff.
        """
        wait_seconds = min(wait_seconds, MAX_COOLDOWN_SECONDS)
        with self._lock:
            now = time.monotonic()
            in_burst = now < self._cooldown_until
            self._cooldown_until = max(self._cooldown_until, now + wait_seconds)
            self._successes = 0
            if not in_burst:
                self._interval = min(
                    max(self._interval, _MIN_THROTTLED_INTERVAL) * 2, _MAX_THROTTLED_INTERVAL
                )


def is_rate_limit_error(exception: BaseException) -> bool:
    """Indica se a exceção é uma resposta HTTP 429 (``openai.RateLimitError``).

    Detecta pelo ``status_code`` para não depender do import do SDK ``openai``.

    Parameters
    ----------
    exception : BaseException
        Exceção levantada pela chamada à API.

    Returns
    -------
    bool
        ``True`` se o status HTTP da resposta for 429.

    Examples
    --------
    >>> is_rate_limit_error(TimeoutError())
    False
    """
    return getattr(exception, "status_code", None) == 429


def get_retry_after_seconds(exception: BaseException) -> float | None:
    """Lê o tempo de espera sugerido pelo servidor em um erro 429.

    Parameters
    ----------
    exception : BaseException
        Erro retornado pela API.

    Returns
    -------
    float | None
        Segundos a esperar, conforme os cabeçalhos ``retry-after-ms`` ou
        ``retry-after`` (em segundos), ou ``None`` se ausentes/ilegíveis.

    Examples
    --------
    >>> get_retry_after_seconds(TimeoutError()) is None
    True
    """
    response: Any = getattr(exception, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None

    for header, factor in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        value = headers.get(header)
        if value is None:
            continue
        try:
            return max(0.0, float(value) * factor)
        except (TypeError, ValueError):
            continue
    return None


def calculate_backoff_seconds(attempt: int) -> float:
    """Calcula o backoff exponencial com jitter, para não sincronizar as threads.

    Parameters
    ----------
    attempt : int
        Número da tentativa (a partir de 1).

    Returns
    -------
    float
        Segundos de espera, limitados a 60 s (mais até 1 s de jitter).

    Examples
    --------
    >>> 2.0 <= calculate_backoff_seconds(1) <= 3.0
    True
    """
    return min(2.0**attempt, _MAX_BACKOFF_SECONDS) + random.uniform(0.0, 1.0)
