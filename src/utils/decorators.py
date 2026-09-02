"""Decoradores genéricos de propósito geral.

Usam o módulo ``logging`` padrão diretamente (não o pacote ``logging`` do
projeto) para evitar dependência circular entre ``utils`` e o pacote de
logging, que por sua vez depende de ``utils.timing``.
"""

import functools
import logging
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from utils.timing import format_duration, measure_execution_time

logger = logging.getLogger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")


def log_execution_time(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """Decora uma função para registrar em log seu tempo total de execução.

    Parameters
    ----------
    func : Callable[_P, _R]
        Função a ser decorada.

    Returns
    -------
    Callable[_P, _R]
        Função decorada, com o mesmo comportamento e retorno da original.

    Examples
    --------
    >>> @log_execution_time
    ... def somar(a: int, b: int) -> int:
    ...     return a + b
    >>> somar(2, 3)
    5
    """

    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with measure_execution_time() as tempo:
            resultado = func(*args, **kwargs)
        logger.info("Função '%s' executada em %s", func.__qualname__, format_duration(tempo.elapsed_seconds))
        return resultado

    return wrapper


def retry_on_exception(
    *,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Decora uma função para reexecutá-la em caso de falha, com espera fixa entre tentativas.

    Útil para operações sujeitas a falhas transitórias (ex.: chamadas a um
    servidor Ollama local ou requisições HTTP).

    Parameters
    ----------
    exceptions : tuple[type[Exception], ...], optional
        Tupla de tipos de exceção que disparam nova tentativa, by default
        ``(Exception,)``.
    max_attempts : int, optional
        Número máximo de tentativas (incluindo a primeira), by default 3.
    delay_seconds : float, optional
        Tempo de espera, em segundos, entre tentativas, by default 1.0.

    Returns
    -------
    Callable[[Callable[_P, _R]], Callable[_P, _R]]
        Decorador configurado.

    Raises
    ------
    ValueError
        Se ``max_attempts`` for menor que 1.

    Examples
    --------
    >>> @retry_on_exception(exceptions=(ValueError,), max_attempts=2, delay_seconds=0)
    ... def sempre_falha() -> None:
    ...     raise ValueError("falha proposital")
    >>> sempre_falha()
    Traceback (most recent call last):
        ...
    ValueError: falha proposital
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts deve ser >= 1, recebido: {max_attempts}")

    def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        @functools.wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            ultima_excecao: Exception | None = None
            for tentativa in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as excecao:
                    ultima_excecao = excecao
                    logger.warning(
                        "Tentativa %d/%d de '%s' falhou: %s",
                        tentativa,
                        max_attempts,
                        func.__qualname__,
                        excecao,
                    )
                    if tentativa < max_attempts:
                        time.sleep(delay_seconds)
            assert ultima_excecao is not None
            raise ultima_excecao

        return wrapper

    return decorator
