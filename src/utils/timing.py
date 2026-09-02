"""Medição e formatação de duração de execução.

Fornece a medição de baixo nível (``perf_counter``) usada tanto por
:mod:`logging.timer` (que registra a duração via ``logging``) quanto por
qualquer trecho de código que precise apenas medir tempo, sem logar.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter


@dataclass
class ExecutionTiming:
    """Resultado de uma medição de tempo de execução.

    Parameters
    ----------
    elapsed_seconds : float
        Duração medida, em segundos.
    """

    elapsed_seconds: float = 0.0


@contextmanager
def measure_execution_time() -> Iterator[ExecutionTiming]:
    """Mede o tempo de execução do bloco ``with`` usando um relógio monotônico.

    Returns
    -------
    Iterator[ExecutionTiming]
        Objeto mutável cujo campo ``elapsed_seconds`` é preenchido ao final
        do bloco ``with``.

    Examples
    --------
    >>> with measure_execution_time() as tempo:
    ...     _ = sum(range(1000))
    >>> tempo.elapsed_seconds >= 0
    True
    """
    resultado = ExecutionTiming()
    inicio = perf_counter()
    try:
        yield resultado
    finally:
        resultado.elapsed_seconds = perf_counter() - inicio


def format_duration(seconds: float) -> str:
    """Formata uma duração em segundos em uma string legível (h/min/s).

    Parameters
    ----------
    seconds : float
        Duração em segundos.

    Returns
    -------
    str
        Duração formatada, por exemplo ``"1h 02min 03.40s"`` ou ``"3.40s"``.

    Raises
    ------
    ValueError
        Se ``seconds`` for negativo.

    Examples
    --------
    >>> format_duration(3.4)
    '3.40s'
    >>> format_duration(3723.4)
    '1h 02min 03.40s'
    """
    if seconds < 0:
        raise ValueError(f"Duração não pode ser negativa: {seconds}")

    horas, resto = divmod(seconds, 3600)
    minutos, segundos = divmod(resto, 60)

    if horas >= 1:
        return f"{int(horas)}h {int(minutos):02d}min {segundos:05.2f}s"
    if minutos >= 1:
        return f"{int(minutos)}min {segundos:05.2f}s"
    return f"{segundos:.2f}s"
