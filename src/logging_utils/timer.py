"""Registro em log da duração de execução de blocos de código nomeados.

Combina :func:`utils.timing.measure_execution_time` com um logger para
instrumentar etapas de pipeline (ex.: "pré-processamento", "treinamento do
modelo X") sem duplicar a lógica de medição de tempo.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from utils.timing import format_duration, measure_execution_time


@contextmanager
def time_block(
    logger: logging.Logger,
    description: str,
    *,
    level: int = logging.INFO,
) -> Iterator[None]:
    """Mede e registra em log a duração de execução de um bloco ``with``.

    Parameters
    ----------
    logger : logging.Logger
        Logger usado para registrar a duração medida.
    description : str
        Descrição do bloco medido (ex.: "Treinamento do modelo BERTimbau"),
        usada na mensagem de log.
    level : int, optional
        Nível de log usado para a mensagem de duração, by default ``logging.INFO``.

    Returns
    -------
    Iterator[None]
        Gerenciador de contexto sem valor de retorno.

    Examples
    --------
    >>> import logging
    >>> logger = logging.getLogger("sentimento_ptbr_llm.exemplo_timer")
    >>> with time_block(logger, "bloco de exemplo"):
    ...     _ = sum(range(1000))
    """
    logger.log(level, "%s: iniciado", description)
    with measure_execution_time() as tempo:
        yield
    logger.log(level, "%s: concluído em %s", description, format_duration(tempo.elapsed_seconds))
