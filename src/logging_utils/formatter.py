"""Formatadores de log usados nos handlers de console e arquivo.

Os formatos seguem ``configs/logging.yaml`` e o padrão definido em
CLAUDE.md: o arquivo de log usa um formato tabular fixo, enquanto o console
delega a formatação visual ao ``RichHandler`` (que já inclui nome do
logger, nível e cores).
"""

import logging

DEFAULT_FILE_LOG_FORMAT = "%(asctime)s \t %(levelname)s \t %(name)s \t %(message)s"
DEFAULT_CONSOLE_LOG_FORMAT = "%(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def build_file_log_formatter(
    log_format: str = DEFAULT_FILE_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
) -> logging.Formatter:
    """Constrói o formatador usado no handler de arquivo.

    Parameters
    ----------
    log_format : str, optional
        String de formato do ``logging``, by default :data:`DEFAULT_FILE_LOG_FORMAT`.
    date_format : str, optional
        String de formato de data/hora, by default :data:`DEFAULT_DATE_FORMAT`.

    Returns
    -------
    logging.Formatter
        Formatador configurado para uso em arquivo.

    Examples
    --------
    >>> isinstance(build_file_log_formatter(), logging.Formatter)
    True
    """
    return logging.Formatter(fmt=log_format, datefmt=date_format)


def build_console_log_formatter(log_format: str = DEFAULT_CONSOLE_LOG_FORMAT) -> logging.Formatter:
    """Constrói o formatador usado no handler de console (``RichHandler``).

    O ``RichHandler`` já exibe nome do logger, nível e cores por conta
    própria; o formato da mensagem em si costuma ficar reduzido a
    ``%(message)s``.

    Parameters
    ----------
    log_format : str, optional
        String de formato do ``logging``, by default :data:`DEFAULT_CONSOLE_LOG_FORMAT`.

    Returns
    -------
    logging.Formatter
        Formatador configurado para uso em console.

    Examples
    --------
    >>> isinstance(build_console_log_formatter(), logging.Formatter)
    True
    """
    return logging.Formatter(fmt=log_format)
