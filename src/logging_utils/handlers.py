"""Construção dos handlers de log de console (Rich) e arquivo.

O handler de console usa ``rich.logging.RichHandler`` para saída colorida
com tracebacks legíveis; o handler de arquivo grava em um arquivo diário
(``logs/log_YYYY-MM-DD.log``), com limpeza de arquivos antigos além da
retenção configurada.
"""

import logging
from datetime import date
from pathlib import Path

from rich.logging import RichHandler

from logging_utils.formatter import build_console_log_formatter, build_file_log_formatter

DEFAULT_FILENAME_PATTERN = "log_{date}.log"


def create_console_handler(
    *,
    level: int = logging.INFO,
    rich_tracebacks: bool = True,
    show_path: bool = True,
) -> RichHandler:
    """Cria o handler de console com formatação e tracebacks enriquecidos pelo Rich.

    Parameters
    ----------
    level : int, optional
        Nível mínimo de log processado pelo handler, by default ``logging.INFO``.
    rich_tracebacks : bool, optional
        Se ``True``, exibe tracebacks de exceção formatados pelo Rich,
        by default True.
    show_path : bool, optional
        Se ``True``, exibe o caminho do arquivo/linha de origem do log,
        by default True.

    Returns
    -------
    RichHandler
        Handler de console configurado.

    Examples
    --------
    >>> handler = create_console_handler()
    >>> isinstance(handler, RichHandler)
    True
    """
    handler = RichHandler(level=level, rich_tracebacks=rich_tracebacks, show_path=show_path)
    handler.setFormatter(build_console_log_formatter())
    return handler


def build_daily_log_file_path(
    log_directory: Path,
    *,
    filename_pattern: str = DEFAULT_FILENAME_PATTERN,
    reference_date: date | None = None,
) -> Path:
    """Monta o caminho do arquivo de log do dia, a partir de um padrão de nome.

    Parameters
    ----------
    log_directory : Path
        Diretório onde os arquivos de log são armazenados.
    filename_pattern : str, optional
        Padrão do nome do arquivo, contendo o marcador ``{date}``,
        by default :data:`DEFAULT_FILENAME_PATTERN`.
    reference_date : date | None, optional
        Data de referência usada para preencher ``{date}`` (formato
        ``YYYY-MM-DD``); usa a data atual quando ``None``, by default None.

    Returns
    -------
    Path
        Caminho completo do arquivo de log do dia.

    Examples
    --------
    >>> from datetime import date
    >>> build_daily_log_file_path(Path("logs"), reference_date=date(2026, 1, 5)).name
    'log_2026-01-05.log'
    """
    log_date = reference_date or date.today()
    filename = filename_pattern.format(date=log_date.isoformat())
    return log_directory / filename


def create_file_handler(
    log_directory: Path,
    *,
    filename_pattern: str = DEFAULT_FILENAME_PATTERN,
    level: int = logging.INFO,
    encoding: str = "utf-8",
) -> logging.FileHandler:
    """Cria o handler de arquivo para o log diário, criando o diretório se necessário.

    Parameters
    ----------
    log_directory : Path
        Diretório onde os arquivos de log são armazenados.
    filename_pattern : str, optional
        Padrão do nome do arquivo, contendo o marcador ``{date}``,
        by default :data:`DEFAULT_FILENAME_PATTERN`.
    level : int, optional
        Nível mínimo de log processado pelo handler, by default ``logging.INFO``.
    encoding : str, optional
        Codificação do arquivo de log, by default "utf-8".

    Returns
    -------
    logging.FileHandler
        Handler de arquivo configurado.

    Examples
    --------
    >>> handler = create_file_handler(Path("logs"))  # doctest: +SKIP
    """
    log_directory.mkdir(parents=True, exist_ok=True)
    file_path = build_daily_log_file_path(log_directory, filename_pattern=filename_pattern)
    handler = logging.FileHandler(file_path, encoding=encoding)
    handler.setLevel(level)
    handler.setFormatter(build_file_log_formatter())
    return handler


def remove_old_log_files(
    log_directory: Path,
    *,
    backup_count: int,
    filename_glob: str = "log_*.log",
) -> int:
    """Remove arquivos de log mais antigos que a retenção configurada.

    Mantém os ``backup_count`` arquivos mais recentes (por data de
    modificação) e remove os demais.

    Parameters
    ----------
    log_directory : Path
        Diretório onde os arquivos de log são armazenados.
    backup_count : int
        Número de arquivos de log recentes a manter.
    filename_glob : str, optional
        Padrão glob usado para localizar arquivos de log, by default "log_*.log".

    Returns
    -------
    int
        Quantidade de arquivos removidos.

    Examples
    --------
    >>> remove_old_log_files(Path("logs"), backup_count=30)  # doctest: +SKIP
    """
    if not log_directory.is_dir():
        return 0

    files = sorted(
        log_directory.glob(filename_glob),
        key=lambda file: file.stat().st_mtime,
        reverse=True,
    )
    files_to_remove = files[backup_count:]
    for file in files_to_remove:
        file.unlink()
    return len(files_to_remove)
