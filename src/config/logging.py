"""Configuração central de logging do projeto.

Lê ``configs/logging.yaml`` e monta o logger raiz com um handler de
console (Rich) e um handler de arquivo diário, compondo as peças
reutilizáveis de :mod:`logging_utils`. Chamado uma única vez, no início da
execução (``src/main.py`` ou ``conftest.py`` dos testes).
"""

import logging
from pathlib import Path

from config.constants import CONFIG_FILE_NAMES, PROJECT_PACKAGE_NAMES
from config.paths import CONFIGS_DIR, PROJECT_ROOT
from exceptions.configuration import InvalidConfigurationError
from io_utils.yaml import read_yaml
from logging_utils.handlers import create_console_handler, create_file_handler, remove_old_log_files
from logging_utils.logger import configure_logger_handlers, set_third_party_loggers_level

DEFAULT_LOGGING_CONFIG_FILE: Path = CONFIGS_DIR / CONFIG_FILE_NAMES["logging"]


def _resolve_log_level(level_name: str) -> int:
    """Converte o nome textual de um nível de log (ex.: "INFO") em seu valor numérico.

    Parameters
    ----------
    level_name : str
        Nome do nível de log, conforme o módulo ``logging`` (ex.: "INFO",
        "WARNING").

    Returns
    -------
    int
        Valor numérico do nível de log.

    Raises
    ------
    InvalidConfigurationError
        Se o nome do nível não for reconhecido pelo módulo ``logging``.

    Examples
    --------
    >>> _resolve_log_level("INFO") == logging.INFO
    True
    """
    level = logging.getLevelName(level_name.upper())
    if not isinstance(level, int):
        raise InvalidConfigurationError(f"nível de log desconhecido: '{level_name}'")
    return level


def configure_logging(
    config_file_path: Path = DEFAULT_LOGGING_CONFIG_FILE,
    *,
    logs_directory: Path | None = None,
) -> None:
    """Configura o logging do projeto a partir de ``configs/logging.yaml``.

    Aplica um handler de console (Rich) e/ou um handler de arquivo diário ao
    logger raiz, conforme habilitados na configuração, define o nível do
    logger raiz e eleva o nível dos pacotes de primeira parte do projeto
    (``project_level``) independentemente do nível padrão do raiz.

    Parameters
    ----------
    config_file_path : Path, optional
        Caminho do arquivo YAML de configuração de logging, by default
        :data:`DEFAULT_LOGGING_CONFIG_FILE`.
    logs_directory : Path | None, optional
        Diretório onde os arquivos de log são gravados; usa o valor de
        ``configs/paths.yaml`` (via ``file.dir`` da própria configuração de
        logging) quando ``None``, by default None.

    Returns
    -------
    None

    Examples
    --------
    >>> configure_logging()  # doctest: +SKIP
    """
    config = read_yaml(config_file_path)

    global_level = _resolve_log_level(config["level"])
    logs_path = logs_directory or (PROJECT_ROOT / config["file"]["dir"])

    handlers: list[logging.Handler] = []

    if config["console"]["enabled"]:
        handlers.append(
            create_console_handler(
                level=global_level,
                rich_tracebacks=config["console"]["rich_tracebacks"],
                show_path=config["console"]["show_path"],
            )
        )

    if config["file"]["enabled"]:
        handlers.append(
            create_file_handler(
                logs_path,
                filename_pattern=config["file"]["filename_pattern"],
                level=global_level,
                encoding=config["file"]["encoding"],
            )
        )
        remove_old_log_files(logs_path, backup_count=config["file"]["backup_count"])

    configure_logger_handlers(
        logging.getLogger(),
        handlers,
        level=_resolve_log_level(config["loggers"]["root_level"]),
        propagate=False,
    )

    set_third_party_loggers_level(
        _resolve_log_level(config["loggers"]["project_level"]), PROJECT_PACKAGE_NAMES
    )
