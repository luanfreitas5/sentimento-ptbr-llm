"""Construção de handlers, formatadores e utilitários de log do projeto.

.. note::
   O pacote chama-se ``logging_utils`` (e não ``logging``) para não colidir
   com o módulo ``logging`` da biblioteca padrão do Python — diferente de
   ``io``, o módulo ``logging`` não é carregado na inicialização do
   interpretador, então a colisão seria uma corrida (race) não
   determinística entre processos, dependendo de qual código importa
   ``logging`` primeiro.

A orquestração central (leitura de ``configs/logging.yaml`` e aplicação
efetiva ao logger raiz) vive em ``src/config/logging.py``; este pacote
fornece as peças reutilizáveis: formatadores, handlers, fábrica de loggers
e um cronômetro para logar a duração de blocos de código.

Modules
-------
formatter
    Formatadores de log para console e arquivo.
handlers
    Construção dos handlers de console (Rich) e arquivo (rotação diária).
logger
    Fábrica de loggers e aplicação de handlers/níveis.
timer
    Registro em log da duração de execução de blocos de código nomeados.
"""

from logging_utils.formatter import build_console_log_formatter, build_file_log_formatter
from logging_utils.handlers import (
    build_daily_log_file_path,
    create_console_handler,
    create_file_handler,
    remove_old_log_files,
)
from logging_utils.logger import (
    configure_logger_handlers,
    get_logger,
    set_third_party_loggers_level,
)
from logging_utils.timer import time_block

__all__: list[str] = [
    "build_console_log_formatter",
    "build_daily_log_file_path",
    "build_file_log_formatter",
    "configure_logger_handlers",
    "create_console_handler",
    "create_file_handler",
    "get_logger",
    "remove_old_log_files",
    "set_third_party_loggers_level",
    "time_block",
]
