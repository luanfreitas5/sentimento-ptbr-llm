"""Fábrica de loggers e aplicação de handlers/níveis.

Fornece as operações de baixo nível sobre ``logging.Logger`` usadas pela
orquestração central em ``src/config/logging.py``, que lê
``configs/logging.yaml`` e monta os handlers via
:mod:`logging_utils.handlers`.
"""

import logging
from collections.abc import Iterable


def get_logger(name: str) -> logging.Logger:
    """Retorna um logger nomeado, seguindo a convenção padrão do módulo ``logging``.

    Parameters
    ----------
    name : str
        Nome do logger, tipicamente ``__name__`` do módulo chamador.

    Returns
    -------
    logging.Logger
        Instância do logger solicitado.

    Examples
    --------
    >>> logger = get_logger("sentimento_ptbr_llm.exemplo")
    >>> logger.name
    'sentimento_ptbr_llm.exemplo'
    """
    return logging.getLogger(name)


def configure_logger_handlers(
    logger: logging.Logger,
    handlers: Iterable[logging.Handler],
    *,
    level: int,
    propagate: bool = False,
) -> None:
    """Substitui os handlers de um logger e define seu nível e propagação.

    Remove quaisquer handlers previamente anexados antes de aplicar os
    novos, evitando duplicação de mensagens em reconfigurações sucessivas
    (ex.: em testes que chamam a configuração de logging múltiplas vezes).

    Parameters
    ----------
    logger : logging.Logger
        Logger a ser configurado.
    handlers : Iterable[logging.Handler]
        Handlers a serem anexados ao logger.
    level : int
        Nível mínimo de log do logger.
    propagate : bool, optional
        Se ``True``, propaga registros para loggers ancestrais,
        by default False.

    Returns
    -------
    None

    Examples
    --------
    >>> logger = get_logger("sentimento_ptbr_llm.exemplo_config")
    >>> configure_logger_handlers(logger, [logging.NullHandler()], level=logging.INFO)
    >>> logger.level == logging.INFO
    True
    """
    logger.handlers.clear()
    for handler in handlers:
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = propagate


def set_third_party_loggers_level(level: int, logger_names: Iterable[str]) -> None:
    """Define o nível de log para uma lista de bibliotecas de terceiros.

    Usado para reduzir a verbosidade de bibliotecas de terceiros
    (ex.: ``urllib3``, ``matplotlib``), mantendo o log do projeto informativo
    sem poluição.

    Parameters
    ----------
    level : int
        Nível mínimo de log a ser aplicado.
    logger_names : Iterable[str]
        Nomes dos loggers de terceiros a ajustar.

    Returns
    -------
    None

    Examples
    --------
    >>> set_third_party_loggers_level(logging.WARNING, ["urllib3"])
    >>> logging.getLogger("urllib3").level == logging.WARNING
    True
    """
    for name in logger_names:
        logging.getLogger(name).setLevel(level)
