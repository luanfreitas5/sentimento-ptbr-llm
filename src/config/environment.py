"""Configuração de variáveis de ambiente e reprodutibilidade do processo.

Compõe :func:`utils.seed.seed_everything` (fixação de sementes) com o
carregamento de variáveis de ambiente via ``python-dotenv``, seguindo
CLAUDE.md ("Secrets: Managed via .env + python-dotenv").
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from config.constants import ENV_FILE_NAME
from config.paths import PROJECT_ROOT
from exceptions.configuration import MissingEnvironmentVariableError
from utils.seed import seed_everything

logger = logging.getLogger(__name__)

DEFAULT_ENV_FILE: Path = PROJECT_ROOT / ENV_FILE_NAME


def configure_environment_variables(env_file_path: Path = DEFAULT_ENV_FILE) -> None:
    """Carrega variáveis de ambiente de um arquivo ``.env``, se ele existir.

    Variáveis já definidas no ambiente do sistema operacional não são
    sobrescritas (``override=False``), permitindo que o operador do
    pipeline substitua valores do ``.env`` pontualmente.

    Parameters
    ----------
    env_file_path : Path, optional
        Caminho do arquivo ``.env``, by default :data:`DEFAULT_ENV_FILE`.

    Returns
    -------
    None

    Examples
    --------
    >>> configure_environment_variables(Path("arquivo_inexistente.env"))
    """
    if not env_file_path.is_file():
        logger.warning(
            "Arquivo de variáveis de ambiente não encontrado em '%s'; "
            "usando apenas variáveis já definidas no ambiente do sistema.",
            env_file_path,
        )
        return
    load_dotenv(dotenv_path=env_file_path, override=False)
    logger.debug("Variáveis de ambiente carregadas de: %s", env_file_path)


def get_required_environment_variable(variable_name: str) -> str:
    """Lê uma variável de ambiente obrigatória, levantando exceção se ausente.

    Parameters
    ----------
    variable_name : str
        Nome da variável de ambiente.

    Returns
    -------
    str
        Valor da variável de ambiente.

    Raises
    ------
    MissingEnvironmentVariableError
        Se a variável não estiver definida.

    Examples
    --------
    >>> import os
    >>> os.environ["EXEMPLO_VARIAVEL"] = "valor"
    >>> get_required_environment_variable("EXEMPLO_VARIAVEL")
    'valor'
    """
    valor = os.environ.get(variable_name)
    if valor is None:
        raise MissingEnvironmentVariableError(variable_name)
    return valor


def configure_reproducibility(random_seed: int, *, deterministic_algorithms: bool = True) -> None:
    """Configura a reprodutibilidade global do processo a partir da semente do projeto.

    Parameters
    ----------
    random_seed : int
        Semente aleatória a ser aplicada (ver ``configs/config.yaml`` ->
        ``reproducibility.random_seed``).
    deterministic_algorithms : bool, optional
        Registrado em log para indicar a intenção de determinismo mesmo em
        operações custosas (ex.: convoluções em GPU); a aplicação efetiva
        ocorre em :func:`utils.seed.seed_everything`, by default True.

    Returns
    -------
    None

    Examples
    --------
    >>> configure_reproducibility(42)
    """
    seed_everything(random_seed)
    logger.info(
        "Reprodutibilidade configurada: semente=%d, algoritmos_deterministicos=%s",
        random_seed,
        deterministic_algorithms,
    )
