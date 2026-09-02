"""Leitura e escrita de arquivos YAML.

Usado principalmente para carregar as configurações versionadas em
``configs/`` (ver ``src/config/settings.py`` e ``src/config/paths.py``).
"""

import logging
from pathlib import Path
from typing import Any

import yaml

from utils.validation import validate_file_exists

logger = logging.getLogger(__name__)


def read_yaml(file_path: Path) -> dict[str, Any]:
    """Lê um arquivo YAML e retorna seu conteúdo como dicionário.

    Parameters
    ----------
    file_path : Path
        Caminho do arquivo YAML a ser lido.

    Returns
    -------
    dict[str, Any]
        Conteúdo do arquivo YAML. Retorna um dicionário vazio se o arquivo
        estiver vazio.

    Raises
    ------
    DataNotFoundError
        Se o arquivo não existir.
    yaml.YAMLError
        Se o conteúdo do arquivo não for um YAML válido.

    Examples
    --------
    >>> read_yaml(Path("configs/config.yaml"))  # doctest: +SKIP
    """
    validate_file_exists(file_path)
    with file_path.open("r", encoding="utf-8") as file:
        content = yaml.safe_load(file)
    logger.debug("Arquivo YAML lido: %s", file_path)
    return content or {}


def write_yaml(data: dict[str, Any], file_path: Path) -> None:
    """Escreve um dicionário em um arquivo YAML, criando diretórios pais se necessário.

    Parameters
    ----------
    data : dict[str, Any]
        Dados a serem serializados em YAML.
    file_path : Path
        Caminho do arquivo YAML de destino.

    Returns
    -------
    None

    Examples
    --------
    >>> write_yaml({"chave": "valor"}, Path("reports/exemplo.yaml"))  # doctest: +SKIP
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)
    logger.debug("Arquivo YAML escrito: %s", file_path)
