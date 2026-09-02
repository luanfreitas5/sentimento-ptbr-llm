"""Leitura e escrita de arquivos JSON.

Usado para artefatos leves como metadados de execução, respostas
estruturadas de LLM e relatórios de avaliação.
"""

import json
import logging
from pathlib import Path
from typing import Any

from utils.validation import validate_file_exists

logger = logging.getLogger(__name__)


def read_json(file_path: Path) -> Any:
    """Lê um arquivo JSON e retorna seu conteúdo desserializado.

    Parameters
    ----------
    file_path : Path
        Caminho do arquivo JSON a ser lido.

    Returns
    -------
    Any
        Conteúdo desserializado do arquivo JSON.

    Raises
    ------
    DataNotFoundError
        Se o arquivo não existir.
    json.JSONDecodeError
        Se o conteúdo do arquivo não for um JSON válido.

    Examples
    --------
    >>> read_json(Path("reports/metrics/exemplo.json"))  # doctest: +SKIP
    """
    validate_file_exists(file_path)
    with file_path.open("r", encoding="utf-8") as file:
        content = json.load(file)
    logger.debug("Arquivo JSON lido: %s", file_path)
    return content


def write_json(data: Any, file_path: Path, *, indent: int = 2) -> None:
    """Escreve dados serializáveis em um arquivo JSON, criando diretórios pais se necessário.

    Parameters
    ----------
    data : Any
        Dados a serem serializados em JSON.
    file_path : Path
        Caminho do arquivo JSON de destino.
    indent : int, optional
        Número de espaços de indentação, by default 2.

    Returns
    -------
    None

    Examples
    --------
    >>> write_json({"f1_macro": 0.82}, Path("reports/metrics/exemplo.json"))  # doctest: +SKIP
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=indent, ensure_ascii=False, sort_keys=False)
    logger.debug("Arquivo JSON escrito: %s", file_path)
