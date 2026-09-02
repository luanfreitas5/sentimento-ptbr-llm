"""Persistência de modelos treinados em disco via ``joblib``.

Usado como mecanismo de baixo nível por ``src/models/persistence.py``. Os
artefatos gerados não são versionados pelo Git (ver ``.gitignore``); o
rastreamento de modelos é feito via DVC/MLflow Model Registry.
"""

import logging
import pickle  # nosec B403 # usado apenas para o tipo de exceção; ver justificativa em load_model
from pathlib import Path
from typing import Any

import joblib

from exceptions.model import ModelPersistenceError
from utils.validation import validate_file_exists

logger = logging.getLogger(__name__)


def save_model(model: Any, file_path: Path) -> None:
    """Salva um modelo treinado em disco, criando diretórios pais se necessário.

    Parameters
    ----------
    model : Any
        Objeto do modelo a ser serializado (ex.: estimador ``scikit-learn``).
    file_path : Path
        Caminho do arquivo de destino.

    Returns
    -------
    None

    Raises
    ------
    ModelPersistenceError
        Se ocorrer falha durante a serialização do modelo.

    Examples
    --------
    >>> save_model(object(), Path("models/artifacts/exemplo.joblib"))  # doctest: +SKIP
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(model, file_path)
    except (OSError, TypeError, ValueError) as exception:
        raise ModelPersistenceError(str(file_path), str(exception)) from exception
    logger.info("Modelo salvo em: %s", file_path)


def load_model(file_path: Path) -> Any:
    """Carrega um modelo previamente salvo com :func:`save_model`.

    Parameters
    ----------
    file_path : Path
        Caminho do arquivo do modelo a ser carregado.

    Returns
    -------
    Any
        Objeto do modelo desserializado.

    Raises
    ------
    DataNotFoundError
        Se o arquivo não existir.
    ModelPersistenceError
        Se ocorrer falha durante a desserialização do modelo.

    Examples
    --------
    >>> load_model(Path("models/artifacts/exemplo.joblib"))  # doctest: +SKIP
    """
    # joblib.load desserializa via pickle, o que permite execução arbitrária
    # de código se o arquivo for adulterado. Aceitável aqui: os artefatos são
    # gerados apenas pelo próprio pipeline de treinamento deste projeto e
    # rastreados via DVC/MLflow Model Registry — nunca uma fonte externa
    # não confiável.
    validate_file_exists(file_path)
    try:
        model = joblib.load(file_path)
    except (OSError, EOFError, ValueError, pickle.UnpicklingError) as exception:
        raise ModelPersistenceError(str(file_path), str(exception)) from exception
    logger.info("Modelo carregado de: %s", file_path)
    return model
