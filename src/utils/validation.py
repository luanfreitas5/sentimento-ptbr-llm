"""Funções utilitárias de validação genérica.

Usadas nos limites do sistema (leitura de arquivos, entrada de dados) para
falhar cedo com uma mensagem clara em pt-BR, em vez de propagar erros
silenciosos adiante no pipeline.
"""

from pathlib import Path
from typing import Any

from exceptions.data import DataNotFoundError, EmptyDatasetError


def validate_file_exists(file_path: Path) -> Path:
    """Valida se um arquivo existe no caminho informado.

    Parameters
    ----------
    file_path : Path
        Caminho do arquivo a ser validado.

    Returns
    -------
    Path
        O mesmo caminho, quando o arquivo existe.

    Raises
    ------
    DataNotFoundError
        Se o arquivo não existir ou não for um arquivo regular.

    Examples
    --------
    >>> validate_file_exists(Path("configs/config.yaml"))  # doctest: +SKIP
    """
    if not file_path.is_file():
        raise DataNotFoundError(str(file_path))
    return file_path


def validate_directory_exists(directory_path: Path, *, create_if_missing: bool = False) -> Path:
    """Valida se um diretório existe, opcionalmente criando-o quando ausente.

    Parameters
    ----------
    directory_path : Path
        Caminho do diretório a ser validado.
    create_if_missing : bool, optional
        Se ``True``, cria o diretório (e pais ausentes) em vez de levantar
        exceção, by default False.

    Returns
    -------
    Path
        O mesmo caminho, garantidamente existente ao final da chamada.

    Raises
    ------
    DataNotFoundError
        Se o diretório não existir e ``create_if_missing`` for ``False``.

    Examples
    --------
    >>> validate_directory_exists(Path("logs"), create_if_missing=True)  # doctest: +SKIP
    """
    if directory_path.is_dir():
        return directory_path
    if create_if_missing:
        directory_path.mkdir(parents=True, exist_ok=True)
        return directory_path
    raise DataNotFoundError(str(directory_path))


def validate_value_in_choices(value: Any, choices: list[Any] | tuple[Any, ...]) -> Any:
    """Valida se um valor pertence a um conjunto de escolhas permitidas.

    Parameters
    ----------
    value : Any
        Valor a ser validado.
    choices : list[Any] | tuple[Any, ...]
        Conjunto de valores permitidos.

    Returns
    -------
    Any
        O próprio valor, quando válido.

    Raises
    ------
    ValueError
        Se o valor não pertencer ao conjunto de escolhas.

    Examples
    --------
    >>> validate_value_in_choices("f1_macro", ["f1_macro", "accuracy"])
    'f1_macro'
    """
    if value not in choices:
        raise ValueError(f"Valor '{value}' inválido. Valores permitidos: {choices}")
    return value


def validate_not_empty_collection(collection: Any, *, collection_name: str) -> Any:
    """Valida se uma coleção (lista, DataFrame, etc.) não está vazia.

    Parameters
    ----------
    collection : Any
        Coleção a ser validada. Deve suportar ``len()``.
    collection_name : str
        Nome descritivo da coleção, usado na mensagem de erro.

    Returns
    -------
    Any
        A própria coleção, quando não vazia.

    Raises
    ------
    EmptyDatasetError
        Se a coleção estiver vazia.

    Examples
    --------
    >>> validate_not_empty_collection([1, 2, 3], collection_name="exemplo")
    [1, 2, 3]
    """
    if len(collection) == 0:
        raise EmptyDatasetError(collection_name)
    return collection
