"""Funções utilitárias de hashing para rastreabilidade de dados.

Usadas para detectar mudanças silenciosas em arquivos de dados (ver
CLAUDE.md, seção "Reprodutibilidade & Determinismo": cada execução deve
registrar o hash do dataset usado, junto do SHA do Git e dos parâmetros).
"""

import hashlib
from pathlib import Path

from utils.validation import validate_file_exists

_DEFAULT_CHUNK_SIZE_BYTES = 65536


def calculate_file_hash(file_path: Path, *, algorithm: str = "sha256") -> str:
    """Calcula o hash de um arquivo, lendo-o em blocos para não estourar a memória.

    Parameters
    ----------
    file_path : Path
        Caminho do arquivo a ser hasheado.
    algorithm : str, optional
        Nome do algoritmo de hash suportado por ``hashlib`` (ex.: ``"sha256"``,
        ``"md5"``), by default "sha256".

    Returns
    -------
    str
        Hash hexadecimal do conteúdo do arquivo.

    Raises
    ------
    DataNotFoundError
        Se o arquivo não existir.

    Examples
    --------
    >>> calculate_file_hash(Path("configs/config.yaml"))  # doctest: +SKIP
    """
    validate_file_exists(file_path)
    hasher = hashlib.new(algorithm)
    with file_path.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(_DEFAULT_CHUNK_SIZE_BYTES), b""):
            hasher.update(bloco)
    return hasher.hexdigest()


def calculate_text_hash(text: str, *, algorithm: str = "sha256") -> str:
    """Calcula o hash de uma string codificada em UTF-8.

    Útil tanto para rastreabilidade (versionar o conteúdo de um prompt, por
    exemplo) quanto para pseudonimização de identificadores diretos.

    Parameters
    ----------
    text : str
        Texto a ser hasheado.
    algorithm : str, optional
        Nome do algoritmo de hash suportado por ``hashlib``, by default "sha256".

    Returns
    -------
    str
        Hash hexadecimal do texto.

    Examples
    --------
    >>> len(calculate_text_hash("exemplo"))
    64
    """
    hasher = hashlib.new(algorithm)
    hasher.update(text.encode("utf-8"))
    return hasher.hexdigest()
