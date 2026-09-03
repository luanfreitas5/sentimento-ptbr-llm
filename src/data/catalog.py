"""Catálogo dos datasets do projeto e seus hashes de rastreabilidade.

Registra, para cada arquivo de dados relevante (bruto, gold set, interim,
processado), seu hash SHA-256 e tamanho em bytes, permitindo detectar
mudanças silenciosas nos dados entre execuções (ver CLAUDE.md,
"Reprodutibilidade & Determinismo").
"""

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from io_utils.json import write_json
from utils.hashing import calculate_file_hash
from utils.validation import validate_file_exists

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetCatalogEntry:
    """Metadados de rastreabilidade de um único dataset.

    Parameters
    ----------
    name : str
        Nome identificador do dataset no catálogo (ex.: ``"raw_tweets"``).
    file_path : str
        Caminho do arquivo, como string (para serialização em JSON).
    sha256_hash : str
        Hash SHA-256 do conteúdo do arquivo.
    size_bytes : int
        Tamanho do arquivo em bytes.
    """

    name: str
    file_path: str
    sha256_hash: str
    size_bytes: int


def build_dataset_catalog_entry(name: str, file_path: Path) -> DatasetCatalogEntry:
    """Constrói a entrada de catálogo de um único dataset.

    Parameters
    ----------
    name : str
        Nome identificador do dataset.
    file_path : Path
        Caminho do arquivo de dados.

    Returns
    -------
    DatasetCatalogEntry
        Entrada com hash SHA-256 e tamanho do arquivo.

    Raises
    ------
    DataNotFoundError
        Se o arquivo não existir.

    Examples
    --------
    >>> build_dataset_catalog_entry("config", Path("configs/config.yaml")).name
    'config'
    """
    validate_file_exists(file_path)
    return DatasetCatalogEntry(
        name=name,
        file_path=str(file_path),
        sha256_hash=calculate_file_hash(file_path),
        size_bytes=file_path.stat().st_size,
    )


def build_dataset_catalog(datasets: dict[str, Path]) -> list[DatasetCatalogEntry]:
    """Constrói o catálogo completo a partir de um mapeamento nome -> caminho.

    Datasets cujo arquivo ainda não existe são ignorados (com aviso em log)
    em vez de interromper a construção do catálogo — comum em estágios
    iniciais do pipeline, quando nem todos os arquivos já foram gerados.

    Parameters
    ----------
    datasets : dict[str, Path]
        Mapeamento entre o nome identificador e o caminho de cada dataset.

    Returns
    -------
    list[DatasetCatalogEntry]
        Entradas de catálogo para os datasets encontrados, na ordem de
        iteração de ``datasets``.

    Examples
    --------
    >>> build_dataset_catalog({"config": Path("configs/config.yaml")})[0].name
    'config'
    """
    entries: list[DatasetCatalogEntry] = []
    for name, file_path in datasets.items():
        if not file_path.is_file():
            logger.warning(
                "Dataset '%s' ignorado no catálogo: arquivo não encontrado (%s)", name, file_path
            )
            continue
        entries.append(build_dataset_catalog_entry(name, file_path))
    return entries


def write_dataset_catalog(entries: list[DatasetCatalogEntry], file_path: Path) -> None:
    """Serializa o catálogo de datasets em um arquivo JSON.

    Parameters
    ----------
    entries : list[DatasetCatalogEntry]
        Entradas de catálogo a serem persistidas.
    file_path : Path
        Caminho do arquivo JSON de destino.

    Returns
    -------
    None

    Examples
    --------
    >>> write_dataset_catalog([], Path("reports/metrics/catalogo_exemplo.json"))  # doctest: +SKIP
    """
    write_json([asdict(entry) for entry in entries], file_path)
    logger.info("Catálogo de %d dataset(s) escrito em: %s", len(entries), file_path)
