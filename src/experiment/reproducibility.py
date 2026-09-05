"""Manifesto de reprodutibilidade por execução de experimento.

Implementa CLAUDE.md, "Reproducibility & Determinism": cada execução deve
registrar o código exato (SHA do Git), os dados exatos (hash do dataset,
ver :mod:`utils.hashing`) e o ambiente exato (versões de bibliotecas) que a
produziram, permitindo reconstruir qualquer resultado publicado.
"""

import logging
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from config.paths import PROJECT_ROOT
from constants.defaults import DEFAULT_RANDOM_SEED
from utils.hashing import calculate_file_hash

logger = logging.getLogger(__name__)

TRACKED_LIBRARY_NAMES: tuple[str, ...] = (
    "numpy",
    "pandas",
    "polars",
    "scikit-learn",
    "scipy",
    "torch",
    "mlflow",
)


def get_current_git_sha(*, short: bool = False) -> str:
    """Obtém o SHA do commit Git atual do repositório do projeto.

    Parameters
    ----------
    short : bool, optional
        Se ``True``, retorna o SHA abreviado, by default False.

    Returns
    -------
    str
        SHA do commit atual (``HEAD``).

    Raises
    ------
    RuntimeError
        Se o ``git`` não estiver disponível ou o comando falhar (ex.: fora
        de um repositório Git).

    Examples
    --------
    >>> get_current_git_sha()  # doctest: +SKIP
    """
    command = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        result = subprocess.run(
            command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exception:
        raise RuntimeError(
            "Não foi possível obter o SHA do commit Git atual. Verifique se o `git` está "
            "instalado e se o diretório do projeto é um repositório Git."
        ) from exception
    return result.stdout.strip()


def collect_library_versions(
    library_names: tuple[str, ...] = TRACKED_LIBRARY_NAMES,
) -> dict[str, str]:
    """Coleta a versão instalada de cada biblioteca rastreada.

    Parameters
    ----------
    library_names : tuple[str, ...], optional
        Nomes das bibliotecas (nomes de distribuição PyPI) a inspecionar,
        by default :data:`TRACKED_LIBRARY_NAMES`.

    Returns
    -------
    dict[str, str]
        Versão de cada biblioteca instalada; bibliotecas não instaladas são
        omitidas do resultado.

    Examples
    --------
    >>> "numpy" in collect_library_versions(("numpy",))
    True
    """
    library_versions: dict[str, str] = {}
    for library_name in library_names:
        try:
            library_versions[library_name] = version(library_name)
        except PackageNotFoundError:
            logger.debug("Biblioteca '%s' não está instalada; omitida do manifesto.", library_name)
    return library_versions


def build_reproducibility_manifest(
    dataset_path: Path,
    *,
    random_seed: int = DEFAULT_RANDOM_SEED,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Monta o manifesto de reprodutibilidade de uma execução.

    Parameters
    ----------
    dataset_path : Path
        Caminho do dataset usado na execução, hasheado para detectar
        mudanças silenciosas.
    random_seed : int, optional
        Semente aleatória usada na execução, by default
        :data:`constants.defaults.DEFAULT_RANDOM_SEED`.
    extra_metadata : dict[str, Any] | None, optional
        Metadados adicionais a incluir (ex.: nome do modelo,
        hiperparâmetros), by default None.

    Returns
    -------
    dict[str, Any]
        Manifesto com ``git_sha``, ``dataset_hash``, ``python_version``,
        ``library_versions``, ``random_seed`` e os campos de
        ``extra_metadata``.

    Raises
    ------
    DataNotFoundError
        Se ``dataset_path`` não existir.
    RuntimeError
        Se o SHA do Git não puder ser obtido.

    Examples
    --------
    >>> build_reproducibility_manifest(Path("configs/config.yaml"))  # doctest: +SKIP
    """
    manifest: dict[str, Any] = {
        "git_sha": get_current_git_sha(),
        "dataset_hash": calculate_file_hash(dataset_path),
        "python_version": platform.python_version(),
        "library_versions": collect_library_versions(),
        "random_seed": random_seed,
    }
    manifest.update(extra_metadata or {})
    logger.info("Manifesto de reprodutibilidade construído (git_sha=%s).", manifest["git_sha"][:8])
    return manifest


def compare_reproducibility_manifests(
    manifest_a: dict[str, Any], manifest_b: dict[str, Any]
) -> dict[str, bool]:
    """Compara dois manifestos de reprodutibilidade campo a campo.

    Parameters
    ----------
    manifest_a : dict[str, Any]
        Primeiro manifesto (ver :func:`build_reproducibility_manifest`).
    manifest_b : dict[str, Any]
        Segundo manifesto, no mesmo formato.

    Returns
    -------
    dict[str, bool]
        Uma entrada por campo presente em ao menos um dos manifestos,
        ``True`` quando os valores coincidem entre os dois manifestos.

    Examples
    --------
    >>> compare_reproducibility_manifests({"git_sha": "abc"}, {"git_sha": "abc"})
    {'git_sha': True}
    >>> compare_reproducibility_manifests({"git_sha": "abc"}, {"git_sha": "def"})
    {'git_sha': False}
    """
    all_keys = set(manifest_a) | set(manifest_b)
    return {key: manifest_a.get(key) == manifest_b.get(key) for key in all_keys}
