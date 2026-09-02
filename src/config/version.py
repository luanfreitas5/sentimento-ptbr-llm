"""Centraliza a versão do projeto e a leitura do ``CHANGELOG.md``.

``pyproject.toml`` é a fonte única de verdade da versão (gerenciada pelo
``commitizen``, que também replica o valor em ``src/__init__.py``). Este
módulo lê ``pyproject.toml`` diretamente para nunca divergir dele.
"""

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10: tomllib só existe a partir do 3.11.
    import tomli as tomllib  # type: ignore[no-redef]

from config.paths import PROJECT_ROOT
from exceptions.configuration import InvalidConfigurationError

DEFAULT_PYPROJECT_FILE: Path = PROJECT_ROOT / "pyproject.toml"
DEFAULT_CHANGELOG_FILE: Path = PROJECT_ROOT / "CHANGELOG.md"

_CHANGELOG_ENTRY_PATTERN = re.compile(r"(## .+?)(?=\n## |\Z)", re.DOTALL)


def get_project_version(pyproject_file_path: Path = DEFAULT_PYPROJECT_FILE) -> str:
    """Lê a versão do projeto declarada em ``pyproject.toml``.

    Parameters
    ----------
    pyproject_file_path : Path, optional
        Caminho do arquivo ``pyproject.toml``, by default
        :data:`DEFAULT_PYPROJECT_FILE`.

    Returns
    -------
    str
        Versão do projeto (ex.: ``"0.2.0"``).

    Raises
    ------
    InvalidConfigurationError
        Se a chave ``[project].version`` não estiver presente no arquivo.

    Examples
    --------
    >>> get_project_version()
    '0.2.0'
    """
    with pyproject_file_path.open("rb") as file:
        dados = tomllib.load(file)
    try:
        return dados["project"]["version"]
    except KeyError as exception:
        raise InvalidConfigurationError(
            f"chave 'project.version' não encontrada em {pyproject_file_path}"
        ) from exception


def get_project_name(pyproject_file_path: Path = DEFAULT_PYPROJECT_FILE) -> str:
    """Lê o nome do projeto declarado em ``pyproject.toml``.

    Parameters
    ----------
    pyproject_file_path : Path, optional
        Caminho do arquivo ``pyproject.toml``, by default
        :data:`DEFAULT_PYPROJECT_FILE`.

    Returns
    -------
    str
        Nome do projeto (ex.: ``"sentimento-ptbr-llm"``).

    Raises
    ------
    InvalidConfigurationError
        Se a chave ``[project].name`` não estiver presente no arquivo.

    Examples
    --------
    >>> get_project_name()
    'sentimento-ptbr-llm'
    """
    with pyproject_file_path.open("rb") as file:
        dados = tomllib.load(file)
    try:
        return dados["project"]["name"]
    except KeyError as exception:
        raise InvalidConfigurationError(
            f"chave 'project.name' não encontrada em {pyproject_file_path}"
        ) from exception


def read_latest_changelog_entry(changelog_file_path: Path = DEFAULT_CHANGELOG_FILE) -> str:
    """Extrai a entrada mais recente do ``CHANGELOG.md`` (formato Keep a Changelog).

    Parameters
    ----------
    changelog_file_path : Path, optional
        Caminho do arquivo de changelog, by default :data:`DEFAULT_CHANGELOG_FILE`.

    Returns
    -------
    str
        Texto da entrada de changelog mais recente (primeiro bloco ``## ...``).
        Retorna string vazia se nenhuma entrada for encontrada.

    Examples
    --------
    >>> read_latest_changelog_entry().startswith("## v0.2.0")
    True
    """
    content = changelog_file_path.read_text(encoding="utf-8")
    recent_changelog_entry = _CHANGELOG_ENTRY_PATTERN.search(content)
    return recent_changelog_entry.group(1).strip() if recent_changelog_entry else ""
