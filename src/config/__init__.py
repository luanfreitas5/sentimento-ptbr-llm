"""Infraestrutura de configuração do projeto ``sentimento-ptbr-llm``.

Reúne o carregamento e a validação de configurações (YAML + ``.env``), a
resolução centralizada de caminhos, a configuração de logging e de
reprodutibilidade, e a leitura da versão do projeto.

Modules
-------
constants
    Constantes internas do subsistema de configuração.
environment
    Carregamento de variáveis de ambiente e configuração de reprodutibilidade.
logging
    Configuração central de logging (console Rich + arquivo diário).
paths
    Centralização de todos os caminhos do projeto via ``pathlib.Path``.
settings
    Configurações validadas com Pydantic (``configs/config.yaml`` + ``.env``).
version
    Versão do projeto e leitura do ``CHANGELOG.md``.
"""

from config.environment import (
    configure_environment_variables,
    configure_reproducibility,
    get_required_environment_variable,
)
from config.logging import configure_logging
from config.paths import PROJECT_ROOT, ProjectPaths, load_project_paths, resolve_project_path
from config.settings import GeneralConfig, Settings, create_settings, load_general_config
from config.version import get_project_name, get_project_version, read_latest_changelog_entry

__all__: list[str] = [
    "configure_environment_variables",
    "configure_reproducibility",
    "get_required_environment_variable",
    "configure_logging",
    "PROJECT_ROOT",
    "ProjectPaths",
    "load_project_paths",
    "resolve_project_path",
    "GeneralConfig",
    "Settings",
    "create_settings",
    "load_general_config",
    "get_project_name",
    "get_project_version",
    "read_latest_changelog_entry",
]
