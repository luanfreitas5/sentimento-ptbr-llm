"""Constantes internas do subsistema de configuração.

Não confundir com o pacote ``src/constants/``, que contém constantes de
domínio (colunas, rótulos, métricas). Este módulo guarda apenas valores
usados pela própria infraestrutura de configuração.
"""

DEFAULT_ENCODING = "utf-8"
ENV_FILE_NAME = ".env"
CONFIGS_DIR_NAME = "configs"
ENVIRONMENT_VARIABLE_PREFIX = "SENTIMENTO_"

# Nomes dos arquivos de configuração versionados em ``configs/``.
CONFIG_FILE_NAMES: dict[str, str] = {
    "general": "config.yaml",
    "paths": "paths.yaml",
    "logging": "logging.yaml",
    "model_params": "model_params.yaml",
    "deploy": "deploy.yaml",
    "llm": "llm.yaml",
    "labeling": "labeling.yaml",
    "evaluation": "evaluation.yaml",
}

# Nomes dos pacotes de primeira parte dentro de ``src/`` (raiz de
# importação do projeto), usados por ``config.logging`` para aplicar o
# nível de log do projeto (``project_level``) a todo o código próprio,
# distinguindo-o de bibliotecas de terceiros.
PROJECT_PACKAGE_NAMES: tuple[str, ...] = (
    "app",
    "config",
    "constants",
    "data",
    "evaluation",
    "exceptions",
    "experiment",
    "features",
    "hypothesaes",
    "inference",
    "io_utils",
    "labeling",
    "llm",
    "logging_utils",
    "metrics",
    "models",
    "parallel",
    "pipelines",
    "preprocessing",
    "schemas",
    "training",
    "utils",
    "visualization",
)
