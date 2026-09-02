"""Configurações do projeto validadas em tempo de execução com Pydantic.

Duas fontes distintas são combinadas:

- :class:`GeneralConfig` — carregada de ``configs/config.yaml`` (versionado
  no Git), validando a configuração geral do projeto.
- :class:`Settings` — carregada de variáveis de ambiente e de um arquivo
  ``.env`` (nunca commitado), contendo segredos e overrides específicos do
  ambiente de execução.

Uma configuração inválida falha aqui, no início da execução, com um erro
tipado e claro — nunca silenciosamente no meio do pipeline.
"""

from pathlib import Path
from typing import Literal

import pydantic
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from config.constants import CONFIG_FILE_NAMES, ENV_FILE_NAME, ENVIRONMENT_VARIABLE_PREFIX
from config.paths import CONFIGS_DIR, PROJECT_ROOT
from constants.labels import SENTIMENT_CLASSES
from exceptions.configuration import InvalidConfigurationError
from io_utils.yaml import read_yaml

DEFAULT_GENERAL_CONFIG_FILE: Path = CONFIGS_DIR / CONFIG_FILE_NAMES["general"]
DEFAULT_ENV_FILE: Path = PROJECT_ROOT / ENV_FILE_NAME


class _StrictBaseModel(BaseModel):
    """Modelo-base que rejeita campos não declarados, para um contrato de configuração estrito."""

    model_config = ConfigDict(extra="forbid")


class ProjectMetadata(_StrictBaseModel):
    """Metadados descritivos do projeto (``configs/config.yaml`` -> ``project``)."""

    name: str
    description: str
    version: str
    language: str


class ReproducibilitySettings(_StrictBaseModel):
    """Parâmetros de reprodutibilidade global (``configs/config.yaml`` -> ``reproducibility``)."""

    random_seed: int = Field(ge=0)
    pythonhashseed: int = Field(ge=0)
    deterministic_algorithms: bool


class ExperimentTrackingSettings(_StrictBaseModel):
    """Configuração de rastreamento de experimentos (``configs/config.yaml`` -> ``experiment``)."""

    name: str
    tracking_uri: str
    registry_stage_default: str


class LabelSettings(_StrictBaseModel):
    """Definição das classes de sentimento e coluna-alvo (``configs/config.yaml`` -> ``labels``)."""

    classes: list[str]
    target_column: str

    @field_validator("classes")
    @classmethod
    def validate_classes_match_known_sentiment_classes(cls, value: list[str]) -> list[str]:
        """Garante que as classes configuradas coincidam com :data:`SENTIMENT_CLASSES`."""
        if set(value) != set(SENTIMENT_CLASSES):
            raise ValueError(f"labels.classes deve corresponder a {SENTIMENT_CLASSES}, recebido: {value}")
        return value


class DataSplitSettings(_StrictBaseModel):
    """Parâmetros de particionamento de dados (``configs/config.yaml`` -> ``data_split``)."""

    test_size: float = Field(gt=0, lt=1)
    validation_size: float = Field(gt=0, lt=1)
    stratify: bool
    random_state: int = Field(ge=0)


class GeneralConfig(_StrictBaseModel):
    """Configuração geral do projeto, espelhando ``configs/config.yaml`` por completo."""

    project: ProjectMetadata
    reproducibility: ReproducibilitySettings
    experiment: ExperimentTrackingSettings
    labels: LabelSettings
    data_split: DataSplitSettings
    stages: list[str] = Field(min_length=1)


def load_general_config(config_file_path: Path = DEFAULT_GENERAL_CONFIG_FILE) -> GeneralConfig:
    """Carrega e valida ``configs/config.yaml`` como :class:`GeneralConfig`.

    Parameters
    ----------
    config_file_path : Path, optional
        Caminho do arquivo de configuração geral, by default
        :data:`DEFAULT_GENERAL_CONFIG_FILE`.

    Returns
    -------
    GeneralConfig
        Configuração geral validada.

    Raises
    ------
    InvalidConfigurationError
        Se o conteúdo do arquivo não satisfizer o contrato de
        :class:`GeneralConfig`.

    Examples
    --------
    >>> load_general_config().labels.target_column
    'sentimento'
    """
    dados = read_yaml(config_file_path)
    try:
        return GeneralConfig.model_validate(dados)
    except pydantic.ValidationError as excecao:
        raise InvalidConfigurationError(str(excecao)) from excecao


class Settings(BaseSettings):
    """Configurações sensíveis ao ambiente, carregadas de variáveis de ambiente/``.env``.

    Nenhum segredo (chaves de API, tokens) deve ser lido de
    ``configs/*.yaml`` — apenas daqui, mantendo o versionamento do Git livre
    de credenciais (ver CLAUDE.md, "Data Privacy & LGPD").
    """

    model_config = SettingsConfigDict(
        env_file=str(DEFAULT_ENV_FILE),
        env_file_encoding="utf-8",
        env_prefix=ENVIRONMENT_VARIABLE_PREFIX,
        extra="ignore",
    )

    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    mlflow_tracking_uri: str | None = None
    ollama_base_url: str | None = None
    huggingface_token: str | None = None


def create_settings(env_file_path: Path | None = None) -> Settings:
    """Cria a instância de :class:`Settings`, opcionalmente a partir de um ``.env`` customizado.

    Parameters
    ----------
    env_file_path : Path | None, optional
        Caminho de um arquivo ``.env`` alternativo (útil em testes); usa
        :data:`DEFAULT_ENV_FILE` quando ``None``, by default None.

    Returns
    -------
    Settings
        Instância de configurações validada.

    Examples
    --------
    >>> create_settings().environment
    'development'
    """
    if env_file_path is not None:
        return Settings(_env_file=env_file_path)  # type: ignore[call-arg]
    return Settings()
