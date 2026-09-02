"""Exceções customizadas do projeto ``sentimento-ptbr-llm``.

Centraliza a hierarquia de exceções usada em todo o projeto, evitando o uso
de exceções genéricas (``Exception``, ``ValueError`` cru) fora dos limites
do sistema. Todas herdam de :class:`exceptions.base.ProjectError`.

Modules
-------
base
    Define :class:`ProjectError`, a exceção-base de todo o projeto.
configuration
    Erros de carregamento/validação de configuração e variáveis de ambiente.
data
    Erros de ausência, vazio ou violação de contrato de dados.
model
    Erros de treinamento, uso indevido e persistência de modelos.
pipeline
    Erros de execução e orquestração de etapas de pipeline.
"""

from exceptions.base import ProjectError
from exceptions.configuration import (
    ConfigurationError,
    ConfigurationFileNotFoundError,
    InvalidConfigurationError,
    MissingEnvironmentVariableError,
)
from exceptions.data import DataError, DataNotFoundError, DataValidationError, EmptyDatasetError
from exceptions.model import ModelError, ModelNotFittedError, ModelPersistenceError, UnsupportedModelError
from exceptions.pipeline import PipelineError, PipelineStageError, UnknownPipelineStageError

__all__: list[str] = [
    "ProjectError",
    "ConfigurationError",
    "ConfigurationFileNotFoundError",
    "InvalidConfigurationError",
    "MissingEnvironmentVariableError",
    "DataError",
    "DataNotFoundError",
    "DataValidationError",
    "EmptyDatasetError",
    "ModelError",
    "ModelNotFittedError",
    "ModelPersistenceError",
    "UnsupportedModelError",
    "PipelineError",
    "PipelineStageError",
    "UnknownPipelineStageError",
]
