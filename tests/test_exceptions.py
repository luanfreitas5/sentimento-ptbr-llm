"""Testes da hierarquia de exceções customizadas do projeto."""

import pytest

from exceptions.base import ProjectError
from exceptions.configuration import (
    ConfigurationError,
    ConfigurationFileNotFoundError,
    InvalidConfigurationError,
    MissingEnvironmentVariableError,
)
from exceptions.data import DataError, DataNotFoundError, DataValidationError, EmptyDatasetError
from exceptions.model import (
    ModelError,
    ModelNotFittedError,
    ModelPersistenceError,
    UnsupportedModelError,
)
from exceptions.pipeline import PipelineError, PipelineStageError, UnknownPipelineStageError


class TestProjectError:
    """Testes da exceção-base do projeto."""

    def test_message_without_context(self) -> None:
        """Sem contexto, a mensagem final é exatamente a mensagem informada."""
        project_error = ProjectError("falha simples")
        assert str(project_error) == "falha simples"
        assert project_error.message == "falha simples"
        assert project_error.context == {}

    def test_message_with_context(self) -> None:
        """Com contexto, a mensagem final inclui os pares chave=valor informados."""
        project_error = ProjectError("falha com contexto", context={"arquivo": "dados.csv"})
        assert "falha com contexto" in str(project_error)
        assert "arquivo='dados.csv'" in str(project_error)

    def test_is_exception_subclass(self) -> None:
        """ProjectError deve poder ser capturada como uma Exception genérica."""
        with pytest.raises(Exception):  # noqa: B017 - verificação intencional de hierarquia ampla
            raise ProjectError("erro")


class TestConfigurationExceptions:
    """Testes das exceções de configuração."""

    def test_configuration_error_is_project_error(self) -> None:
        """ConfigurationError deve herdar de ProjectError."""
        assert issubclass(ConfigurationError, ProjectError)

    def test_configuration_file_not_found_error_message(self) -> None:
        """A mensagem deve citar o caminho do arquivo ausente."""
        configuration_error = ConfigurationFileNotFoundError("configs/inexistente.yaml")
        assert "configs/inexistente.yaml" in str(configuration_error)
        assert isinstance(configuration_error, ConfigurationError)

    def test_invalid_configuration_error_message(self) -> None:
        """A mensagem deve citar o detalhe do problema de validação."""
        validation_error = InvalidConfigurationError("campo obrigatório ausente")
        assert "campo obrigatório ausente" in str(validation_error)

    def test_missing_environment_variable_error_message(self) -> None:
        """A mensagem deve citar o nome da variável de ambiente ausente."""
        missing_env_var_error = MissingEnvironmentVariableError("MLFLOW_TRACKING_URI")
        assert "MLFLOW_TRACKING_URI" in str(missing_env_var_error)


class TestDataExceptions:
    """Testes das exceções relacionadas a dados."""

    def test_data_error_is_project_error(self) -> None:
        """DataError deve herdar de ProjectError."""
        assert issubclass(DataError, ProjectError)

    def test_data_not_found_error_message(self) -> None:
        """A mensagem deve citar a fonte de dados ausente."""
        data_error = DataNotFoundError("data/raw/tweets.parquet")
        assert "data/raw/tweets.parquet" in str(data_error)

    def test_empty_dataset_error_message(self) -> None:
        """A mensagem deve citar o dataset vazio."""
        empty_dataset_error = EmptyDatasetError("corpus_treino")
        assert "corpus_treino" in str(empty_dataset_error)

    def test_data_validation_error_message(self) -> None:
        """A mensagem deve citar o nome do schema e o detalhe da falha."""
        validation_error = DataValidationError(
            schema_name="LabeledCorpusSchema", detail="coluna ausente: sentimento"
        )
        assert "LabeledCorpusSchema" in str(validation_error)
        assert "coluna ausente: sentimento" in str(validation_error)


class TestModelExceptions:
    """Testes das exceções relacionadas a modelos."""

    def test_model_error_is_project_error(self) -> None:
        """ModelError deve herdar de ProjectError."""
        assert issubclass(ModelError, ProjectError)

    def test_model_not_fitted_error_message(self) -> None:
        """A mensagem deve citar o nome do modelo não treinado."""
        fitted_model_error = ModelNotFittedError("regressao_logistica")
        assert "regressao_logistica" in str(fitted_model_error)

    def test_model_persistence_error_message(self) -> None:
        """A mensagem deve citar o caminho e o detalhe da falha de persistência."""
        persistence_error = ModelPersistenceError("models/artifacts/modelo.joblib", "disco cheio")
        assert "models/artifacts/modelo.joblib" in str(persistence_error)
        assert "disco cheio" in str(persistence_error)

    def test_unsupported_model_error_message(self) -> None:
        """A mensagem deve citar o modelo solicitado e os modelos disponíveis."""
        unsupported_model_error = UnsupportedModelError(
            "modelo_desconhecido", ["svm", "random_forest"]
        )
        assert "modelo_desconhecido" in str(unsupported_model_error)
        assert "svm" in str(unsupported_model_error)


class TestPipelineExceptions:
    """Testes das exceções relacionadas à execução de pipelines."""

    def test_pipeline_error_is_project_error(self) -> None:
        """PipelineError deve herdar de ProjectError."""
        assert issubclass(PipelineError, ProjectError)

    def test_pipeline_stage_error_message(self) -> None:
        """A mensagem deve citar a etapa e o detalhe da falha."""
        pipeline_stage_error = PipelineStageError("preprocessing", "arquivo de entrada corrompido")
        assert "preprocessing" in str(pipeline_stage_error)
        assert "arquivo de entrada corrompido" in str(pipeline_stage_error)

    def test_unknown_pipeline_stage_error_message(self) -> None:
        """A mensagem deve citar a etapa desconhecida e as etapas disponíveis."""
        unknown_stage_error = UnknownPipelineStageError(
            "etapa_invalida", ["ingestion", "preprocessing"]
        )
        assert "etapa_invalida" in str(unknown_stage_error)
        assert "ingestion" in str(unknown_stage_error)
