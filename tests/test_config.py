"""Testes da infraestrutura de configuração do projeto (``src/config``)."""

import logging
import os
from pathlib import Path
from typing import Any

import pytest

from config.environment import (
    configure_environment_variables,
    configure_reproducibility,
    get_required_environment_variable,
)
from config.logging import _resolve_log_level, configure_logging
from config.paths import PROJECT_ROOT, load_project_paths, resolve_project_path
from config.settings import Settings, create_settings, load_general_config
from config.version import get_project_name, get_project_version, read_latest_changelog_entry
from exceptions.configuration import InvalidConfigurationError, MissingEnvironmentVariableError
from io_utils.yaml import write_yaml


class TestPaths:
    """Testes da resolução centralizada de caminhos do projeto."""

    def test_project_root_points_to_repository_root(self) -> None:
        """PROJECT_ROOT deve apontar para o diretório que contém pyproject.toml."""
        assert (PROJECT_ROOT / "pyproject.toml").is_file()

    def test_resolve_project_path_joins_with_root(self) -> None:
        """resolve_project_path deve juntar o caminho relativo à raiz do projeto."""
        assert resolve_project_path("configs") == PROJECT_ROOT / "configs"

    def test_load_project_paths_resolves_all_fields(
        self, tmp_path: Path, minimal_paths_config_dict: dict[str, Any]
    ) -> None:
        """Todos os campos de ProjectPaths devem ser resolvidos como caminhos absolutos."""
        caminho_config = tmp_path / "paths.yaml"
        write_yaml(minimal_paths_config_dict, caminho_config)

        caminhos = load_project_paths(caminho_config)

        assert caminhos.data_raw_dir == PROJECT_ROOT / "data/raw"
        assert caminhos.logs_dir == PROJECT_ROOT / "logs"
        assert caminhos.models_registry_dir == PROJECT_ROOT / "models/registry"


class TestEnvironment:
    """Testes de carregamento de variáveis de ambiente e reprodutibilidade."""

    def test_configure_environment_variables_loads_env_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deve carregar variáveis definidas em um arquivo .env customizado."""
        monkeypatch.delenv("VARIAVEL_DE_TESTE_ENV", raising=False)
        arquivo_env = tmp_path / ".env"
        arquivo_env.write_text("VARIAVEL_DE_TESTE_ENV=valor_do_arquivo\n")

        try:
            configure_environment_variables(arquivo_env)
            assert get_required_environment_variable("VARIAVEL_DE_TESTE_ENV") == "valor_do_arquivo"
        finally:
            # load_dotenv escreve diretamente em os.environ, fora do rastreamento
            # do monkeypatch; a limpeza manual evita vazamento entre testes.
            os.environ.pop("VARIAVEL_DE_TESTE_ENV", None)

    def test_configure_environment_variables_warns_when_file_missing(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Deve registrar um aviso, sem levantar exceção, quando o .env não existe."""
        with caplog.at_level(logging.WARNING, logger="config.environment"):
            configure_environment_variables(tmp_path / "inexistente.env")
        assert any("não encontrado" in registro.message for registro in caplog.records)

    def test_get_required_environment_variable_raises_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Deve levantar MissingEnvironmentVariableError se a variável não estiver definida."""
        monkeypatch.delenv("VARIAVEL_AUSENTE_DE_TESTE", raising=False)
        with pytest.raises(MissingEnvironmentVariableError):
            get_required_environment_variable("VARIAVEL_AUSENTE_DE_TESTE")

    def test_configure_reproducibility_makes_random_state_deterministic(self) -> None:
        """A mesma semente aplicada duas vezes deve produzir a mesma sequência aleatória."""
        import random

        configure_reproducibility(7)
        primeira_sequencia = [random.random() for _ in range(3)]

        configure_reproducibility(7)
        segunda_sequencia = [random.random() for _ in range(3)]

        assert primeira_sequencia == segunda_sequencia


class TestLoggingConfig:
    """Testes da configuração central de logging."""

    def test_resolve_log_level_accepts_known_level(self) -> None:
        """Deve converter o nome de um nível conhecido em seu valor numérico."""
        assert _resolve_log_level("INFO") == logging.INFO

    def test_resolve_log_level_rejects_unknown_level(self) -> None:
        """Deve levantar InvalidConfigurationError para um nome de nível desconhecido."""
        with pytest.raises(InvalidConfigurationError):
            _resolve_log_level("NIVEL_INEXISTENTE")

    def test_configure_logging_attaches_console_and_file_handlers(
        self,
        tmp_path: Path,
        minimal_logging_config_dict: dict[str, Any],
        reset_root_logger: logging.Logger,
    ) -> None:
        """Deve anexar handlers de console e arquivo ao logger raiz e definir seu nível."""
        caminho_config = tmp_path / "logging.yaml"
        write_yaml(minimal_logging_config_dict, caminho_config)
        diretorio_logs = tmp_path / "logs"

        configure_logging(caminho_config, logs_directory=diretorio_logs)

        assert len(reset_root_logger.handlers) == 2
        assert reset_root_logger.level == logging.WARNING
        assert any(diretorio_logs.glob("log_*.log"))

    def test_configure_logging_respects_disabled_handlers(
        self,
        tmp_path: Path,
        minimal_logging_config_dict: dict[str, Any],
        reset_root_logger: logging.Logger,
    ) -> None:
        """Quando console e arquivo estão desabilitados, nenhum handler deve ser anexado."""
        minimal_logging_config_dict["console"]["enabled"] = False
        minimal_logging_config_dict["file"]["enabled"] = False
        caminho_config = tmp_path / "logging.yaml"
        write_yaml(minimal_logging_config_dict, caminho_config)

        configure_logging(caminho_config, logs_directory=tmp_path / "logs")

        assert reset_root_logger.handlers == []


class TestSettings:
    """Testes do carregamento e validação de configurações."""

    def test_load_general_config_accepts_valid_dict(
        self, tmp_path: Path, minimal_general_config_dict: dict[str, Any]
    ) -> None:
        """Um config.yaml válido deve ser carregado como GeneralConfig."""
        caminho_config = tmp_path / "config.yaml"
        write_yaml(minimal_general_config_dict, caminho_config)

        configuracao = load_general_config(caminho_config)

        assert configuracao.labels.target_column == "sentimento"
        assert configuracao.reproducibility.random_seed == 42

    def test_load_general_config_rejects_unknown_sentiment_classes(
        self, tmp_path: Path, minimal_general_config_dict: dict[str, Any]
    ) -> None:
        """Classes de sentimento divergentes das conhecidas devem ser rejeitadas."""
        minimal_general_config_dict["labels"]["classes"] = ["ruim", "bom"]
        caminho_config = tmp_path / "config.yaml"
        write_yaml(minimal_general_config_dict, caminho_config)

        with pytest.raises(InvalidConfigurationError):
            load_general_config(caminho_config)

    def test_load_general_config_rejects_out_of_range_test_size(
        self, tmp_path: Path, minimal_general_config_dict: dict[str, Any]
    ) -> None:
        """test_size fora do intervalo (0, 1) deve ser rejeitado."""
        minimal_general_config_dict["data_split"]["test_size"] = 1.5
        caminho_config = tmp_path / "config.yaml"
        write_yaml(minimal_general_config_dict, caminho_config)

        with pytest.raises(InvalidConfigurationError):
            load_general_config(caminho_config)

    def test_load_general_config_rejects_unexpected_extra_field(
        self, tmp_path: Path, minimal_general_config_dict: dict[str, Any]
    ) -> None:
        """Um campo de nível raiz não declarado deve ser rejeitado (extra='forbid')."""
        minimal_general_config_dict["campo_desconhecido"] = "valor"
        caminho_config = tmp_path / "config.yaml"
        write_yaml(minimal_general_config_dict, caminho_config)

        with pytest.raises(InvalidConfigurationError):
            load_general_config(caminho_config)

    def test_create_settings_reads_values_from_custom_env_file(self, tmp_path: Path) -> None:
        """Settings deve ler variáveis de um arquivo .env customizado, com o prefixo do projeto."""
        arquivo_env = tmp_path / ".env"
        arquivo_env.write_text("SENTIMENTO_ENVIRONMENT=production\nSENTIMENTO_LOG_LEVEL=DEBUG\n")

        configuracoes = create_settings(arquivo_env)

        assert isinstance(configuracoes, Settings)
        assert configuracoes.environment == "production"
        assert configuracoes.log_level == "DEBUG"

    def test_create_settings_uses_defaults_when_env_file_absent(self, tmp_path: Path) -> None:
        """Sem um .env, Settings deve usar os valores padrão sem levantar exceção."""
        configuracoes = create_settings(tmp_path / "inexistente.env")
        assert configuracoes.environment == "development"
        assert configuracoes.mlflow_tracking_uri is None


class TestVersion:
    """Testes de leitura da versão do projeto e do changelog."""

    def test_get_project_version_matches_pyproject(self) -> None:
        """A versão lida deve corresponder à declarada em pyproject.toml."""
        assert get_project_version() == "0.2.0"

    def test_get_project_name_matches_pyproject(self) -> None:
        """O nome lido deve corresponder ao declarado em pyproject.toml."""
        assert get_project_name() == "sentimento-ptbr-llm"

    def test_get_project_version_raises_for_missing_version_key(self, tmp_path: Path) -> None:
        """Deve levantar InvalidConfigurationError se a chave project.version estiver ausente."""
        pyproject_incompleto = tmp_path / "pyproject.toml"
        pyproject_incompleto.write_text('[project]\nname = "exemplo"\n')

        with pytest.raises(InvalidConfigurationError):
            get_project_version(pyproject_incompleto)

    def test_read_latest_changelog_entry_returns_most_recent_section(self) -> None:
        """Deve retornar a primeira seção '## ...' do CHANGELOG.md real do projeto."""
        entrada = read_latest_changelog_entry()
        assert entrada.startswith("## v0.2.0")

    def test_read_latest_changelog_entry_returns_empty_string_when_no_entries(self, tmp_path: Path) -> None:
        """Deve retornar string vazia se o changelog não tiver nenhuma seção '## '."""
        changelog_vazio = tmp_path / "CHANGELOG.md"
        changelog_vazio.write_text("# Changelog\n\nNenhuma entrada ainda.\n")

        assert read_latest_changelog_entry(changelog_vazio) == ""
