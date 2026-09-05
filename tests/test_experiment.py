"""Testes de rastreamento, registro e reprodutibilidade de experimentos (``src/experiment``)."""

import subprocess
from pathlib import Path

import pytest

from config.paths import PROJECT_ROOT
from exceptions.data import DataValidationError
from exceptions.model import ModelError
from experiment.registry import (
    get_latest_model_version,
    register_model_version,
    transition_model_stage,
)
from experiment.reproducibility import (
    build_reproducibility_manifest,
    collect_library_versions,
    compare_reproducibility_manifests,
    get_current_git_sha,
)
from experiment.tracker import (
    build_experiment_run_metrics_dataframe,
    log_run_artifact,
    log_run_metrics,
    log_run_parameters,
    track_experiment_run,
)


@pytest.fixture
def local_mlflow_tracking(tmp_path: Path) -> None:
    """Configura um MLflow Tracking Store local e isolado (arquivo) para os testes."""
    import mlflow

    mlflow.set_tracking_uri((tmp_path / "mlruns").as_posix())
    mlflow.set_experiment("teste-sentimento-ptbr-llm")


@pytest.fixture
def local_mlflow_registry(tmp_path: Path) -> None:
    """Configura um MLflow Tracking Store local baseado em SQLite, com suporte a Model Registry."""
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("teste-registry")


class TestGetCurrentGitSha:
    """Testes de :func:`experiment.reproducibility.get_current_git_sha`."""

    def test_returns_non_empty_sha(self) -> None:
        """Deve retornar uma string não vazia dentro de um repositório Git."""
        git_sha = get_current_git_sha()
        assert isinstance(git_sha, str)
        assert len(git_sha) > 0

    def test_short_sha_is_shorter_than_full_sha(self) -> None:
        """O SHA abreviado deve ser mais curto que o SHA completo."""
        full_sha = get_current_git_sha(short=False)
        short_sha = get_current_git_sha(short=True)
        assert len(short_sha) < len(full_sha)

    def test_raises_runtime_error_when_git_is_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deve levantar ``RuntimeError`` quando o comando ``git`` falha."""

        def _fake_run(*args: object, **kwargs: object) -> None:
            raise FileNotFoundError("git não encontrado")

        monkeypatch.setattr(subprocess, "run", _fake_run)
        with pytest.raises(RuntimeError, match="git"):
            get_current_git_sha()


class TestCollectLibraryVersions:
    """Testes de :func:`experiment.reproducibility.collect_library_versions`."""

    def test_includes_installed_library(self) -> None:
        """Uma biblioteca instalada deve aparecer no resultado."""
        versions = collect_library_versions(("numpy",))
        assert "numpy" in versions

    def test_omits_uninstalled_library(self) -> None:
        """Uma biblioteca inexistente deve ser omitida, sem levantar exceção."""
        versions = collect_library_versions(("biblioteca-inexistente-xyz",))
        assert versions == {}


class TestBuildReproducibilityManifest:
    """Testes de :func:`experiment.reproducibility.build_reproducibility_manifest`."""

    def test_contains_expected_keys(self) -> None:
        """O manifesto deve conter todas as chaves essenciais de reprodutibilidade."""
        manifest = build_reproducibility_manifest(PROJECT_ROOT / "configs" / "config.yaml")
        assert {
            "git_sha",
            "dataset_hash",
            "python_version",
            "library_versions",
            "random_seed",
        } <= set(manifest.keys())

    def test_merges_extra_metadata(self) -> None:
        """Deve incluir os campos adicionais de ``extra_metadata`` no manifesto."""
        manifest = build_reproducibility_manifest(
            PROJECT_ROOT / "configs" / "config.yaml", extra_metadata={"model_name": "naive_bayes"}
        )
        assert manifest["model_name"] == "naive_bayes"


class TestCompareReproducibilityManifests:
    """Testes de :func:`experiment.reproducibility.compare_reproducibility_manifests`."""

    def test_flags_matching_and_differing_fields(self) -> None:
        """Deve indicar corretamente quais campos coincidem e quais diferem."""
        result = compare_reproducibility_manifests(
            {"git_sha": "abc", "random_seed": 42}, {"git_sha": "abc", "random_seed": 7}
        )
        assert result == {"git_sha": True, "random_seed": False}


class TestTrackExperimentRun:
    """Testes de :func:`experiment.tracker.track_experiment_run`."""

    def test_yields_a_non_empty_run_id(self, local_mlflow_tracking: None) -> None:
        """Deve abrir uma execução MLflow válida e retornar seu ``run_id``."""
        with track_experiment_run("modelo_teste") as run_id:
            assert isinstance(run_id, str)
            assert len(run_id) > 0


class TestLogRunParametersAndMetrics:
    """Testes de :func:`experiment.tracker.log_run_parameters` e :func:`experiment.tracker.log_run_metrics`."""

    def test_logs_parameters_and_metrics_without_raising(self, local_mlflow_tracking: None) -> None:
        """Registrar parâmetros e métricas válidas não deve levantar exceções."""
        with track_experiment_run("modelo_teste"):
            log_run_parameters({"random_state": 42})
            log_run_metrics({"f1_macro": 0.82})

    def test_log_run_metrics_raises_on_unknown_metric_name(
        self, local_mlflow_tracking: None
    ) -> None:
        """Deve levantar ``DataValidationError`` para um nome de métrica desconhecido."""
        with track_experiment_run("modelo_teste"), pytest.raises(DataValidationError):
            log_run_metrics({"metrica_invalida": 0.5})


class TestLogRunArtifact:
    """Testes de :func:`experiment.tracker.log_run_artifact`."""

    def test_logs_artifact_without_raising(
        self, local_mlflow_tracking: None, tmp_path: Path
    ) -> None:
        """Registrar um arquivo local como artefato não deve levantar exceções."""
        artifact_path = tmp_path / "relatorio.txt"
        artifact_path.write_text("conteudo de exemplo", encoding="utf-8")
        with track_experiment_run("modelo_teste"):
            log_run_artifact(artifact_path)


class TestBuildExperimentRunMetricsDataframe:
    """Testes de :func:`experiment.tracker.build_experiment_run_metrics_dataframe`."""

    def test_returns_one_row_per_metric(self) -> None:
        """Deve retornar uma linha por métrica, validada contra o schema do projeto."""
        result = build_experiment_run_metrics_dataframe(
            "run123", "naive_bayes", {"f1_macro": 0.82}, git_sha="abc123", dataset_hash="def456"
        )
        assert result.height == 1
        assert result["metric_name"].to_list() == ["f1_macro"]

    def test_raises_on_unknown_metric_name(self) -> None:
        """Deve levantar ``DataValidationError`` para um nome de métrica desconhecido."""
        with pytest.raises(DataValidationError):
            build_experiment_run_metrics_dataframe(
                "run123",
                "naive_bayes",
                {"metrica_invalida": 0.5},
                git_sha="abc",
                dataset_hash="def",
            )


class TestValidateStage:
    """Testes de validação de estágio, comuns às funções de ``experiment.registry``."""

    def test_transition_model_stage_raises_on_invalid_stage(self) -> None:
        """Deve levantar ``ValueError`` para um estágio de destino desconhecido."""
        with pytest.raises(ValueError, match="stage"):
            transition_model_stage("modelo", "1", "EstagioInvalido")

    def test_get_latest_model_version_raises_on_invalid_stage(self) -> None:
        """Deve levantar ``ValueError`` para um estágio de consulta desconhecido."""
        with pytest.raises(ValueError, match="stage"):
            get_latest_model_version("modelo", stage="EstagioInvalido")


@pytest.mark.integration
class TestModelRegistryLifecycle:
    """Testes de integração do ciclo de vida completo do MLflow Model Registry."""

    def test_register_model_version_returns_version_number(
        self, local_mlflow_registry: None
    ) -> None:
        """Registrar um modelo deve retornar o número da versão criada."""
        import mlflow.sklearn
        from sklearn.dummy import DummyClassifier

        model = DummyClassifier(strategy="most_frequent").fit([[0], [1]], ["positivo", "negativo"])
        with mlflow.start_run() as run:
            mlflow.sklearn.log_model(model, "modelo")
            model_uri = f"runs:/{run.info.run_id}/modelo"

        version_number = register_model_version(model_uri, "modelo-registrado-teste")
        assert version_number == "1"

    def test_register_promote_and_query_latest_version(self, local_mlflow_registry: None) -> None:
        """Deve registrar, promover e consultar a versão mais recente de um modelo."""
        import mlflow.sklearn
        from sklearn.dummy import DummyClassifier

        model = DummyClassifier(strategy="most_frequent").fit([[0], [1]], ["positivo", "negativo"])
        with mlflow.start_run():
            mlflow.sklearn.log_model(model, "modelo", registered_model_name="modelo-teste")

        latest_version = get_latest_model_version("modelo-teste", stage="None")
        transition_model_stage("modelo-teste", latest_version, "Staging")
        assert get_latest_model_version("modelo-teste", stage="Staging") == latest_version

        with pytest.raises(ModelError):
            get_latest_model_version("modelo-teste", stage="Production")
