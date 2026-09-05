"""Testes dos pipelines de orquestração ponta a ponta (``src/pipelines``).

Cada estágio é testado isoladamente, com dublês (dublês de teste/fakes)
substituindo dependências pesadas ou externas (rede, LLM local, PyTorch),
para que os testes permaneçam rápidos e determinísticos (ver CLAUDE.md,
"Testing").
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from config.paths import ProjectPaths
from data.loader import read_dataset_file
from data.writer import write_dataset, write_labeled_corpus
from exceptions.data import EmptyDatasetError
from exceptions.pipeline import PipelineStageError, UnknownPipelineStageError
from labeling.automatic import LexicalHeuristicLabeler
from pipelines import training_deep_learning, workflow
from pipelines.comparative_evaluation import run_comparative_evaluation_stage
from pipelines.features import run_features_stage
from pipelines.ingestion import run_ingestion_stage
from pipelines.labeling import run_labeling_stage
from pipelines.llm_evaluation import run_llm_evaluation_stage
from pipelines.preprocessing import run_preprocessing_stage
from pipelines.training_classical import run_training_classical_stage
from pipelines.training_deep_learning import run_training_deep_learning_stage
from pipelines.workflow import run_full_workflow, run_pipeline_stage


@pytest.fixture
def pipeline_paths(tmp_path: Path) -> ProjectPaths:
    """Constrói um :class:`ProjectPaths` isolado em diretório temporário, para testes de pipeline."""
    return ProjectPaths(
        data_raw_dir=tmp_path / "data" / "raw",
        data_external_dir=tmp_path / "data" / "external",
        data_interim_dir=tmp_path / "data" / "interim",
        data_processed_dir=tmp_path / "data" / "processed",
        raw_tweets_file=tmp_path / "data" / "raw" / "tweets.parquet",
        tweetsentbr_file=tmp_path / "data" / "external" / "tweetsentbr.parquet",
        repro_file=tmp_path / "data" / "external" / "repro.parquet",
        normalized_corpus_file=tmp_path / "data" / "interim" / "normalizado.parquet",
        labeled_corpus_file=tmp_path / "data" / "processed" / "rotulado.parquet",
        training_corpus_file=tmp_path / "data" / "processed" / "treino.parquet",
        validation_corpus_file=tmp_path / "data" / "processed" / "validacao.parquet",
        test_corpus_file=tmp_path / "data" / "processed" / "teste.parquet",
        models_checkpoints_dir=tmp_path / "models" / "checkpoints",
        models_artifacts_dir=tmp_path / "models" / "artifacts",
        models_registry_dir=tmp_path / "models" / "registry",
        mlflow_tracking_dir=tmp_path / "mlruns",
        logs_dir=tmp_path / "logs",
        reports_figures_dir=tmp_path / "reports" / "figures",
        reports_tables_dir=tmp_path / "reports" / "tables",
        reports_metrics_dir=tmp_path / "reports" / "metrics",
        reports_statistics_dir=tmp_path / "reports" / "statistics",
        reports_ablation_dir=tmp_path / "reports" / "ablation",
        reports_interpretability_dir=tmp_path / "reports" / "interpretability",
        reports_model_cards_dir=tmp_path / "reports" / "model_cards",
        reports_datasheets_dir=tmp_path / "reports" / "datasheets",
        docs_root_dir=tmp_path / "docs",
        docs_guides_dir=tmp_path / "docs" / "guides",
        docs_assets_dir=tmp_path / "docs" / "assets",
    )


def _fake_scrape_func(query: str) -> list[dict[str, str]]:
    """Dublê de coleta: retorna um único tweet sintético por consulta."""
    return [
        {
            "id": f"{query}-1",
            "text": f"tweet de teste sobre {query}",
            "data_source": "scraping",
            "data_collected": "2026-01-01",
        }
    ]


class TestRunIngestionStage:
    """Testes de :func:`pipelines.ingestion.run_ingestion_stage`."""

    def test_collects_tweets_downloads_gold_sets_and_builds_catalog(
        self, pipeline_paths: ProjectPaths
    ) -> None:
        """Deve gravar o corpus bruto, baixar o gold set informado e montar o catálogo."""
        raw_tweets_path = run_ingestion_stage(
            pipeline_paths,
            scrape_func=_fake_scrape_func,
            queries=["python", "django"],
            external_download_funcs={"tweetsentbr": lambda: b"conteudo-gold-set"},
            max_workers=1,
        )

        assert raw_tweets_path == pipeline_paths.raw_tweets_file
        assert raw_tweets_path.is_file()
        assert pipeline_paths.tweetsentbr_file.read_bytes() == b"conteudo-gold-set"
        assert (pipeline_paths.data_raw_dir / "catalog.json").is_file()

        collected = read_dataset_file(raw_tweets_path)
        assert collected.height == 2

    def test_raises_when_no_tweets_collected(self, pipeline_paths: ProjectPaths) -> None:
        """Deve propagar ``EmptyDatasetError`` quando nenhuma consulta retorna tweets."""
        with pytest.raises(EmptyDatasetError):
            run_ingestion_stage(pipeline_paths, scrape_func=lambda query: [], queries=["python"])


class TestRunPreprocessingStage:
    """Testes de :func:`pipelines.preprocessing.run_preprocessing_stage`."""

    def test_writes_normalized_corpus_with_normalized_text_column(
        self, pipeline_paths: ProjectPaths
    ) -> None:
        """Deve carregar o corpus bruto e gravar o corpus normalizado correspondente."""
        raw_corpus = pl.DataFrame(
            {
                "id": ["1", "2"],
                "text": ["RT @a: muito bom!! 😍", "RT @b: péssimo produto"],
                "data_source": ["scraping", "scraping"],
                "data_collected": ["2026-01-01", "2026-01-01"],
            }
        )
        write_dataset(raw_corpus, pipeline_paths.raw_tweets_file)

        normalized_path = run_preprocessing_stage(pipeline_paths, apply_inclusion_filters=False)

        assert normalized_path == pipeline_paths.normalized_corpus_file
        normalized_corpus = read_dataset_file(normalized_path)
        assert normalized_corpus.height == 2
        assert "text_normalized" in normalized_corpus.columns


class TestRunLabelingStage:
    """Testes de :func:`pipelines.labeling.run_labeling_stage`."""

    def test_writes_labeled_corpus_via_lexical_heuristic_without_human_validation(
        self, pipeline_paths: ProjectPaths
    ) -> None:
        """Deve rotular o corpus via heurística léxica e não gerar amostra de validação humana.

        Com um único rotulador, a razão de concordância é sempre 1.0
        (nenhuma discordância possível), portanto nenhuma amostra deve ser
        sinalizada para validação humana.
        """
        normalized_corpus = pl.DataFrame(
            {
                "id": ["1", "2"],
                "text": ["adorei o produto", "produto pessimo"],
                "text_normalized": ["adorei o produto", "produto pessimo"],
            }
        )
        write_dataset(normalized_corpus, pipeline_paths.normalized_corpus_file)

        labeled_path = run_labeling_stage(
            pipeline_paths, {"heuristica_lexica": LexicalHeuristicLabeler()}
        )

        assert labeled_path == pipeline_paths.labeled_corpus_file
        labeled_corpus = read_dataset_file(labeled_path)
        assert labeled_corpus.sort("id")["sentiment_label"].to_list() == ["positivo", "negativo"]
        assert not (pipeline_paths.reports_tables_dir / "human_validation_sample.csv").is_file()

    def test_applies_human_validation_labels_and_gold_set_without_raising(
        self, pipeline_paths: ProjectPaths
    ) -> None:
        """Deve sobrescrever o rótulo por validação humana e apenas alertar em desacordo com o gold set."""
        normalized_corpus = pl.DataFrame(
            {
                "id": ["1", "2"],
                "text": ["adorei o produto", "produto pessimo"],
                "text_normalized": ["adorei o produto", "produto pessimo"],
            }
        )
        write_dataset(normalized_corpus, pipeline_paths.normalized_corpus_file)
        human_validation_labels = pl.DataFrame({"id": ["1"], "sentiment_label": ["neutro"]})
        gold_set = pl.DataFrame({"id": ["1", "2"], "sentiment_label": ["positivo", "negativo"]})

        labeled_path = run_labeling_stage(
            pipeline_paths,
            {"heuristica_lexica": LexicalHeuristicLabeler()},
            select_for_human_validation=False,
            human_validation_labels=human_validation_labels,
            gold_set=gold_set,
        )

        labeled_corpus = read_dataset_file(labeled_path)
        assert labeled_corpus.filter(pl.col("id") == "1")["sentiment_label"].to_list() == ["neutro"]
        assert labeled_corpus.filter(pl.col("id") == "2")["sentiment_label"].to_list() == [
            "negativo"
        ]


class TestRunFeaturesStage:
    """Testes de :func:`pipelines.features.run_features_stage`."""

    def test_writes_split_corpora_and_tfidf_features(self, pipeline_paths: ProjectPaths) -> None:
        """Deve particionar o corpus rotulado e calcular a matriz TF-IDF do conjunto de treino."""
        n_rows = 10
        labeled_corpus = pl.DataFrame(
            {
                "id": [str(index) for index in range(n_rows)],
                "text": [
                    "bom produto" if index % 2 == 0 else "produto ruim" for index in range(n_rows)
                ],
                "sentiment_label": [
                    "positivo" if index % 2 == 0 else "negativo" for index in range(n_rows)
                ],
            }
        )
        write_labeled_corpus(labeled_corpus, pipeline_paths.labeled_corpus_file)

        artifacts = run_features_stage(
            pipeline_paths,
            tfidf_overrides={"min_document_frequency": 1, "max_document_frequency_ratio": 1.0},
            test_size=0.2,
            validation_size=0.2,
        )

        assert artifacts.training_corpus_path.is_file()
        assert artifacts.validation_corpus_path.is_file()
        assert artifacts.test_corpus_path.is_file()
        assert artifacts.tfidf_features_path.is_file()

        training_split = read_dataset_file(artifacts.training_corpus_path)
        assert training_split.height == 6
        tfidf_features = read_dataset_file(artifacts.tfidf_features_path)
        assert tfidf_features.height > 0


class TestRunTrainingClassicalStage:
    """Testes de :func:`pipelines.training_classical.run_training_classical_stage`."""

    def test_trains_and_saves_each_classical_model(self, pipeline_paths: ProjectPaths) -> None:
        """Deve treinar cada modelo clássico configurado e salvar seu checkpoint em disco."""
        X_train = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [2, 0], [0, 2]])
        y_train = ["positivo", "negativo", "neutro", "negativo", "positivo", "neutro"]
        X_val = np.array([[1, 0], [0, 1]])
        y_val = ["positivo", "negativo"]

        results = run_training_classical_stage(
            X_train,
            y_train,
            X_val,
            y_val,
            model_names=("naive_bayes", "logistic_regression"),
            checkpoints_dir=pipeline_paths.models_checkpoints_dir,
        )

        assert set(results) == {"naive_bayes", "logistic_regression"}
        for model_name, result in results.items():
            assert (pipeline_paths.models_checkpoints_dir / f"{model_name}.joblib").is_file()
            assert "f1_macro" in result.metrics


class _FakeDeepLearningClassifier:
    """Dublê leve de classificador de deep learning, sem dependência de PyTorch."""

    def fit(self, X: Sequence[Any], y: Sequence[str]) -> "_FakeDeepLearningClassifier":
        """Simula o treino sem nenhum cálculo real."""
        return self

    def predict(self, X: Sequence[Any]) -> list[str]:
        """Sempre prediz a classe positiva, para simplicidade do dublê."""
        return ["positivo" for _ in X]

    def predict_proba(self, X: Sequence[Any]) -> list[list[float]]:
        """Retorna uma distribuição de probabilidade fixa por amostra."""
        return [[0.1, 0.1, 0.8] for _ in X]


class TestRunTrainingDeepLearningStage:
    """Testes de :func:`pipelines.training_deep_learning.run_training_deep_learning_stage`."""

    def test_trains_and_saves_each_deep_learning_model_with_fakes(
        self, pipeline_paths: ProjectPaths, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deve treinar e salvar cada modelo, sem depender de PyTorch/Transformers instalados.

        Substitui :func:`models.factory.create_classifier` e
        :func:`models.persistence.save_classifier` por dublês, mantendo o
        teste rápido e livre de dependências pesadas/opcionais.
        """
        saved_paths: list[Path] = []

        def _fake_create_classifier(
            model_name: str, **overrides: Any
        ) -> _FakeDeepLearningClassifier:
            return _FakeDeepLearningClassifier()

        def _fake_save_classifier(model: Any, file_path: Path, *, backend: str = "joblib") -> Path:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text("modelo-fake")
            saved_paths.append(file_path)
            return file_path

        monkeypatch.setattr(training_deep_learning, "create_classifier", _fake_create_classifier)
        monkeypatch.setattr(training_deep_learning, "save_classifier", _fake_save_classifier)

        results = run_training_deep_learning_stage(
            ["texto um", "texto dois", "texto tres"],
            ["positivo", "negativo", "neutro"],
            None,
            None,
            model_names=("lstm",),
            checkpoints_dir=pipeline_paths.models_checkpoints_dir,
        )

        assert "lstm" in results
        assert len(saved_paths) == 1
        assert saved_paths[0].is_file()


class _FakeSentimentClassifier:
    """Dublê de classificador de sentimento, sem dependência de LLM/Ollama."""

    classes_: tuple[str, ...] = ("negativo", "neutro", "positivo")

    def predict(self, X: Sequence[str]) -> list[str]:
        """Classifica pela presença de palavras-chave simples."""
        return [self._label(text) for text in X]

    def predict_proba(self, X: Sequence[str]) -> list[list[float]]:
        """Atribui alta probabilidade à classe predita, distribuindo o restante."""
        return [self._probabilities(text) for text in X]

    def _label(self, text: str) -> str:
        """Determina o rótulo a partir de palavras-chave presentes no texto."""
        if "bom" in text:
            return "positivo"
        if "ruim" in text:
            return "negativo"
        return "neutro"

    def _probabilities(self, text: str) -> list[float]:
        """Monta a distribuição de probabilidade correspondente ao rótulo predito."""
        probabilities = [0.1, 0.1, 0.1]
        probabilities[self.classes_.index(self._label(text))] = 0.8
        return probabilities


class TestRunLlmEvaluationStage:
    """Testes de :func:`pipelines.llm_evaluation.run_llm_evaluation_stage`."""

    def test_classifies_and_evaluates_with_fake_classifier(self) -> None:
        """Deve classificar e avaliar o conjunto de teste usando um classificador injetado."""
        test_dataframe = pl.DataFrame(
            {
                "id": ["1", "2", "3"],
                "text": ["produto muito bom", "isso é ruim", "texto qualquer"],
                "sentiment_label": ["positivo", "negativo", "neutro"],
            }
        )

        predictions, evaluation_result = run_llm_evaluation_stage(
            test_dataframe, classifier=_FakeSentimentClassifier(), max_workers=1
        )

        assert predictions.height == 3
        assert predictions.sort("id")["sentiment_label"].to_list() == [
            "positivo",
            "negativo",
            "neutro",
        ]
        assert evaluation_result.point_metrics["f1_macro"] == pytest.approx(1.0)


class TestRunComparativeEvaluationStage:
    """Testes de :func:`pipelines.comparative_evaluation.run_comparative_evaluation_stage`."""

    def test_builds_merged_report_and_pairwise_tests_for_two_models(self, tmp_path: Path) -> None:
        """Deve mesclar os relatórios e comparar dois modelos via teste de McNemar."""
        y_true = ["positivo", "negativo", "positivo", "negativo", "neutro", "neutro"]
        model_predictions = {
            "model_a": ["positivo", "negativo", "positivo", "negativo", "neutro", "positivo"],
            "model_b": ["positivo", "positivo", "positivo", "negativo", "neutro", "neutro"],
        }
        output_path = tmp_path / "comparativo.csv"

        result = run_comparative_evaluation_stage(
            model_predictions, y_true, output_path=output_path
        )

        assert output_path.is_file()
        assert set(result.pairwise_significance_tests) == {"model_a_vs_model_b"}
        assert result.friedman_test is None
        assert result.nemenyi_test is None
        assert result.slice_reports == {}
        assert set(result.merged_report["model_name"].unique().to_list()) == {
            "model_a",
            "model_b",
        }

    def test_includes_friedman_and_nemenyi_for_three_or_more_models_with_fold_scores(
        self, tmp_path: Path
    ) -> None:
        """Deve incluir Friedman e Nemenyi quando há scores por dobra de três ou mais modelos."""
        y_true = ["positivo", "negativo", "positivo", "negativo", "neutro", "neutro"]
        model_predictions = {
            "model_a": ["positivo", "negativo", "positivo", "negativo", "neutro", "positivo"],
            "model_b": ["positivo", "positivo", "positivo", "negativo", "neutro", "neutro"],
            "model_c": ["negativo", "negativo", "positivo", "positivo", "neutro", "neutro"],
        }
        cross_validation_fold_scores = {
            "model_a": [0.80, 0.82, 0.79],
            "model_b": [0.75, 0.78, 0.74],
            "model_c": [0.70, 0.71, 0.69],
        }

        result = run_comparative_evaluation_stage(
            model_predictions,
            y_true,
            cross_validation_fold_scores=cross_validation_fold_scores,
            output_path=tmp_path / "comparativo_multimodelo.csv",
        )

        assert result.friedman_test is not None
        assert result.nemenyi_test is not None
        assert len(result.pairwise_significance_tests) == 3

    def test_includes_slice_reports_when_slice_labels_informed(self, tmp_path: Path) -> None:
        """Deve avaliar por fatia dos dados quando ``slice_labels`` for informado."""
        y_true = ["positivo", "negativo", "positivo", "negativo", "neutro", "neutro"]
        model_predictions = {
            "model_a": ["positivo", "negativo", "positivo", "negativo", "neutro", "positivo"],
            "model_b": ["positivo", "positivo", "positivo", "negativo", "neutro", "neutro"],
        }
        slice_labels = ["twitter", "twitter", "reddit", "reddit", "twitter", "reddit"]

        result = run_comparative_evaluation_stage(
            model_predictions,
            y_true,
            slice_labels=slice_labels,
            output_path=tmp_path / "comparativo_fatias.csv",
        )

        assert set(result.slice_reports) == {"model_a", "model_b"}
        assert sorted(result.slice_reports["model_a"]["slice"].to_list()) == ["reddit", "twitter"]


class TestRunPipelineStage:
    """Testes de :func:`pipelines.workflow.run_pipeline_stage`."""

    def test_raises_unknown_pipeline_stage_error_for_unregistered_name(self) -> None:
        """Deve levantar ``UnknownPipelineStageError`` para um nome de etapa não registrado."""
        with pytest.raises(UnknownPipelineStageError):
            run_pipeline_stage("etapa_inexistente")

    def test_propagates_project_error_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Deve repropagar sem modificação uma exceção já tipada do projeto."""

        def _fake_stage(**kwargs: Any) -> None:
            raise EmptyDatasetError("fonte_de_teste")

        monkeypatch.setitem(workflow.STAGE_REGISTRY, "fake_stage", _fake_stage)

        with pytest.raises(EmptyDatasetError):
            run_pipeline_stage("fake_stage")

    def test_wraps_unexpected_exception_in_pipeline_stage_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deve encapsular uma exceção inesperada em ``PipelineStageError``."""

        def _fake_stage(**kwargs: Any) -> None:
            raise ValueError("falha inesperada")

        monkeypatch.setitem(workflow.STAGE_REGISTRY, "fake_stage", _fake_stage)

        with pytest.raises(PipelineStageError):
            run_pipeline_stage("fake_stage")

    def test_returns_result_and_forwards_stage_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deve repassar os argumentos recebidos à função do estágio e retornar seu resultado."""

        def _fake_stage(*, value: int) -> int:
            return value * 2

        monkeypatch.setitem(workflow.STAGE_REGISTRY, "fake_stage", _fake_stage)

        assert run_pipeline_stage("fake_stage", value=21) == 42


class TestRunFullWorkflow:
    """Testes de :func:`pipelines.workflow.run_full_workflow`."""

    def test_executes_stages_in_order_and_collects_results(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deve executar os estágios na ordem informada e agregar os resultados por nome."""
        call_order: list[str] = []

        def _fake_stage_a(**kwargs: Any) -> str:
            call_order.append("a")
            return "resultado_a"

        def _fake_stage_b(**kwargs: Any) -> str:
            call_order.append("b")
            return "resultado_b"

        monkeypatch.setitem(workflow.STAGE_REGISTRY, "fake_stage_a", _fake_stage_a)
        monkeypatch.setitem(workflow.STAGE_REGISTRY, "fake_stage_b", _fake_stage_b)

        results = run_full_workflow(["fake_stage_a", "fake_stage_b"])

        assert call_order == ["a", "b"]
        assert results == {"fake_stage_a": "resultado_a", "fake_stage_b": "resultado_b"}

    def test_stops_at_first_failing_stage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Deve interromper a execução (falha rápida) no primeiro estágio que falhar."""

        def _fake_stage_ok(**kwargs: Any) -> str:
            return "ok"

        def _fake_stage_fail(**kwargs: Any) -> None:
            raise ValueError("falha")

        monkeypatch.setitem(workflow.STAGE_REGISTRY, "fake_stage_ok", _fake_stage_ok)
        monkeypatch.setitem(workflow.STAGE_REGISTRY, "fake_stage_fail", _fake_stage_fail)

        with pytest.raises(PipelineStageError):
            run_full_workflow(["fake_stage_ok", "fake_stage_fail", "fake_stage_ok"])
