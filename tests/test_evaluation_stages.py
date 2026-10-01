"""Testes dos estágios de treino por categoria, ``evaluate`` e ``report``.

Cobrem o vetorizador TF-IDF persistido, o baseline, a descoberta de checkpoints
por categoria, a ablação, a correção de Holm e o fluxo ponta a ponta
``treino -> evaluate -> report`` com dados sintéticos pequenos. O LLM e os
modelos neurais são substituídos por dublês: nenhum teste usa rede, GPU ou Ollama.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from config.paths import ProjectPaths
from constants.labels import SENTIMENT_CLASSES
from data.loader import load_training_example_dataset
from data.writer import write_training_example_dataset
from evaluation import predictions as predictions_module
from evaluation.ablation import calculate_paired_bootstrap_difference, run_pipeline_ablation
from evaluation.predictions import (
    CATEGORY_CLASSICAL,
    CATEGORY_DEEP_LEARNING,
    CATEGORY_LLM,
    CATEGORY_TRANSFORMER,
    discover_checkpoints,
    normalize_predictions,
)
from exceptions.data import DataNotFoundError, EmptyDatasetError
from features.lexical import LexicalTfidfVectorizer
from models.factory import create_classifier
from models.llm import LLMSentimentClassifier
from models.persistence import save_classifier
from pipelines import training_deep_learning, training_llm
from pipelines.evaluate import adjust_p_values_holm, run_evaluate_stage
from pipelines.features import TFIDF_VECTORIZER_FILE_NAME, run_features_stage
from pipelines.report import run_report_stage
from pipelines.training_classical import run_training_classical_stage
from pipelines.training_llm import (
    LLM_SPEC_SUFFIX,
    run_training_llm_stage,
    select_sample_indices,
)
from pipelines.training_transformer import run_training_transformer_stage
from reporting.tables import (
    build_results_table,
    dataframe_to_latex,
    dataframe_to_markdown,
    format_interval,
)
from visualization.evaluation import plot_ablation_impact, plot_model_comparison

_WORDS = {"positivo": "bom otimo", "negativo": "ruim pessimo", "neutro": "normal comum"}


def _synthetic_split(split: str, n_per_class: int, *, offset: int) -> pl.DataFrame:
    """Gera uma partição sintética, separável por palavras-chave, com ids únicos."""
    rows = [
        {
            "id": str(offset + index * 3 + class_index),
            "text": f"{_WORDS[label]} item{index % 5}",
            "sentiment_label": label,
            "split": split,
        }
        for index in range(n_per_class)
        for class_index, label in enumerate(SENTIMENT_CLASSES)
    ]
    return pl.DataFrame(rows)


@pytest.fixture
def stage_paths(tmp_path: Path) -> ProjectPaths:
    """Caminhos isolados em ``tmp_path`` com as três partições sintéticas já gravadas."""
    paths = ProjectPaths(
        data_raw_dir=tmp_path / "data" / "raw",
        data_external_dir=tmp_path / "data" / "external",
        data_interim_dir=tmp_path / "data" / "interim",
        data_processed_dir=tmp_path / "data" / "processed",
        raw_tweets_file=tmp_path / "data" / "raw" / "tweets.parquet",
        tweetsentbr_file=tmp_path / "data" / "external" / "tweetsentbr.parquet",
        repro_file=tmp_path / "data" / "external" / "repro.parquet",
        normalized_corpus_file=tmp_path / "data" / "interim" / "normalizado.parquet",
        labeled_corpus_file=tmp_path / "data" / "processed" / "rotulado.parquet",
        huggingface_labeled_file=tmp_path / "data" / "processed" / "tweets_hf.parquet",
        openai_labeled_file=tmp_path / "data" / "processed" / "tweets_openai.parquet",
        labeling_checkpoints_dir=tmp_path / "data" / "interim" / "checkpoints",
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
    write_training_example_dataset(
        _synthetic_split("treino", 20, offset=0), paths.training_corpus_file
    )
    write_training_example_dataset(
        _synthetic_split("validacao", 8, offset=1000), paths.validation_corpus_file
    )
    write_training_example_dataset(
        _synthetic_split("teste", 10, offset=2000), paths.test_corpus_file
    )
    return paths


class _FakeNeuralModel:
    """Modelo neural falso (nível de módulo: o callback de checkpoint o serializa com joblib)."""

    def fit(self, X: Sequence[Any], y: Sequence[str]) -> "_FakeNeuralModel":  # noqa: N803
        return self

    def predict(self, X: Sequence[Any]) -> np.ndarray:  # noqa: N803
        return np.array(["positivo"] * len(X))


class _FakeLlmBackend:
    """Backend que "classifica" por palavra-chave, no formato JSON instruído pelo prompt."""

    def generate(self, prompt: str) -> str:
        text = prompt.rsplit('Texto: "', maxsplit=1)[-1]
        label = "positivo" if "bom" in text else "negativo" if "ruim" in text else "neutro"
        return json.dumps({"sentiment_label": label, "confidence_score": 0.9})


def _fake_create_classifier(model_name: str, /, **overrides: Any) -> Any:
    """Dublê de ``create_classifier``: ``llm`` usa o backend falso; os demais, a fábrica real."""
    if model_name != "llm":
        return create_classifier(model_name, **overrides)
    return LLMSentimentClassifier(_FakeLlmBackend(), few_shot=overrides.get("few_shot", True))


def _fit_vectorizer(paths: ProjectPaths) -> LexicalTfidfVectorizer:
    """Ajusta e persiste o vetorizador no treino, como faz a etapa ``features``."""
    train = load_training_example_dataset(paths.training_corpus_file)
    vectorizer = LexicalTfidfVectorizer(
        ngram_range=(1, 1), min_document_frequency=1, max_document_frequency_ratio=1.0
    ).fit(train["text"].to_list())
    save_classifier(vectorizer, paths.models_checkpoints_dir / TFIDF_VECTORIZER_FILE_NAME)
    return vectorizer


class TestLexicalTfidfVectorizer:
    """Testes de :class:`features.lexical.LexicalTfidfVectorizer`."""

    def test_transform_has_one_row_per_text_and_a_column_per_term(self) -> None:
        """A matriz tem ``(n_textos, n_termos)`` e ordem alfabética estável das colunas."""
        vectorizer = LexicalTfidfVectorizer(
            ngram_range=(1, 1), min_document_frequency=1, max_document_frequency_ratio=1.0
        ).fit(["bom dia", "bom produto"])

        matrix = vectorizer.transform(["bom dia", "produto", "nada conhecido"])

        assert matrix.shape == (3, 3)
        assert list(vectorizer.vocabulary_) == ["bom", "dia", "produto"]
        assert matrix[2].sum() == 0  # sem nenhum termo do vocabulário: linha de zeros

    def test_unseen_terms_do_not_leak_into_the_vocabulary(self) -> None:
        """Termos só vistos na transformação são ignorados (sem vazamento do teste)."""
        vectorizer = LexicalTfidfVectorizer(
            ngram_range=(1, 1), min_document_frequency=1, max_document_frequency_ratio=1.0
        ).fit(["bom dia"])

        assert vectorizer.transform(["bom novidade"]).shape == (1, 2)
        assert "novidade" not in vectorizer.vocabulary_

    def test_transform_before_fit_raises(self) -> None:
        """Usar o vetorizador sem ajuste levanta ``RuntimeError`` orientado."""
        with pytest.raises(RuntimeError, match="não ajustado"):
            LexicalTfidfVectorizer().transform(["texto"])

    def test_fit_on_empty_input_raises(self) -> None:
        """Ajustar sem textos levanta ``EmptyDatasetError``."""
        with pytest.raises(EmptyDatasetError):
            LexicalTfidfVectorizer().fit([])


class TestFeaturesStagePersistsVectorizer:
    """A etapa ``features`` grava o vetorizador usado pelos modelos clássicos e por ``evaluate``."""

    def test_vectorizer_is_saved_next_to_the_checkpoints(self, stage_paths: ProjectPaths) -> None:
        """O artefato ``tfidf_vectorizer.joblib`` é gerado e pode ser recarregado."""
        from data.writer import write_labeled_corpus
        from models.persistence import load_classifier

        corpus = pl.DataFrame(
            {
                "id": [str(i) for i in range(30)],
                "text": [_WORDS[SENTIMENT_CLASSES[i % 3]] for i in range(30)],
                "sentiment_label": [SENTIMENT_CLASSES[i % 3] for i in range(30)],
            }
        )
        write_labeled_corpus(corpus, stage_paths.labeled_corpus_file)

        artifacts = run_features_stage(
            stage_paths,
            tfidf_overrides={"min_document_frequency": 1, "max_document_frequency_ratio": 1.0},
        )

        assert artifacts.tfidf_vectorizer_path.name == TFIDF_VECTORIZER_FILE_NAME
        assert load_classifier(artifacts.tfidf_vectorizer_path).vocabulary_


class TestBaselineAndHelpers:
    """Baseline, subamostragem determinística e correção de Holm."""

    def test_dummy_baseline_is_available_in_the_factory(self) -> None:
        """O baseline ``dummy`` sorteia pela distribuição do treino (``stratified``)."""
        model = create_classifier("dummy", strategy="most_frequent")
        model.fit(np.zeros((4, 1)), ["positivo", "positivo", "negativo", "positivo"])
        assert set(model.predict(np.zeros((3, 1)))) == {"positivo"}

    def test_select_sample_indices_is_deterministic_and_bounded(self) -> None:
        """Mesma semente, mesma subamostra; sem limite, todos os índices."""
        assert select_sample_indices(10, None, 1) == list(range(10))
        assert select_sample_indices(10, 20, 1) == list(range(10))
        first = select_sample_indices(100, 10, 7)
        assert first == select_sample_indices(100, 10, 7)
        assert len(set(first)) == 10
        assert first == sorted(first)

    def test_holm_adjustment_is_monotonic_and_capped(self) -> None:
        """Holm: p ajustado >= p bruto, monotônico na ordem de significância e <= 1."""
        adjusted = adjust_p_values_holm([0.01, 0.04, 0.03, 0.9])
        assert adjusted == pytest.approx([0.04, 0.09, 0.09, 0.9])
        assert all(value <= 1.0 for value in adjusted)


class TestDiscoverCheckpoints:
    """Descoberta e categorização dos artefatos de modelo."""

    def test_categories_follow_file_type_and_model_name(self, tmp_path: Path) -> None:
        """``.joblib`` é clássico, ``.pt`` separa DL de Transformer e ``.llm.json`` é LLM."""
        for name in (
            "svm.joblib",
            TFIDF_VECTORIZER_FILE_NAME,
            "lstm.pt",
            "bertimbau.pt",
            f"llama3_2{LLM_SPEC_SUFFIX}",
            "notas.txt",
        ):
            (tmp_path / name).write_text("x")
        (tmp_path / "diagnostics").mkdir()

        found = {item.name: item.category for item in discover_checkpoints(tmp_path)}

        assert found == {
            "svm": CATEGORY_CLASSICAL,
            "lstm": CATEGORY_DEEP_LEARNING,
            "bertimbau": CATEGORY_TRANSFORMER,
            "llama3_2": CATEGORY_LLM,
        }

    def test_filter_by_model_name_and_missing_directory(self, tmp_path: Path) -> None:
        """``model_names`` restringe a lista; diretório inexistente devolve lista vazia."""
        (tmp_path / "svm.joblib").write_text("x")
        (tmp_path / "dummy.joblib").write_text("x")

        assert [item.name for item in discover_checkpoints(tmp_path, ["svm"])] == ["svm"]
        assert discover_checkpoints(tmp_path / "inexistente") == []

    def test_normalize_predictions_decodes_integer_labels_and_reorders_scores(self) -> None:
        """Rótulos inteiros (XGBoost) viram texto; scores seguem ``SENTIMENT_CLASSES``."""

        class _Model:
            classes_ = np.array([2, 0, 1])

        labels, scores = normalize_predictions(
            _Model(), np.array([2, 0]), np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        )

        assert labels == ["positivo", "negativo"]
        assert scores is not None
        assert scores[0].tolist() == pytest.approx([0.2, 0.1, 0.7])  # negativo, neutro, positivo


class TestAblation:
    """Ablação do pipeline clássico."""

    def test_returns_one_row_per_component_with_paired_interval(self) -> None:
        """Cada componente vira uma linha, com impacto e IC pareado ordenado."""
        train = _synthetic_split("treino", 15, offset=0)
        val = _synthetic_split("validacao", 6, offset=1000)

        table = run_pipeline_ablation(
            train["text"].to_list(),
            train["sentiment_label"].to_list(),
            val["text"].to_list(),
            val["sentiment_label"].to_list(),
            components={
                "sem_bigramas": {"tfidf": {"ngram_range": [1, 1]}},
                "sem_class_weight": {"model": {"class_weight": None}},
            },
            tfidf_params={
                "ngram_range": [1, 2],
                "min_document_frequency": 1,
                "max_document_frequency_ratio": 1.0,
            },
            model_params={"max_iter": 200},
            n_bootstrap=20,
        )

        assert set(table["component"]) == {"sem_bigramas", "sem_class_weight"}
        assert (table["impact_ci_low"] <= table["impact_ci_high"]).all()

    def test_empty_components_raise(self) -> None:
        """Sem componentes não há ablação a fazer."""
        with pytest.raises(EmptyDatasetError):
            run_pipeline_ablation(["a"], ["positivo"], ["a"], ["positivo"], components={})

    def test_paired_difference_is_zero_for_identical_predictions(self) -> None:
        """Configurações idênticas têm diferença nula em todas as reamostragens."""
        y_true = ["positivo", "negativo", "neutro"] * 5
        assert calculate_paired_bootstrap_difference(y_true, y_true, y_true, n_bootstrap=20) == (
            0.0,
            0.0,
        )


class TestNeuralStagesDelegation:
    """``training_deep_learning`` e ``training_transformer`` compartilham o mesmo laço de treino."""

    def test_transformer_stage_saves_one_torch_checkpoint_per_model(
        self, stage_paths: ProjectPaths, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cada Transformer é treinado com validação e salvo em ``<modelo>.pt``."""

        saved: list[tuple[Path, str]] = []

        def _fake_save(model: Any, file_path: Path, *, backend: str = "joblib") -> Path:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text("fake")
            saved.append((file_path, backend))
            return file_path

        monkeypatch.setattr(
            training_deep_learning, "create_classifier", lambda *a, **k: _FakeNeuralModel()
        )
        monkeypatch.setattr(training_deep_learning, "save_classifier", _fake_save)

        results = run_training_transformer_stage(
            ["a", "b", "c"],
            ["positivo", "negativo", "neutro"],
            ["a", "b"],
            ["positivo", "negativo"],
            model_names=("bertimbau", "distilbert"),
            checkpoints_dir=stage_paths.models_checkpoints_dir,
        )

        assert set(results) == {"bertimbau", "distilbert"}
        assert all("f1_macro" in result.metrics for result in results.values())
        assert {(path.name, backend) for path, backend in saved} == {
            ("bertimbau.pt", "torch"),
            ("distilbert.pt", "torch"),
        }


class TestTrainingLlmStage:
    """O estágio ``training_llm`` grava a especificação reproduzível e as métricas de validação."""

    def test_writes_spec_with_validation_metrics(
        self, stage_paths: ProjectPaths, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Com o backend falso, o LLM acerta a validação sintética e a especificação é gravada."""
        monkeypatch.setattr(training_llm, "create_classifier", _fake_create_classifier)
        train = load_training_example_dataset(stage_paths.training_corpus_file)
        val = load_training_example_dataset(stage_paths.validation_corpus_file)

        results = run_training_llm_stage(
            train["text"].to_list(),
            train["sentiment_label"].to_list(),
            val["text"].to_list(),
            val["sentiment_label"].to_list(),
            model_params={"llama3_2": {"model_name": "llama3.2", "few_shot": True}},
            checkpoints_dir=stage_paths.models_checkpoints_dir,
            max_validation_samples=12,
        )

        specification = json.loads(
            (stage_paths.models_checkpoints_dir / f"llama3_2{LLM_SPEC_SUFFIX}").read_text()
        )
        assert results["llama3_2"].metrics["f1_macro"] == pytest.approx(1.0)
        assert specification["n_validation_samples"] == 12
        assert specification["overrides"]["model_name"] == "llama3.2"


class TestEvaluateAndReportEndToEnd:
    """Treino sintético -> ``evaluate`` -> ``report``, cobrindo as quatro saídas pedidas."""

    @pytest.fixture
    def trained_paths(
        self, stage_paths: ProjectPaths, monkeypatch: pytest.MonkeyPatch
    ) -> ProjectPaths:
        """Treina baseline + clássicos e prepara o LLM falso nas partições sintéticas."""
        vectorizer = _fit_vectorizer(stage_paths)
        train = load_training_example_dataset(stage_paths.training_corpus_file)
        val = load_training_example_dataset(stage_paths.validation_corpus_file)
        run_training_classical_stage(
            vectorizer.transform(train["text"].to_list()),
            train["sentiment_label"].to_list(),
            vectorizer.transform(val["text"].to_list()),
            val["sentiment_label"].to_list(),
            model_names=("dummy", "naive_bayes", "logistic_regression"),
            checkpoints_dir=stage_paths.models_checkpoints_dir,
        )
        monkeypatch.setattr(training_llm, "create_classifier", _fake_create_classifier)
        monkeypatch.setattr(predictions_module, "create_classifier", _fake_create_classifier)
        run_training_llm_stage(
            train["text"].to_list(),
            train["sentiment_label"].to_list(),
            val["text"].to_list(),
            val["sentiment_label"].to_list(),
            model_params={"llama3_2": {"model_name": "llama3.2"}},
            checkpoints_dir=stage_paths.models_checkpoints_dir,
        )
        return stage_paths

    def test_evaluate_writes_metrics_with_intervals_significance_and_ablation(
        self, trained_paths: ProjectPaths
    ) -> None:
        """Métricas com IC, McNemar com Holm, ablação (na validação) e resumo são gravados."""
        result = run_evaluate_stage(
            trained_paths,
            n_bootstrap=20,
            llm_max_test_samples=12,
            ablation_config={
                "model": "logistic_regression",
                "model_params": {"max_iter": 200},
                "tfidf_params": {
                    "ngram_range": [1, 1],
                    "min_document_frequency": 1,
                    "max_document_frequency_ratio": 1.0,
                },
                "components": {"sem_class_weight": {"model": {"class_weight": None}}},
            },
        )

        table = result.metrics_table
        assert set(table["model"]) == {"dummy", "naive_bayes", "logistic_regression", "llama3_2"}
        assert (table["f1_macro_ci_low"] <= table["f1_macro_ci_high"]).all()
        llm_row = table.filter(pl.col("model") == "llama3_2").row(0, named=True)
        assert llm_row["n_test"] == 12  # LLM só vê a subamostra configurada
        assert llm_row["category"] == CATEGORY_LLM
        dummy_f1 = table.filter(pl.col("model") == "dummy")["f1_macro"][0]
        best_f1 = table["f1_macro"].max()
        assert best_f1 >= dummy_f1  # nenhum modelo pior que o baseline neste problema separável
        assert not result.mcnemar_table.is_empty()
        assert {"p_value_holm", "significant"} <= set(result.mcnemar_table.columns)
        assert result.ablation_table is not None
        assert result.files["ablation"].is_file()
        assert result.summary["best_model"]["model"] == table["model"][0]
        assert set(result.summary["best_by_category"]) == {CATEGORY_CLASSICAL, CATEGORY_LLM}

    def test_skip_ablation_flag_omits_the_ablation(self, trained_paths: ProjectPaths) -> None:
        """``skip_ablation`` dispensa a ablação mesmo com a configuração presente."""
        result = run_evaluate_stage(
            trained_paths,
            n_bootstrap=10,
            ablation_config={"components": {"x": {}}},
            skip_ablation=True,
        )
        assert result.ablation_table is None
        assert "ablation" not in result.files

    def test_evaluate_without_models_points_to_the_make_targets(
        self, stage_paths: ProjectPaths
    ) -> None:
        """Sem nenhum checkpoint, o erro orienta quais alvos do Makefile rodar."""
        with pytest.raises(DataNotFoundError, match="make classical"):
            run_evaluate_stage(stage_paths)

    def test_report_generates_figures_tables_model_cards_and_datasheet(
        self, trained_paths: ProjectPaths
    ) -> None:
        """``report`` consome ``evaluate`` e gera figuras, tabelas, Model Cards e Datasheet."""
        run_evaluate_stage(
            trained_paths,
            n_bootstrap=10,
            llm_max_test_samples=12,
            ablation_config={
                "model": "logistic_regression",
                "model_params": {"max_iter": 200},
                "tfidf_params": {
                    "ngram_range": [1, 1],
                    "min_document_frequency": 1,
                    "max_document_frequency_ratio": 1.0,
                },
                "components": {"sem_class_weight": {"model": {"class_weight": None}}},
            },
        )

        report = run_report_stage(trained_paths)

        figure_names = {path.name for path in report.figures}
        assert {"comparacao_modelos.png", "comparacao_modelos.svg", "ablacao_pipeline.png"} <= (
            figure_names
        )
        assert "matriz_confusao_llama3_2.png" in figure_names
        assert {path.suffix for path in report.tables} == {".csv", ".md", ".tex"}
        card_names = {path.name for path in report.model_cards}
        assert card_names == {
            "model_card_ml_classico_resultados.md",
            "model_card_llm_local_resultados.md",
        }
        classical_card = (
            trained_paths.reports_model_cards_dir / "model_card_ml_classico_resultados.md"
        ).read_text(encoding="utf-8")
        assert "IC 95%" in classical_card
        assert "Ablação" in classical_card
        datasheet = report.datasheet.read_text(encoding="utf-8")
        assert "treino" in datasheet and "teste" in datasheet
        assert "item1" not in datasheet  # nenhum texto de tweet no datasheet (LGPD)

    def test_report_requires_evaluate_first(self, stage_paths: ProjectPaths) -> None:
        """Sem as saídas de ``evaluate``, o erro manda rodar ``make evaluate``."""
        with pytest.raises(DataNotFoundError, match="make evaluate"):
            run_report_stage(stage_paths)


class TestReportingTablesAndFigures:
    """Formatação das tabelas e gráficos da avaliação."""

    def test_format_interval(self) -> None:
        """Valor com e sem intervalo."""
        assert format_interval(0.5, 0.4, 0.6) == "0.500 [0.400, 0.600]"
        assert format_interval(0.5, None, None) == "0.500"

    def test_markdown_and_latex_escape_and_null_handling(self) -> None:
        """Nulos viram traço e o LaTeX escapa ``_`` e ``%``."""
        table = pl.DataFrame(
            {"modelo": ["a_b"], "valor": [None]}, schema={"modelo": pl.String, "valor": pl.Float64}
        )
        assert "| a_b | — |" in dataframe_to_markdown(table)
        latex = dataframe_to_latex(table, caption="100%", label="tab:x")
        assert r"a\_b" in latex and r"100\%" in latex and r"\toprule" in latex

    def test_results_table_requires_rows(self) -> None:
        """Tabela vazia levanta ``EmptyDatasetError``."""
        with pytest.raises(EmptyDatasetError):
            build_results_table(pl.DataFrame())

    def test_plots_reject_empty_tables(self) -> None:
        """Os gráficos recusam tabelas vazias em vez de desenhar eixos vazios."""
        with pytest.raises(EmptyDatasetError):
            plot_model_comparison(pl.DataFrame())
        with pytest.raises(EmptyDatasetError):
            plot_ablation_impact(pl.DataFrame())
