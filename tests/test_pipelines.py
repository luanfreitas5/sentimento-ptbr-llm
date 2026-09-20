"""Testes dos pipelines de orquestração ponta a ponta (``src/pipelines``).

Cada estágio é testado isoladamente, com dublês (dublês de teste/fakes)
substituindo dependências pesadas ou externas (rede, LLM local, PyTorch),
para que os testes permaneçam rápidos e determinísticos (ver CLAUDE.md,
"Testing").
"""

from collections.abc import Sequence
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from config.paths import ProjectPaths
from data.loader import read_dataset_file
from data.writer import write_dataset, write_labeled_corpus
from exceptions.configuration import InvalidConfigurationError
from exceptions.data import DataNotFoundError, DataValidationError, EmptyDatasetError
from exceptions.pipeline import (
    IncompleteLabelingError,
    PipelineStageError,
    UnknownPipelineStageError,
)
from io_utils.json import read_json
from pipelines import hypothesaes_analysis, training_deep_learning, workflow
from pipelines.comparative_evaluation import run_comparative_evaluation_stage
from pipelines.features import run_features_stage
from pipelines.hypothesaes_analysis import run_hypothesaes_analysis_stage
from pipelines.ingestion import run_ingestion_stage
from pipelines.labeling import LabelingSource, run_labeling_stage
from pipelines.preprocessing import run_preprocessing_stage
from pipelines.training_classical import run_training_classical_stage
from pipelines.training_deep_learning import run_training_deep_learning_stage
from pipelines.workflow import run_full_workflow, run_pipeline_stage


@pytest.fixture
def pipeline_paths(tmp_path: Path) -> ProjectPaths:
    """Constrói um :class:`ProjectPaths` isolado em diretório temporário,
    para testes de pipeline."""
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

    def test_loads_raw_batch_and_writes_normalized_corpus(
        self, pipeline_paths: ProjectPaths
    ) -> None:
        """Deve carregar o lote bruto de data/raw/ e gravar o corpus normalizado correspondente."""
        raw_batch = pl.DataFrame(
            {
                "tweet_id": ["1", "2"],
                "user_id": ["u1", "u1"],
                "text": ["RT @a: muito bom!! 😍", "RT @b: péssimo produto"],
                "created_at": [datetime(2026, 1, 1), datetime(2026, 1, 1)],
                "language": ["pt", "pt"],
                "is_reply": [False, False],
                "is_retweet": [False, False],
                "like_count": [0, 0],
                "reply_count": [0, 0],
                "retweet_count": [0, 0],
                "quote_count": [0, 0],
                "source_query": ["teste", "teste"],
                "source_group": ["teste", "teste"],
            }
        )
        write_dataset(raw_batch, pipeline_paths.data_raw_dir / "usuario_teste.parquet")

        normalized_path = run_preprocessing_stage(
            pipeline_paths, show_progress=False, apply_inclusion_filters=False
        )

        assert normalized_path == pipeline_paths.normalized_corpus_file
        normalized_corpus = read_dataset_file(normalized_path)
        assert normalized_corpus.height == 2
        assert "text_normalized" in normalized_corpus.columns

    def test_excludes_retweets_before_normalization(self, pipeline_paths: ProjectPaths) -> None:
        """Um tweet marcado como retweet deve ser removido antes da normalização."""
        raw_batch = pl.DataFrame(
            {
                "tweet_id": ["1", "2"],
                "user_id": ["u1", "u1"],
                "text": ["muito bom o produto", "RT texto duplicado"],
                "created_at": [datetime(2026, 1, 1), datetime(2026, 1, 1)],
                "language": ["pt", "pt"],
                "is_reply": [False, False],
                "is_retweet": [False, True],
                "like_count": [0, 0],
                "reply_count": [0, 0],
                "retweet_count": [0, 0],
                "quote_count": [0, 0],
                "source_query": ["teste", "teste"],
                "source_group": ["teste", "teste"],
            }
        )
        write_dataset(raw_batch, pipeline_paths.data_raw_dir / "usuario_teste.parquet")

        run_preprocessing_stage(pipeline_paths, show_progress=False, apply_inclusion_filters=False)

        normalized_corpus = read_dataset_file(pipeline_paths.normalized_corpus_file)
        assert normalized_corpus.height == 1


def _keyword_classifier(texts: Sequence[str]) -> list[tuple[str, float] | None]:
    """Dublê de classificador em lote: positivo (0.9) se houver "adorei"; negativo (0.8) senão."""
    return [("positivo", 0.9) if "adorei" in text else ("negativo", 0.8) for text in texts]


def _build_source(
    name: str = "huggingface", classifier: Any = _keyword_classifier, batch_size: int = 8
) -> LabelingSource:
    """Monta uma fonte de rotulagem com um classificador injetado (sem modelo nem rede)."""
    return LabelingSource(
        name=name,
        model_name=f"modelo-{name}",
        prompt_name="prompt_de_teste",
        prompt_template='Tweet: "{{TEXTO}}"',
        temperature=0.0,
        batch_size=batch_size,
        open_classifier=lambda: nullcontext(classifier),
    )


def _write_normalized_corpus(paths: ProjectPaths) -> pl.DataFrame:
    """Grava um corpus normalizado sintético de dois tweets."""
    normalized_corpus = pl.DataFrame(
        {
            "id": ["1", "2"],
            "text": ["adorei o produto @fulano", "produto pessimo"],
            "text_normalized": ["adorei o produto [MENCAO]", "produto pessimo"],
        }
    )
    write_dataset(normalized_corpus, paths.normalized_corpus_file)
    return normalized_corpus


class TestRunLabelingStage:
    """Testes de :func:`pipelines.labeling.run_labeling_stage`."""

    def test_writes_one_independent_base_per_source_with_standard_columns(
        self, pipeline_paths: ProjectPaths
    ) -> None:
        """Cada fonte gera sua própria base, com as mesmas colunas padronizadas e os mesmos ids."""
        _write_normalized_corpus(pipeline_paths)

        written = run_labeling_stage(
            pipeline_paths,
            [_build_source("huggingface"), _build_source("openai")],
            select_for_human_validation=False,
            show_progress=False,
        )

        hf_base = read_dataset_file(written["huggingface"])
        oa_base = read_dataset_file(written["openai"])
        assert written["huggingface"] == pipeline_paths.huggingface_labeled_file
        assert written["openai"] == pipeline_paths.openai_labeled_file
        for base in (hf_base, oa_base):
            assert base.columns == [
                "id",
                "text",
                "text_normalized",
                "sentiment_label",
                "confidence_score",
            ]
        assert hf_base["id"].to_list() == oa_base["id"].to_list() == ["1", "2"]
        assert hf_base["sentiment_label"].to_list() == ["positivo", "negativo"]
        assert hf_base["confidence_score"].to_list() == [0.9, 0.8]

    def test_sends_sanitized_text_to_the_model_but_keeps_original_text_in_the_base(
        self, pipeline_paths: ProjectPaths
    ) -> None:
        """O LLM recebe ``text_normalized`` (LGPD); a base guarda o ``text`` original."""
        _write_normalized_corpus(pipeline_paths)
        received: list[str] = []

        def _recording_classifier(texts: Sequence[str]) -> list[tuple[str, float] | None]:
            received.extend(texts)
            return _keyword_classifier(texts)

        run_labeling_stage(
            pipeline_paths,
            [_build_source(classifier=_recording_classifier)],
            select_for_human_validation=False,
            show_progress=False,
        )

        assert received == ["adorei o produto [MENCAO]", "produto pessimo"]
        base = read_dataset_file(pipeline_paths.huggingface_labeled_file)
        assert base["text"].to_list()[0] == "adorei o produto @fulano"

    def test_records_model_and_prompt_in_metadata_file(self, pipeline_paths: ProjectPaths) -> None:
        """Modelo, prompt e hash dos dados ficam num ``.meta.json`` ao lado da base."""
        _write_normalized_corpus(pipeline_paths)

        run_labeling_stage(
            pipeline_paths,
            [_build_source("openai")],
            select_for_human_validation=False,
            show_progress=False,
        )

        metadata = read_json(pipeline_paths.openai_labeled_file.with_suffix(".meta.json"))
        assert metadata["model"] == "modelo-openai"
        assert metadata["prompt_name"] == "prompt_de_teste"
        assert metadata["n_tweets"] == 2
        assert len(metadata["normalized_corpus_sha256"]) == 64

    def test_builds_downstream_corpus_from_the_configured_source(
        self, pipeline_paths: ProjectPaths
    ) -> None:
        """``sentiment_label`` vem da fonte de ``downstream_source``; as duas ficam preservadas."""
        _write_normalized_corpus(pipeline_paths)

        def _always_neutral(texts: Sequence[str]) -> list[tuple[str, float] | None]:
            return [("neutro", 0.6) for _ in texts]

        written = run_labeling_stage(
            pipeline_paths,
            [_build_source("huggingface"), _build_source("openai", classifier=_always_neutral)],
            downstream_source="openai",
            select_for_human_validation=False,
            show_progress=False,
        )

        labeled_corpus = read_dataset_file(written["labeled_corpus"]).sort("id")
        assert labeled_corpus["sentiment_label"].to_list() == ["neutro", "neutro"]
        assert labeled_corpus["sentiment_label_huggingface"].to_list() == ["positivo", "negativo"]
        assert labeled_corpus["sentiment_label_openai"].to_list() == ["neutro", "neutro"]

    def test_skips_downstream_corpus_while_its_source_is_missing(
        self, pipeline_paths: ProjectPaths
    ) -> None:
        """Sem a base da fonte ``downstream_source``, o corpus das etapas seguintes não é gerado."""
        _write_normalized_corpus(pipeline_paths)

        written = run_labeling_stage(
            pipeline_paths,
            [_build_source("openai")],
            downstream_source="huggingface",
            show_progress=False,
        )

        assert set(written) == {"openai"}
        assert not pipeline_paths.labeled_corpus_file.exists()

    def test_resumes_from_checkpoint_without_relabeling_finished_tweets(
        self, pipeline_paths: ProjectPaths
    ) -> None:
        """Após uma falha parcial, a nova execução só reprocessa os tweets pendentes."""
        _write_normalized_corpus(pipeline_paths)

        def _fails_on_second_tweet(texts: Sequence[str]) -> list[tuple[str, float] | None]:
            return [None if "pessimo" in text else ("positivo", 0.9) for text in texts]

        with pytest.raises(IncompleteLabelingError):
            run_labeling_stage(
                pipeline_paths,
                [_build_source(classifier=_fails_on_second_tweet)],
                select_for_human_validation=False,
                show_progress=False,
            )
        assert not pipeline_paths.huggingface_labeled_file.exists()

        received: list[str] = []

        def _recording_classifier(texts: Sequence[str]) -> list[tuple[str, float] | None]:
            received.extend(texts)
            return [("negativo", 0.7) for _ in texts]

        run_labeling_stage(
            pipeline_paths,
            [_build_source(classifier=_recording_classifier)],
            select_for_human_validation=False,
            show_progress=False,
        )

        assert received == ["produto pessimo"]
        base = read_dataset_file(pipeline_paths.huggingface_labeled_file)
        assert base["sentiment_label"].to_list() == ["positivo", "negativo"]

    def test_raises_data_validation_error_for_label_outside_the_classes(
        self, pipeline_paths: ProjectPaths
    ) -> None:
        """Um rótulo fora das classes do projeto nunca chega a ser gravado na base."""
        _write_normalized_corpus(pipeline_paths)

        def _invalid_label(texts: Sequence[str]) -> list[tuple[str, float] | None]:
            return [("muito_positivo", 0.9) for _ in texts]

        with pytest.raises(DataValidationError):
            run_labeling_stage(
                pipeline_paths,
                [_build_source(classifier=_invalid_label)],
                show_progress=False,
            )
        assert not pipeline_paths.huggingface_labeled_file.exists()

    def test_raises_for_unknown_downstream_source(self, pipeline_paths: ProjectPaths) -> None:
        """Uma fonte de ``downstream_source`` desconhecida deve falhar com erro de configuração."""
        _write_normalized_corpus(pipeline_paths)

        with pytest.raises(InvalidConfigurationError):
            run_labeling_stage(
                pipeline_paths,
                [_build_source()],
                downstream_source="gemini",
                show_progress=False,
            )

    def test_applies_human_validation_labels_and_gold_set_without_raising(
        self, pipeline_paths: ProjectPaths
    ) -> None:
        """Deve sobrescrever o rótulo por validação humana e apenas alertar
        em desacordo com o gold set."""
        _write_normalized_corpus(pipeline_paths)
        human_validation_labels = pl.DataFrame({"id": ["1"], "sentiment_label": ["neutro"]})
        gold_set = pl.DataFrame({"id": ["1", "2"], "sentiment_label": ["positivo", "negativo"]})

        written = run_labeling_stage(
            pipeline_paths,
            [_build_source()],
            select_for_human_validation=False,
            human_validation_labels=human_validation_labels,
            gold_set=gold_set,
            show_progress=False,
        )

        labeled_corpus = read_dataset_file(written["labeled_corpus"])
        assert labeled_corpus.filter(pl.col("id") == "1")["sentiment_label"].to_list() == ["neutro"]
        assert labeled_corpus.filter(pl.col("id") == "2")["sentiment_label"].to_list() == [
            "negativo"
        ]

    def test_does_not_flag_confident_samples_for_human_validation(
        self, pipeline_paths: ProjectPaths
    ) -> None:
        """Com confiança acima do limiar padrão (0.5) não há amostra de validação humana."""
        _write_normalized_corpus(pipeline_paths)

        run_labeling_stage(pipeline_paths, [_build_source()], show_progress=False)

        assert not (pipeline_paths.reports_tables_dir / "human_validation_sample.csv").is_file()

    def test_flags_low_confidence_sample_for_human_validation(
        self, pipeline_paths: ProjectPaths
    ) -> None:
        """Uma amostra com confiança abaixo do limiar deve gerar a amostra de validação humana."""
        _write_normalized_corpus(pipeline_paths)

        run_labeling_stage(
            pipeline_paths,
            [_build_source()],
            low_confidence_threshold=0.85,
            human_validation_sample_size=1,
            show_progress=False,
        )

        assert (pipeline_paths.reports_tables_dir / "human_validation_sample.csv").is_file()

    def test_raises_for_empty_normalized_corpus(self, pipeline_paths: ProjectPaths) -> None:
        """Um corpus normalizado vazio deve levantar ``EmptyDatasetError``."""
        pipeline_paths.normalized_corpus_file.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"id": [], "text": [], "text_normalized": []}).cast(
            {"id": pl.String, "text": pl.String, "text_normalized": pl.String}
        ).write_parquet(pipeline_paths.normalized_corpus_file)

        with pytest.raises(EmptyDatasetError):
            run_labeling_stage(pipeline_paths, [_build_source()], show_progress=False)


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
        X_train = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [2, 0], [0, 2]])  # noqa: N806
        y_train = ["positivo", "negativo", "neutro", "negativo", "positivo", "neutro"]
        X_val = np.array([[1, 0], [0, 1]])  # noqa: N806
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

    def fit(self, X: Sequence[Any], y: Sequence[str]) -> "_FakeDeepLearningClassifier":  # noqa: N803
        """Simula o treino sem nenhum cálculo real."""
        return self

    def predict(self, X: Sequence[Any]) -> list[str]:  # noqa: N803
        """Sempre prediz a classe positiva, para simplicidade do dublê."""
        return ["positivo" for _ in X]

    def predict_proba(self, X: Sequence[Any]) -> list[list[float]]:  # noqa: N803
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


_LABEL_CYCLE_HF = ["positivo", "negativo", "neutro", "positivo", "negativo", "neutro"]
_LABEL_CYCLE_OPENAI = ["positivo", "negativo", "positivo", "positivo", "neutro", "neutro"]


def _write_comparison_bases(paths: ProjectPaths, n_tweets: int = 60) -> None:
    """Grava as duas bases rotuladas sintéticas (mesmos tweets; 4 de cada 6 concordam)."""
    ids = [str(index) for index in range(n_tweets)]
    texts = [" ".join(["palavra"] * (1 + index % 9)) for index in range(n_tweets)]
    for path, cycle, confidence_shift in (
        (paths.huggingface_labeled_file, _LABEL_CYCLE_HF, 0.0),
        (paths.openai_labeled_file, _LABEL_CYCLE_OPENAI, -0.1),
    ):
        write_dataset(
            pl.DataFrame(
                {
                    "id": ids,
                    "text": texts,
                    "text_normalized": texts,
                    "sentiment_label": [cycle[index % 6] for index in range(n_tweets)],
                    "confidence_score": [
                        round(min(1.0, max(0.0, 0.55 + 0.07 * (index % 6) + confidence_shift)), 4)
                        for index in range(n_tweets)
                    ],
                }
            ),
            path,
        )


class TestRunComparativeEvaluationStage:
    """Testes de :func:`pipelines.comparative_evaluation.run_comparative_evaluation_stage`."""

    def test_runs_full_analysis_and_writes_tables_figures_and_metrics(
        self, pipeline_paths: ProjectPaths
    ) -> None:
        """Uma única chamada carrega, valida, compara e grava tabelas, gráficos e métricas."""
        _write_comparison_bases(pipeline_paths)

        result = run_comparative_evaluation_stage(
            pipeline_paths, n_bootstrap=30, n_length_bins=3, run_hypotheses=False
        )

        expected_tables = {
            "distribuicao_classes",
            "matriz_concordancia",
            "confianca_por_modelo",
            "maiores_divergencias",
            "conflitos_de_confianca",
            "concordancia_por_tamanho",
            "casos_ambiguos",
            "resumo_ambiguidade",
            "transicoes_divergencia",
            "exemplos_transicoes",
        }
        assert set(result.tables) == expected_tables
        assert all(path.is_file() for path in result.tables.values())
        assert result.metrics_path.is_file()
        assert (
            pipeline_paths.reports_tables_dir / "comparativo_hf_openai" / "resumo_comparativo.md"
        ).is_file()
        figure_names = {path.name for path in result.figures}
        assert {"distribuicao_classes.png", "matriz_concordancia.svg"} <= figure_names
        assert all(path.is_file() for path in result.figures)

    def test_agreement_metrics_match_the_synthetic_bases(
        self, pipeline_paths: ProjectPaths
    ) -> None:
        """Em 4 de cada 6 tweets os modelos concordam: concordância = 2/3 e divergência = 1/3."""
        _write_comparison_bases(pipeline_paths)

        result = run_comparative_evaluation_stage(
            pipeline_paths, n_bootstrap=30, run_hypotheses=False
        )

        agreement = result.summary["agreement"]
        assert agreement["n_tweets"] == 60
        assert agreement["agreement_rate"] == pytest.approx(2 / 3)
        assert agreement["divergence_rate"] == pytest.approx(1 / 3)
        lower, upper = agreement["agreement_rate_ci"]
        assert lower <= agreement["agreement_rate"] <= upper
        assert result.summary["hypotheses"] is None

    def test_pairs_tweets_by_id_regardless_of_row_order(self, pipeline_paths: ProjectPaths) -> None:
        """A comparação usa o ``id`` como chave: embaralhar a base OpenAI não muda o resultado."""
        _write_comparison_bases(pipeline_paths)
        baseline = run_comparative_evaluation_stage(
            pipeline_paths, n_bootstrap=30, run_hypotheses=False
        ).summary["agreement"]["n_agree"]
        shuffled = read_dataset_file(pipeline_paths.openai_labeled_file).reverse()
        write_dataset(shuffled, pipeline_paths.openai_labeled_file)

        result = run_comparative_evaluation_stage(
            pipeline_paths, n_bootstrap=30, run_hypotheses=False
        )

        assert result.summary["agreement"]["n_agree"] == baseline

    def test_raises_with_guidance_when_a_base_is_missing(
        self, pipeline_paths: ProjectPaths
    ) -> None:
        """Sem uma das bases, falha com ``DataNotFoundError`` indicando o comando de rotulagem."""
        _write_comparison_bases(pipeline_paths)
        pipeline_paths.openai_labeled_file.unlink()

        with pytest.raises(DataNotFoundError, match="pipeline-labeling-openai"):
            run_comparative_evaluation_stage(pipeline_paths, run_hypotheses=False)

    def test_raises_when_bases_do_not_contain_the_same_tweets(
        self, pipeline_paths: ProjectPaths
    ) -> None:
        """Bases com conjuntos de ``id`` diferentes não são comparáveis: erro explícito."""
        _write_comparison_bases(pipeline_paths)
        truncated = read_dataset_file(pipeline_paths.openai_labeled_file).head(50)
        write_dataset(truncated, pipeline_paths.openai_labeled_file)

        with pytest.raises(DataValidationError, match="mesmos tweets"):
            run_comparative_evaluation_stage(pipeline_paths, run_hypotheses=False)

    def test_runs_hypothesaes_last_and_records_its_outcome(
        self, pipeline_paths: ProjectPaths, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Com ``run_hypotheses=True``, o HypotheSAEs roda depois das tabelas e vai ao resumo."""
        from diagnostics import model_disagreement

        _write_comparison_bases(pipeline_paths)
        captured: dict[str, Any] = {}

        def _fake_run_hypotheses(frame: pl.DataFrame, paths: ProjectPaths, **kwargs: Any) -> list:
            captured["n_tweets"] = frame.height
            captured["tables_already_written"] = (
                kwargs["output_dir"] / "casos_ambiguos.csv"
            ).is_file()
            captured["targets"] = kwargs["targets"]
            return [
                model_disagreement.HypothesisTargetOutcome(
                    "disagreement", "gate_reprovado", "sem sinal", 0, None, None
                )
            ]

        monkeypatch.setattr(model_disagreement, "run_disagreement_hypotheses", _fake_run_hypotheses)

        result = run_comparative_evaluation_stage(
            pipeline_paths,
            n_bootstrap=30,
            run_hypotheses=True,
            hypotheses_targets=("disagreement",),
        )

        assert captured == {
            "n_tweets": 60,
            "tables_already_written": True,
            "targets": ("disagreement",),
        }
        assert result.summary["hypotheses"][0]["status"] == "gate_reprovado"
        assert "gate_reprovado" in read_json(result.metrics_path)["hypotheses"][0]["status"]


def _fake_extract_local_embeddings(texts: list[str], **kwargs: Any) -> dict[str, np.ndarray]:
    """Dublê de :func:`hypothesaes.embedding.extract_local_embeddings`:
    vetores fixos, sem modelo real."""
    return {text: np.zeros(4, dtype=np.float32) for text in texts}


def _fake_train_sae(**kwargs: Any) -> str:
    """Dublê de :func:`hypothesaes.quickstart.train_sae`: retorna um
    sentinela, sem treinar de fato."""
    return "sae-fake"


def _fake_interpret_sae(**kwargs: Any) -> pd.DataFrame:
    """Dublê de :func:`hypothesaes.quickstart.interpret_sae`: dois padrões fixos, sem chamar LLM."""
    return pd.DataFrame({"neuron_idx": [0, 1], "interpretation": ["padrão a", "padrão b"]})


def _fake_generate_hypotheses(*, selection_method: str, **kwargs: Any) -> pd.DataFrame:
    """Dublê de :func:`hypothesaes.quickstart.generate_hypotheses`: duas
    hipóteses fixas, sem LLM."""
    return pd.DataFrame(
        {
            "neuron_idx": [0, 1],
            f"target_{selection_method}": [0.5, -0.3],
            "interpretation": ["hipótese de baixa confiança", "hipótese de alta confiança"],
        }
    )


def _fake_evaluate_hypotheses(**kwargs: Any) -> tuple[dict[str, Any], pd.DataFrame]:
    """Dublê de :func:`hypothesaes.quickstart.evaluate_hypotheses`: métricas fixas, sem LLM."""
    metrics = {"Significant": (1, 2, 0.5)}
    evaluation_df = pd.DataFrame({"hypothesis": ["hipótese de baixa confiança"], "f1": [0.8]})
    return metrics, evaluation_df


class TestRunHypothesaesAnalysisStage:
    """Testes de :func:`pipelines.hypothesaes_analysis.run_hypothesaes_analysis_stage`.

    O SAE, os embeddings e todas as chamadas de LLM são substituídos por
    dublês (ver funções ``_fake_*`` acima), mantendo o teste rápido e livre
    de ``torch``/``sentence-transformers`` — apenas a integração entre a
    etapa e ``evaluation.hypothesaes_report``/``visualization.hypothesaes``
    é exercitada de verdade.
    """

    @staticmethod
    def _labeled_corpus_with_confidence(n_rows: int = 20) -> pl.DataFrame:
        """Corpus rotulado sintético, com 1/4 das amostras marcadas como baixa confiança."""
        return pl.DataFrame(
            {
                "id": [str(index) for index in range(n_rows)],
                "text": [f"tweet numero {index}" for index in range(n_rows)],
                "sentiment_label": [
                    "positivo" if index % 2 == 0 else "negativo" for index in range(n_rows)
                ],
                "confidence_score": [0.2 if index % 4 == 0 else 0.9 for index in range(n_rows)],
            }
        )

    def _apply_fakes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            hypothesaes_analysis, "extract_local_embeddings", _fake_extract_local_embeddings
        )
        monkeypatch.setattr(hypothesaes_analysis, "train_sae", _fake_train_sae)
        monkeypatch.setattr(hypothesaes_analysis, "interpret_sae", _fake_interpret_sae)
        monkeypatch.setattr(hypothesaes_analysis, "generate_hypotheses", _fake_generate_hypotheses)

    def test_saves_artifacts_and_returns_them(
        self, pipeline_paths: ProjectPaths, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deve sinalizar tweets de baixa confiança, gerar padrões/hipóteses
        e salvar tudo em disco."""
        self._apply_fakes(monkeypatch)
        labeled_corpus = self._labeled_corpus_with_confidence()

        artifacts = run_hypothesaes_analysis_stage(
            labeled_corpus, pipeline_paths, n_selected_neurons=2, random_seed=42
        )

        assert artifacts.low_confidence_tweets.height == 5
        assert artifacts.patterns.height == 2
        assert artifacts.hypotheses.height == 2
        assert artifacts.top_hypotheses_table.height == 2
        assert artifacts.holdout_metrics is None
        assert artifacts.figure_paths[0].is_file()
        assert artifacts.figure_paths[1].is_file()
        assert artifacts.summary_path.is_file()

        interpretability_dir = pipeline_paths.reports_interpretability_dir
        assert (interpretability_dir / "hypothesaes_tweets_baixa_confianca.csv").is_file()
        assert (interpretability_dir / "hypothesaes_descoberta_padroes.csv").is_file()
        assert (interpretability_dir / "hypothesaes_hipoteses_inconsistencia.csv").is_file()
        assert (interpretability_dir / "hypothesaes_top_hipoteses.csv").is_file()

    def test_raises_when_confidence_column_is_missing(self, pipeline_paths: ProjectPaths) -> None:
        """Deve levantar ``DataValidationError`` quando a coluna de confiança está ausente."""
        labeled_corpus = pl.DataFrame(
            {
                "id": ["1"],
                "text": ["tweet"],
                "sentiment_label": ["positivo"],
            }
        )
        with pytest.raises(DataValidationError):
            run_hypothesaes_analysis_stage(labeled_corpus, pipeline_paths)

    def test_raises_for_empty_corpus(self, pipeline_paths: ProjectPaths) -> None:
        """Deve levantar ``EmptyDatasetError`` quando o corpus rotulado está vazio."""
        empty_corpus = pl.DataFrame(
            schema={
                "id": pl.Utf8,
                "text": pl.Utf8,
                "sentiment_label": pl.Utf8,
                "confidence_score": pl.Float64,
            }
        )
        with pytest.raises(EmptyDatasetError):
            run_hypothesaes_analysis_stage(empty_corpus, pipeline_paths)

    def test_evaluates_on_holdout_when_enabled(
        self, pipeline_paths: ProjectPaths, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Com ``evaluate_on_holdout=True``, deve avaliar as hipóteses no
        holdout e salvar as métricas."""
        self._apply_fakes(monkeypatch)
        monkeypatch.setattr(hypothesaes_analysis, "evaluate_hypotheses", _fake_evaluate_hypotheses)
        labeled_corpus = self._labeled_corpus_with_confidence()

        artifacts = run_hypothesaes_analysis_stage(
            labeled_corpus,
            pipeline_paths,
            n_selected_neurons=2,
            evaluate_on_holdout=True,
            holdout_size=0.2,
            random_seed=42,
        )

        assert artifacts.holdout_metrics == {"Significant": [1, 2, 0.5]}
        interpretability_dir = pipeline_paths.reports_interpretability_dir
        assert (interpretability_dir / "hypothesaes_avaliacao_holdout.csv").is_file()
        assert (interpretability_dir / "hypothesaes_avaliacao_metricas.json").is_file()


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
