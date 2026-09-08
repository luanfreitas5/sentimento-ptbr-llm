"""Testes dos contratos de dados (schemas pandera.polars) do projeto."""

from datetime import datetime

import polars as pl
import pytest

from exceptions.data import DataValidationError
from schemas.dataset import validate_labeled_corpus, validate_raw_tweet_dataset
from schemas.experiment import validate_experiment_run_metric
from schemas.labeling import validate_labeling_result
from schemas.prediction import validate_prediction
from schemas.training import validate_training_example


def _minimal_raw_tweet_frame(tweet_ids: list[str] | None = None) -> pl.DataFrame:
    """Constrói um DataFrame mínimo válido contra o novo ``RawTweetSchema``, para testes."""
    ids = tweet_ids if tweet_ids is not None else ["1"]
    n = len(ids)
    return pl.DataFrame({
        "tweet_id": ids,
        "user_id": ["u1"] * n,
        "text": [f"texto {tweet_id}" for tweet_id in ids],
        "created_at": [datetime(2026, 1, 1)] * n,
        "language": ["pt"] * n,
        "is_reply": [False] * n,
        "is_retweet": [False] * n,
        "like_count": [0] * n,
        "reply_count": [0] * n,
        "retweet_count": [0] * n,
        "quote_count": [0] * n,
        "source_query": pl.Series("source_query", [None] * n, dtype=pl.Utf8),
        "source_group": pl.Series("source_group", [None] * n, dtype=pl.Utf8),
    })


class TestDatasetSchemas:
    """Testes dos schemas de corpus bruto e rotulado."""

    def test_validate_raw_tweet_dataset_accepts_valid_dataframe(self) -> None:
        """Um DataFrame com todas as colunas obrigatórias e tweet_id único deve ser aceito."""
        result = validate_raw_tweet_dataset(_minimal_raw_tweet_frame(["1", "2"]))
        assert result.height == 2

    def test_validate_raw_tweet_dataset_allows_null_source_query_and_group(self) -> None:
        """source_query/source_group nulos (comum quando a coleta é por usuário, não por termo) são aceitos."""
        result = validate_raw_tweet_dataset(_minimal_raw_tweet_frame())
        assert result["source_query"].null_count() == 1
        assert result["source_group"].null_count() == 1

    def test_validate_raw_tweet_dataset_rejects_extra_column(self) -> None:
        """Uma coluna extra não declarada deve ser rejeitada (schema strict)."""
        df = _minimal_raw_tweet_frame().with_columns(pl.lit("valor").alias("extra_column"))
        with pytest.raises(DataValidationError):
            validate_raw_tweet_dataset(df)

    def test_validate_raw_tweet_dataset_rejects_duplicate_tweet_id(self) -> None:
        """tweet_id duplicado deve violar a restrição de unicidade."""
        with pytest.raises(DataValidationError):
            validate_raw_tweet_dataset(_minimal_raw_tweet_frame(["1", "1"]))

    def test_validate_raw_tweet_dataset_rejects_negative_engagement_count(self) -> None:
        """Uma contagem de engajamento negativa deve violar o contrato (like_count >= 0)."""
        df = _minimal_raw_tweet_frame().with_columns(pl.Series("like_count", [-1]))
        with pytest.raises(DataValidationError):
            validate_raw_tweet_dataset(df)

    def test_validate_labeled_corpus_accepts_valid_dataframe(
        self, sample_labeled_corpus: pl.DataFrame
    ) -> None:
        """Um corpus rotulado válido deve ser aceito."""
        result = validate_labeled_corpus(sample_labeled_corpus)
        assert result.height == 3

    def test_validate_labeled_corpus_allows_extra_column(
        self, sample_labeled_corpus: pl.DataFrame
    ) -> None:
        """Colunas extras (ex.: metadados) devem ser permitidas (schema não estrito)."""
        df = sample_labeled_corpus.with_columns(pl.lit("scraping").alias("data_source"))
        result = validate_labeled_corpus(df)
        assert "data_source" in result.columns

    def test_validate_labeled_corpus_rejects_invalid_label(
        self, sample_labeled_corpus: pl.DataFrame
    ) -> None:
        """Um rótulo fora das classes conhecidas deve ser rejeitado."""
        df = sample_labeled_corpus.with_columns(
            pl.Series("sentiment_label", ["muito_positivo", "negativo", "neutro"])
        )
        with pytest.raises(DataValidationError):
            validate_labeled_corpus(df)


class TestLabelingResultSchema:
    """Testes do schema de resultados de rotulagem em cascata."""

    def test_validate_labeling_result_accepts_valid_dataframe(self) -> None:
        """Um resultado de rotulagem válido deve ser aceito."""
        df = pl.DataFrame({
            "id": ["1"],
            "tagger": ["heuristica_lexica"],
            "sentiment_label": ["positivo"],
            "confidence_score": [0.9],
            "weight": [1.0],
        })
        assert validate_labeling_result(df).height == 1

    def test_validate_labeling_result_rejects_confidence_out_of_range(self) -> None:
        """Confiança fora do intervalo [0, 1] deve ser rejeitada."""
        df = pl.DataFrame({
            "id": ["1"],
            "tagger": ["heuristica_lexica"],
            "sentiment_label": ["positivo"],
            "confidence_score": [1.5],
            "weight": [1.0],
        })
        with pytest.raises(DataValidationError):
            validate_labeling_result(df)

    def test_validate_labeling_result_rejects_non_positive_weight(self) -> None:
        """Peso não positivo deve ser rejeitado."""
        df = pl.DataFrame({
            "id": ["1"],
            "tagger": ["heuristica_lexica"],
            "sentiment_label": ["positivo"],
            "confidence_score": [0.9],
            "weight": [0.0],
        })
        with pytest.raises(DataValidationError):
            validate_labeling_result(df)


class TestPredictionSchema:
    """Testes do schema de predições de sentimento."""

    def test_validate_prediction_accepts_valid_dataframe(self) -> None:
        """Uma predição válida deve ser aceita."""
        df = pl.DataFrame({
            "id": ["1"],
            "text": ["ótimo produto"],
            "sentiment_label": ["positivo"],
            "confidence_score": [0.95],
        })
        assert validate_prediction(df).height == 1

    def test_validate_prediction_rejects_unknown_label(self) -> None:
        """Um rótulo predito fora das classes conhecidas deve ser rejeitado."""
        df = pl.DataFrame({
            "id": ["1"],
            "text": ["ótimo produto"],
            "sentiment_label": ["desconhecido"],
            "confidence_score": [0.95],
        })
        with pytest.raises(DataValidationError):
            validate_prediction(df)


class TestTrainingExampleSchema:
    """Testes do schema de exemplos de treino/validação/teste."""

    def test_validate_training_example_accepts_valid_dataframe(self) -> None:
        """Um exemplo de treino válido deve ser aceito."""
        df = pl.DataFrame({
            "id": ["1"],
            "text": ["ótimo produto"],
            "sentiment_label": ["positivo"],
            "split": ["treino"],
        })
        assert validate_training_example(df).height == 1

    def test_validate_training_example_rejects_unknown_split(self) -> None:
        """Um valor de split fora de treino/validacao/teste deve ser rejeitado."""
        df = pl.DataFrame({
            "id": ["1"],
            "text": ["ótimo produto"],
            "sentiment_label": ["positivo"],
            "split": ["outro"],
        })
        with pytest.raises(DataValidationError):
            validate_training_example(df)


class TestExperimentRunMetricSchema:
    """Testes do schema de métricas de execuções de experimento."""

    def test_validate_experiment_run_metric_accepts_valid_dataframe(self) -> None:
        """Um registro de métrica válido deve ser aceito."""
        df = pl.DataFrame({
            "run_id": ["abc123"],
            "model_name": ["logistic_regression"],
            "metric_name": ["f1_macro"],
            "metric_value": [0.82],
            "git_sha": ["deadbeef"],
            "dataset_hash": ["0f3123a4"],
        })
        assert validate_experiment_run_metric(df).height == 1

    def test_validate_experiment_run_metric_rejects_unknown_metric_name(self) -> None:
        """Um nome de métrica não reconhecido deve ser rejeitado."""
        df = pl.DataFrame({
            "run_id": ["abc123"],
            "model_name": ["logistic_regression"],
            "metric_name": ["metrica_inexistente"],
            "metric_value": [0.82],
            "git_sha": ["deadbeef"],
            "dataset_hash": ["0f3123a4"],
        })
        with pytest.raises(DataValidationError):
            validate_experiment_run_metric(df)
