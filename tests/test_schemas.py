"""Testes dos contratos de dados (schemas pandera.polars) do projeto."""

import polars as pl
import pytest

from exceptions.data import DataValidationError
from schemas.dataset import validate_labeled_corpus, validate_raw_tweet_dataset
from schemas.experiment import validate_experiment_run_metric
from schemas.labeling import validate_labeling_result
from schemas.prediction import validate_prediction
from schemas.training import validate_training_example


class TestDatasetSchemas:
    """Testes dos schemas de corpus bruto e rotulado."""

    def test_validate_raw_tweet_dataset_accepts_valid_dataframe(self) -> None:
        """Um DataFrame com todas as colunas obrigatórias e id único deve ser aceito."""
        df = pl.DataFrame(
            {
                "id": ["1", "2"],
                "texto": ["ótimo produto", "não gostei"],
                "fonte_dados": ["scraping", "scraping"],
                "data_coleta": ["2026-01-01", "2026-01-02"],
            }
        )
        resultado = validate_raw_tweet_dataset(df)
        assert resultado.height == 2

    def test_validate_raw_tweet_dataset_rejects_extra_column(self) -> None:
        """Uma coluna extra não declarada deve ser rejeitada (schema strict)."""
        df = pl.DataFrame(
            {
                "id": ["1"],
                "texto": ["ótimo produto"],
                "fonte_dados": ["scraping"],
                "data_coleta": ["2026-01-01"],
                "coluna_extra": ["valor"],
            }
        )
        with pytest.raises(DataValidationError):
            validate_raw_tweet_dataset(df)

    def test_validate_raw_tweet_dataset_rejects_duplicate_id(self) -> None:
        """Ids duplicados devem violar a restrição de unicidade."""
        df = pl.DataFrame(
            {
                "id": ["1", "1"],
                "texto": ["a", "b"],
                "fonte_dados": ["scraping", "scraping"],
                "data_coleta": ["2026-01-01", "2026-01-02"],
            }
        )
        with pytest.raises(DataValidationError):
            validate_raw_tweet_dataset(df)

    def test_validate_labeled_corpus_accepts_valid_dataframe(self, sample_labeled_corpus: pl.DataFrame) -> None:
        """Um corpus rotulado válido deve ser aceito."""
        resultado = validate_labeled_corpus(sample_labeled_corpus)
        assert resultado.height == 3

    def test_validate_labeled_corpus_allows_extra_column(self, sample_labeled_corpus: pl.DataFrame) -> None:
        """Colunas extras (ex.: metadados) devem ser permitidas (schema não estrito)."""
        df = sample_labeled_corpus.with_columns(pl.lit("scraping").alias("fonte_dados"))
        resultado = validate_labeled_corpus(df)
        assert "fonte_dados" in resultado.columns

    def test_validate_labeled_corpus_rejects_invalid_label(self, sample_labeled_corpus: pl.DataFrame) -> None:
        """Um rótulo fora das classes conhecidas deve ser rejeitado."""
        df = sample_labeled_corpus.with_columns(pl.Series("sentimento", ["muito_positivo", "negativo", "neutro"]))
        with pytest.raises(DataValidationError):
            validate_labeled_corpus(df)


class TestLabelingResultSchema:
    """Testes do schema de resultados de rotulagem em cascata."""

    def test_validate_labeling_result_accepts_valid_dataframe(self) -> None:
        """Um resultado de rotulagem válido deve ser aceito."""
        df = pl.DataFrame(
            {
                "id": ["1"],
                "rotulador": ["heuristica_lexica"],
                "sentimento_predito": ["positivo"],
                "confianca": [0.9],
                "peso": [1.0],
            }
        )
        assert validate_labeling_result(df).height == 1

    def test_validate_labeling_result_rejects_confidence_out_of_range(self) -> None:
        """Confiança fora do intervalo [0, 1] deve ser rejeitada."""
        df = pl.DataFrame(
            {
                "id": ["1"],
                "rotulador": ["heuristica_lexica"],
                "sentimento_predito": ["positivo"],
                "confianca": [1.5],
                "peso": [1.0],
            }
        )
        with pytest.raises(DataValidationError):
            validate_labeling_result(df)

    def test_validate_labeling_result_rejects_non_positive_weight(self) -> None:
        """Peso não positivo deve ser rejeitado."""
        df = pl.DataFrame(
            {
                "id": ["1"],
                "rotulador": ["heuristica_lexica"],
                "sentimento_predito": ["positivo"],
                "confianca": [0.9],
                "peso": [0.0],
            }
        )
        with pytest.raises(DataValidationError):
            validate_labeling_result(df)


class TestPredictionSchema:
    """Testes do schema de predições de sentimento."""

    def test_validate_prediction_accepts_valid_dataframe(self) -> None:
        """Uma predição válida deve ser aceita."""
        df = pl.DataFrame(
            {"id": ["1"], "texto": ["ótimo produto"], "sentimento_predito": ["positivo"], "confianca": [0.95]}
        )
        assert validate_prediction(df).height == 1

    def test_validate_prediction_rejects_unknown_label(self) -> None:
        """Um rótulo predito fora das classes conhecidas deve ser rejeitado."""
        df = pl.DataFrame(
            {"id": ["1"], "texto": ["ótimo produto"], "sentimento_predito": ["desconhecido"], "confianca": [0.95]}
        )
        with pytest.raises(DataValidationError):
            validate_prediction(df)


class TestTrainingExampleSchema:
    """Testes do schema de exemplos de treino/validação/teste."""

    def test_validate_training_example_accepts_valid_dataframe(self) -> None:
        """Um exemplo de treino válido deve ser aceito."""
        df = pl.DataFrame({"id": ["1"], "texto": ["ótimo produto"], "sentimento": ["positivo"], "split": ["treino"]})
        assert validate_training_example(df).height == 1

    def test_validate_training_example_rejects_unknown_split(self) -> None:
        """Um valor de split fora de treino/validacao/teste deve ser rejeitado."""
        df = pl.DataFrame({"id": ["1"], "texto": ["ótimo produto"], "sentimento": ["positivo"], "split": ["outro"]})
        with pytest.raises(DataValidationError):
            validate_training_example(df)


class TestExperimentRunMetricSchema:
    """Testes do schema de métricas de execuções de experimento."""

    def test_validate_experiment_run_metric_accepts_valid_dataframe(self) -> None:
        """Um registro de métrica válido deve ser aceito."""
        df = pl.DataFrame(
            {
                "run_id": ["abc123"],
                "model_name": ["logistic_regression"],
                "metric_name": ["f1_macro"],
                "metric_value": [0.82],
                "git_sha": ["deadbeef"],
                "dataset_hash": ["0f3123a4"],
            }
        )
        assert validate_experiment_run_metric(df).height == 1

    def test_validate_experiment_run_metric_rejects_unknown_metric_name(self) -> None:
        """Um nome de métrica não reconhecido deve ser rejeitado."""
        df = pl.DataFrame(
            {
                "run_id": ["abc123"],
                "model_name": ["logistic_regression"],
                "metric_name": ["metrica_inexistente"],
                "metric_value": [0.82],
                "git_sha": ["deadbeef"],
                "dataset_hash": ["0f3123a4"],
            }
        )
        with pytest.raises(DataValidationError):
            validate_experiment_run_metric(df)
