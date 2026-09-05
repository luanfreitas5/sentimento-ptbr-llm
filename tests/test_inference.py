"""Testes da camada comum de inferência (``src/inference``)."""

from collections.abc import Sequence

import numpy as np
import pytest

from exceptions.data import EmptyDatasetError
from inference.batch import run_batch_inference
from inference.llm_batch import run_llm_batch_inference
from inference.online import OnlinePredictor
from inference.postprocessing import build_prediction_dataframe, standardize_prediction_output
from inference.predictor import Predictor


class _FakeTextClassifier:
    """Classificador de teste: prediz com base na presença da palavra "bom" no texto."""

    def __init__(self) -> None:
        self.classes_ = np.array(["negativo", "positivo"])

    def fit(self, X: Sequence[str], y: Sequence[str]) -> "_FakeTextClassifier":
        """No-op: o classificador de teste não aprende parâmetros."""
        return self

    def predict(self, X: Sequence[str]) -> np.ndarray:
        """Prediz 'positivo' para textos contendo 'bom', 'negativo' caso contrário."""
        return np.array(["positivo" if "bom" in text else "negativo" for text in X])

    def predict_proba(self, X: Sequence[str]) -> np.ndarray:
        """Retorna probabilidades fixas de 0.9/0.1, conforme a predição de :meth:`predict`."""
        return np.array([[0.1, 0.9] if "bom" in text else [0.9, 0.1] for text in X])


class _FlakyTextClassifier(_FakeTextClassifier):
    """Classificador de teste que falha propositalmente para o texto ``"erro"``."""

    def predict(self, X: Sequence[str]) -> np.ndarray:
        """Levanta ``ValueError`` se algum texto do lote for ``"erro"``; delega o restante."""
        if "erro" in X:
            raise ValueError("falha simulada de inferência")
        return super().predict(X)

    def predict_proba(self, X: Sequence[str]) -> np.ndarray:
        """Levanta ``ValueError`` se algum texto do lote for ``"erro"``; delega o restante."""
        if "erro" in X:
            raise ValueError("falha simulada de inferência")
        return super().predict_proba(X)


class TestStandardizePredictionOutput:
    """Testes da padronização de rótulo/confiança."""

    def test_normalizes_label_and_clips_confidence_above_one(self) -> None:
        """Um rótulo com variação de caixa deve ser normalizado; confiança > 1 deve ser 1.0."""
        label, confidence = standardize_prediction_output(" Positivo ", 1.5)
        assert label == "positivo"
        assert confidence == 1.0

    def test_clips_negative_confidence_to_zero(self) -> None:
        """Uma confiança negativa deve ser restrita a 0.0."""
        _, confidence = standardize_prediction_output("negativo", -0.2)
        assert confidence == 0.0

    def test_raises_for_unknown_label(self) -> None:
        """Um rótulo fora das classes conhecidas deve levantar ``ValueError``."""
        with pytest.raises(ValueError, match="não pertence"):
            standardize_prediction_output("irritado", 0.5)


class TestBuildPredictionDataframe:
    """Testes da montagem/validação do DataFrame de predições."""

    def test_builds_and_validates_dataframe(self) -> None:
        """Deve montar um DataFrame válido contra ``PredictionSchema``."""
        dataframe = build_prediction_dataframe(
            ["1", "2"], ["bom dia", "péssimo dia"], ["positivo", "negativo"], [0.9, 0.8]
        )
        assert dataframe.height == 2
        assert dataframe["confidence"].to_list() == [0.9, 0.8]

    def test_raises_for_empty_ids(self) -> None:
        """Um lote vazio deve levantar ``EmptyDatasetError``."""
        with pytest.raises(EmptyDatasetError):
            build_prediction_dataframe([], [], [], [])

    def test_supports_extra_columns(self) -> None:
        """Colunas adicionais (ex.: probabilidades por classe) devem ser preservadas."""
        dataframe = build_prediction_dataframe(
            ["1"], ["bom"], ["positivo"], [0.9], extra_columns={"prob_positivo": [0.9]}
        )
        assert "prob_positivo" in dataframe.columns


class TestPredictor:
    """Testes da interface única de inferência sobre um modelo treinado."""

    def test_predict_one_from_features_returns_expected_label(self) -> None:
        """Deve retornar o rótulo, a confiança e as probabilidades por classe esperados."""
        predictor = Predictor(_FakeTextClassifier(), allowed_labels=("positivo", "negativo"))
        record = predictor.predict_one_from_features(["produto muito bom"])
        assert record["sentiment_label"] == "positivo"
        assert record["confidence"] == pytest.approx(0.9)
        assert record["probabilities"]["positivo"] == pytest.approx(0.9)

    def test_predict_returns_validated_dataframe(self) -> None:
        """``predict`` deve retornar um DataFrame com uma linha por amostra de entrada."""
        predictor = Predictor(_FakeTextClassifier())
        dataframe = predictor.predict(["bom dia", "péssimo atendimento"])
        assert dataframe.height == 2
        assert dataframe["sentiment_label"].to_list() == ["positivo", "negativo"]

    def test_predict_proba_has_one_column_per_class(self) -> None:
        """``predict_proba`` deve retornar uma matriz ``(n_amostras, n_classes)``."""
        predictor = Predictor(_FakeTextClassifier())
        probabilities = predictor.predict_proba(["bom", "ruim"])
        assert probabilities.shape == (2, 2)

    def test_predict_raises_for_empty_texts(self) -> None:
        """Um lote vazio deve levantar ``EmptyDatasetError``."""
        predictor = Predictor(_FakeTextClassifier())
        with pytest.raises(EmptyDatasetError):
            predictor.predict([])


class TestOnlinePredictor:
    """Testes da inferência ponto a ponto."""

    def test_predict_delegates_to_predictor(self) -> None:
        """Deve classificar uma única amostra, delegando a ``Predictor``."""
        online_predictor = OnlinePredictor(
            Predictor(_FakeTextClassifier(), allowed_labels=("positivo", "negativo"))
        )
        record = online_predictor.predict("um texto muito bom")
        assert record["sentiment_label"] == "positivo"

    def test_raises_for_none_sample(self) -> None:
        """Uma amostra ``None`` deve levantar ``ValueError``."""
        online_predictor = OnlinePredictor(Predictor(_FakeTextClassifier()))
        with pytest.raises(ValueError, match="vazia"):
            online_predictor.predict(None)

    def test_raises_for_empty_string_sample(self) -> None:
        """Uma amostra de texto vazia deve levantar ``ValueError``."""
        online_predictor = OnlinePredictor(Predictor(_FakeTextClassifier()))
        with pytest.raises(ValueError, match="vazia"):
            online_predictor.predict("")


class TestRunBatchInference:
    """Testes da inferência em lote sequencial, em blocos."""

    def test_processes_all_samples_across_multiple_chunks(self) -> None:
        """Deve processar corretamente todas as amostras, mesmo com um último bloco parcial."""
        predictor = Predictor(_FakeTextClassifier(), allowed_labels=("positivo", "negativo"))
        texts = ["bom", "ruim", "bom", "ruim", "bom"]

        result = run_batch_inference(predictor, texts, batch_size=2, show_progress=False)

        assert result.height == 5
        assert result["sentiment_label"].to_list() == [
            "positivo",
            "negativo",
            "positivo",
            "negativo",
            "positivo",
        ]

    def test_raises_for_invalid_batch_size(self) -> None:
        """``batch_size`` menor que 1 deve levantar ``ValueError``."""
        predictor = Predictor(_FakeTextClassifier())
        with pytest.raises(ValueError, match="batch_size"):
            run_batch_inference(predictor, ["bom"], batch_size=0, show_progress=False)

    def test_raises_for_empty_texts(self) -> None:
        """Um lote vazio deve levantar ``EmptyDatasetError``."""
        predictor = Predictor(_FakeTextClassifier())
        with pytest.raises(EmptyDatasetError):
            run_batch_inference(predictor, [], show_progress=False)


class TestRunLLMBatchInference:
    """Testes da inferência em lote concorrente para LLMs."""

    def test_preserves_original_order_regardless_of_completion_order(self) -> None:
        """A ordem das linhas do resultado deve corresponder à ordem original de ``texts``."""
        predictor = Predictor(_FakeTextClassifier(), allowed_labels=("positivo", "negativo"))
        texts = ["bom", "ruim", "bom", "ruim"]

        result = run_llm_batch_inference(predictor, texts, show_progress=False)

        assert result["text"].to_list() == texts

    def test_isolates_failures_without_raising(self) -> None:
        """Uma falha em um único texto não deve interromper o processamento dos demais."""
        predictor = Predictor(_FlakyTextClassifier(), allowed_labels=("positivo", "negativo"))

        result = run_llm_batch_inference(
            predictor, ["bom dia", "erro", "péssimo dia"], show_progress=False
        )

        assert result.height == 2
        assert "erro" not in result["text"].to_list()

    def test_raises_for_empty_texts(self) -> None:
        """Um lote vazio deve levantar ``EmptyDatasetError``."""
        predictor = Predictor(_FakeTextClassifier())
        with pytest.raises(EmptyDatasetError):
            run_llm_batch_inference(predictor, [], show_progress=False)
