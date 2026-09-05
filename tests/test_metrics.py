"""Testes das métricas de avaliação (``src/metrics``)."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from exceptions.data import EmptyDatasetError
from exceptions.model import ModelError
from metrics.classification import (
    calculate_accuracy,
    calculate_classification_metrics,
    calculate_confusion_matrix,
    calculate_matthews_correlation_coefficient,
    calculate_per_class_report,
    calculate_precision_recall_f1,
)
from metrics.confidence import (
    calculate_average_confidence,
    calculate_confidence_accuracy_correlation,
    calculate_multiclass_brier_score,
    calculate_selective_prediction_accuracy,
)
from metrics.operational import (
    calculate_latency_statistics,
    calculate_operational_metrics,
    count_trainable_parameters,
    measure_inference_latency,
)
from metrics.ranking import calculate_pr_auc_ovr, calculate_ranking_metrics, calculate_roc_auc_ovr


class TestCalculateConfusionMatrix:
    """Testes de :func:`metrics.classification.calculate_confusion_matrix`."""

    def test_builds_matrix_in_labels_order(self) -> None:
        """A matriz deve seguir a ordem de classes informada em ``labels``."""
        matrix = calculate_confusion_matrix(
            ["positivo", "negativo"], ["positivo", "positivo"], labels=["negativo", "positivo"]
        )
        assert matrix.tolist() == [[0, 1], [0, 1]]

    def test_raises_on_empty_y_true(self) -> None:
        """Deve levantar ``EmptyDatasetError`` quando ``y_true`` está vazio."""
        with pytest.raises(EmptyDatasetError):
            calculate_confusion_matrix([], [])

    def test_raises_on_mismatched_lengths(self) -> None:
        """Deve levantar ``ValueError`` quando os tamanhos diferem."""
        with pytest.raises(ValueError, match="mesmo tamanho"):
            calculate_confusion_matrix(["positivo"], ["positivo", "negativo"])


class TestCalculateAccuracy:
    """Testes de :func:`metrics.classification.calculate_accuracy`."""

    def test_returns_one_for_perfect_predictions(self) -> None:
        """Predições idênticas aos rótulos verdadeiros devem gerar acurácia 1.0."""
        assert calculate_accuracy(["positivo", "negativo"], ["positivo", "negativo"]) == 1.0

    def test_returns_fraction_of_correct_predictions(self) -> None:
        """A acurácia deve ser a fração exata de acertos."""
        assert calculate_accuracy(["positivo", "negativo"], ["positivo", "positivo"]) == 0.5

    def test_raises_on_empty_y_true(self) -> None:
        """Deve levantar ``EmptyDatasetError`` quando ``y_true`` está vazio."""
        with pytest.raises(EmptyDatasetError):
            calculate_accuracy([], [])


class TestCalculatePrecisionRecallF1:
    """Testes de :func:`metrics.classification.calculate_precision_recall_f1`."""

    def test_returns_expected_keys(self) -> None:
        """O dicionário retornado deve conter precisão, revocação e F1."""
        result = calculate_precision_recall_f1(["positivo", "negativo"], ["positivo", "negativo"])
        assert set(result.keys()) == {"precision", "recall", "f1"}

    def test_perfect_predictions_yield_score_one(self) -> None:
        """Predições perfeitas devem gerar F1 macro igual a 1.0."""
        result = calculate_precision_recall_f1(
            ["positivo", "negativo", "neutro"], ["positivo", "negativo", "neutro"], average="macro"
        )
        assert result["f1"] == 1.0


class TestCalculatePerClassReport:
    """Testes de :func:`metrics.classification.calculate_per_class_report`."""

    def test_returns_one_entry_per_label(self) -> None:
        """Deve retornar uma entrada por classe informada em ``labels``."""
        report = calculate_per_class_report(
            ["positivo", "negativo"], ["positivo", "positivo"], labels=["negativo", "positivo"]
        )
        assert set(report.keys()) == {"negativo", "positivo"}

    def test_reports_full_recall_for_always_correct_class(self) -> None:
        """A classe sempre acertada deve ter revocação 1.0."""
        report = calculate_per_class_report(
            ["positivo", "negativo"], ["positivo", "positivo"], labels=["negativo", "positivo"]
        )
        assert report["positivo"]["recall"] == 1.0

    def test_reports_zero_recall_for_never_correct_class(self) -> None:
        """A classe nunca acertada deve ter revocação 0.0."""
        report = calculate_per_class_report(
            ["positivo", "negativo"], ["positivo", "positivo"], labels=["negativo", "positivo"]
        )
        assert report["negativo"]["recall"] == 0.0


class TestCalculateMatthewsCorrelationCoefficient:
    """Testes de :func:`metrics.classification.calculate_matthews_correlation_coefficient`."""

    def test_returns_one_for_perfect_agreement(self) -> None:
        """Concordância perfeita deve gerar MCC igual a 1.0."""
        labels = ["positivo", "negativo", "positivo", "negativo"]
        assert calculate_matthews_correlation_coefficient(labels, labels) == 1.0

    def test_raises_on_empty_y_true(self) -> None:
        """Deve levantar ``EmptyDatasetError`` quando ``y_true`` está vazio."""
        with pytest.raises(EmptyDatasetError):
            calculate_matthews_correlation_coefficient([], [])


class TestCalculateClassificationMetrics:
    """Testes de :func:`metrics.classification.calculate_classification_metrics`."""

    def test_returns_all_expected_metric_names(self) -> None:
        """Deve retornar exatamente as métricas principal e secundárias do projeto."""
        metrics = calculate_classification_metrics(
            ["positivo", "negativo"], ["positivo", "positivo"]
        )
        assert set(metrics.keys()) == {
            "f1_macro",
            "precision_macro",
            "recall_macro",
            "f1_weighted",
            "accuracy",
            "mcc",
        }

    @given(st.lists(st.sampled_from(["positivo", "negativo", "neutro"]), min_size=1, max_size=30))
    def test_metrics_values_are_bounded(self, y_true: list[str]) -> None:
        """Todas as métricas devem estar dentro de seus intervalos teóricos válidos."""
        metrics = calculate_classification_metrics(y_true, list(y_true))
        assert 0.0 <= metrics["f1_macro"] <= 1.0
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert -1.0 <= metrics["mcc"] <= 1.0


class TestCalculateRankingMetrics:
    """Testes de :mod:`metrics.ranking`."""

    def setup_method(self) -> None:
        """Prepara um cenário perfeitamente separável para os testes de ranqueamento."""
        self.y_true = ["negativo", "positivo", "negativo", "positivo"]
        self.y_score = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]])
        self.labels = ["negativo", "positivo"]

    def test_roc_auc_ovr_is_perfect_for_separable_case(self) -> None:
        """Um cenário perfeitamente separável deve gerar ROC-AUC igual a 1.0."""
        assert calculate_roc_auc_ovr(self.y_true, self.y_score, labels=self.labels) == 1.0

    def test_pr_auc_ovr_is_perfect_for_separable_case(self) -> None:
        """Um cenário perfeitamente separável deve gerar PR-AUC igual a 1.0."""
        assert calculate_pr_auc_ovr(self.y_true, self.y_score, labels=self.labels) == 1.0

    def test_calculate_ranking_metrics_returns_expected_keys(self) -> None:
        """Deve retornar as chaves ``roc_auc_ovr`` e ``pr_auc_ovr``."""
        metrics = calculate_ranking_metrics(self.y_true, self.y_score, labels=self.labels)
        assert set(metrics.keys()) == {"roc_auc_ovr", "pr_auc_ovr"}

    def test_raises_on_empty_y_true(self) -> None:
        """Deve levantar ``EmptyDatasetError`` quando ``y_true`` está vazio."""
        with pytest.raises(EmptyDatasetError):
            calculate_roc_auc_ovr([], np.empty((0, 2)), labels=self.labels)

    def test_raises_on_column_count_mismatch(self) -> None:
        """Deve levantar ``ValueError`` quando o número de colunas não bate com ``labels``."""
        with pytest.raises(ValueError, match="coluna por classe"):
            calculate_roc_auc_ovr(self.y_true, self.y_score[:, :1], labels=self.labels)


class TestCalculateAverageConfidence:
    """Testes de :func:`metrics.confidence.calculate_average_confidence`."""

    def test_averages_the_winning_probability(self) -> None:
        """Deve calcular a média das maiores probabilidades por linha."""
        y_score = np.array([[0.9, 0.1], [0.6, 0.4]])
        assert calculate_average_confidence(y_score) == 0.75

    def test_raises_on_empty_input(self) -> None:
        """Deve levantar ``EmptyDatasetError`` quando ``y_score`` está vazio."""
        with pytest.raises(EmptyDatasetError):
            calculate_average_confidence(np.empty((0, 2)))


class TestCalculateConfidenceAccuracyCorrelation:
    """Testes de :func:`metrics.confidence.calculate_confidence_accuracy_correlation`."""

    def test_returns_value_within_valid_range(self) -> None:
        """A correlação ponto-bisserial deve estar entre -1.0 e 1.0."""
        y_true = ["positivo", "negativo", "positivo", "negativo"]
        y_pred = ["positivo", "negativo", "negativo", "negativo"]
        confidences = np.array([0.95, 0.90, 0.55, 0.60])
        correlation = calculate_confidence_accuracy_correlation(y_true, y_pred, confidences)
        assert -1.0 <= correlation <= 1.0

    def test_raises_on_empty_input(self) -> None:
        """Deve levantar ``EmptyDatasetError`` quando ``y_true`` está vazio."""
        with pytest.raises(EmptyDatasetError):
            calculate_confidence_accuracy_correlation([], [], np.empty(0))


class TestCalculateMulticlassBrierScore:
    """Testes de :func:`metrics.confidence.calculate_multiclass_brier_score`."""

    def test_returns_zero_for_perfectly_confident_and_correct_predictions(self) -> None:
        """Predições perfeitas e totalmente confiantes devem gerar Brier score 0.0."""
        y_true = ["positivo", "negativo"]
        y_score = np.array([[0.0, 1.0], [1.0, 0.0]])
        assert (
            calculate_multiclass_brier_score(y_true, y_score, labels=["negativo", "positivo"])
            == 0.0
        )

    def test_raises_on_empty_input(self) -> None:
        """Deve levantar ``EmptyDatasetError`` quando ``y_true`` está vazio."""
        with pytest.raises(EmptyDatasetError):
            calculate_multiclass_brier_score([], np.empty((0, 2)))


class TestCalculateSelectivePredictionAccuracy:
    """Testes de :func:`metrics.confidence.calculate_selective_prediction_accuracy`."""

    def test_keeps_only_most_confident_fraction(self) -> None:
        """Deve calcular a acurácia apenas sobre a fração mais confiante."""
        y_true = ["positivo", "negativo", "positivo", "negativo"]
        y_pred = ["positivo", "negativo", "negativo", "negativo"]
        confidences = np.array([0.95, 0.90, 0.55, 0.60])
        assert (
            calculate_selective_prediction_accuracy(y_true, y_pred, confidences, coverage=0.5)
            == 1.0
        )

    def test_raises_on_invalid_coverage(self) -> None:
        """Deve levantar ``ValueError`` quando ``coverage`` está fora de ``(0, 1]``."""
        with pytest.raises(ValueError, match="coverage"):
            calculate_selective_prediction_accuracy(
                ["positivo"], ["positivo"], np.array([0.9]), coverage=0.0
            )

    def test_raises_on_empty_input(self) -> None:
        """Deve levantar ``EmptyDatasetError`` quando ``y_true`` está vazio."""
        with pytest.raises(EmptyDatasetError):
            calculate_selective_prediction_accuracy([], [], np.empty(0), coverage=0.5)


class TestCalculateLatencyStatistics:
    """Testes de :func:`metrics.operational.calculate_latency_statistics`."""

    def test_computes_expected_mean(self) -> None:
        """A média deve corresponder à média aritmética das latências."""
        result = calculate_latency_statistics([10.0, 12.0, 11.0, 50.0])
        assert result["mean_ms"] == 20.75

    def test_raises_on_empty_input(self) -> None:
        """Deve levantar ``EmptyDatasetError`` quando ``latencies_ms`` está vazio."""
        with pytest.raises(EmptyDatasetError):
            calculate_latency_statistics([])


class TestMeasureInferenceLatency:
    """Testes de :func:`metrics.operational.measure_inference_latency`."""

    def test_returns_non_negative_statistics(self) -> None:
        """As estatísticas de latência medidas devem ser não negativas."""
        result = measure_inference_latency(lambda x: x + 1, 1, n_repeats=3)
        assert result["mean_ms"] >= 0.0

    def test_raises_on_invalid_n_repeats(self) -> None:
        """Deve levantar ``ValueError`` quando ``n_repeats`` é menor que 1."""
        with pytest.raises(ValueError, match="n_repeats"):
            measure_inference_latency(lambda x: x, 1, n_repeats=0)


class TestCountTrainableParameters:
    """Testes de :func:`metrics.operational.count_trainable_parameters`."""

    class _FakeParameter:
        """Parâmetro de teste, com a interface mínima de um tensor PyTorch."""

        def __init__(self, n_elements: int, *, requires_grad: bool = True) -> None:
            self._n_elements = n_elements
            self.requires_grad = requires_grad

        def numel(self) -> int:
            """Retorna o número de elementos do parâmetro falso."""
            return self._n_elements

    class _FakeModel:
        """Modelo de teste com a interface mínima de ``torch.nn.Module``."""

        def __init__(self, parameters: list["TestCountTrainableParameters._FakeParameter"]) -> None:
            self._parameters = parameters

        def parameters(self) -> list["TestCountTrainableParameters._FakeParameter"]:
            """Retorna os parâmetros falsos configurados no modelo."""
            return self._parameters

    def test_counts_only_trainable_parameters(self) -> None:
        """Deve somar apenas os parâmetros com ``requires_grad=True``."""
        model = self._FakeModel(
            [self._FakeParameter(10), self._FakeParameter(5, requires_grad=False)]
        )
        assert count_trainable_parameters(model) == 10

    def test_raises_model_error_when_model_lacks_parameters_method(self) -> None:
        """Deve levantar ``ModelError`` quando o modelo não expõe ``parameters()``."""
        with pytest.raises(ModelError):
            count_trainable_parameters(object())


class TestCalculateOperationalMetrics:
    """Testes de :func:`metrics.operational.calculate_operational_metrics`."""

    def test_includes_computational_cost_when_model_is_given(self) -> None:
        """Deve incluir ``computational_cost`` quando um modelo é informado."""
        model = TestCountTrainableParameters._FakeModel(
            [TestCountTrainableParameters._FakeParameter(7)]
        )
        result = calculate_operational_metrics(lambda x: x, 1, n_repeats=2, model=model)
        assert result["computational_cost"] == 7.0

    def test_omits_computational_cost_when_model_is_none(self) -> None:
        """Não deve incluir ``computational_cost`` quando nenhum modelo é informado."""
        result = calculate_operational_metrics(lambda x: x, 1, n_repeats=2)
        assert "computational_cost" not in result
