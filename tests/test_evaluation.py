"""Testes da avaliação rigorosa de classificadores (``src/evaluation``)."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from evaluation.ablation import calculate_ablation_impact, identify_most_impactful_component
from evaluation.calibration import (
    calculate_calibration_metrics,
    calculate_expected_calibration_error,
    calculate_reliability_curve,
)
from evaluation.evaluator import (
    EvaluationResult,
    calculate_bootstrap_confidence_intervals,
    evaluate_classifier,
)
from evaluation.reports import (
    build_evaluation_report,
    merge_evaluation_reports,
    save_evaluation_report,
)
from evaluation.significance import (
    run_friedman_test,
    run_mcnemar_test,
    run_nemenyi_post_hoc_test,
    run_wilcoxon_signed_rank_test,
)
from evaluation.slice_evaluation import evaluate_metrics_by_slice, identify_underperforming_slices
from exceptions.data import EmptyDatasetError


class TestCalculateBootstrapConfidenceIntervals:
    """Testes de :func:`evaluation.evaluator.calculate_bootstrap_confidence_intervals`."""

    def test_includes_all_classification_metrics(self) -> None:
        """Deve retornar um intervalo para cada métrica de classificação."""
        y_true = ["positivo", "negativo", "positivo", "negativo", "neutro"]
        y_pred = ["positivo", "negativo", "negativo", "negativo", "neutro"]
        intervals = calculate_bootstrap_confidence_intervals(y_true, y_pred, n_bootstrap=20)
        assert "f1_macro" in intervals
        assert "mcc" in intervals

    def test_interval_bounds_are_ordered(self) -> None:
        """O limite inferior de cada intervalo não deve exceder o limite superior."""
        y_true = ["positivo", "negativo"] * 5
        y_pred = ["positivo", "negativo"] * 5
        intervals = calculate_bootstrap_confidence_intervals(y_true, y_pred, n_bootstrap=20)
        for lower_bound, upper_bound in intervals.values():
            assert lower_bound <= upper_bound

    def test_raises_on_empty_y_true(self) -> None:
        """Deve levantar ``EmptyDatasetError`` quando ``y_true`` está vazio."""
        with pytest.raises(EmptyDatasetError):
            calculate_bootstrap_confidence_intervals([], [])


class TestEvaluateClassifier:
    """Testes de :func:`evaluation.evaluator.evaluate_classifier`."""

    def test_returns_evaluation_result_with_expected_structure(self) -> None:
        """Deve retornar um :class:`EvaluationResult` com todos os componentes preenchidos."""
        y_true = ["positivo", "negativo", "positivo", "negativo", "neutro"]
        y_pred = ["positivo", "negativo", "negativo", "negativo", "neutro"]
        result = evaluate_classifier(y_true, y_pred, n_bootstrap=20)
        assert isinstance(result, EvaluationResult)
        assert "f1_macro" in result.point_metrics
        assert "f1_macro" in result.confidence_intervals
        assert set(result.per_class_report.keys()) == {"negativo", "neutro", "positivo"}
        assert result.confusion_matrix.shape == (3, 3)

    def test_includes_ranking_metrics_when_y_score_is_given(self) -> None:
        """Deve incluir métricas de ranqueamento quando ``y_score`` é informado."""
        y_true = ["negativo", "positivo", "neutro", "negativo", "positivo", "neutro"]
        y_pred = ["negativo", "positivo", "neutro", "negativo", "positivo", "neutro"]
        y_score = np.array(
            [
                [0.9, 0.05, 0.05],
                [0.05, 0.9, 0.05],
                [0.05, 0.05, 0.9],
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
            ]
        )
        result = evaluate_classifier(
            y_true,
            y_pred,
            y_score=y_score,
            labels=["negativo", "positivo", "neutro"],
            n_bootstrap=10,
        )
        assert "roc_auc_ovr" in result.point_metrics

    def test_raises_on_empty_y_true(self) -> None:
        """Deve levantar ``EmptyDatasetError`` quando ``y_true`` está vazio."""
        with pytest.raises(EmptyDatasetError):
            evaluate_classifier([], [])


class TestRunMcnemarTest:
    """Testes de :func:`evaluation.significance.run_mcnemar_test`."""

    def test_returns_statistic_and_p_value(self) -> None:
        """Deve retornar as chaves ``statistic`` e ``p_value``."""
        y_true = ["positivo", "negativo", "positivo", "negativo"]
        y_pred_a = ["positivo", "negativo", "positivo", "negativo"]
        y_pred_b = ["negativo", "negativo", "positivo", "negativo"]
        result = run_mcnemar_test(y_true, y_pred_a, y_pred_b)
        assert set(result.keys()) == {"statistic", "p_value"}
        assert 0.0 <= result["p_value"] <= 1.0

    def test_raises_on_empty_y_true(self) -> None:
        """Deve levantar ``EmptyDatasetError`` quando ``y_true`` está vazio."""
        with pytest.raises(EmptyDatasetError):
            run_mcnemar_test([], [], [])


class TestRunWilcoxonSignedRankTest:
    """Testes de :func:`evaluation.significance.run_wilcoxon_signed_rank_test`."""

    def test_returns_statistic_and_p_value(self) -> None:
        """Deve retornar as chaves ``statistic`` e ``p_value``."""
        result = run_wilcoxon_signed_rank_test([0.80, 0.82, 0.79], [0.75, 0.78, 0.74])
        assert set(result.keys()) == {"statistic", "p_value"}

    def test_raises_on_empty_input(self) -> None:
        """Deve levantar ``EmptyDatasetError`` quando ``scores_a`` está vazio."""
        with pytest.raises(EmptyDatasetError):
            run_wilcoxon_signed_rank_test([], [])


class TestRunFriedmanTest:
    """Testes de :func:`evaluation.significance.run_friedman_test`."""

    def test_returns_statistic_and_p_value(self) -> None:
        """Deve retornar as chaves ``statistic`` e ``p_value``."""
        result = run_friedman_test([0.80, 0.82, 0.79], [0.75, 0.78, 0.74], [0.70, 0.71, 0.69])
        assert set(result.keys()) == {"statistic", "p_value"}

    def test_raises_on_fewer_than_three_models(self) -> None:
        """Deve levantar ``ValueError`` quando menos de 3 modelos são informados."""
        with pytest.raises(ValueError, match="3 modelos"):
            run_friedman_test([0.8, 0.7], [0.6, 0.5])


class TestRunNemenyiPostHocTest:
    """Testes de :func:`evaluation.significance.run_nemenyi_post_hoc_test`."""

    def test_ranks_best_model_first(self) -> None:
        """O modelo com maior métrica em todas as dobras deve ter rank médio 1.0."""
        scores_matrix = np.array([[0.80, 0.75, 0.70], [0.82, 0.78, 0.71], [0.79, 0.74, 0.69]])
        result = run_nemenyi_post_hoc_test(scores_matrix)
        assert result["average_ranks"].tolist() == [1.0, 2.0, 3.0]
        assert result["critical_difference"] > 0

    def test_raises_on_single_model(self) -> None:
        """Deve levantar ``ValueError`` quando há apenas um modelo (coluna)."""
        with pytest.raises(ValueError, match="2 modelos"):
            run_nemenyi_post_hoc_test(np.array([[0.8], [0.7]]))

    def test_raises_on_empty_input(self) -> None:
        """Deve levantar ``EmptyDatasetError`` quando a matriz está vazia."""
        with pytest.raises(EmptyDatasetError):
            run_nemenyi_post_hoc_test(np.empty((0, 0)))


class TestCalculateReliabilityCurve:
    """Testes de :func:`evaluation.calibration.calculate_reliability_curve`."""

    def test_bins_samples_by_confidence(self) -> None:
        """Deve distribuir as amostras nas faixas de confiança corretas."""
        y_true = ["positivo", "positivo", "negativo", "negativo"]
        y_pred = ["positivo", "positivo", "negativo", "positivo"]
        confidences = np.array([0.9, 0.95, 0.85, 0.3])
        curve = calculate_reliability_curve(y_true, y_pred, confidences, n_bins=2)
        assert curve["bin_counts"].tolist() == [1, 3]

    def test_raises_on_empty_input(self) -> None:
        """Deve levantar ``EmptyDatasetError`` quando ``y_true`` está vazio."""
        with pytest.raises(EmptyDatasetError):
            calculate_reliability_curve([], [], np.empty(0))

    def test_raises_on_invalid_n_bins(self) -> None:
        """Deve levantar ``ValueError`` quando ``n_bins`` é menor que 1."""
        with pytest.raises(ValueError, match="n_bins"):
            calculate_reliability_curve(["positivo"], ["positivo"], np.array([0.9]), n_bins=0)


class TestCalculateExpectedCalibrationError:
    """Testes de :func:`evaluation.calibration.calculate_expected_calibration_error`."""

    def test_computes_expected_value(self) -> None:
        """Deve calcular corretamente o ECE para um único bin."""
        y_true = ["positivo", "positivo", "negativo", "negativo"]
        y_pred = ["positivo", "positivo", "negativo", "positivo"]
        confidences = np.array([0.9, 0.9, 0.9, 0.9])
        ece = calculate_expected_calibration_error(y_true, y_pred, confidences, n_bins=1)
        assert round(ece, 4) == 0.15


class TestCalculateCalibrationMetrics:
    """Testes de :func:`evaluation.calibration.calculate_calibration_metrics`."""

    def test_returns_expected_keys(self) -> None:
        """Deve retornar as chaves ``brier_score`` e ``expected_calibration_error``."""
        y_true = ["positivo", "negativo"]
        y_score = np.array([[0.1, 0.9], [0.8, 0.2]])
        result = calculate_calibration_metrics(y_true, y_score, labels=["negativo", "positivo"])
        assert set(result.keys()) == {"brier_score", "expected_calibration_error"}


class TestEvaluateMetricsBySlice:
    """Testes de :func:`evaluation.slice_evaluation.evaluate_metrics_by_slice`."""

    def test_returns_one_row_per_slice(self) -> None:
        """Deve retornar uma linha por valor único de fatia."""
        y_true = ["positivo", "positivo", "negativo", "negativo"]
        y_pred = ["positivo", "negativo", "negativo", "negativo"]
        slices = ["twitter", "twitter", "reddit", "reddit"]
        result = evaluate_metrics_by_slice(y_true, y_pred, slices)
        assert sorted(result["slice"].to_list()) == ["reddit", "twitter"]
        assert "f1_macro" in result.columns

    def test_raises_on_length_mismatch(self) -> None:
        """Deve levantar ``ValueError`` quando os tamanhos não coincidem."""
        with pytest.raises(ValueError, match="mesmo tamanho"):
            evaluate_metrics_by_slice(["positivo"], ["positivo"], ["a", "b"])

    def test_raises_on_empty_input(self) -> None:
        """Deve levantar ``EmptyDatasetError`` quando ``y_true`` está vazio."""
        with pytest.raises(EmptyDatasetError):
            evaluate_metrics_by_slice([], [], [])


class TestIdentifyUnderperformingSlices:
    """Testes de :func:`evaluation.slice_evaluation.identify_underperforming_slices`."""

    def test_filters_slices_below_threshold(self) -> None:
        """Deve manter apenas as fatias abaixo do limiar informado."""
        metrics = pl.DataFrame({"slice": ["a", "b"], "f1_macro": [0.9, 0.4]})
        result = identify_underperforming_slices(metrics, threshold=0.65)
        assert result["slice"].to_list() == ["b"]

    def test_raises_on_empty_input(self) -> None:
        """Deve levantar ``EmptyDatasetError`` quando ``slice_metrics`` está vazio."""
        with pytest.raises(EmptyDatasetError):
            identify_underperforming_slices(pl.DataFrame({"slice": [], "f1_macro": []}))


class TestCalculateAblationImpact:
    """Testes de :func:`evaluation.ablation.calculate_ablation_impact`."""

    def test_ranks_components_by_impact(self) -> None:
        """O componente que mais reduz a métrica deve aparecer primeiro."""
        baseline = {"f1_macro": 0.80}
        ablated = {
            "sem_embeddings_contextuais": {"f1_macro": 0.65},
            "sem_autoencoder": {"f1_macro": 0.78},
        }
        result = calculate_ablation_impact(baseline, ablated)
        assert result["component"].to_list() == ["sem_embeddings_contextuais", "sem_autoencoder"]

    def test_raises_on_empty_ablated_metrics(self) -> None:
        """Deve levantar ``EmptyDatasetError`` quando ``ablated_metrics`` está vazio."""
        with pytest.raises(EmptyDatasetError):
            calculate_ablation_impact({"f1_macro": 0.8}, {})

    def test_raises_on_unknown_metric_name(self) -> None:
        """Deve levantar ``ValueError`` quando a métrica não existe no baseline."""
        with pytest.raises(ValueError, match="metric_name"):
            calculate_ablation_impact(
                {"f1_macro": 0.8}, {"componente": {"f1_macro": 0.7}}, metric_name="inexistente"
            )


class TestIdentifyMostImpactfulComponent:
    """Testes de :func:`evaluation.ablation.identify_most_impactful_component`."""

    def test_returns_first_row_as_dict(self) -> None:
        """Deve retornar a primeira linha (maior impacto) como dicionário."""
        impact = pl.DataFrame(
            {
                "component": ["a", "b"],
                "baseline_value": [0.8, 0.8],
                "ablated_value": [0.5, 0.7],
                "impact": [0.3, 0.1],
            }
        )
        assert identify_most_impactful_component(impact)["component"] == "a"

    def test_raises_on_empty_input(self) -> None:
        """Deve levantar ``EmptyDatasetError`` quando o DataFrame está vazio."""
        with pytest.raises(EmptyDatasetError):
            identify_most_impactful_component(pl.DataFrame({"component": []}))


class TestBuildEvaluationReport:
    """Testes de :func:`evaluation.reports.build_evaluation_report`."""

    def test_includes_one_row_per_metric(self) -> None:
        """Deve gerar uma linha por métrica em ``point_metrics``."""
        result = EvaluationResult(
            point_metrics={"f1_macro": 0.80, "mcc": 0.6},
            confidence_intervals={"f1_macro": (0.75, 0.85)},
            per_class_report={},
            confusion_matrix=np.zeros((3, 3)),
        )
        report = build_evaluation_report("naive_bayes", result)
        assert report.height == 2
        assert report.filter(pl.col("metric_name") == "f1_macro")["ci_lower"].item() == 0.75

    def test_uses_none_for_metric_without_confidence_interval(self) -> None:
        """Deve usar ``None`` quando a métrica não tem intervalo de confiança."""
        result = EvaluationResult(
            point_metrics={"mcc": 0.6},
            confidence_intervals={},
            per_class_report={},
            confusion_matrix=np.zeros((3, 3)),
        )
        report = build_evaluation_report("naive_bayes", result)
        assert report["ci_lower"].item() is None


class TestMergeEvaluationReports:
    """Testes de :func:`evaluation.reports.merge_evaluation_reports`."""

    def test_concatenates_reports_vertically(self) -> None:
        """Deve concatenar todos os relatórios verticalmente, preservando a ordem."""
        report_a = pl.DataFrame(
            {"model_name": ["a"], "metric_name": ["f1_macro"], "metric_value": [0.8]}
        )
        report_b = pl.DataFrame(
            {"model_name": ["b"], "metric_name": ["f1_macro"], "metric_value": [0.7]}
        )
        merged = merge_evaluation_reports([report_a, report_b])
        assert merged["model_name"].to_list() == ["a", "b"]

    def test_raises_on_empty_list(self) -> None:
        """Deve levantar ``EmptyDatasetError`` quando a lista está vazia."""
        with pytest.raises(EmptyDatasetError):
            merge_evaluation_reports([])


class TestSaveEvaluationReport:
    """Testes de :func:`evaluation.reports.save_evaluation_report`."""

    def test_writes_csv_file(self, tmp_path: Path) -> None:
        """Deve criar o arquivo CSV no caminho informado, incluindo diretórios pais."""
        output_path = tmp_path / "subdir" / "relatorio.csv"
        save_evaluation_report(
            pl.DataFrame({"metric_name": ["f1_macro"], "metric_value": [0.8]}), output_path
        )
        assert output_path.is_file()
