"""Testes de temperature scaling, tabela de confiabilidade e predição com abstenção."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from evaluation.abstention import (
    calculate_risk_coverage_curve,
    evaluate_abstention_at_threshold,
    select_confidence_threshold,
)
from evaluation.calibration import (
    apply_temperature_scaling,
    calculate_negative_log_likelihood,
    calculate_reliability_table,
    fit_temperature_scaling,
)
from exceptions.data import EmptyDatasetError

LABELS = ["negativo", "neutro", "positivo"]


def _overconfident_scores(
    n_samples: int, accuracy: float, seed: int = 0
) -> tuple[list[str], np.ndarray]:
    """Gera um modelo que diz 0,95 de confiança mas acerta só ``accuracy`` das vezes."""
    generator = np.random.default_rng(seed)
    true_codes = generator.integers(0, 3, n_samples)
    correct = generator.random(n_samples) < accuracy
    pred_codes = np.where(correct, true_codes, (true_codes + 1) % 3)
    scores = np.full((n_samples, 3), 0.025)
    scores[np.arange(n_samples), pred_codes] = 0.95
    return [LABELS[code] for code in true_codes], scores


class TestTemperatureScaling:
    """Testes de :func:`apply_temperature_scaling` e :func:`fit_temperature_scaling`."""

    def test_rows_sum_to_one_and_argmax_is_preserved(self) -> None:
        """Cada linha continua somando 1 e a classe predita não muda."""
        scores = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
        scaled = apply_temperature_scaling(scores, 3.0)
        assert np.allclose(scaled.sum(axis=1), 1.0)
        assert (scaled.argmax(axis=1) == scores.argmax(axis=1)).all()

    def test_temperature_one_is_identity(self) -> None:
        """T = 1 devolve as probabilidades originais."""
        scores = np.array([[0.7, 0.2, 0.1]])
        assert np.allclose(apply_temperature_scaling(scores, 1.0), scores)

    def test_non_positive_temperature_raises(self) -> None:
        """Temperatura não positiva é rejeitada."""
        with pytest.raises(ValueError, match="temperature"):
            apply_temperature_scaling(np.array([[0.5, 0.5]]), 0.0)

    def test_overconfident_model_gets_temperature_above_one_and_lower_nll(self) -> None:
        """Modelo excessivamente confiante recebe T > 1 e a log-verossimilhança melhora."""
        y_true, scores = _overconfident_scores(2000, accuracy=0.6)
        temperature = fit_temperature_scaling(y_true, scores, labels=LABELS)
        before = calculate_negative_log_likelihood(y_true, scores, labels=LABELS).mean()
        after = calculate_negative_log_likelihood(
            y_true, apply_temperature_scaling(scores, temperature), labels=LABELS
        ).mean()
        assert temperature > 1.0
        assert after < before

    @given(temperature=st.floats(min_value=0.1, max_value=10.0))
    def test_scaling_never_changes_the_predicted_class(self, temperature: float) -> None:
        """Invariante: para qualquer T, o argmax por linha é preservado."""
        scores = np.array([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6], [0.1, 0.8, 0.1]])
        scaled = apply_temperature_scaling(scores, temperature)
        assert (scaled.argmax(axis=1) == scores.argmax(axis=1)).all()

    def test_nll_rejects_empty_and_unknown_labels(self) -> None:
        """Entrada vazia e rótulo desconhecido levantam erro."""
        with pytest.raises(EmptyDatasetError):
            calculate_negative_log_likelihood([], np.empty((0, 3)), labels=LABELS)
        with pytest.raises(ValueError, match="fora de"):
            calculate_negative_log_likelihood(["x"], np.array([[1.0, 0.0, 0.0]]), labels=LABELS)


class TestReliabilityTable:
    """Testes de :func:`calculate_reliability_table`."""

    def test_reports_n_and_interval_per_bin(self) -> None:
        """Cada faixa traz n e um IC que contém a acurácia observada."""
        y_true = ["a", "a", "b", "b"]
        y_pred = ["a", "b", "b", "b"]
        confidences = np.array([0.9, 0.9, 0.6, 0.6])
        table = calculate_reliability_table(y_true, y_pred, confidences, n_bins=10)
        assert table["n"].to_list() == [2, 2]
        assert (table["ci_low"] <= table["accuracy"]).all()
        assert (table["accuracy"] <= table["ci_high"]).all()
        assert (table["ci_low"] >= 0).all()
        assert (table["ci_high"] <= 1).all()

    def test_small_bins_have_wide_intervals(self) -> None:
        """Faixa com poucas amostras tem IC mais largo que a mesma taxa com muitas."""
        small = calculate_reliability_table(["a"] * 3, ["a"] * 3, np.array([0.9] * 3), n_bins=1)
        large = calculate_reliability_table(
            ["a"] * 300, ["a"] * 300, np.array([0.9] * 300), n_bins=1
        )
        assert small["ci_low"][0] < large["ci_low"][0]


class TestRiskCoverage:
    """Testes de :func:`calculate_risk_coverage_curve`."""

    def test_full_coverage_keeps_everyone_and_risk_is_one_minus_accuracy(self) -> None:
        """Com cobertura 1,0 ninguém é abstido e risco = 1 - acurácia."""
        y_true, scores = _overconfident_scores(500, accuracy=0.7)
        y_pred = [LABELS[i] for i in scores.argmax(axis=1)]
        curve = calculate_risk_coverage_curve(
            y_true, y_pred, scores.max(axis=1), coverages=[1.0], labels=LABELS, n_bootstrap=20
        )
        accuracy = np.mean([a == b for a, b in zip(y_true, y_pred, strict=True)])
        assert curve["n_answered"][0] == 500
        assert curve["accuracy"][0] == pytest.approx(accuracy)
        assert curve["risk"][0] == pytest.approx(1 - accuracy)

    def test_informative_confidence_makes_f1_grow_as_coverage_drops(self) -> None:
        """Se a confiança é informativa, abster-se nos menos confiantes eleva a F1-macro."""
        generator = np.random.default_rng(1)
        n_samples = 1500
        true_codes = generator.integers(0, 3, n_samples)
        confidence = generator.random(n_samples)
        correct = generator.random(n_samples) < confidence  # mais confiante => mais acerto
        pred_codes = np.where(correct, true_codes, (true_codes + 1) % 3)
        curve = calculate_risk_coverage_curve(
            [LABELS[c] for c in true_codes],
            [LABELS[c] for c in pred_codes],
            confidence,
            coverages=[0.3, 0.6, 1.0],
            labels=LABELS,
            n_bootstrap=20,
        )
        f1_values = curve.sort("coverage")["f1_macro"].to_list()
        assert f1_values[0] > f1_values[1] > f1_values[2]
        assert (curve["f1_macro_ci_low"] <= curve["f1_macro_ci_high"]).all()

    def test_invalid_inputs_raise(self) -> None:
        """Entrada vazia e cobertura fora de (0, 1] levantam erro."""
        with pytest.raises(EmptyDatasetError):
            calculate_risk_coverage_curve([], [], np.array([]), labels=LABELS)
        with pytest.raises(ValueError, match="coverage"):
            calculate_risk_coverage_curve(
                ["negativo"], ["negativo"], np.array([0.9]), coverages=[0.0], labels=LABELS
            )


class TestThresholdAndAbstention:
    """Testes de :func:`select_confidence_threshold` e :func:`evaluate_abstention_at_threshold`."""

    def test_threshold_keeps_the_requested_fraction(self) -> None:
        """O limiar mantém (aproximadamente) a cobertura pedida."""
        confidences = np.linspace(0.34, 1.0, 1000)
        threshold = select_confidence_threshold(confidences, 0.8)
        assert (confidences >= threshold).mean() == pytest.approx(0.8, abs=0.01)

    def test_threshold_validates_inputs(self) -> None:
        """Vazio e cobertura inválida levantam erro."""
        with pytest.raises(EmptyDatasetError):
            select_confidence_threshold(np.array([]), 0.5)
        with pytest.raises(ValueError, match="coverage"):
            select_confidence_threshold(np.array([0.5]), 1.5)

    def test_counts_add_up_and_the_error_was_the_abstained_row(self) -> None:
        """Respondidos + abstidos = total; o único erro estava entre os abstidos."""
        y_true = ["negativo", "neutro", "positivo", "positivo"]
        y_pred = ["negativo", "neutro", "negativo", "positivo"]
        confidences = np.array([0.9, 0.8, 0.4, 0.7])
        result = evaluate_abstention_at_threshold(
            y_true, y_pred, confidences, 0.6, labels=LABELS, n_bootstrap=10
        )
        assert result["n_answered"] == 3
        assert result["n_abstained"] == 1
        assert result["coverage"] == pytest.approx(0.75)
        assert result["f1_macro_answered"] == pytest.approx(1.0)

    def test_total_and_zero_abstention_give_nan_instead_of_errors(self) -> None:
        """Abstenção total ou nula deixa a métrica sem amostras como ``nan``."""
        args = (["negativo"], ["negativo"], np.array([0.9]))
        everyone = evaluate_abstention_at_threshold(*args, 2.0, labels=LABELS, n_bootstrap=5)
        nobody = evaluate_abstention_at_threshold(*args, 0.0, labels=LABELS, n_bootstrap=5)
        assert np.isnan(everyone["f1_macro_answered"])
        assert np.isnan(nobody["f1_macro_abstained"])
