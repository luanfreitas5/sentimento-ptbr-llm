"""Testes do gate de sanidade dos embeddings (``diagnostics.sanity``)."""

import numpy as np
import pytest

from diagnostics.sanity import (
    SanityGateResult,
    assert_sanity_gate_passed,
    evaluate_sanity_gate,
)
from exceptions.data import DataValidationError
from exceptions.pipeline import SanityGateFailedError


def _split(x: np.ndarray, y: np.ndarray, n_train: int = 300) -> tuple[np.ndarray, ...]:
    """Divide embeddings/alvo em treino e holdout disjuntos."""
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]


@pytest.fixture
def informative_binary() -> tuple[np.ndarray, ...]:
    """Alvo binário fortemente ligado a uma dimensão dos embeddings."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 8))
    y = (x[:, 0] + 0.3 * rng.normal(size=500) > 0).astype(float)
    return _split(x, y)


@pytest.fixture
def noise_binary() -> tuple[np.ndarray, ...]:
    """Alvo binário independente dos embeddings (ruído puro)."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 8))
    y = rng.integers(0, 2, size=500).astype(float)
    return _split(x, y)


class TestEvaluateSanityGate:
    """Decisão go/no-go do gate."""

    def test_informative_binary_target_is_go(self, informative_binary: tuple) -> None:
        """Sinal forte: AUC alta, IC acima de 0,5 e permutação significativa."""
        result = evaluate_sanity_gate(
            *informative_binary,
            target_name="disc",
            classification=True,
            n_bootstrap=200,
            n_permutations=200,
        )
        assert result.go
        assert result.metric_name == "roc_auc"
        assert result.holdout_score > 0.8
        assert result.ci_lower > 0.5

    def test_noise_binary_target_is_no_go(self, noise_binary: tuple) -> None:
        """Ruído puro: não passa no gate."""
        result = evaluate_sanity_gate(
            *noise_binary,
            target_name="ruido",
            classification=True,
            n_bootstrap=200,
            n_permutations=200,
        )
        assert not result.go
        assert result.reason

    def test_continuous_target_uses_r2(self) -> None:
        """Alvo contínuo usa R²."""
        rng = np.random.default_rng(2)
        x = rng.normal(size=(500, 6))
        y = 2.0 * x[:, 1] + 0.1 * rng.normal(size=500)
        result = evaluate_sanity_gate(
            *_split(x, y),
            target_name="incerteza",
            classification=False,
            n_bootstrap=200,
            n_permutations=200,
        )
        assert result.metric_name == "r2"
        assert result.go
        assert result.holdout_score > 0.9

    def test_constant_target_is_no_go_without_fitting(self) -> None:
        """Alvo com uma única classe é no-go, sem exceção do sklearn."""
        rng = np.random.default_rng(3)
        x = rng.normal(size=(50, 4))
        result = evaluate_sanity_gate(
            x[:30], np.zeros(30), x[30:], np.zeros(20), target_name="const", classification=True
        )
        assert not result.go
        assert "constante" in result.reason
        assert np.isnan(result.holdout_score)

    def test_is_deterministic_for_same_seed(self, informative_binary: tuple) -> None:
        """Mesma semente, mesmo resultado (reprodutibilidade)."""
        kwargs = {
            "target_name": "d",
            "classification": True,
            "n_bootstrap": 100,
            "n_permutations": 100,
            "random_seed": 7,
        }
        first = evaluate_sanity_gate(*informative_binary, **kwargs)
        second = evaluate_sanity_gate(*informative_binary, **kwargs)
        assert first == second

    def test_min_effect_can_flip_decision(self, informative_binary: tuple) -> None:
        """Uma margem mínima inatingível reprova até um alvo informativo."""
        result = evaluate_sanity_gate(
            *informative_binary,
            target_name="d",
            classification=True,
            n_bootstrap=100,
            n_permutations=100,
            min_effect=0.6,
        )
        assert not result.go
        assert "IC inferior" in result.reason

    def test_row_mismatch_raises(self) -> None:
        """Número de linhas incompatível entre embeddings e alvo."""
        with pytest.raises(DataValidationError, match="número de linhas"):
            evaluate_sanity_gate(
                np.zeros((3, 2)),
                np.zeros(2),
                np.zeros((2, 2)),
                np.zeros(2),
                target_name="x",
                classification=False,
            )

    def test_dimension_mismatch_raises(self) -> None:
        """Dimensão de embedding diferente entre treino e holdout."""
        with pytest.raises(DataValidationError, match="dimensão"):
            evaluate_sanity_gate(
                np.zeros((3, 2)),
                np.zeros(3),
                np.zeros((2, 5)),
                np.zeros(2),
                target_name="x",
                classification=False,
            )

    def test_non_2d_embeddings_raise(self) -> None:
        """Embeddings devem ser 2D."""
        with pytest.raises(DataValidationError, match="2D"):
            evaluate_sanity_gate(
                np.zeros(3),
                np.zeros(3),
                np.zeros(2),
                np.zeros(2),
                target_name="x",
                classification=False,
            )

    def test_empty_partition_raises(self) -> None:
        """Partição vazia falha cedo."""
        with pytest.raises(DataValidationError, match="vazia"):
            evaluate_sanity_gate(
                np.zeros((0, 2)),
                np.zeros(0),
                np.zeros((2, 2)),
                np.zeros(2),
                target_name="x",
                classification=False,
            )


class TestAssertSanityGatePassed:
    """Interrupção do fluxo quando o gate reprova."""

    def test_go_passes_silently(self) -> None:
        """Resultado go não levanta exceção."""
        ok = SanityGateResult("t", "roc_auc", 0.7, 0.6, 0.8, 0.01, 10, 10, True, "ok")
        assert_sanity_gate_passed(ok)

    def test_no_go_raises_with_summary(self) -> None:
        """Resultado no-go levanta exceção com métrica, IC e p-valor."""
        bad = SanityGateResult("alvo_x", "roc_auc", 0.51, 0.48, 0.55, 0.4, 10, 10, False, "acaso")
        with pytest.raises(SanityGateFailedError, match="alvo_x") as info:
            assert_sanity_gate_passed(bad)
        assert "roc_auc=0.5100" in str(info.value)

    def test_to_dict_is_serializable(self) -> None:
        """``to_dict`` expõe todos os campos (para o MLflow)."""
        result = SanityGateResult("t", "r2", 0.1, 0.0, 0.2, 0.03, 5, 5, True, "ok")
        assert result.to_dict()["target_name"] == "t"
        assert set(result.to_dict()) >= {"holdout_score", "ci_lower", "permutation_pvalue", "go"}
