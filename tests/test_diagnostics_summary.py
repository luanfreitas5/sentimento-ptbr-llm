"""Testes dos resumos em texto das etapas do diagnóstico."""

from pathlib import Path
from types import SimpleNamespace

import polars as pl

from diagnostics.summary import (
    format_comparison_summary,
    format_hypotheses_summary,
    format_validation_summary,
    save_summary_report,
)


def _gate(go: bool) -> SimpleNamespace:
    return SimpleNamespace(
        go=go,
        reason="ok",
        metric_name="roc_auc",
        holdout_score=0.71,
        ci_lower=0.6,
        ci_upper=0.8,
        permutation_pvalue=0.01,
        n_train=100,
        n_holdout=50,
    )


def test_hypotheses_summary_lists_top_hypotheses() -> None:
    """O resumo traz o gate e as hipóteses ordenadas pelo score de separação."""
    table = pl.DataFrame({"hypothesis": ["a", "b"], "separation_score": [0.1, 0.9]})
    result = SimpleNamespace(target_slug="t", gate=_gate(True), hypotheses=table, output_path="x")
    text = format_hypotheses_summary(result)
    assert "APROVADO" in text
    assert text.index(" b ") < text.index(" a ")


def test_hypotheses_summary_gate_failed() -> None:
    """Gate reprovado informa que nenhuma hipótese foi gerada."""
    result = SimpleNamespace(target_slug="t", gate=_gate(False), hypotheses=None, output_path=None)
    assert "REPROVADO" in format_hypotheses_summary(result)
    assert "Nenhuma hipótese" in format_hypotheses_summary(result)


def test_validation_summary_counts_survivors() -> None:
    """Sobreviventes aparecem primeiro e a contagem é informada."""
    validated = pl.DataFrame(
        {
            "hypothesis": ["a", "b"],
            "survives": [False, True],
            "regression_pval_adjusted": [0.5, 0.01],
        }
    )
    outcome = SimpleNamespace(validated=validated, n_survivors=1, sample_path=None)
    text = format_validation_summary("t", outcome, correction="bonferroni")
    assert "1/2" in text
    assert text.index("[SOBREVIVE] b") < text.index("[descartada] a")


def test_comparison_summary_and_save(tmp_path: Path) -> None:
    """A comparação lista métricas por modelo e o arquivo é gravado."""
    result = pl.DataFrame({"model": ["m1"], "delta_mcc": [0.05]})
    text = format_comparison_summary(
        result, {"disagreement_rate": 0.1, "mean_agreement": 0.9}, None, gold_name="g"
    )
    assert "Modelo: m1" in text
    assert "delta_mcc: 0.05" in text
    path = save_summary_report(text, tmp_path / "novo", "comparacao_g")
    assert path.name == "resumo_comparacao_g.txt"
    assert "Modelo: m1" in path.read_text(encoding="utf-8")
