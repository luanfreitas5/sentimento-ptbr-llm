"""Testes da validação humana (n=500) e do erro de reconstrução do autoencoder."""

import numpy as np
import polars as pl
import pytest

from diagnostics.human_validation import (
    STRATA,
    assign_confidence_strata,
    draw_stratified_sample,
    evaluate_raters,
)
from diagnostics.reconstruction import flag_anomalies, summarize_error_by_group
from exceptions.data import EmptyDatasetError


def _table(n: int = 90) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    labels = np.array(["negativo", "neutro", "positivo"])
    return pl.DataFrame(
        {
            "id": [str(i) for i in range(n)],
            "conf_hf": rng.uniform(0.3, 1.0, n),
            "lab_huggingface": rng.choice(labels, n),
            "lab_openai": rng.choice(labels, n),
            "lab_unb": rng.choice(labels, n),
        }
    )


def test_strata_cover_all_rows_and_sample_is_balanced() -> None:
    table = assign_confidence_strata(_table())
    sample = draw_stratified_sample(table, sample_size=30, random_seed=1)
    assert sample.height == 30
    counts = sample["stratum"].value_counts()
    assert set(counts["stratum"]) == set(STRATA)
    assert counts["count"].to_list() == [10, 10, 10]
    # peso = N_faixa / n_faixa
    assert sample["weight"].min() == pytest.approx(90 / 3 / 10, rel=0.1)


def test_evaluate_raters_perfect_agreement() -> None:
    table = draw_stratified_sample(
        assign_confidence_strata(_table()), sample_size=30, random_seed=1
    )
    joined = table.with_columns(
        pl.col("lab_huggingface").alias("rotulo_humano"),
        pl.col("lab_huggingface").alias("lab_openai"),
    )
    result = evaluate_raters(joined, n_bootstrap=20, confidence_level=0.95, random_seed=0)
    primary = result.filter(pl.col("rotulador") == "huggingface_primario").row(0, named=True)
    assert primary["concordancia_ponderada"] == pytest.approx(1.0)
    assert primary["ic_inferior"] <= primary["concordancia_ponderada"] <= primary["ic_superior"]


def test_evaluate_raters_rejects_empty() -> None:
    with pytest.raises(EmptyDatasetError):
        evaluate_raters(pl.DataFrame(), n_bootstrap=5, confidence_level=0.95, random_seed=0)


def test_flag_anomalies_marks_top_tail() -> None:
    mask, threshold = flag_anomalies(np.arange(100, dtype=float), 95)
    assert int(mask.sum()) == 5
    assert threshold == pytest.approx(94.05)
    with pytest.raises(EmptyDatasetError):
        flag_anomalies(np.array([]), 95)


def test_summarize_error_by_group_has_both_groupings() -> None:
    scored = pl.DataFrame(
        {
            "reconstruction_error": [0.1, 0.2, 0.9, 0.3],
            "is_anomaly": [False, False, True, False],
            "lab_huggingface": ["neutro", "neutro", "positivo", None],
            "conf_hf": [0.9, 0.4, 0.3, 0.8],
        }
    )
    summary = summarize_error_by_group(scored)
    assert set(summary["grupo"]) == {"classe_primario", "confianca_primario"}
    assert summary["n"].sum() == 6  # 3 posts rotulados contados em cada agrupamento


def test_mcnemar_comparisons_counts_exclusive_hits() -> None:
    from diagnostics.human_validation import run_mcnemar_comparisons

    joined = pl.DataFrame(
        {
            "rotulo_humano": ["positivo"] * 6,
            "lab_huggingface": ["positivo", "positivo", "positivo", "neutro", "neutro", "neutro"],
            "lab_openai": ["positivo", "neutro", "neutro", "positivo", "positivo", "positivo"],
            "lab_unb": ["positivo"] * 6,
        }
    )
    result = run_mcnemar_comparisons(joined).filter(
        (pl.col("rotulador_a") == "huggingface_primario") & (pl.col("rotulador_b") == "gold_openai")
    )
    row = result.row(0, named=True)
    assert (row["so_a_acerta"], row["so_b_acerta"]) == (2, 3)
    assert 0.0 <= row["p_valor"] <= 1.0
