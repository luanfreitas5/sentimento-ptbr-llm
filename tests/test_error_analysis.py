"""Testes de :mod:`evaluation.error_analysis` (fatias com IC e análise de erros)."""

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from evaluation.error_analysis import (
    ALL_DIMENSIONS,
    CATEGORY_IRONY,
    CATEGORY_LABEL_NOISE,
    CATEGORY_NEGATION,
    CATEGORY_NO_CONTEXT,
    CATEGORY_OTHER,
    SOURCE_BOTH_WRONG,
    SOURCE_CORRECT,
    SOURCE_LABEL_ERROR,
    SOURCE_MODEL_ERROR,
    SOURCE_REPRODUCES_LABEL,
    build_slice_flags,
    build_top_error_table,
    classify_error_source,
    evaluate_slices_with_uncertainty,
    select_best_model_by_category,
    suggest_error_category,
    summarize_error_patterns,
    summarize_error_sources,
)
from evaluation.predictions import ModelPredictions
from exceptions.data import EmptyDatasetError

LABELS = ["positivo", "negativo", "neutro"]


def _predictions(
    name: str, category: str, y_pred: list[str], scores: np.ndarray | None
) -> ModelPredictions:
    """Cria predições sintéticas sobre todo o teste."""
    return ModelPredictions(
        name=name,
        category=category,
        indices=list(range(len(y_pred))),
        y_pred=y_pred,
        y_score=scores,
        inference_ms_per_sample=0.0,
    )


class TestBuildSliceFlags:
    """Detecção das fatias linguísticas."""

    def test_detects_each_binary_dimension(self) -> None:
        """Emoji, negação, hashtag, gíria e ironia aparente são marcados."""
        flags = build_slice_flags(
            ["adorei 😍", "não gostei", "mano que #top", "claro que sim", "texto simples aqui"]
        )
        assert flags["emoji"].to_list()[0] == "com_emoji"
        assert flags["negacao"].to_list()[1] == "com_negacao"
        assert flags["hashtag"].to_list()[2] == "com_hashtag"
        assert flags["giria"].to_list()[2] == "com_giria"
        assert flags["ironia_aparente"].to_list()[3] == "com_ironia_aparente"
        assert flags.row(4, named=True)["emoji"] == "sem_emoji"

    def test_has_all_dimensions_and_length_terciles(self) -> None:
        """Todas as dimensões existem e o comprimento é dividido em tercis."""
        flags = build_slice_flags(["a", "a b c", "a b c d e f g"])
        assert set(flags.columns) == set(ALL_DIMENSIONS)
        assert flags["comprimento_texto"].to_list() == ["1_curto", "2_medio", "3_longo"]

    def test_empty_input_raises(self) -> None:
        """Entrada vazia levanta ``EmptyDatasetError``."""
        with pytest.raises(EmptyDatasetError):
            build_slice_flags([])


class TestEvaluateSlicesWithUncertainty:
    """Macro-F1 por fatia com IC, diferença contra o resto e fatias pequenas."""

    @staticmethod
    def _setup() -> tuple[list[str], list[str], pl.DataFrame]:
        """Cenário em que a fatia ``com_emoji`` é claramente pior que o resto."""
        rng = np.random.default_rng(0)
        y_true = [LABELS[i] for i in rng.integers(0, 3, 600)]
        in_slice = np.arange(600) < 200
        y_pred = [
            label if (not flagged or rng.random() < 0.3) else LABELS[(LABELS.index(label) + 1) % 3]
            for label, flagged in zip(y_true, in_slice, strict=True)
        ]
        flags = pl.DataFrame({"emoji": ["com_emoji" if f else "sem_emoji" for f in in_slice]})
        return y_true, y_pred, flags

    def test_detects_a_clearly_worse_slice(self) -> None:
        """A fatia ruim tem IC da diferença inteiro abaixo de zero."""
        y_true, y_pred, flags = self._setup()
        table = evaluate_slices_with_uncertainty(
            y_true, y_pred, flags, dimensions=["emoji"], n_bootstrap=200
        )
        row = table.filter(pl.col("slice") == "com_emoji").row(0, named=True)
        assert row["n"] == 200
        assert row["f1_macro_ci_low"] <= row["f1_macro"] <= row["f1_macro_ci_high"]
        assert row["f1_macro_difference"] < 0
        assert row["significantly_worse"] is True

    def test_small_slice_is_flagged_and_never_significant(self) -> None:
        """Fatia abaixo do tamanho mínimo é marcada e nunca vira 'pior significativa'."""
        y_true, y_pred, flags = self._setup()
        table = evaluate_slices_with_uncertainty(
            y_true, y_pred, flags, dimensions=["emoji"], n_bootstrap=50, min_slice_size=500
        )
        assert table["small_sample"].all()
        assert not table["significantly_worse"].any()

    def test_is_deterministic_for_the_same_seed(self) -> None:
        """A mesma semente produz exatamente a mesma tabela."""
        y_true, y_pred, flags = self._setup()
        options = {"dimensions": ["emoji"], "n_bootstrap": 30}
        first = evaluate_slices_with_uncertainty(y_true, y_pred, flags, **options)
        second = evaluate_slices_with_uncertainty(y_true, y_pred, flags, **options)
        assert first.equals(second)

    def test_rejects_mismatched_sizes_and_unknown_dimension(self) -> None:
        """Tamanhos diferentes e dimensão inexistente levantam ``ValueError``."""
        flags = pl.DataFrame({"emoji": ["a", "b"]})
        with pytest.raises(ValueError, match="mesmo tamanho"):
            evaluate_slices_with_uncertainty(["positivo"], ["positivo", "neutro"], flags)
        with pytest.raises(ValueError, match="ausentes"):
            evaluate_slices_with_uncertainty(
                ["positivo"] * 2, ["positivo"] * 2, flags, dimensions=["x"]
            )

    def test_empty_input_raises(self) -> None:
        """Entrada vazia levanta ``EmptyDatasetError``."""
        with pytest.raises(EmptyDatasetError):
            evaluate_slices_with_uncertainty([], [], pl.DataFrame())

    @settings(max_examples=15, deadline=None)
    @given(st.lists(st.sampled_from(LABELS), min_size=40, max_size=80))
    def test_perfect_predictions_never_flag_a_slice_as_worse(self, y_true: list[str]) -> None:
        """Invariante: predição perfeita nunca deixa uma fatia pior que o resto."""
        flags = pl.DataFrame({"d": ["a" if i % 2 else "b" for i in range(len(y_true))]})
        table = evaluate_slices_with_uncertainty(
            y_true, y_true, flags, dimensions=["d"], n_bootstrap=10
        )
        assert not table["significantly_worse"].any()
        assert (table["accuracy"] == 1.0).all()


class TestErrorSources:
    """Decomposição entre erro do modelo e erro do rótulo de treino."""

    @pytest.mark.parametrize(
        ("label", "predicted", "reference", "expected"),
        [
            ("positivo", "positivo", "positivo", SOURCE_CORRECT),
            ("positivo", "negativo", "positivo", SOURCE_MODEL_ERROR),
            ("negativo", "positivo", "positivo", SOURCE_LABEL_ERROR),
            ("negativo", "negativo", "positivo", SOURCE_REPRODUCES_LABEL),
            ("negativo", "neutro", "positivo", SOURCE_BOTH_WRONG),
        ],
    )
    def test_classify_error_source(
        self, label: str, predicted: str, reference: str, expected: str
    ) -> None:
        """Cada combinação rótulo/predição/referência cai em uma origem."""
        assert classify_error_source(label, predicted, reference) == expected

    def test_summary_counts_only_tweets_covered_by_the_reference(self) -> None:
        """Tweets sem gold ficam fora da contagem."""
        item = _predictions("m", "classical", ["positivo", "negativo", "neutro"], None)
        references = {"gold": pl.DataFrame({"id": ["0", "1"], "label": ["positivo", "neutro"]})}
        table = summarize_error_sources(
            [item], ["positivo", "positivo", "neutro"], ["0", "1", "2"], references
        )
        row = table.row(0, named=True)
        assert row["n"] == 2
        assert row[SOURCE_CORRECT] == 1
        assert row[SOURCE_BOTH_WRONG] == 1
        shares = sum(row[f"{name}_share"] for name in (SOURCE_CORRECT, SOURCE_BOTH_WRONG))
        assert shares == pytest.approx(1.0)

    def test_without_references_returns_empty(self) -> None:
        """Sem referência não há decomposição."""
        assert summarize_error_sources([], [], [], {}).is_empty()


class TestSuggestErrorCategory:
    """Prioridade das regras de categoria sugerida."""

    def test_priority_order(self) -> None:
        """Ruído de rótulo > ironia > contexto ausente > negação > outro."""
        long_text = "eu gostei do filme de ontem"
        common = {"label": "positivo", "predicted": "negativo", "confidence": 0.9}
        assert (
            suggest_error_category(long_text, error_source=SOURCE_LABEL_ERROR, flags={}, **common)
            == CATEGORY_LABEL_NOISE
        )
        assert (
            suggest_error_category(
                long_text,
                error_source=None,
                flags={"ironia_aparente": "com_ironia_aparente"},
                **common,
            )
            == CATEGORY_IRONY
        )
        assert (
            suggest_error_category("oi", error_source=None, flags={}, **common)
            == CATEGORY_NO_CONTEXT
        )
        assert (
            suggest_error_category(
                long_text, error_source=None, flags={"negacao": "com_negacao"}, **common
            )
            == CATEGORY_NEGATION
        )
        assert (
            suggest_error_category(long_text, error_source=None, flags={}, **common)
            == CATEGORY_OTHER
        )


class TestTopErrorTable:
    """Maiores erros por paradigma, com texto mascarado e coluna manual vazia."""

    @staticmethod
    def _inputs() -> tuple[pl.DataFrame, list[ModelPredictions], pl.DataFrame]:
        """Teste sintético com um clássico (com scores) e um LLM (sem scores)."""
        test = pl.DataFrame(
            {
                "id": ["a", "b", "c", "d"],
                "text": [
                    "fala @fulano veja https://x.com ruim",
                    "bom demais ontem sim amigo",
                    "oi",
                    "ok",
                ],
                "sentiment_label": ["positivo", "positivo", "negativo", "neutro"],
            }
        )
        scores = np.array([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.7, 0.2, 0.1], [0.2, 0.2, 0.6]])
        classical = _predictions(
            "m1", "classical", ["negativo", "positivo", "positivo", "neutro"], scores
        )
        llm = _predictions("m2", "llm", ["negativo", "negativo", "negativo", "neutro"], None)
        return test, [classical, llm], build_slice_flags(test["text"].to_list())

    def test_orders_by_confidence_masks_text_and_leaves_manual_blank(self) -> None:
        """Ordena pela confiança errada, mascara dados pessoais e deixa a coluna manual vazia."""
        test, predictions, flags = self._inputs()
        table = build_top_error_table(
            predictions, {"classical": "m1", "llm": "m2"}, test, flags, {}, top_n=10
        )
        classical = table.filter(pl.col("category") == "classical")
        assert classical["id"].to_list() == ["a", "c"]
        assert "@fulano" not in classical["text"][0]
        assert "https://" not in classical["text"][0]
        assert (table["manual_category"] == "").all()
        llm = table.filter(pl.col("category") == "llm")
        assert llm["confidence"].is_null().all()
        assert llm["id"].to_list() == ["a", "b"]

    def test_top_n_limits_rows_per_paradigm(self) -> None:
        """``top_n`` limita as linhas por paradigma."""
        test, predictions, flags = self._inputs()
        table = build_top_error_table(predictions, {"classical": "m1"}, test, flags, {}, top_n=1)
        assert table.height == 1

    def test_gold_marks_label_noise(self) -> None:
        """Quando o modelo acerta o gold e o rótulo de treino não, é ruído de rótulo."""
        test, predictions, flags = self._inputs()
        gold = {"consenso_gold": pl.DataFrame({"id": ["c"], "label": ["positivo"]})}
        table = build_top_error_table(predictions, {"classical": "m1"}, test, flags, gold)
        row = table.filter(pl.col("id") == "c").row(0, named=True)
        assert row["error_source"] == SOURCE_LABEL_ERROR
        assert row["suggested_category"] == CATEGORY_LABEL_NOISE

    def test_patterns_cover_all_errors_and_shares_sum_to_one(self) -> None:
        """O resumo de padrões cobre todos os erros e as frações somam 1."""
        test, predictions, flags = self._inputs()
        patterns = summarize_error_patterns(predictions, {"classical": "m1"}, test, flags, {})
        assert patterns["n_errors"].sum() == 2
        assert patterns["share"].sum() == pytest.approx(1.0)

    def test_no_best_models_returns_empty(self) -> None:
        """Sem modelos escolhidos a tabela é vazia."""
        test, predictions, flags = self._inputs()
        assert build_top_error_table(predictions, {}, test, flags, {}).is_empty()


def test_select_best_model_by_category() -> None:
    """Escolhe o maior macro-F1 de cada categoria."""
    table = pl.DataFrame(
        {"model": ["a", "b", "c"], "category": ["x", "x", "y"], "f1_macro": [0.5, 0.7, 0.2]}
    )
    assert select_best_model_by_category(table) == {"x": "b", "y": "c"}


class TestPlotSliceDifferences:
    """Figura de diferença de macro-F1 por fatia."""

    @staticmethod
    def _table() -> pl.DataFrame:
        return pl.DataFrame(
            {
                "model": ["m", "m", "m", "outro"],
                "dimension": ["emoji", "negacao", "giria", "emoji"],
                "slice": ["com_emoji", "com_negacao", "com_giria", "com_emoji"],
                "n": [100, 200, 5, 100],
                "small_sample": [False, False, True, False],
                "f1_macro_difference": [-0.08, 0.01, -0.3, -0.02],
                "difference_ci_low": [-0.12, -0.03, -0.5, -0.05],
                "difference_ci_high": [-0.04, 0.05, -0.1, 0.01],
            }
        )

    def test_omits_small_slices_and_labels_axes(self) -> None:
        """Fatias pequenas não são desenhadas; o eixo x tem rótulo."""
        from visualization.evaluation import plot_slice_differences

        axis = plot_slice_differences(self._table(), model="m").axes[0]
        assert len(axis.get_yticklabels()) == 2
        assert "IC 95%" in axis.get_xlabel()

    def test_without_usable_slices_raises(self) -> None:
        """Modelo sem fatias utilizáveis levanta ``EmptyDatasetError``."""
        from visualization.evaluation import plot_slice_differences

        with pytest.raises(EmptyDatasetError):
            plot_slice_differences(self._table(), model="inexistente")
