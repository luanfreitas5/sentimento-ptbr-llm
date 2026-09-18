"""Testes dos alvos de diagnóstico e do contrato de dados (``diagnostics.targets``)."""

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from constants.labels import SENTIMENT_CLASSES
from diagnostics.targets import (
    TARGET_NAMES,
    adapt_labeled_corpus,
    build_disagreement_target,
    build_gold_error_target,
    build_pseudo_label_target,
    build_target,
    build_uncertainty_target,
)
from exceptions.data import DataValidationError, EmptyDatasetError
from schemas.diagnostics import (
    list_model_label_columns,
    validate_binary_target,
    validate_continuous_target,
    validate_diagnostic_corpus,
)


@pytest.fixture
def diagnostic_corpus() -> pl.DataFrame:
    """Corpus sintético com dois modelos, agreement e gold."""
    return pl.DataFrame(
        {
            "id": ["1", "2", "3", "4"],
            "text_normalized": ["ótimo", "péssimo", "normal", "sei lá"],
            "agreement_score": [0.9, 0.8, 0.5, 0.4],
            "lab_a": ["positivo", "negativo", "neutro", "positivo"],
            "lab_b": ["positivo", "negativo", "positivo", None],
            "gold_label": ["positivo", "neutro", "neutro", "negativo"],
        }
    )


class TestAdaptLabeledCorpus:
    """Adaptador do corpus rotulado atual para o contrato."""

    def test_keeps_only_contract_columns_and_drops_user_id(self) -> None:
        """Minimização (LGPD): ``user_id`` e texto bruto não passam para o contrato."""
        raw = pl.DataFrame(
            {
                "id": [1, 2],
                "user_id": ["u1", "u2"],
                "text": ["@fulano oi", "tchau"],
                "text_normalized": ["[MENCAO] oi", "tchau"],
                "confidence_score": [0.9, 0.3],
                "sentiment_label_huggingface": ["positivo", "negativo"],
                "sentiment_label_llm_relabel": [None, "neutro"],
            }
        )
        adapted = adapt_labeled_corpus(raw)
        assert adapted.columns == [
            "id",
            "text_normalized",
            "agreement_score",
            "lab_huggingface",
            "lab_llm_relabel",
        ]
        assert adapted["id"].to_list() == ["1", "2"]

    def test_includes_gold_column_when_requested(self) -> None:
        """A coluna gold é renomeada para ``gold_label``."""
        raw = pl.DataFrame(
            {
                "id": ["1"],
                "text_normalized": ["a"],
                "confidence_score": [0.9],
                "sentiment_label_huggingface": ["positivo"],
                "sentiment_label_llm_relabel": ["positivo"],
                "gold": ["positivo"],
            }
        )
        assert "gold_label" in adapt_labeled_corpus(raw, gold_column="gold").columns

    def test_missing_source_column_raises(self) -> None:
        """Coluna de origem ausente falha cedo, listando as ausentes."""
        with pytest.raises(DataValidationError, match="colunas ausentes"):
            adapt_labeled_corpus(pl.DataFrame({"id": ["1"]}))


class TestContractValidation:
    """Validação do contrato de entrada e saída."""

    def test_valid_corpus_passes(self, diagnostic_corpus: pl.DataFrame) -> None:
        """Um corpus bem formado é devolvido intacto."""
        assert validate_diagnostic_corpus(diagnostic_corpus).height == 4

    def test_requires_a_model_column(self) -> None:
        """Sem colunas ``lab_*`` o contrato é violado."""
        df = pl.DataFrame({"id": ["1"], "text_normalized": ["a"], "agreement_score": [0.5]})
        with pytest.raises(DataValidationError, match="nenhuma coluna de rótulo"):
            validate_diagnostic_corpus(df)

    def test_rejects_unknown_label(self, diagnostic_corpus: pl.DataFrame) -> None:
        """Rótulos fora de ``SENTIMENT_CLASSES`` são rejeitados."""
        broken = diagnostic_corpus.with_columns(pl.lit("misto").alias("lab_a"))
        with pytest.raises(DataValidationError, match="fora de"):
            validate_diagnostic_corpus(broken)

    def test_rejects_agreement_out_of_range(self, diagnostic_corpus: pl.DataFrame) -> None:
        """``agreement_score`` deve estar em [0, 1]."""
        broken = diagnostic_corpus.with_columns(pl.lit(1.5).alias("agreement_score"))
        with pytest.raises(DataValidationError):
            validate_diagnostic_corpus(broken)

    def test_rejects_duplicated_ids(self, diagnostic_corpus: pl.DataFrame) -> None:
        """``id`` deve ser único."""
        broken = diagnostic_corpus.with_columns(pl.lit("1").alias("id"))
        with pytest.raises(DataValidationError):
            validate_diagnostic_corpus(broken)

    def test_binary_target_rejects_values_outside_0_1(self) -> None:
        """Alvo binário só aceita 0/1."""
        with pytest.raises(DataValidationError):
            validate_binary_target(pl.DataFrame({"id": ["1"], "target": [2]}))

    def test_continuous_target_rejects_values_outside_unit_interval(self) -> None:
        """Alvo contínuo deve estar em [0, 1]."""
        with pytest.raises(DataValidationError):
            validate_continuous_target(pl.DataFrame({"id": ["1"], "target": [1.2]}))

    def test_list_model_label_columns_preserves_order(
        self, diagnostic_corpus: pl.DataFrame
    ) -> None:
        """Só colunas com prefixo ``lab_`` são listadas, na ordem original."""
        assert list_model_label_columns(diagnostic_corpus) == ["lab_a", "lab_b"]


class TestBuildTargets:
    """Construtores dos quatro alvos."""

    def test_disagreement_only_where_all_models_labeled(
        self, diagnostic_corpus: pl.DataFrame
    ) -> None:
        """Linhas com algum modelo nulo saem; discordância = rótulos distintos."""
        result = build_disagreement_target(diagnostic_corpus)
        assert result["id"].to_list() == ["1", "2", "3"]
        assert result["target"].to_list() == [0, 0, 1]

    def test_disagreement_needs_two_models(self, diagnostic_corpus: pl.DataFrame) -> None:
        """Um único modelo não permite medir discordância."""
        with pytest.raises(DataValidationError, match="ao menos dois modelos"):
            build_disagreement_target(diagnostic_corpus, model_columns=["lab_a"])

    def test_disagreement_with_three_models(self) -> None:
        """Com três modelos, qualquer rótulo distinto marca discordância."""
        df = pl.DataFrame(
            {
                "id": ["1", "2"],
                "text_normalized": ["a", "b"],
                "agreement_score": [0.5, 0.5],
                "lab_a": ["positivo", "positivo"],
                "lab_b": ["positivo", "positivo"],
                "lab_c": ["positivo", "negativo"],
            }
        )
        assert build_disagreement_target(df)["target"].to_list() == [0, 1]

    def test_disagreement_empty_when_no_overlap(self) -> None:
        """Sem sobreposição de rótulos, o alvo é vazio e o erro é explícito."""
        df = pl.DataFrame(
            {
                "id": ["1", "2"],
                "text_normalized": ["a", "b"],
                "agreement_score": [0.5, 0.5],
                "lab_a": ["positivo", None],
                "lab_b": [None, "positivo"],
            }
        )
        with pytest.raises(EmptyDatasetError):
            build_disagreement_target(df)

    def test_uncertainty_is_complement_of_agreement(self, diagnostic_corpus: pl.DataFrame) -> None:
        """Incerteza = 1 - agreement_score."""
        result = build_uncertainty_target(diagnostic_corpus)
        assert result["target"].to_list() == pytest.approx([0.1, 0.2, 0.5, 0.6])

    def test_uncertainty_empty_corpus_raises(self, diagnostic_corpus: pl.DataFrame) -> None:
        """Corpus vazio não gera alvo."""
        with pytest.raises(EmptyDatasetError):
            build_uncertainty_target(diagnostic_corpus.head(0))

    def test_pseudo_label_one_vs_rest(self, diagnostic_corpus: pl.DataFrame) -> None:
        """One-vs-rest do modelo; nulos do modelo saem."""
        result = build_pseudo_label_target(
            diagnostic_corpus, model_column="lab_b", label="positivo"
        )
        assert result["target"].to_list() == [1, 0, 1]

    def test_pseudo_label_rejects_unknown_label(self, diagnostic_corpus: pl.DataFrame) -> None:
        """Classe fora do conjunto conhecido é erro de contrato."""
        with pytest.raises(DataValidationError, match="fora de"):
            build_pseudo_label_target(diagnostic_corpus, model_column="lab_a", label="raiva")

    def test_pseudo_label_rejects_unknown_model(self, diagnostic_corpus: pl.DataFrame) -> None:
        """Modelo inexistente lista os disponíveis."""
        with pytest.raises(DataValidationError, match="desconhecidas"):
            build_pseudo_label_target(diagnostic_corpus, model_column="lab_x", label="positivo")

    def test_gold_error(self, diagnostic_corpus: pl.DataFrame) -> None:
        """Erro = predição diferente do gold."""
        result = build_gold_error_target(diagnostic_corpus, model_column="lab_a")
        assert result["target"].to_list() == [0, 1, 0, 1]

    def test_gold_error_requires_gold_column(self, diagnostic_corpus: pl.DataFrame) -> None:
        """Sem ``gold_label`` o alvo de erro não existe."""
        with pytest.raises(DataValidationError, match="gold_label"):
            build_gold_error_target(diagnostic_corpus.drop("gold_label"), model_column="lab_a")


class TestBuildTargetDispatcher:
    """Despachante ``build_target``."""

    @pytest.mark.parametrize("target_name", ["disagreement", "uncertainty"])
    def test_targets_without_extra_arguments(
        self, diagnostic_corpus: pl.DataFrame, target_name: str
    ) -> None:
        """Alvos sem argumentos extras funcionam só com o corpus."""
        assert build_target(diagnostic_corpus, target_name).height > 0  # type: ignore[arg-type]

    def test_pseudo_label_requires_model_and_label(self, diagnostic_corpus: pl.DataFrame) -> None:
        """``pseudo_label`` exige modelo e classe."""
        with pytest.raises(DataValidationError, match="model_column"):
            build_target(diagnostic_corpus, "pseudo_label", label="positivo")
        with pytest.raises(DataValidationError, match="label"):
            build_target(diagnostic_corpus, "pseudo_label", model_column="lab_a")

    def test_gold_error_requires_model(self, diagnostic_corpus: pl.DataFrame) -> None:
        """``gold_error`` exige o modelo avaliado."""
        with pytest.raises(DataValidationError, match="model_column"):
            build_target(diagnostic_corpus, "gold_error")

    def test_unknown_target_raises(self, diagnostic_corpus: pl.DataFrame) -> None:
        """Alvo desconhecido lista os disponíveis."""
        with pytest.raises(DataValidationError, match="desconhecido"):
            build_target(diagnostic_corpus, "outro")  # type: ignore[arg-type]

    def test_dispatcher_validates_corpus(self) -> None:
        """O corpus é validado antes de construir qualquer alvo."""
        with pytest.raises(DataValidationError):
            build_target(pl.DataFrame({"id": ["1"]}), "uncertainty")

    def test_target_names_cover_the_four_allowed_targets(self) -> None:
        """Os quatro alvos do CLAUDE.md estão registrados."""
        assert set(TARGET_NAMES) == {"disagreement", "uncertainty", "pseudo_label", "gold_error"}


_LABELS = st.sampled_from(SENTIMENT_CLASSES)


@given(pairs=st.lists(st.tuples(_LABELS, _LABELS), min_size=1, max_size=30))
def test_disagreement_is_symmetric_and_matches_inequality(
    pairs: list[tuple[str, str]],
) -> None:
    """Invariante: trocar a ordem dos modelos não muda o alvo, que é ``a != b``."""
    ids = [str(i) for i in range(len(pairs))]
    base = {
        "id": ids,
        "text_normalized": ["t"] * len(pairs),
        "agreement_score": [0.5] * len(pairs),
    }
    forward = pl.DataFrame({**base, "lab_a": [p[0] for p in pairs], "lab_b": [p[1] for p in pairs]})
    backward = pl.DataFrame(
        {**base, "lab_a": [p[1] for p in pairs], "lab_b": [p[0] for p in pairs]}
    )
    expected = [int(a != b) for a, b in pairs]
    assert build_disagreement_target(forward)["target"].to_list() == expected
    assert build_disagreement_target(backward)["target"].to_list() == expected


@given(scores=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=30))
def test_uncertainty_stays_in_unit_interval(scores: list[float]) -> None:
    """Invariante: incerteza sempre em [0, 1] e monotônica decrescente no agreement."""
    df = pl.DataFrame(
        {
            "id": [str(i) for i in range(len(scores))],
            "text_normalized": ["t"] * len(scores),
            "agreement_score": scores,
            "lab_a": ["positivo"] * len(scores),
        }
    )
    target = build_uncertainty_target(df)["target"].to_list()
    assert all(0.0 <= value <= 1.0 for value in target)
    assert target == pytest.approx([1.0 - score for score in scores])
