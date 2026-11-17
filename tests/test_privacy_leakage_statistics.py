"""Testes de privacidade (LGPD), vazamento entre partições, referência gold e estatística.

Usam apenas DataFrames sintéticos pequenos: nenhum dado real, rede ou GPU.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from data.authors import (
    AUTHOR_KEY_COLUMN,
    build_author_key_table,
    load_author_key_table,
    write_author_key_table,
)
from data.leakage import audit_split_leakage
from data.splitter import create_grouped_stratified_split, remove_cross_split_duplicates
from evaluation.gold_reference import (
    CONSENSUS_REFERENCE,
    TRAINING_LABELER_NAME,
    build_gold_references,
    calculate_krippendorff_alpha_nominal,
    evaluate_models_against_references,
    summarize_training_label_noise,
)
from evaluation.significance import (
    adjust_p_values,
    calculate_mcnemar_effect,
    calculate_paired_bootstrap_f1_difference,
    run_corrected_resampled_ttest,
)
from exceptions.configuration import MissingEnvironmentVariableError
from exceptions.data import DataNotFoundError, EmptyDatasetError
from io_utils.csv import write_csv
from utils.privacy import (
    count_personal_data,
    mask_personal_data,
    mask_text_columns,
    pseudonymize_value,
    resolve_pseudonymization_key,
    validate_cpf,
)

VALID_CPF = "529.982.247-25"


class TestPrivacy:
    """Pseudonimização, mascaramento e contagem de dados pessoais."""

    def test_pseudonymization_is_deterministic_and_keyed(self) -> None:
        """Mesma chave -> mesmo pseudônimo; chave diferente -> pseudônimo diferente."""
        assert pseudonymize_value("123", "k") == pseudonymize_value("123", "k")
        assert pseudonymize_value("123", "k") != pseudonymize_value("123", "outra")
        assert "123" not in pseudonymize_value("123", "k")

    def test_resolve_key_requires_environment_variable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sem a variável, deve levantar erro em vez de usar um valor padrão."""
        monkeypatch.delenv("PSEUDONYMIZATION_KEY", raising=False)
        with pytest.raises(MissingEnvironmentVariableError):
            resolve_pseudonymization_key()
        monkeypatch.setenv("PSEUDONYMIZATION_KEY", "segredo")
        assert resolve_pseudonymization_key() == "segredo"

    def test_validate_cpf(self) -> None:
        """Aceita CPF válido e rejeita dígitos repetidos, tamanho errado e DV errado."""
        assert validate_cpf(VALID_CPF)
        assert not validate_cpf("111.111.111-11")
        assert not validate_cpf("123")
        assert not validate_cpf("529.982.247-24")

    def test_mask_personal_data_covers_all_categories(self) -> None:
        """Mascara @menção, URL, e-mail, CPF válido e telefone, preservando o resto."""
        text = f"@fulano veja https://x.com/a1 ou a@b.com cpf {VALID_CPF} tel (11) 91234-5678 ótimo"
        masked = mask_personal_data(text)
        assert masked == "@usuario veja <url> ou <email> cpf <cpf> tel <telefone> ótimo"

    def test_mask_keeps_invalid_cpf_like_numbers(self) -> None:
        """Um número de 11 dígitos sem DV de CPF não deve ser tratado como CPF."""
        assert "<cpf>" not in mask_personal_data("pedido 123.456.789-00")

    def test_mask_mentions_can_be_disabled(self) -> None:
        """Com ``mask_mentions=False`` a @menção fica intacta."""
        assert mask_personal_data("oi @fulano", mask_mentions=False) == "oi @fulano"

    def test_count_personal_data_reports_counts_only(self) -> None:
        """Conta textos por categoria e devolve o total inspecionado."""
        counts = count_personal_data(["oi @alguem", "veja http://x.com", "limpo", "a@b.com"])
        assert counts["mencao"] == 1
        assert counts["url"] == 1
        assert counts["email"] == 1
        assert counts["total"] == 4

    def test_count_ignores_the_mask_placeholder(self) -> None:
        """Texto já mascarado não deve ser contado como contendo @menção."""
        assert count_personal_data([mask_personal_data("oi @fulano")])["mencao"] == 0

    @given(text=st.text(max_size=80))
    def test_mask_is_idempotent(self, text: str) -> None:
        """Invariante: mascarar um texto já mascarado não o altera."""
        once = mask_personal_data(text)
        assert mask_personal_data(once) == once

    def test_mask_text_columns_only_touches_text_columns(self) -> None:
        """Colunas de texto são mascaradas; as demais ficam como estão."""
        frame = pl.DataFrame(
            {"text_normalized": ["oi @fulano"], "outra": ["@fulano"], "top_example_1": ["a@b.com"]}
        )
        masked = mask_text_columns(frame)
        assert masked["text_normalized"].to_list() == ["oi @usuario"]
        assert masked["top_example_1"].to_list() == ["<email>"]
        assert masked["outra"].to_list() == ["@fulano"]

    def test_write_csv_masks_when_requested(self, tmp_path: Path) -> None:
        """``write_csv(mask_text=True)`` não grava a @menção; o padrão não altera nada."""
        frame = pl.DataFrame({"text": ["oi @fulano"]})
        masked_path = tmp_path / "mascarado.csv"
        raw_path = tmp_path / "cru.csv"
        write_csv(frame, masked_path, mask_text=True)
        write_csv(frame, raw_path)
        assert "@fulano" not in masked_path.read_text(encoding="utf-8")
        assert "@fulano" in raw_path.read_text(encoding="utf-8")


class TestAuthors:
    """Tabela ``id -> author_key``."""

    def test_table_has_ids_and_no_raw_author(self, tmp_path: Path) -> None:
        """Mesmo autor -> mesma chave; o identificador original não aparece na saída."""
        raw = pl.DataFrame({"id_screen_name": [987654321, 987654321, 5, None]})
        table = build_author_key_table(raw, key="k")
        assert table["id"].to_list() == ["t0000001", "t0000002", "t0000003", "t0000004"]
        keys = table[AUTHOR_KEY_COLUMN].to_list()
        assert keys[0] == keys[1]
        assert keys[0] != keys[2]
        assert keys[3] is None
        assert "987654321" not in "".join(k for k in keys if k)

        path = write_author_key_table(table, tmp_path)
        assert path.is_file()
        assert load_author_key_table(tmp_path).height == 4

    def test_missing_author_column_raises(self) -> None:
        """Sem a coluna do autor deve levantar ``KeyError`` claro."""
        with pytest.raises(KeyError):
            build_author_key_table(pl.DataFrame({"x": [1]}), key="k")

    def test_missing_table_raises_with_instruction(self, tmp_path: Path) -> None:
        """Sem a tabela, o erro orienta como gerá-la."""
        with pytest.raises(DataNotFoundError):
            load_author_key_table(tmp_path)


def _authors_frame(n_authors: int = 60, per_author: int = 5) -> pl.DataFrame:
    """Corpus sintético: ``per_author`` tweets por autor, três classes alternadas."""
    labels = ["positivo", "negativo", "neutro"]
    rows = [
        {
            "sentiment_label": labels[(author + tweet) % 3],
            AUTHOR_KEY_COLUMN: f"a{author}",
            "text": f"texto {author} {tweet}",
        }
        for author in range(n_authors)
        for tweet in range(per_author)
    ]
    return pl.DataFrame(rows)


class TestGroupedSplit:
    """Split por autor e remoção de duplicatas entre partições."""

    def test_no_author_in_two_splits(self) -> None:
        """Invariante central: cada autor cai em uma única partição."""
        split = create_grouped_stratified_split(
            _authors_frame(), label_column="sentiment_label", group_column=AUTHOR_KEY_COLUMN
        )
        splits_per_author = split.group_by(AUTHOR_KEY_COLUMN).agg(pl.col("split").n_unique())
        assert splits_per_author["split"].max() == 1
        assert set(split["split"].unique()) == {"treino", "validacao", "teste"}

    def test_proportions_are_approximately_respected(self) -> None:
        """Teste ~20% e validação ~10%, aceitando a granularidade de grupos inteiros."""
        split = create_grouped_stratified_split(
            _authors_frame(200, 5),
            label_column="sentiment_label",
            group_column=AUTHOR_KEY_COLUMN,
            test_size=0.2,
            validation_size=0.1,
        )
        shares = split["split"].value_counts().to_dict(as_series=False)
        share = dict(zip(shares["split"], np.array(shares["count"]) / split.height, strict=True))
        assert share["teste"] == pytest.approx(0.2, abs=0.04)
        assert share["validacao"] == pytest.approx(0.1, abs=0.04)

    def test_is_deterministic_for_a_seed(self) -> None:
        """Mesma semente -> mesma partição; semente diferente -> partição diferente."""
        frame = _authors_frame()
        first = create_grouped_stratified_split(
            frame,
            label_column="sentiment_label",
            group_column=AUTHOR_KEY_COLUMN,
            random_seed=1,
        )
        again = create_grouped_stratified_split(
            frame,
            label_column="sentiment_label",
            group_column=AUTHOR_KEY_COLUMN,
            random_seed=1,
        )
        other = create_grouped_stratified_split(
            frame,
            label_column="sentiment_label",
            group_column=AUTHOR_KEY_COLUMN,
            random_seed=2,
        )
        assert first["split"].to_list() == again["split"].to_list()
        assert first["split"].to_list() != other["split"].to_list()

    def test_null_group_rows_become_their_own_groups(self) -> None:
        """Linhas sem autor não devem ser todas agrupadas na mesma partição."""
        frame = pl.DataFrame(
            {
                "sentiment_label": ["positivo", "negativo"] * 50,
                AUTHOR_KEY_COLUMN: [None] * 100,
            }
        )
        split = create_grouped_stratified_split(
            frame, label_column="sentiment_label", group_column=AUTHOR_KEY_COLUMN
        )
        assert split["split"].n_unique() == 3

    def test_invalid_sizes_and_columns_raise(self) -> None:
        """Frações que somam 1 ou coluna ausente devem falhar com erro claro."""
        frame = _authors_frame(10, 2)
        with pytest.raises(ValueError):
            create_grouped_stratified_split(
                frame,
                label_column="sentiment_label",
                group_column=AUTHOR_KEY_COLUMN,
                test_size=0.6,
                validation_size=0.4,
            )
        with pytest.raises(KeyError):
            create_grouped_stratified_split(frame, label_column="sentiment_label", group_column="x")

    def test_remove_cross_split_duplicates_keeps_test_intact(self) -> None:
        """Cópias (mesmo texto normalizado) saem do treino/validação; o teste não perde linhas."""
        frame = pl.DataFrame(
            {
                "text": ["Ótimo!", "ótimo", "ÓTIMO", "ruim", "ruim"],
                "split": ["treino", "teste", "validacao", "treino", "treino"],
            }
        )
        result = remove_cross_split_duplicates(frame)
        assert result.filter(pl.col("split") == "teste").height == 1
        assert result.filter(pl.col("split") == "validacao").height == 0
        assert result.filter(pl.col("split") == "treino")["text"].to_list() == ["ruim", "ruim"]


class TestLeakageAudit:
    """Auditoria de vazamento."""

    def test_random_split_leaks_authors_but_grouped_does_not(self) -> None:
        """O split agrupado zera a sobreposição de autores medida pela auditoria."""
        frame = _authors_frame()
        grouped = create_grouped_stratified_split(
            frame, label_column="sentiment_label", group_column=AUTHOR_KEY_COLUMN
        )
        report = audit_split_leakage(grouped, group_column=AUTHOR_KEY_COLUMN)
        assert report["treino->teste"]["author_overlap_share"] == 0.0

        shuffled = frame.with_columns(pl.Series("split", ["treino", "teste"] * (frame.height // 2)))
        leaked = audit_split_leakage(shuffled, group_column=AUTHOR_KEY_COLUMN)
        assert leaked["treino->teste"]["author_overlap_share"] > 0.5

    def test_duplicate_text_share(self) -> None:
        """Conta o texto do teste que também está no treino, ignorando caixa e pontuação."""
        frame = pl.DataFrame(
            {"text": ["Oi, mundo!", "oi mundo", "outro"], "split": ["treino", "teste", "teste"]}
        )
        report = audit_split_leakage(frame)["treino->teste"]
        assert report["duplicate_text_count"] == 1
        assert report["duplicate_text_share"] == pytest.approx(0.5)
        assert "author_overlap_share" not in report

    def test_missing_column_raises(self) -> None:
        """Sem a coluna de partição deve levantar ``KeyError``."""
        with pytest.raises(KeyError):
            audit_split_leakage(pl.DataFrame({"text": ["a"]}))


class TestSignificanceAdditions:
    """Correção de comparações múltiplas, efeito, bootstrap pareado e t corrigido."""

    def test_holm_matches_known_values(self) -> None:
        """Valores conhecidos de Holm e Benjamini-Hochberg."""
        holm = adjust_p_values([0.01, 0.04, 0.03], method="holm")
        assert holm == pytest.approx([0.03, 0.06, 0.06])
        bh = adjust_p_values([0.01, 0.04, 0.03], method="bh")
        assert bh == pytest.approx([0.03, 0.04, 0.04])

    def test_unknown_method_and_empty(self) -> None:
        """Método desconhecido levanta erro; lista vazia devolve lista vazia."""
        with pytest.raises(ValueError):
            adjust_p_values([0.1], method="inexistente")
        assert adjust_p_values([]) == []

    @given(
        p_values=st.lists(st.floats(min_value=0, max_value=1), min_size=1, max_size=20),
        method=st.sampled_from(["holm", "bonferroni", "bh"]),
    )
    def test_adjusted_p_values_are_never_smaller_than_raw(
        self, p_values: list[float], method: str
    ) -> None:
        """Invariante: a correção nunca diminui o p-valor e o mantém em [0, 1]."""
        adjusted = adjust_p_values(p_values, method=method)
        for raw, corrected in zip(p_values, adjusted, strict=True):
            assert corrected >= raw - 1e-12
            assert 0 <= corrected <= 1 + 1e-12

    def test_mcnemar_effect(self) -> None:
        """Acertos exclusivos, razão de chances e diferença de acurácia."""
        y_true = ["a", "a", "a", "b"]
        effect = calculate_mcnemar_effect(y_true, ["a", "a", "b", "b"], ["b", "a", "a", "b"])
        assert effect["a_only_correct"] == 1.0
        assert effect["b_only_correct"] == 1.0
        assert effect["odds_ratio"] == pytest.approx(1.0)
        assert effect["accuracy_difference"] == pytest.approx(0.0)
        with pytest.raises(EmptyDatasetError):
            calculate_mcnemar_effect([], [], [])

    def test_paired_bootstrap_detects_clear_difference(self) -> None:
        """Modelo perfeito contra um que só chuta uma classe: IC acima de zero."""
        y_true = ["a", "b", "c"] * 60
        good = list(y_true)
        bad = ["a"] * len(y_true)
        result = calculate_paired_bootstrap_f1_difference(y_true, good, bad, n_bootstrap=200)
        assert result["ci_low"] > 0
        # o chute em "a" tem F1 0,5 em "a" e 0 nas outras: macro = 1/6
        assert result["f1_macro_difference"] == pytest.approx(1 - 1 / 6, abs=1e-9)

    def test_paired_bootstrap_identical_models_have_zero_difference(self) -> None:
        """Modelos idênticos: diferença exatamente zero e IC degenerado em zero."""
        y_true = ["a", "b"] * 30
        result = calculate_paired_bootstrap_f1_difference(y_true, y_true, y_true, n_bootstrap=50)
        assert result["f1_macro_difference"] == 0.0
        assert result["ci_low"] == result["ci_high"] == 0.0

    def test_corrected_ttest_is_more_conservative_than_plain(self) -> None:
        """A correção de Nadeau-Bengio aumenta o p-valor em relação ao teste t comum."""
        from scipy.stats import ttest_rel

        scores_a = [0.80, 0.82, 0.81, 0.79, 0.83]
        scores_b = [0.78, 0.80, 0.80, 0.78, 0.81]
        corrected = run_corrected_resampled_ttest(scores_a, scores_b, n_train=800, n_test=200)
        plain = ttest_rel(scores_a, scores_b).pvalue
        assert corrected["p_value"] > plain
        assert corrected["mean_difference"] == pytest.approx(0.016)

    def test_corrected_ttest_validates_inputs(self) -> None:
        """Poucas dobras ou tamanhos diferentes devem falhar."""
        with pytest.raises(ValueError):
            run_corrected_resampled_ttest([0.8], [0.7], n_train=10, n_test=5)
        with pytest.raises(EmptyDatasetError):
            run_corrected_resampled_ttest([], [], n_train=10, n_test=5)


class TestGoldReference:
    """Avaliação contra referências independentes e ruído do rótulo de treino."""

    def test_consensus_keeps_only_agreeing_tweets(self) -> None:
        """O consenso exige o mesmo rótulo em todos os gold sets."""
        gold_a = pl.DataFrame(
            {"id": ["1", "2", "3"], "sentiment_label": ["positivo", "negativo", "neutro"]}
        )
        gold_b = pl.DataFrame(
            {"id": ["1", "2", "3"], "sentiment_label": ["positivo", "neutro", "neutro"]}
        )
        references = build_gold_references({"a": gold_a, "b": gold_b})
        assert sorted(references[CONSENSUS_REFERENCE]["id"].to_list()) == ["1", "3"]
        assert set(references) == {"a", "b", CONSENSUS_REFERENCE}

    def test_labels_outside_sentiment_classes_are_dropped(self) -> None:
        """Rótulos como "indefinido" não podem virar uma quarta classe sem suporte."""
        gold = pl.DataFrame(
            {"id": ["1", "2", "3"], "sentiment_label": ["positivo", "indefinido", None]}
        )
        assert build_gold_references({"g": gold})["g"]["id"].to_list() == ["1"]

    def test_single_gold_has_no_consensus(self) -> None:
        """Com um só gold set não há referência de consenso."""
        gold = pl.DataFrame({"id": ["1"], "sentiment_label": ["positivo"]})
        assert CONSENSUS_REFERENCE not in build_gold_references({"a": gold})
        with pytest.raises(EmptyDatasetError):
            build_gold_references({})

    def test_krippendorff_alpha_known_cases(self) -> None:
        """Concordância perfeita dá 1; concordância no nível do acaso dá ~0; oposta dá < 0."""
        perfect = np.array([["a", "a"], ["b", "b"], ["a", "a"], ["b", "b"]])
        assert calculate_krippendorff_alpha_nominal(perfect) == pytest.approx(1.0)
        opposite = np.array([["a", "b"], ["b", "a"], ["a", "b"], ["b", "a"]])
        assert calculate_krippendorff_alpha_nominal(opposite) < 0
        with pytest.raises(ValueError):
            calculate_krippendorff_alpha_nominal(np.array(["a", "b"]))

    def test_models_scored_only_on_covered_tweets_and_labeler_is_included(self) -> None:
        """Só os tweets da referência entram; o rótulo de treino aparece como um "modelo"."""
        predictions = pl.DataFrame(
            {
                "id": ["1", "2", "3", "4"],
                "y_true": ["positivo", "negativo", "neutro", "positivo"],
                "pred_bom": ["positivo", "negativo", "neutro", "positivo"],
                "pred_ruim": ["negativo", "negativo", "positivo", None],
            }
        )
        reference = {
            "g": pl.DataFrame({"id": ["1", "2", "3"], "label": ["positivo", "negativo", "neutro"]})
        }
        table = evaluate_models_against_references(predictions, reference, n_bootstrap=20)
        assert set(table["model"]) == {"bom", "ruim", TRAINING_LABELER_NAME}
        assert set(table["n"]) == {3}
        best = table.row(0, named=True)
        assert best["f1_macro"] == pytest.approx(1.0)
        assert best["f1_macro_ci_low"] <= best["f1_macro"] <= best["f1_macro_ci_high"]

    def test_models_without_overlap_return_empty_table(self) -> None:
        """Referência que não cobre nenhum tweet do teste não gera linhas nem erro."""
        predictions = pl.DataFrame({"id": ["1"], "y_true": ["positivo"], "pred_m": ["positivo"]})
        reference = {"g": pl.DataFrame({"id": ["9"], "label": ["positivo"]})}
        assert evaluate_models_against_references(predictions, reference, n_bootstrap=5).is_empty()

    def test_missing_columns_raise(self) -> None:
        """Tabela de predições sem ``y_true`` deve falhar."""
        with pytest.raises(KeyError):
            evaluate_models_against_references(pl.DataFrame({"id": ["1"]}), {})

    def test_training_label_noise_summary(self) -> None:
        """Mede a concordância do rótulo de treino com cada referência."""
        training = pl.DataFrame(
            {"id": [str(i) for i in range(10)], "label": ["positivo"] * 5 + ["negativo"] * 5}
        )
        reference = {
            "g": pl.DataFrame(
                {"id": [str(i) for i in range(10)], "label": ["positivo"] * 8 + ["negativo"] * 2}
            )
        }
        summary = summarize_training_label_noise(training, reference, n_bootstrap=20)
        row = summary.row(0, named=True)
        assert row["n"] == 10
        assert row["accuracy"] == pytest.approx(0.7)
        assert -1 <= row["krippendorff_alpha"] <= 1
