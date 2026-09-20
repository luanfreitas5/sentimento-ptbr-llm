"""Testes da comparação Hugging Face × OpenAI (``evaluation.llm_comparison`` e derivados)."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from config.paths import ProjectPaths
from constants.labels import SENTIMENT_CLASSES
from diagnostics import model_disagreement
from diagnostics.model_disagreement import (
    HypothesisTargetOutcome,
    build_diagnostic_corpus,
    build_hypothesis_evidence,
    run_disagreement_hypotheses,
)
from diagnostics.sae_runner import DiscoveryData
from evaluation.llm_comparison import (
    analyze_by_text_length,
    build_agreement_matrix,
    build_comparison_frame,
    calculate_agreement_summary,
    calculate_class_distribution,
    compare_confidence_scores,
    find_ambiguous_cases,
    find_confidence_conflicts,
    find_top_divergences,
    summarize_classification_differences,
)
from exceptions.data import DataValidationError, EmptyDatasetError
from exceptions.pipeline import SanityGateFailedError
from visualization.comparison import (
    plot_agreement_by_text_length,
    plot_class_distribution_by_model,
    plot_confidence_comparison,
    plot_divergence_transitions,
)


def _base(
    labels: list[str], confidences: list[float], texts: list[str] | None = None
) -> pl.DataFrame:
    """Base rotulada sintética com ids ``"0"``, ``"1"``..."""
    n_tweets = len(labels)
    texts = texts or [f"tweet {index}" for index in range(n_tweets)]
    return pl.DataFrame(
        {
            "id": [str(index) for index in range(n_tweets)],
            "text": texts,
            "text_normalized": texts,
            "sentiment_label": labels,
            "confidence_score": confidences,
        }
    )


@pytest.fixture
def frame() -> pl.DataFrame:
    """Seis tweets: 3 concordantes, 1 neutro-vs-polar e 2 de polaridade oposta."""
    hf = _base(
        ["positivo", "negativo", "neutro", "positivo", "negativo", "positivo"],
        [0.95, 0.90, 0.60, 0.85, 0.92, 0.97],
        ["ótimo", "péssimo demais", "sem opinião aqui", "gostei muito disso", "não gostei", "top"],
    )
    openai = _base(
        ["positivo", "negativo", "neutro", "neutro", "positivo", "negativo"],
        [0.90, 0.88, 0.55, 0.70, 0.86, 0.30],
        ["ótimo", "péssimo demais", "sem opinião aqui", "gostei muito disso", "não gostei", "top"],
    )
    return build_comparison_frame(hf, openai)


class TestBuildComparisonFrame:
    """Testes de :func:`evaluation.llm_comparison.build_comparison_frame`."""

    def test_joins_by_id_and_derives_agreement_columns(self, frame: pl.DataFrame) -> None:
        """Une pelo ``id`` e deriva ``agree``, ``label_distance`` e ``word_count``."""
        assert frame.height == 6
        assert frame["agree"].to_list() == [True, True, True, False, False, False]
        assert frame["label_distance"].to_list() == [0, 0, 0, 1, 2, 2]
        assert frame["word_count"].to_list() == [1, 2, 3, 3, 2, 1]

    def test_row_order_of_the_second_base_does_not_matter(self) -> None:
        """O ``id`` é a chave: embaralhar a base OpenAI produz o mesmo resultado."""
        hf = _base(["positivo", "negativo"], [0.9, 0.8])
        openai = _base(["positivo", "positivo"], [0.7, 0.6])

        straight = build_comparison_frame(hf, openai).sort("id")
        reversed_ = build_comparison_frame(hf, openai.reverse()).sort("id")

        assert straight.equals(reversed_)

    def test_raises_when_bases_have_different_tweets(self) -> None:
        """Conjuntos de ``id`` diferentes não são comparáveis."""
        hf = _base(["positivo", "negativo"], [0.9, 0.8])
        openai = _base(["positivo"], [0.7])
        with pytest.raises(DataValidationError, match="mesmos tweets"):
            build_comparison_frame(hf, openai)

    def test_raises_for_empty_base(self) -> None:
        """Uma base vazia deve levantar ``EmptyDatasetError``."""
        hf = _base(["positivo"], [0.9])
        with pytest.raises(EmptyDatasetError):
            build_comparison_frame(hf, hf.clear())

    def test_raises_for_base_violating_the_contract(self) -> None:
        """A validação do contrato é refeita ao carregar: classe inválida é rejeitada."""
        hf = _base(["positivo"], [0.9])
        with pytest.raises(DataValidationError):
            build_comparison_frame(hf, _base(["excelente"], [0.9]))


class TestClassDistributionAndMatrix:
    """Distribuição de classes e matriz de concordância."""

    def test_distribution_has_every_class_for_both_models(self, frame: pl.DataFrame) -> None:
        """Todas as classes aparecem para os dois modelos, e as proporções somam 1 por modelo."""
        distribution = calculate_class_distribution(frame)

        assert distribution.height == 2 * len(SENTIMENT_CLASSES)
        for model in ("huggingface", "openai"):
            assert distribution.filter(pl.col("model") == model)[
                "proportion"
            ].sum() == pytest.approx(1.0)
        hf_positive = distribution.filter(
            (pl.col("model") == "huggingface") & (pl.col("sentiment_label") == "positivo")
        )
        assert hf_positive["count"].to_list() == [3]

    def test_matrix_rows_are_huggingface_and_columns_openai(self, frame: pl.DataFrame) -> None:
        """Ordem negativo/neutro/positivo, com o Hugging Face nas linhas e a OpenAI nas colunas."""
        matrix = build_agreement_matrix(frame)

        assert matrix.sum() == frame.height
        assert matrix.trace() == 3
        assert matrix[2, 1] == 1  # HF positivo -> OpenAI neutro
        assert matrix[0, 2] == 1  # HF negativo -> OpenAI positivo
        assert matrix[2, 0] == 1  # HF positivo -> OpenAI negativo


class TestAgreementSummary:
    """Testes de :func:`evaluation.llm_comparison.calculate_agreement_summary`."""

    def test_point_estimates_and_confidence_intervals(self, frame: pl.DataFrame) -> None:
        """Concordância = 3/6, divergência = 3/6 e IC bootstrap contendo a estimativa."""
        summary = calculate_agreement_summary(frame, n_bootstrap=200, random_seed=1)

        assert summary["n_tweets"] == 6
        assert summary["n_agree"] == 3
        assert summary["agreement_rate"] == pytest.approx(0.5)
        assert summary["divergence_rate"] == pytest.approx(0.5)
        assert summary["opposite_polarity_rate"] == pytest.approx(2 / 3)
        lower, upper = summary["agreement_rate_ci"]
        assert 0.0 <= lower <= 0.5 <= upper <= 1.0
        assert summary["cohen_kappa_ci"][0] <= summary["cohen_kappa_ci"][1]

    def test_is_reproducible_with_the_same_seed(self, frame: pl.DataFrame) -> None:
        """Mesma semente produz exatamente os mesmos intervalos."""
        first = calculate_agreement_summary(frame, n_bootstrap=50, random_seed=7)
        second = calculate_agreement_summary(frame, n_bootstrap=50, random_seed=7)
        assert first["agreement_rate_ci"] == second["agreement_rate_ci"]
        assert first["cohen_kappa_ci"] == second["cohen_kappa_ci"]

    def test_perfect_agreement(self) -> None:
        """Duas bases idênticas dão concordância 1, divergência 0 e kappa 1."""
        base = _base(["positivo", "negativo", "neutro", "positivo"], [0.9, 0.8, 0.7, 0.6])
        summary = calculate_agreement_summary(build_comparison_frame(base, base), n_bootstrap=20)

        assert summary["agreement_rate"] == 1.0
        assert summary["divergence_rate"] == 0.0
        assert summary["cohen_kappa"] == pytest.approx(1.0)
        assert summary["opposite_polarity_rate"] == 0.0

    def test_raises_for_empty_frame(self, frame: pl.DataFrame) -> None:
        """Um DataFrame vazio deve levantar ``EmptyDatasetError``."""
        with pytest.raises(EmptyDatasetError):
            calculate_agreement_summary(frame.clear())

    @settings(max_examples=25, deadline=None)
    @given(
        st.lists(
            st.tuples(st.sampled_from(SENTIMENT_CLASSES), st.sampled_from(SENTIMENT_CLASSES)),
            min_size=2,
            max_size=30,
        )
    )
    def test_agreement_and_divergence_always_sum_to_one(self, pairs: list[tuple[str, str]]) -> None:
        """Invariante: concordância + divergência = 1 e kappa <= 1, para qualquer rotulagem."""
        n_tweets = len(pairs)
        frame = build_comparison_frame(
            _base([hf for hf, _ in pairs], [0.9] * n_tweets),
            _base([oa for _, oa in pairs], [0.8] * n_tweets),
        )
        summary = calculate_agreement_summary(frame, n_bootstrap=5)

        assert summary["agreement_rate"] + summary["divergence_rate"] == pytest.approx(1.0)
        assert summary["cohen_kappa"] <= 1.0 + 1e-9


class TestConfidenceComparison:
    """Testes de :func:`evaluation.llm_comparison.compare_confidence_scores`."""

    def test_describes_each_model_and_splits_by_agreement(self, frame: pl.DataFrame) -> None:
        """Estatísticas por modelo, com a confiança média separada em concordância/divergência."""
        table, summary = compare_confidence_scores(frame)

        assert table["model"].to_list() == ["huggingface", "openai"]
        hf_row = table.filter(pl.col("model") == "huggingface").to_dicts()[0]
        assert hf_row["mean"] == pytest.approx(np.mean([0.95, 0.90, 0.60, 0.85, 0.92, 0.97]))
        assert hf_row["mean_when_agree"] == pytest.approx(np.mean([0.95, 0.90, 0.60]))
        assert -1.0 <= summary["spearman_rho"] <= 1.0
        assert summary["wilcoxon_paired"] is not None

    def test_wilcoxon_is_none_when_confidences_are_identical(self) -> None:
        """Sem nenhuma diferença pareada, o teste de Wilcoxon não é aplicável."""
        base = _base(["positivo", "negativo", "neutro"], [0.9, 0.8, 0.7])
        _, summary = compare_confidence_scores(build_comparison_frame(base, base))

        assert summary["wilcoxon_paired"] is None
        assert summary["mean_absolute_difference"] == 0.0


class TestDivergenceAnalyses:
    """Maiores divergências, conflitos de confiança, ambiguidade e transições."""

    def test_top_divergences_rank_opposite_polarity_first_then_confidence(
        self, frame: pl.DataFrame
    ) -> None:
        """Polaridade oposta vem antes de neutro-vs-polar; empate desfeito pela menor confiança."""
        top = find_top_divergences(frame)

        assert top["id"].to_list() == ["4", "5", "3"]
        assert top["label_distance"].to_list() == [2, 2, 1]
        assert top["min_confidence"].to_list() == [0.86, 0.3, 0.7]

    def test_top_divergences_respects_top_n_and_excludes_agreements(
        self, frame: pl.DataFrame
    ) -> None:
        """``top_n`` limita a saída, que nunca contém tweets concordantes."""
        assert find_top_divergences(frame, top_n=1).height == 1
        assert min(find_top_divergences(frame)["label_distance"].to_list()) >= 1

    def test_confidence_conflicts_find_high_versus_low(self, frame: pl.DataFrame) -> None:
        """Só o tweet 5 (HF 0.97 vs OpenAI 0.30) tem um modelo confiante e o outro não."""
        conflicts = find_confidence_conflicts(frame, high_threshold=0.8, low_threshold=0.5)

        assert conflicts["id"].to_list() == ["5"]
        assert conflicts["more_confident_model"].to_list() == ["huggingface"]
        assert conflicts["confidence_gap"].to_list() == [pytest.approx(0.67)]

    def test_confidence_conflicts_flag_the_openai_side_too(self) -> None:
        """O conflito também é detectado quando a OpenAI é a confiante."""
        hf = _base(["positivo"], [0.3])
        openai = _base(["positivo"], [0.95])
        conflicts = find_confidence_conflicts(build_comparison_frame(hf, openai))

        assert conflicts["more_confident_model"].to_list() == ["openai"]

    def test_ambiguous_cases_are_classified_and_summarized(self) -> None:
        """Cada tweet ambíguo recebe uma categoria; o resumo usa todo o corpus como denominador."""
        hf = _base(
            ["positivo", "positivo", "positivo", "positivo", "positivo"],
            [0.9, 0.4, 0.4, 0.9, 0.9],
        )
        openai = _base(
            ["negativo", "negativo", "positivo", "positivo", "positivo"],
            [0.9, 0.9, 0.45, 0.4, 0.9],
        )
        cases, summary = find_ambiguous_cases(build_comparison_frame(hf, openai), low_threshold=0.5)

        by_id = dict(zip(cases["id"].to_list(), cases["ambiguity_type"].to_list(), strict=True))
        assert by_id == {
            "0": "divergente_alta_confianca",
            "1": "divergente_baixa_confianca",
            "2": "concordante_baixa_confianca_ambos",
            "3": "concordante_baixa_confianca_um",
        }
        assert summary["n_tweets"].sum() == 4
        assert summary["proportion"].sum() == pytest.approx(4 / 5)

    def test_transitions_count_each_class_pair_and_pick_confident_examples(
        self, frame: pl.DataFrame
    ) -> None:
        """Transições HF→OpenAI ordenadas por frequência, com exemplos de maior confiança."""
        transitions, examples = summarize_classification_differences(
            frame, examples_per_transition=1
        )

        assert transitions["n_tweets"].sum() == 3
        assert transitions["share_of_divergences"].sum() == pytest.approx(1.0)
        assert transitions.height == 3
        assert examples.height == 3
        assert set(examples["id"].to_list()) == {"3", "4", "5"}

    def test_transitions_are_empty_when_everything_agrees(self) -> None:
        """Sem divergências, as tabelas de transição são vazias (sem divisão por zero)."""
        base = _base(["positivo", "negativo"], [0.9, 0.8])
        transitions, examples = summarize_classification_differences(
            build_comparison_frame(base, base)
        )
        assert transitions.is_empty()
        assert examples.is_empty()


class TestTextLengthAnalysis:
    """Testes de :func:`evaluation.llm_comparison.analyze_by_text_length`."""

    def test_bins_cover_all_tweets_with_valid_intervals(self, frame: pl.DataFrame) -> None:
        """As faixas cobrem todos os tweets; taxas e ICs de Wilson ficam em [0, 1]."""
        analysis = analyze_by_text_length(frame, n_bins=3)

        assert analysis["n_tweets"].sum() == frame.height
        for row in analysis.to_dicts():
            assert 0.0 <= row["agreement_rate_ci_lower"] <= row["agreement_rate"]
            assert row["agreement_rate"] <= row["agreement_rate_ci_upper"] <= 1.0
            assert row["min_words"] <= row["max_words"]

    def test_bins_are_ordered_by_length(self, frame: pl.DataFrame) -> None:
        """As faixas vêm em ordem crescente de tamanho."""
        analysis = analyze_by_text_length(frame, n_bins=3)
        assert analysis["min_words"].to_list() == sorted(analysis["min_words"].to_list())

    def test_merges_bins_when_all_tweets_have_the_same_length(self) -> None:
        """Se todos têm o mesmo tamanho, os quantis repetidos se fundem numa única faixa."""
        base = _base(["positivo", "negativo", "neutro"], [0.9, 0.8, 0.7], ["a b", "c d", "e f"])
        analysis = analyze_by_text_length(build_comparison_frame(base, base), n_bins=4)

        assert analysis.height == 1
        assert analysis["n_tweets"].to_list() == [3]

    def test_raises_for_empty_frame(self, frame: pl.DataFrame) -> None:
        """Um DataFrame vazio deve levantar ``EmptyDatasetError``."""
        with pytest.raises(EmptyDatasetError):
            analyze_by_text_length(frame.clear())


class TestComparisonPlots:
    """Cada gráfico devolve uma figura com eixos rotulados (CLAUDE.md, "Visualization")."""

    def test_class_distribution_plot_labels_axes(self, frame: pl.DataFrame) -> None:
        """O gráfico de distribuição tem título e eixos rotulados."""
        figure = plot_class_distribution_by_model(calculate_class_distribution(frame))
        axis = figure.axes[0]
        assert axis.get_title()
        assert axis.get_xlabel() == "Classe de sentimento"
        assert axis.get_ylabel() == "Tweets (%)"

    def test_confidence_plot_has_two_panels_and_colorbar(self, frame: pl.DataFrame) -> None:
        """Dois painéis (histograma e densidade) mais a barra de cores."""
        assert len(plot_confidence_comparison(frame).axes) == 3

    def test_length_plot_labels_axes(self, frame: pl.DataFrame) -> None:
        """O gráfico por tamanho tem eixos rotulados e limite em 0-100%."""
        axis = plot_agreement_by_text_length(analyze_by_text_length(frame, n_bins=3)).axes[0]
        assert axis.get_ylabel() == "Concordância (%)"
        assert axis.get_ylim() == (0, 100)

    def test_transition_plot_requires_divergences(self, frame: pl.DataFrame) -> None:
        """Sem divergências não há o que plotar: erro explícito; com elas, a figura é gerada."""
        transitions, _ = summarize_classification_differences(frame)
        assert plot_divergence_transitions(transitions).axes[0].get_xlabel() == "Tweets divergentes"
        with pytest.raises(EmptyDatasetError):
            plot_divergence_transitions(transitions.clear())

    @pytest.mark.parametrize(
        "plot", [plot_class_distribution_by_model, plot_agreement_by_text_length]
    )
    def test_plots_reject_empty_tables(self, plot: Any) -> None:
        """Tabelas vazias levantam ``EmptyDatasetError`` em vez de gerar um gráfico vazio."""
        with pytest.raises(EmptyDatasetError):
            plot(pl.DataFrame())


class _FakeSae:
    """SAE de mentira: a ativação do neurônio ``k`` é o valor da coluna ``k`` dos embeddings."""

    def compute_activations(self, embeddings: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Devolve os próprios embeddings como ativações."""
        return embeddings


def _build_discovery(embeddings: np.ndarray, frame: pl.DataFrame) -> DiscoveryData:
    """Dados de descoberta sintéticos: as ativações do SAE de mentira são os embeddings."""
    return DiscoveryData(
        partitioned=frame.select("id"),
        embeddings=embeddings,
        sae=_FakeSae(),
        cache_name="teste",
        corpus_hash="0" * 64,
    )


class TestModelDisagreementHypotheses:
    """Testes de :mod:`diagnostics.model_disagreement` (SAE, embeddings e LLM são dublês)."""

    def test_diagnostic_corpus_uses_min_confidence_and_only_normalized_text(
        self, frame: pl.DataFrame
    ) -> None:
        """Contrato de diagnóstico: ``agreement_score`` = menor confiança; sem texto original."""
        corpus = build_diagnostic_corpus(frame)

        assert corpus.columns == [
            "id",
            "text_normalized",
            "agreement_score",
            "lab_huggingface",
            "lab_openai",
        ]
        assert corpus["agreement_score"].to_list() == [0.90, 0.88, 0.55, 0.70, 0.86, 0.30]

    def test_evidence_lists_top_activating_tweets_per_hypothesis(self, frame: pl.DataFrame) -> None:
        """Por hipótese, os tweets de maior ativação positiva, com rótulos e confianças."""
        embeddings = np.array(
            [[0.0, 0.1], [0.5, 0.0], [0.0, 0.0], [0.9, 0.2], [0.0, 0.0], [0.7, 0.3]],
            dtype=float,
        )
        discovery = _build_discovery(embeddings, frame)
        hypotheses = pl.DataFrame({"hypothesis": ["menciona gostar"], "neuron_idx": [0]})

        evidence = build_hypothesis_evidence(
            hypotheses, discovery, frame, target="disagreement", top_tweets_per_hypothesis=2
        )

        assert evidence["id"].to_list() == ["3", "5"]
        assert evidence["rank"].to_list() == [1, 2]
        assert evidence["activation"].to_list() == [0.9, 0.7]
        assert evidence["target"].to_list() == ["disagreement"] * 2
        assert {"label_huggingface", "confidence_openai", "text_normalized"} <= set(
            evidence.columns
        )

    def test_evidence_skips_neurons_that_never_activate(self, frame: pl.DataFrame) -> None:
        """Neurônios sem ativação positiva não geram evidência (e o resultado mantém o esquema)."""
        discovery = _build_discovery(np.zeros((6, 2)), frame)
        evidence = build_hypothesis_evidence(
            pl.DataFrame({"hypothesis": ["x"], "neuron_idx": [1]}),
            discovery,
            frame,
            target="uncertainty",
            top_tweets_per_hypothesis=3,
        )
        assert evidence.is_empty()
        assert "hypothesis" in evidence.columns

    def test_run_records_gate_failure_as_result_and_keeps_going(
        self,
        frame: pl.DataFrame,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Reprovar o gate de sanidade é um resultado (registrado); os demais alvos continuam."""
        from diagnostics.settings import load_diagnostics_settings

        paths = SimpleNamespace()
        discovery = _build_discovery(np.tile(np.array([[0.4, 0.0]]), (6, 1)), frame)
        monkeypatch.setattr(model_disagreement, "prepare_discovery_data", lambda *args: discovery)

        def _fake_target_run(target: str, *args: Any, **kwargs: Any) -> Any:
            if target == "disagreement":
                raise SanityGateFailedError(target, "sem sinal")
            return SimpleNamespace(
                hypotheses=pl.DataFrame({"hypothesis": ["hipótese A"], "neuron_idx": [0]}),
                output_path=tmp_path / "uncertainty.parquet",
            )

        monkeypatch.setattr(model_disagreement, "run_target_diagnostics", _fake_target_run)
        assert load_diagnostics_settings().random_seed == 42

        outcomes = run_disagreement_hypotheses(
            frame,
            paths,  # type: ignore[arg-type]
            output_dir=tmp_path / "saida",
            targets=("disagreement", "uncertainty"),
            top_tweets_per_hypothesis=2,
            track=False,
        )

        assert [outcome.status for outcome in outcomes] == ["gate_reprovado", "concluido"]
        assert outcomes[0].n_hypotheses == 0 and outcomes[0].evidence_path is None
        assert outcomes[1].n_hypotheses == 1
        assert isinstance(outcomes[1], HypothesisTargetOutcome)
        assert outcomes[1].evidence_path is not None and outcomes[1].evidence_path.is_file()
        assert (tmp_path / "saida" / "corpus_diagnostico.parquet").is_file()

    def test_run_rejects_unknown_targets(self, frame: pl.DataFrame, tmp_path: Path) -> None:
        """Só ``disagreement`` e ``uncertainty`` são alvos válidos desta etapa."""
        with pytest.raises(DataValidationError, match="alvos inválidos"):
            run_disagreement_hypotheses(
                frame,
                SimpleNamespace(),  # type: ignore[arg-type]
                output_dir=tmp_path,
                targets=("gold_error",),
                track=False,
            )


def test_project_paths_expose_the_new_labeling_locations() -> None:
    """``ProjectPaths`` resolve as duas bases e o diretório de checkpoints a partir do YAML."""
    from config.paths import load_project_paths

    paths: ProjectPaths = load_project_paths()

    assert paths.huggingface_labeled_file.name == "tweets_data_huggingface.parquet"
    assert paths.openai_labeled_file.name == "tweets_data_openai.parquet"
    assert paths.labeling_checkpoints_dir.name == "labeling_checkpoints"
