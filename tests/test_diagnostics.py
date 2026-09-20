"""Testes da camada de diagnóstico HypotheSAEs (``diagnostics``), sem rede."""

import asyncio
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st

import main
from config.settings import load_general_config
from constants.labels import SENTIMENT_CLASSES
from data.gold import (
    GOLD_COLUMNS,
    load_gold_set,
    normalize_gold_label,
    sanitize_tweet_text,
    split_gold_discovery_eval,
)
from diagnostics import comparison as comparison_module
from diagnostics import hypotheses, llm_client, sae_runner, validation
from diagnostics import hypotheses as hypotheses_module
from diagnostics import sae_runner as sae_runner_module
from diagnostics.annotation import (
    AnnotationResult,
    annotate_concepts,
    annotate_concepts_async,
    assert_failure_rate_acceptable,
    build_annotation_prompt,
    enforce_annotation_budget,
    parse_yes_no,
    truncate_words,
)
from diagnostics.comparison import build_dry_run_report as build_comparison_dry_run_report
from diagnostics.comparison import (
    calculate_disagreement_and_agreement,
    classify_texts,
    classify_texts_async,
    compare_prompt_versions,
    render_prompt,
    slugify_model,
    summarize_hypothesis_weakening,
)
from diagnostics.cost import (
    ANNOTATION_PROMPT_OVERHEAD_TOKENS,
    CostEstimate,
    calculate_cost_usd,
    estimate_annotation_cost,
    estimate_classification_cost,
    estimate_hypothesis_generation_cost,
    format_cost_report,
)
from diagnostics.gold_eval import (
    OVERALL_GROUP,
    build_labeled_evaluation_frame,
    calculate_scores_with_ci,
    evaluate_models_by_concept,
    load_human_labels,
)
from diagnostics.hypotheses import (
    OUTPUT_COLUMNS,
    build_generation_kwargs,
    build_partition_slice,
    build_target_slug,
    compute_neuron_statistics,
    parse_arguments,
    resolve_target_arguments,
    run_target_diagnostics,
)
from diagnostics.hypotheses import build_dry_run_report as build_hypotheses_dry_run_report
from diagnostics.hypotheses import main as run_hypotheses_cli
from diagnostics.llm_client import (
    AsyncLLMClient,
    CallStats,
    DiskCompletionCache,
    _resolve_endpoint,
    estimate_tokens,
)
from diagnostics.prompt_synthesis import (
    RULES_HEADER,
    build_rules_request,
    insert_rules_into_prompt,
    parse_rules,
    synthesize_rules,
    write_prompt_v2,
)
from diagnostics.sae_runner import (
    DISCOVERY_PARTITION,
    HOLDOUT_PARTITION,
    SAE_VALIDATION_PARTITION,
    DiscoveryData,
    assign_partitions,
    build_embedding_cache_name,
    prepare_discovery_data,
)
from diagnostics.sampling import (
    HUMAN_LABEL_COLUMN,
    SAMPLE_COLUMNS,
    pseudonymize_id,
    resolve_sample_salt,
    sample_tweets_by_concept,
    write_labeling_sample,
)
from diagnostics.sanity import SanityGateResult, assert_sanity_gate_passed, evaluate_sanity_gate
from diagnostics.settings import (
    DEFAULT_DIAGNOSTICS_CONFIG_FILE,
    PricingSettings,
    load_diagnostics_settings,
)
from diagnostics.targets import (
    TARGET_NAMES,
    adapt_labeled_corpus,
    build_disagreement_target,
    build_gold_error_target,
    build_pseudo_label_target,
    build_target,
    build_uncertainty_target,
)
from diagnostics.tracking import flatten_params, keep_finite_metrics
from diagnostics.validation import build_dry_run_report as build_validation_dry_run_report
from diagnostics.validation import (
    run_validation_stage,
    select_holdout_rows,
    select_top_hypotheses,
    stratified_subsample_indices,
    validate_hypotheses,
)
from exceptions.configuration import (
    ConfigurationFileNotFoundError,
    InvalidConfigurationError,
    MissingEnvironmentVariableError,
)
from exceptions.data import DataNotFoundError, DataValidationError, EmptyDatasetError
from exceptions.pipeline import PipelineStageError, SanityGateFailedError
from pipelines.diagnostics_analysis import (
    _load_settings,
    _resolve_target_arguments,
    run_diagnostics_stage,
)
from pipelines.workflow import STAGE_REGISTRY
from schemas.diagnostics import (
    list_model_label_columns,
    validate_binary_target,
    validate_continuous_target,
    validate_diagnostic_corpus,
)

# ============================================================================
# annotation
# ============================================================================

TEMPLATE = 'PROPERTY: "{hypothesis}"\nTEXT: "{text}"\nOutput:'


class _FakeAnnotationClient:
    """Cliente que responde "Yes" quando o texto contém "ironia" e "No" caso contrário."""

    def __init__(self, unparseable: bool = False) -> None:
        self.stats = CallStats()
        self.unparseable = unparseable
        self.prompts: list[str] = []

    async def complete_many(self, prompts: list[str], **kwargs: Any) -> list[str | None]:
        self.prompts.extend(prompts)
        self.stats.n_requests += len(prompts)
        if self.unparseable:
            return ["PROPERTY: eco do prompt"] * len(prompts)
        return ["Yes. tem" if "ironia" in p.split("TEXT:")[1] else "No." for p in prompts]


class TestParseYesNo:
    """Interpretação da resposta do anotador."""

    @pytest.mark.parametrize(
        ("completion", "expected"),
        [
            ('Yes. "sun sets" is a natural scene.', 1),
            ("yes", 1),
            ("Sim, o texto expressa isso.", 1),
            ("No. Não menciona.", 0),
            ("Não, é neutro.", 0),
            ("  \n nao.", 0),
            ("<think>hmm</think> Yes.", 1),
            ("PROPERTY: eco do prompt", None),
            ("", None),
            (None, None),
            ("123 !!!", None),
        ],
    )
    def test_parse(self, completion: str | None, expected: int | None) -> None:
        """Yes/Sim -> 1, No/Não -> 0, o resto -> None."""
        assert parse_yes_no(completion) == expected


class TestPromptHelpers:
    """Truncamento e montagem do prompt."""

    def test_truncate_words(self) -> None:
        """Mantém só as primeiras palavras."""
        assert truncate_words("a b c d e", 3) == "a b c"

    def test_build_prompt_truncates_and_fills(self) -> None:
        """O texto é truncado antes de entrar no template."""
        prompt = build_annotation_prompt(TEMPLATE, "ironia", "um dois três quatro", max_words=2)
        assert 'PROPERTY: "ironia"' in prompt
        assert 'TEXT: "um dois"' in prompt

    def test_braces_in_text_do_not_break_formatting(self) -> None:
        """Chaves no tweet não quebram o ``str.format`` (são argumentos, não template)."""
        assert "{x}" in build_annotation_prompt(TEMPLATE, "c", "olha {x}", max_words=5)


class TestBudget:
    """Trava contra explosão de chamadas (N x C)."""

    def test_within_budget(self) -> None:
        """Dentro do orçamento devolve o nº de chamadas."""
        assert enforce_annotation_budget(10, 5, max_calls=50) == 50

    def test_over_budget_raises_with_hint(self) -> None:
        """Acima do orçamento pede subamostragem."""
        with pytest.raises(PipelineStageError, match="subamostre"):
            enforce_annotation_budget(1000, 100, max_calls=500)

    def test_failure_rate_check(self) -> None:
        """Taxa de falha acima do limite aborta; abaixo passa."""
        assert_failure_rate_acceptable(AnnotationResult({}, 10, 1), max_rate=0.2)
        with pytest.raises(PipelineStageError, match="Yes/No"):
            assert_failure_rate_acceptable(AnnotationResult({}, 10, 5), max_rate=0.2)

    def test_failure_rate_zero_total(self) -> None:
        """Sem anotações, a taxa é zero (sem divisão por zero)."""
        assert AnnotationResult({}, 0, 0).failure_rate == 0.0


class TestAnnotateConcepts:
    """Anotação N x C com cliente falso."""

    def test_annotations_are_aligned_per_concept(self) -> None:
        """Cada conceito recebe um vetor 0/1 na ordem dos textos."""
        client = _FakeAnnotationClient()
        result = asyncio.run(
            annotate_concepts_async(
                client,  # type: ignore[arg-type]
                ["tem ironia aqui", "texto liso"],
                ["c1", "c2"],
                model="m",
                template=TEMPLATE,
                max_words=10,
            )
        )
        assert set(result.annotations) == {"c1", "c2"}
        np.testing.assert_array_equal(result.annotations["c1"], [1, 0])
        assert result.n_total == 4
        assert result.n_failed == 0

    def test_unparseable_answers_are_counted_and_treated_as_zero(self) -> None:
        """Respostas ilegíveis contam como falha e viram 0."""
        client = _FakeAnnotationClient(unparseable=True)
        result = asyncio.run(
            annotate_concepts_async(
                client,  # type: ignore[arg-type]
                ["a", "b"],
                ["c"],
                model="m",
                template=TEMPLATE,
                max_words=5,
            )
        )
        assert result.n_failed == 2
        assert result.failure_rate == 1.0
        np.testing.assert_array_equal(result.annotations["c"], [0, 0])

    def test_sync_wrapper_enforces_failure_rate(self, diagnostics_settings: Any) -> None:
        """O modelo que não segue o formato faz o fluxo abortar (fora do dry-run)."""
        with pytest.raises(PipelineStageError, match="Yes/No"):
            annotate_concepts(
                diagnostics_settings,
                ["a", "b"],
                ["c"],
                client=_FakeAnnotationClient(unparseable=True),  # type: ignore[arg-type]
                template=TEMPLATE,
            )

    def test_sync_wrapper_returns_stats(self, diagnostics_settings: Any) -> None:
        """O wrapper devolve resultado e contadores do cliente."""
        client = _FakeAnnotationClient()
        result, stats = annotate_concepts(
            diagnostics_settings,
            ["tem ironia"],
            ["c"],
            client=client,  # type: ignore[arg-type]
            template=TEMPLATE,
        )
        assert stats.n_requests == 1
        np.testing.assert_array_equal(result.annotations["c"], [1])

    def test_sync_wrapper_respects_budget(self, diagnostics_settings: Any) -> None:
        """Estourar ``validation.max_annotation_calls`` falha antes de qualquer chamada."""
        settings = diagnostics_settings.model_copy(
            update={
                "validation": diagnostics_settings.validation.model_copy(
                    update={"max_annotation_calls": 3}
                )
            }
        )
        client = SimpleNamespace(stats=CallStats())
        with pytest.raises(PipelineStageError, match="orçamento"):
            annotate_concepts(
                settings,
                ["a", "b"],
                ["c1", "c2"],
                client=client,  # type: ignore[arg-type]
                template=TEMPLATE,
            )


# ============================================================================
# comparison
# ============================================================================

CLASSES = ["negativo", "neutro", "positivo"]
OPTIONS = {"n_folds": 6, "n_bootstrap": 50, "confidence_level": 0.9, "random_seed": 0}


def _truth(n: int = 60) -> list[str]:
    return [CLASSES[i % 3] for i in range(n)]


class TestHelpers:
    """Funções pequenas."""

    def test_slugify_model(self) -> None:
        """Sufixo seguro para nome de coluna."""
        assert slugify_model("gemma2:9b") == "gemma2_9b"
        assert slugify_model("org/Modelo-X 1") == "org_Modelo_X_1"

    def test_render_prompt_keeps_json_braces(self) -> None:
        """``str.replace`` preserva as chaves do JSON do prompt."""
        assert render_prompt('{"label": "x"} {{TEXTO}}', "oi") == '{"label": "x"} oi'


class _FakeClassifierClient:
    """Cliente falso: devolve JSON de rótulo, ou lixo para o texto "quebrado"."""

    def __init__(self) -> None:
        self.stats = CallStats()

    async def complete_many(self, prompts: list[str], **_: Any) -> list[str | None]:
        out: list[str | None] = []
        for prompt in prompts:
            if "quebrado" in prompt:
                out.append("não sei")
            elif "falhou" in prompt:
                out.append(None)
            else:
                out.append('{"label": "positivo", "confidence": 0.9}')
        return out


class TestClassify:
    """Classificação com um template."""

    def test_labels_and_unparseable(self) -> None:
        """Rótulos válidos são extraídos; resposta ilegível ou ausente vira ``None``."""
        labels = asyncio.run(
            classify_texts_async(
                _FakeClassifierClient(),  # type: ignore[arg-type]
                ["bom", "quebrado", "falhou"],
                "Tweet: {{TEXTO}}",
                model="m",
                namespace="v1",
            )
        )
        assert labels == ["positivo", None, None]

    def test_sync_wrapper(self, diagnostics_settings: Any) -> None:
        """Versão síncrona devolve rótulos e contadores."""
        labels, stats = classify_texts(
            diagnostics_settings,
            ["bom"],
            "Tweet: {{TEXTO}}",
            model="m",
            namespace="v1",
            client=_FakeClassifierClient(),  # type: ignore[arg-type]
        )
        assert labels == ["positivo"]
        assert isinstance(stats, CallStats)


class TestComparePromptVersions:
    """Comparação pareada."""

    def test_better_v2_has_positive_delta_and_small_pvalue(self) -> None:
        """Se o v2 corrige erros do v1, o ganho é positivo e o McNemar é significativo."""
        truth = _truth()
        v1 = [t if i % 2 else CLASSES[(CLASSES.index(t) + 1) % 3] for i, t in enumerate(truth)]
        v2 = list(truth)
        result = compare_prompt_versions(truth, v1, v2, **OPTIONS)
        assert result.delta_mcc > 0
        assert result.delta_mcc_ci[0] <= result.delta_mcc <= result.delta_mcc_ci[1] + 1e-9
        assert result.mcnemar_pvalue < 0.05
        assert result.scores_v2["mcc"] == pytest.approx(1.0)
        assert result.n_unparsed == 0

    def test_identical_versions_have_zero_delta_and_p_one(self) -> None:
        """Versões idênticas: ganho zero, McNemar/Wilcoxon sem evidência."""
        truth = _truth()
        preds = [CLASSES[(CLASSES.index(t) + (i % 4 == 0)) % 3] for i, t in enumerate(truth)]
        result = compare_prompt_versions(truth, preds, list(preds), **OPTIONS)
        assert result.delta_mcc == pytest.approx(0.0)
        assert result.wilcoxon_pvalue == 1.0
        assert result.mcnemar_pvalue >= 0.99

    def test_unparsed_pairs_are_dropped_and_counted(self) -> None:
        """Pares com ``None`` em qualquer versão saem e são contados."""
        truth = _truth()
        v1: list[str | None] = list(truth)
        v2: list[str | None] = list(truth)
        v1[0], v2[1] = None, None
        result = compare_prompt_versions(truth, v1, v2, **OPTIONS)
        assert result.n_unparsed == 2
        assert result.n_paired == len(truth) - 2

    def test_size_mismatch_raises(self) -> None:
        """Vetores de tamanhos diferentes são erro."""
        with pytest.raises(DataValidationError, match="tamanhos"):
            compare_prompt_versions(["a"], ["a", "b"], ["a"], **OPTIONS)

    def test_too_few_pairs_raises(self) -> None:
        """Menos pares do que folds não permite o Wilcoxon."""
        with pytest.raises(DataValidationError, match="pares válidos"):
            compare_prompt_versions(_truth(3), _truth(3), _truth(3), **OPTIONS)

    def test_row_is_flat(self) -> None:
        """``to_row`` produz uma linha achatada com o modelo."""
        truth = _truth()
        row = compare_prompt_versions(truth, truth, truth, **OPTIONS).to_row("gemma2:9b")
        assert row["model"] == "gemma2:9b"
        assert {"v1_mcc", "v2_mcc", "delta_mcc", "mcnemar_pvalue", "wilcoxon_pvalue"} <= set(row)

    def test_is_reproducible(self) -> None:
        """Mesma semente, mesmo resultado."""
        truth = _truth()
        v1 = [CLASSES[(CLASSES.index(t) + (i % 3 == 0)) % 3] for i, t in enumerate(truth)]
        first = compare_prompt_versions(truth, v1, truth, **OPTIONS)
        second = compare_prompt_versions(truth, v1, truth, **OPTIONS)
        assert first == second


class TestDisagreement:
    """Discordância e agreement entre modelos."""

    def test_rates(self) -> None:
        """Metade dos tweets diverge; agreement médio = (0,5 + 1,0) / 2."""
        result = calculate_disagreement_and_agreement({"a": ["x", "y"], "b": ["x", "x"]})
        assert result == {"disagreement_rate": 0.5, "mean_agreement": 0.75, "n_tweets": 2.0}

    def test_none_rows_are_ignored(self) -> None:
        """Só entram tweets em que todos os modelos têm predição."""
        result = calculate_disagreement_and_agreement({"a": ["x", None], "b": ["y", "x"]})
        assert result["n_tweets"] == 1.0
        assert result["disagreement_rate"] == 1.0

    def test_no_comparable_rows_gives_nan(self) -> None:
        """Sem tweets comparáveis, as taxas são ``nan``."""
        result = calculate_disagreement_and_agreement({"a": [None], "b": ["x"]})
        assert math.isnan(result["disagreement_rate"])


class TestWeakening:
    """As hipóteses principais enfraquecem após o v2?"""

    def _table(self, separations: list[float], pvals: list[float]) -> pl.DataFrame:
        return pl.DataFrame({"separation_score": separations, "regression_pval": pvals})

    def test_weaker_after_is_flagged(self) -> None:
        """Separação menor e menos significativas depois: ``weakened``."""
        before = self._table([0.4, 0.3], [0.001, 0.002])
        after = self._table([0.1, 0.05], [0.4, 0.6])
        summary = summarize_hypothesis_weakening(before, after, top_k=2)
        assert summary["weakened"] is True
        assert summary["n_significant_before"] == 2.0
        assert summary["n_significant_after"] == 0.0

    def test_unchanged_is_not_weakened(self) -> None:
        """Sem mudança, não há enfraquecimento."""
        table = self._table([0.4], [0.001])
        assert summarize_hypothesis_weakening(table, table)["weakened"] is False

    def test_empty_after_counts_as_zero(self) -> None:
        """Tabela vazia depois do v2 conta como zero de separação."""
        before = self._table([0.4], [0.001])
        after = pl.DataFrame(
            {"separation_score": [], "regression_pval": []},
            schema={"separation_score": pl.Float64, "regression_pval": pl.Float64},
        )
        assert summarize_hypothesis_weakening(before, after)["mean_abs_separation_after"] == 0.0


def test_dry_run_report(diagnostics_settings: Any) -> None:
    """A estimativa cobre gold x 2 versões x modelos e afirma que não houve rede."""
    report = build_comparison_dry_run_report(
        diagnostics_settings, n_gold=100, models=["m1", "m2"], prompt_tokens=800
    )
    assert "100 tweets do gold x 2 versões x 2 modelo(s)" in report
    assert "Nenhuma chamada de rede" in report
    assert "400" in report  # 100 tweets x 2 versões x 2 modelos = 400 chamadas


# ============================================================================
# cost
# ============================================================================


class TestCalculateCost:
    """Conversão de tokens em USD."""

    def test_local_model_is_free_by_default(self) -> None:
        """Preço padrão zero (Ollama)."""
        assert calculate_cost_usd(PricingSettings(), "gemma2:9b", 5_000_000, 1_000_000) == 0.0

    def test_model_specific_price_overrides_default(self) -> None:
        """Preço do modelo prevalece sobre o padrão."""
        pricing = PricingSettings.model_validate(
            {
                "default": {"input": 1.0, "output": 1.0},
                "models": {"m": {"input": 2.0, "output": 4.0}},
            }
        )
        assert calculate_cost_usd(pricing, "m", 1_000_000, 500_000) == pytest.approx(4.0)
        assert calculate_cost_usd(pricing, "outro", 1_000_000, 0) == pytest.approx(1.0)


class TestEstimates:
    """Contagem de chamadas por etapa."""

    def test_annotation_is_cartesian_product(self, diagnostics_settings: Any) -> None:
        """N tweets x C conceitos chamadas, cada uma com o cabeçalho do prompt."""
        estimate = estimate_annotation_cost(diagnostics_settings, n_tweets=50, n_concepts=10)
        assert estimate.n_calls == 500
        assert estimate.input_tokens >= 500 * ANNOTATION_PROMPT_OVERHEAD_TOKENS

    def test_hypothesis_generation_counts_interpretation_and_scoring(
        self, diagnostics_settings: Any
    ) -> None:
        """Interpretação (neurônios x candidatas) + pontuação (cada candidata x exemplos)."""
        hyp = diagnostics_settings.hypotheses
        n_interp = hyp.n_selected_neurons * hyp.n_candidate_interpretations
        estimate = estimate_hypothesis_generation_cost(
            diagnostics_settings, n_discovery_tweets=3000
        )
        assert estimate.n_calls == n_interp + n_interp * hyp.n_scoring_examples

    def test_scoring_examples_are_capped_by_available_tweets(
        self, diagnostics_settings: Any
    ) -> None:
        """Poucos tweets de descoberta reduzem as chamadas de pontuação."""
        hyp = diagnostics_settings.hypotheses
        n_interp = hyp.n_selected_neurons * hyp.n_candidate_interpretations
        estimate = estimate_hypothesis_generation_cost(diagnostics_settings, n_discovery_tweets=10)
        assert estimate.n_calls == n_interp + n_interp * 10

    def test_classification_scales_with_prompts(self, diagnostics_settings: Any) -> None:
        """Chamadas = tweets x (modelos x versões)."""
        estimate = estimate_classification_cost(
            diagnostics_settings, n_tweets=100, n_prompts=4, prompt_tokens=800
        )
        assert estimate.n_calls == 400


class TestFormatReport:
    """Relatório textual."""

    def test_report_has_total_and_ollama_note(self) -> None:
        """Inclui o total e a observação sobre custo local."""
        report = format_cost_report(
            {"a": CostEstimate(2, 10, 4, 0.0), "b": CostEstimate(3, 20, 6, 0.0)}, provider="ollama"
        )
        assert "TOTAL" in report
        assert "Ollama local" in report

    def test_addition_sums_fields(self) -> None:
        """A soma de estimativas soma todos os campos."""
        total = CostEstimate(1, 2, 3, 0.5) + CostEstimate(4, 5, 6, 1.5)
        assert (total.n_calls, total.input_tokens, total.output_tokens) == (5, 7, 9)
        assert total.cost_usd == pytest.approx(2.0)


# ============================================================================
# gold
# ============================================================================


def _write_gold(path: Path, n_rows: int = 40, label_column: str = "sentiment_label") -> Path:
    labels = ["Positive", "Negative", "neutral", "pos"]
    pl.DataFrame(
        {
            "id": list(range(n_rows)),
            "text": [f"oi @user{i} veja http://x.com/{i}" for i in range(n_rows)],
            label_column: [labels[i % 4] for i in range(n_rows)],
        }
    ).write_parquet(path)
    return path


class TestNormalization:
    """Rótulos e sanitização."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("Positive", "positivo"), (" NEG ", "negativo"), ("neutro", "neutro"), ("misto", None)],
    )
    def test_labels(self, raw: str, expected: str | None) -> None:
        """Aliases em inglês/abreviados viram classes em pt-BR."""
        assert normalize_gold_label(raw) == expected

    def test_sanitize_removes_mentions_and_urls(self) -> None:
        """LGPD: menções e URLs viram tokens."""
        cleaned = sanitize_tweet_text("oi @fulano   veja http://x.com/abc")
        assert "@fulano" not in cleaned
        assert "http" not in cleaned
        assert "[MENCAO]" in cleaned
        assert "[URL]" in cleaned


class TestLoadGoldSet:
    """Leitura e validação do arquivo."""

    def test_loads_contract_columns(self, tmp_path: Path) -> None:
        """Devolve id/text_normalized/gold_label, sem texto bruto."""
        gold = load_gold_set(_write_gold(tmp_path / "g.parquet"))
        assert tuple(gold.columns) == GOLD_COLUMNS
        assert set(gold["gold_label"]) == {"positivo", "negativo", "neutro"}
        assert not any("@user" in text for text in gold["text_normalized"])

    def test_missing_file_gives_actionable_error(self, tmp_path: Path) -> None:
        """Arquivo ausente orienta a colocar o gold em data/external/."""
        with pytest.raises(DataNotFoundError, match="data/external"):
            load_gold_set(tmp_path / "nao_existe.parquet")

    def test_missing_column_raises(self, tmp_path: Path) -> None:
        """Coluna de rótulo ausente falha cedo."""
        path = _write_gold(tmp_path / "g.parquet", label_column="outra")
        with pytest.raises(DataValidationError, match="colunas ausentes"):
            load_gold_set(path)

    def test_unknown_label_raises(self, tmp_path: Path) -> None:
        """Rótulo desconhecido não é descartado em silêncio."""
        path = tmp_path / "g.parquet"
        pl.DataFrame({"id": [1], "text": ["a"], "sentiment_label": ["misto"]}).write_parquet(path)
        with pytest.raises(DataValidationError, match="não reconhecidos"):
            load_gold_set(path)

    def test_all_blank_texts_raise_empty(self, tmp_path: Path) -> None:
        """Sem nenhum texto válido, o dataset é vazio."""
        path = tmp_path / "g.parquet"
        pl.DataFrame({"id": [1], "text": ["   "], "sentiment_label": ["pos"]}).write_parquet(path)
        with pytest.raises(EmptyDatasetError):
            load_gold_set(path)

    def test_max_tweets_subsamples(self, tmp_path: Path) -> None:
        """``max_tweets`` limita o tamanho mantendo todas as classes."""
        gold = load_gold_set(_write_gold(tmp_path / "g.parquet", 80), max_tweets=20)
        assert 15 <= gold.height <= 24
        assert gold["gold_label"].n_unique() == 3


class TestSplit:
    """Divisão descoberta/avaliação."""

    def test_split_is_disjoint_and_complete(self, tmp_path: Path) -> None:
        """G_disc e G_eval são disjuntos por id e cobrem o gold."""
        gold = load_gold_set(_write_gold(tmp_path / "g.parquet", 80))
        g_disc, g_eval = split_gold_discovery_eval(gold, eval_fraction=0.5, random_seed=0)
        assert set(g_disc["id"]).isdisjoint(set(g_eval["id"]))
        assert g_disc.height + g_eval.height == gold.height
        assert set(g_eval["gold_label"]) == set(gold["gold_label"])

    def test_split_is_deterministic(self, tmp_path: Path) -> None:
        """Mesma semente, mesma divisão."""
        gold = load_gold_set(_write_gold(tmp_path / "g.parquet", 80))
        first = split_gold_discovery_eval(gold, eval_fraction=0.4, random_seed=5)
        second = split_gold_discovery_eval(gold, eval_fraction=0.4, random_seed=5)
        assert first[1].equals(second[1])

    @pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_fraction_raises(self, fraction: float) -> None:
        """A fração deve estar em (0, 1)."""
        gold = pl.DataFrame({"id": ["1"], "text_normalized": ["a"], "gold_label": ["positivo"]})
        with pytest.raises(DataValidationError):
            split_gold_discovery_eval(gold, eval_fraction=fraction, random_seed=0)


# ============================================================================
# gold_eval
# ============================================================================


def _write_labels(path: Path, labels: list[str | None]) -> Path:
    pl.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(len(labels))],
            "concept": ["c1"] * len(labels),
            "text": ["t"] * len(labels),
            "rotulo_humano": labels,
        }
    ).write_csv(path)
    return path


class TestLoadHumanLabels:
    """Leitura do CSV rotulado."""

    def test_blank_rows_are_dropped_and_labels_normalized(self, tmp_path: Path) -> None:
        """Linhas em branco saem; caixa e espaços são normalizados."""
        path = _write_labels(tmp_path / "r.csv", [" Positivo ", None, "NEGATIVO", ""])
        labeled = load_human_labels(path)
        assert labeled["rotulo_humano"].to_list() == ["positivo", "negativo"]

    def test_unknown_label_raises(self, tmp_path: Path) -> None:
        """Rótulo fora das classes é erro."""
        with pytest.raises(DataValidationError, match="fora de"):
            load_human_labels(_write_labels(tmp_path / "r.csv", ["positivo", "misto"]))

    def test_nothing_labeled_raises(self, tmp_path: Path) -> None:
        """CSV sem nenhum rótulo é vazio."""
        with pytest.raises(EmptyDatasetError):
            load_human_labels(_write_labels(tmp_path / "r.csv", [None, None]))

    def test_missing_columns_raise(self, tmp_path: Path) -> None:
        """Faltando a coluna do rótulo humano."""
        path = tmp_path / "r.csv"
        pl.DataFrame({"sample_id": ["a"], "concept": ["c"]}).write_csv(path)
        with pytest.raises(DataValidationError, match="colunas ausentes"):
            load_human_labels(path)


def _frame() -> pl.DataFrame:
    """Dois conceitos x 12 tweets; ``lab_perfeito`` acerta tudo e ``lab_ruim`` erra tudo."""
    human = [CLASSES[i % 3] for i in range(24)]
    return pl.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(24)],
            "concept": ["c1"] * 12 + ["c2"] * 12,
            "rotulo_humano": human,
            "lab_perfeito": human,
            "lab_ruim": [CLASSES[(i + 1) % 3] for i in range(24)],
        }
    )


class TestBuildFrame:
    """Junção rótulos humanos + chave + predições."""

    def test_joins_predictions_through_key(self) -> None:
        """A chave liga ``sample_id`` ao ``id`` do corpus; o ``id`` não vaza no resultado."""
        labels = pl.DataFrame(
            {"sample_id": ["s1"], "concept": ["c"], "rotulo_humano": ["positivo"]}
        )
        key = pl.DataFrame({"sample_id": ["s1"], "id": ["99"], "concept": ["c"]})
        corpus = pl.DataFrame({"id": ["99", "100"], "lab_a": ["positivo", "negativo"]})
        frame = build_labeled_evaluation_frame(labels, key, corpus)
        assert frame.columns == ["sample_id", "concept", "rotulo_humano", "lab_a"]
        assert frame["lab_a"].to_list() == ["positivo"]


class TestEvaluateModels:
    """MCC e macro-F1 por conceito e modelo, com IC."""

    def test_row_per_group_and_model(self) -> None:
        """(2 conceitos + total) x 2 modelos."""
        result = evaluate_models_by_concept(
            _frame(), model_columns=["lab_perfeito", "lab_ruim"], n_bootstrap=30, random_seed=0
        )
        assert result.height == 6
        assert set(result["concept"]) == {"c1", "c2", OVERALL_GROUP}

    def test_perfect_model_scores_one_and_bad_model_is_worse(self) -> None:
        """MCC 1,0 para o modelo perfeito; negativo para o que erra tudo."""
        result = evaluate_models_by_concept(
            _frame(), model_columns=["lab_perfeito", "lab_ruim"], n_bootstrap=30, random_seed=0
        )
        overall = result.filter(pl.col("concept") == OVERALL_GROUP)
        perfect = overall.filter(pl.col("model") == "lab_perfeito").row(0, named=True)
        bad = overall.filter(pl.col("model") == "lab_ruim").row(0, named=True)
        assert perfect["mcc"] == pytest.approx(1.0)
        assert perfect["mcc_ci_low"] <= perfect["mcc"] <= perfect["mcc_ci_high"]
        assert bad["mcc"] < 0
        assert perfect["n"] == 24

    def test_nulls_in_predictions_are_ignored(self) -> None:
        """Predições ausentes não entram no ``n``."""
        frame = _frame().with_columns(
            pl.when(pl.col("sample_id") == "s0")
            .then(None)
            .otherwise(pl.col("lab_perfeito"))
            .alias("lab_perfeito")
        )
        result = evaluate_models_by_concept(
            frame, model_columns=["lab_perfeito"], n_bootstrap=20, random_seed=0
        )
        assert result.filter(pl.col("concept") == OVERALL_GROUP)["n"].item() == 23

    def test_tiny_group_returns_nan(self) -> None:
        """Com menos de 2 amostras, métricas e ICs são ``nan`` (sem exceção)."""
        scores = calculate_scores_with_ci(
            ["positivo"], ["positivo"], n_bootstrap=10, confidence_level=0.9, seed=0
        )
        assert all(math.isnan(value) for value in scores.values())

    def test_bootstrap_is_reproducible(self) -> None:
        """Mesma semente, mesmo IC."""
        args = (["positivo", "negativo", "neutro"] * 5, ["positivo", "neutro", "neutro"] * 5)
        options = {"n_bootstrap": 50, "confidence_level": 0.95, "seed": 7}
        assert calculate_scores_with_ci(*args, **options) == calculate_scores_with_ci(
            *args, **options
        )


# ============================================================================
# hypotheses
# ============================================================================


class _FakeSae:
    """SAE falso: as ativações são o valor absoluto das duas primeiras dimensões."""

    def compute_activations(self, embeddings: np.ndarray, show_progress: bool = True) -> np.ndarray:
        return np.abs(embeddings[:, :2])


def _discovery(corpus: pl.DataFrame, settings: Any, *, signal: float) -> DiscoveryData:
    """Monta um ``DiscoveryData`` com embeddings em que a dimensão 0 carrega o alvo."""
    partitioned = assign_partitions(
        corpus,
        holdout_size=settings.splits.holdout_size,
        validation_size=settings.splits.validation_size,
        random_seed=settings.random_seed,
    )
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(partitioned.height, 6))
    target = build_disagreement_target(corpus)
    positives = set(target.filter(pl.col("target") == 1)["id"])
    is_positive = np.array([tweet_id in positives for tweet_id in partitioned["id"]])
    embeddings[:, 0] += signal * is_positive
    return DiscoveryData(partitioned, embeddings, _FakeSae(), "cache", "h" * 64)


def _fake_generate(**_: Any) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "neuron_idx": [0, 1, 1],
            "target_lasso": [0.9, 0.4, 0.3],
            "interpretation": ["menciona ironia", "usa gíria", "usa gíria"],
            "f1_fidelity_score": [0.7, 0.6, 0.5],
        }
    )


def _fake_score(**kwargs: Any) -> tuple[dict[str, Any], pd.DataFrame]:
    names = list(kwargs["hypothesis_annotations"])
    return {}, pd.DataFrame(
        {
            "hypothesis": names,
            "separation_score": [0.2] * len(names),
            "separation_pval": [0.01] * len(names),
            "regression_coef": [1.0] * len(names),
            "regression_pval": [0.02] * len(names),
            "feature_prevalence": [0.4] * len(names),
        }
    )


def _fast_settings(settings: Any) -> Any:
    """Reduz bootstrap/permutações do gate para manter o teste rápido."""
    return settings.model_copy(
        update={
            "sanity": settings.sanity.model_copy(update={"n_bootstrap": 100, "n_permutations": 100})
        }
    )


class TestSlugAndKwargs:
    """Identificador do alvo e argumentos de ``generate_hypotheses``."""

    def test_slug(self) -> None:
        """Junta apenas as partes informadas."""
        assert build_target_slug("disagreement") == "disagreement"
        assert (
            build_target_slug("pseudo_label", model_column="lab_a", label="negativo")
            == "pseudo_label_lab_a_negativo"
        )

    def test_generation_kwargs_follow_settings(self, diagnostics_settings: Any) -> None:
        """Os argumentos vêm do YAML (lasso, 3 candidatas, provedor único)."""
        kwargs = build_generation_kwargs(diagnostics_settings, cache_name="c", classification=True)
        assert kwargs["selection_method"] == "lasso"
        assert kwargs["n_candidate_interpretations"] == 3
        assert kwargs["interpret_llm_kwargs"]["provider"] == diagnostics_settings.llm.provider
        assert kwargs["n_workers_annotation"] <= diagnostics_settings.llm.max_concurrency
        assert "Considere fenômenos" in kwargs["task_specific_instructions"]


class TestPartitionSlice:
    """Alinhamento entre alvo, partição e embeddings."""

    def test_slice_is_aligned_and_disjoint(
        self, diagnostic_corpus_large: pl.DataFrame, diagnostics_settings: Any
    ) -> None:
        """Textos, alvo e embeddings têm o mesmo tamanho; treino e holdout não se cruzam."""
        discovery = _discovery(diagnostic_corpus_large, diagnostics_settings, signal=0.0)
        target = build_disagreement_target(diagnostic_corpus_large)
        train = build_partition_slice(target, discovery, DISCOVERY_PARTITION)
        test = build_partition_slice(target, discovery, HOLDOUT_PARTITION)
        assert len(train.texts) == train.labels.size == train.embeddings.shape[0]
        assert set(train.texts).isdisjoint(set(test.texts))

    def test_empty_partition_raises(
        self, diagnostic_corpus_large: pl.DataFrame, diagnostics_settings: Any
    ) -> None:
        """Alvo sem linhas na partição pedida é erro explícito."""
        discovery = _discovery(diagnostic_corpus_large, diagnostics_settings, signal=0.0)
        empty_target = pl.DataFrame({"id": ["nao-existe"], "target": [1]})
        with pytest.raises(EmptyDatasetError):
            build_partition_slice(empty_target, discovery, DISCOVERY_PARTITION)


class TestNeuronStatistics:
    """Estatísticas por hipótese sem chamadas de LLM."""

    def test_output_schema_and_dedup(
        self,
        diagnostic_corpus_large: pl.DataFrame,
        diagnostics_settings: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Colunas do esquema de saída; interpretações repetidas ou nulas são removidas."""
        monkeypatch.setattr(hypotheses, "score_hypotheses", _fake_score)
        discovery = _discovery(diagnostic_corpus_large, diagnostics_settings, signal=3.0)
        target = build_disagreement_target(diagnostic_corpus_large)
        slice_ = build_partition_slice(target, discovery, DISCOVERY_PARTITION)
        generated = pl.from_pandas(_fake_generate()).vstack(
            pl.DataFrame(
                {
                    "neuron_idx": [0],
                    "target_lasso": [0.1],
                    "interpretation": [None],
                    "f1_fidelity_score": [0.0],
                },
                schema_overrides={"interpretation": pl.String},
            )
        )
        table = compute_neuron_statistics(
            generated,
            discovery.sae,
            slice_,
            settings=diagnostics_settings,
            classification=True,
            target_slug="disagreement",
            partition_name="discovery",
        )
        assert set(table.columns) <= set(OUTPUT_COLUMNS)
        assert {"hypothesis", "separation_score", "regression_pval", "feature_prevalence"} <= set(
            table.columns
        )
        assert sorted(table["hypothesis"].to_list()) == ["menciona ironia", "usa gíria"]
        assert set(table["partition"]) == {"discovery"}

    def test_no_valid_hypotheses_returns_empty_table(
        self, diagnostic_corpus_large: pl.DataFrame, diagnostics_settings: Any
    ) -> None:
        """Sem interpretações válidas, devolve tabela vazia (sem chamar o scorer)."""
        discovery = _discovery(diagnostic_corpus_large, diagnostics_settings, signal=0.0)
        target = build_disagreement_target(diagnostic_corpus_large)
        slice_ = build_partition_slice(target, discovery, DISCOVERY_PARTITION)
        empty = pl.DataFrame(
            {"neuron_idx": [0], "interpretation": [None]},
            schema_overrides={"interpretation": pl.String},
        )
        table = compute_neuron_statistics(
            empty,
            discovery.sae,
            slice_,
            settings=diagnostics_settings,
            classification=True,
            target_slug="t",
            partition_name="discovery",
        )
        assert table.is_empty()


class TestRunTargetDiagnostics:
    """Fluxo completo com gate, hipóteses e gravação."""

    def test_passing_gate_writes_outputs(
        self,
        diagnostic_corpus_large: pl.DataFrame,
        diagnostics_settings: Any,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Com sinal nos embeddings, o gate passa e a tabela é gravada em parquet e CSV."""
        settings = _fast_settings(diagnostics_settings)
        monkeypatch.setattr(hypotheses, "score_hypotheses", _fake_score)
        discovery = _discovery(diagnostic_corpus_large, settings, signal=5.0)
        paths = SimpleNamespace(reports_interpretability_dir=tmp_path)
        result = run_target_diagnostics(
            "disagreement",
            diagnostic_corpus_large,
            discovery,
            settings,
            paths,  # type: ignore[arg-type]
            track=False,
            generate_fn=_fake_generate,
        )
        assert result.gate.go
        assert result.output_path is not None
        assert result.output_path.is_file()
        assert result.output_path.with_suffix(".csv").is_file()
        assert result.hypotheses is not None
        assert result.hypotheses.height == 2

    def test_failing_gate_aborts_and_records_reason(
        self, diagnostic_corpus_large: pl.DataFrame, diagnostics_settings: Any, tmp_path: Path
    ) -> None:
        """Gate reprovado: não gera hipóteses (o LLM nem é chamado) e grava o motivo."""
        base = _fast_settings(diagnostics_settings)
        settings = base.model_copy(
            update={"sanity": base.sanity.model_copy(update={"min_effect": 0.6})}
        )
        discovery = _discovery(diagnostic_corpus_large, settings, signal=5.0)

        def _must_not_run(**_: Any) -> None:
            raise AssertionError("generate_hypotheses não deve rodar com o gate reprovado")

        paths = SimpleNamespace(reports_interpretability_dir=tmp_path)
        with pytest.raises(SanityGateFailedError, match="disagreement"):
            run_target_diagnostics(
                "disagreement",
                diagnostic_corpus_large,
                discovery,
                settings,
                paths,  # type: ignore[arg-type]
                track=False,
                generate_fn=_must_not_run,
            )
        assert (tmp_path / settings.data.output_dir / "disagreement_gate_reprovado.json").is_file()


class TestCli:
    """Argumentos, dry-run e erros de uso."""

    def test_parse_arguments(self) -> None:
        """``--target`` é obrigatório e restrito aos quatro alvos."""
        assert parse_arguments(["--target", "disagreement", "--dry-run"]).dry_run is True
        with pytest.raises(SystemExit):
            parse_arguments(["--target", "inexistente"])
        with pytest.raises(SystemExit):
            parse_arguments([])

    def test_resolve_target_arguments_uses_yaml_defaults(self, diagnostics_settings: Any) -> None:
        """Sem flags, usa modelo e classe de ``configs/diagnostics.yaml -> targets``."""
        args = parse_arguments(["--target", "pseudo_label"])
        resolved = resolve_target_arguments("pseudo_label", args, diagnostics_settings)
        assert resolved["model_column"] == "lab_huggingface"
        assert resolved["label"] == "negativo"
        override = parse_arguments(["--target", "pseudo_label", "--label", "positivo"])
        assert (
            resolve_target_arguments("pseudo_label", override, diagnostics_settings)["label"]
            == "positivo"
        )

    def test_dry_run_report_has_no_network(
        self, diagnostic_corpus_large: pl.DataFrame, diagnostics_settings: Any
    ) -> None:
        """O relatório traz o nº de chamadas e diz que não houve rede."""
        report = build_hypotheses_dry_run_report(
            "disagreement", diagnostic_corpus_large, diagnostics_settings
        )
        assert "DRY-RUN" in report
        assert "Nenhuma chamada de rede" in report
        assert "generate_hypotheses" in report

    def test_main_dry_run_prints_report(
        self,
        diagnostic_corpus_large: pl.DataFrame,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """``--dry-run`` imprime a estimativa e sai com 0, sem carregar embeddings nem SAE."""
        monkeypatch.setattr(hypotheses, "configure_environment_variables", lambda: None)
        monkeypatch.setattr(hypotheses, "configure_logging", lambda: None)
        monkeypatch.setattr(hypotheses, "load_project_paths", SimpleNamespace)
        monkeypatch.setattr(
            hypotheses, "load_diagnostic_corpus", lambda *_: diagnostic_corpus_large
        )

        def _must_not_run(*_: Any, **__: Any) -> None:
            raise AssertionError("dry-run não pode preparar embeddings/SAE")

        monkeypatch.setattr(hypotheses, "prepare_discovery_data", _must_not_run)
        assert run_hypotheses_cli(["--target", "disagreement", "--dry-run"]) == 0
        assert "DRY-RUN do alvo 'disagreement'" in capsys.readouterr().out

    def test_gold_error_requires_corpus(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """O alvo ``gold_error`` exige ``--corpus`` com as predições sobre o gold."""
        monkeypatch.setattr(hypotheses, "configure_environment_variables", lambda: None)
        monkeypatch.setattr(hypotheses, "configure_logging", lambda: None)
        monkeypatch.setattr(hypotheses, "load_project_paths", SimpleNamespace)
        with pytest.raises(DataValidationError, match="--corpus"):
            run_hypotheses_cli(["--target", "gold_error", "--dry-run"])


# ============================================================================
# llm_client
# ============================================================================


class _FakeCompletions:
    """Dublê de ``client.chat.completions`` que registra chamadas e concorrência."""

    def __init__(self, responder: Any = lambda kwargs: "Yes") -> None:
        self.responder = responder
        self.calls = 0
        self.active = 0
        self.max_active = 0

    async def create(self, **kwargs: Any) -> Any:
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)
        self.active -= 1
        content = self.responder(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _factory(completions: _FakeCompletions) -> Any:
    return lambda settings: SimpleNamespace(chat=SimpleNamespace(completions=completions))


class TestDiskCompletionCache:
    """Cache em disco por chave SHA-256."""

    def test_put_then_get(self, tmp_path: Path) -> None:
        """O que é gravado é lido de volta."""
        cache = DiskCompletionCache(tmp_path)
        cache.put("ab" + "0" * 62, "olá")
        assert cache.get("ab" + "0" * 62) == "olá"

    def test_missing_key_returns_none(self, tmp_path: Path) -> None:
        """Chave ausente devolve ``None``."""
        assert DiskCompletionCache(tmp_path).get("ff" + "0" * 62) is None

    def test_corrupted_entry_is_ignored(self, tmp_path: Path) -> None:
        """Arquivo corrompido não derruba a execução."""
        cache = DiskCompletionCache(tmp_path)
        key = "cd" + "0" * 62
        cache.put(key, "ok")
        (tmp_path / "cd" / f"{key}.json").write_text("{quebrado", encoding="utf-8")
        assert cache.get(key) is None

    def test_key_depends_on_full_content(self) -> None:
        """A chave muda com qualquer campo, inclusive o texto COMPLETO do prompt."""
        base = {
            "model": "m",
            "prompt": "a" * 300,
            "system_prompt": None,
            "temperature": 0.0,
            "max_tokens": None,
            "namespace": "",
        }
        other_tail = {**base, "prompt": "a" * 299 + "b"}
        other_ns = {**base, "namespace": "v2"}
        keys = {
            DiskCompletionCache.build_key(**variant) for variant in (base, other_tail, other_ns)
        }
        assert len(keys) == 3


class TestAsyncLLMClient:
    """Comportamento do cliente: cache, dry-run, semáforo e retentativas."""

    def test_completion_is_cached(self, diagnostics_settings: Any, tmp_path: Path) -> None:
        """Duas chamadas idênticas geram uma única requisição."""
        fake = _FakeCompletions()
        client = AsyncLLMClient(
            diagnostics_settings.llm,
            cache=DiskCompletionCache(tmp_path),
            client_factory=_factory(fake),
        )

        async def _run() -> list[str]:
            return [await client.complete("p", model="m"), await client.complete("p", model="m")]

        assert asyncio.run(_run()) == ["Yes", "Yes"]
        assert fake.calls == 1
        assert client.stats.n_cache_hits == 1
        assert client.stats.n_requests == 1

    def test_dry_run_makes_no_request(self, diagnostics_settings: Any) -> None:
        """Em dry-run a fábrica nunca é chamada; só as estatísticas são contadas."""

        def _boom(settings: Any) -> Any:
            raise AssertionError("não deve criar cliente em dry-run")

        client = AsyncLLMClient(diagnostics_settings.llm, dry_run=True, client_factory=_boom)
        assert asyncio.run(client.complete("prompt qualquer", model="m", max_tokens=50)) == ""
        assert client.stats.n_requests == 1
        assert client.stats.output_tokens == 50
        assert client.stats.input_tokens >= 1

    def test_concurrency_is_limited_by_semaphore(self, diagnostics_settings: Any) -> None:
        """Nunca há mais requisições simultâneas que ``max_concurrency``."""
        settings = diagnostics_settings.llm.model_copy(update={"max_concurrency": 2})
        fake = _FakeCompletions()
        client = AsyncLLMClient(settings, client_factory=_factory(fake))
        results = asyncio.run(client.complete_many([f"p{i}" for i in range(10)], model="m"))
        assert results == ["Yes"] * 10
        assert fake.max_active <= 2

    def test_transient_error_is_retried(
        self, diagnostics_settings: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Erro transitório é retentado com espera (mockada) e depois tem sucesso."""

        async def _no_sleep(_: float) -> None:
            return None

        monkeypatch.setattr(llm_client, "_transient_exceptions", lambda: (RuntimeError,))
        monkeypatch.setattr(llm_client.asyncio, "sleep", _no_sleep)
        attempts = {"n": 0}

        def _flaky(kwargs: Any) -> str:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("rate limit")
            return "No"

        client = AsyncLLMClient(
            diagnostics_settings.llm, client_factory=_factory(_FakeCompletions(_flaky))
        )
        assert asyncio.run(client.complete("p", model="m")) == "No"
        assert attempts["n"] == 2

    def test_exhausted_retries_raise_and_complete_many_returns_none(
        self, diagnostics_settings: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Depois das retentativas, ``complete`` levanta e ``complete_many`` devolve ``None``."""

        async def _no_sleep(_: float) -> None:
            return None

        def _always_fail(kwargs: Any) -> str:
            raise RuntimeError("fora do ar")

        monkeypatch.setattr(llm_client, "_transient_exceptions", lambda: (RuntimeError,))
        monkeypatch.setattr(llm_client.asyncio, "sleep", _no_sleep)
        settings = diagnostics_settings.llm.model_copy(update={"max_retries": 1})
        client = AsyncLLMClient(settings, client_factory=_factory(_FakeCompletions(_always_fail)))
        with pytest.raises(PipelineStageError):
            asyncio.run(client.complete("p", model="m"))
        assert asyncio.run(client.complete_many(["p"], model="m")) == [None]
        assert client.stats.n_failures == 2


class TestResolveEndpoint:
    """Resolução de endpoint e chave (só do ambiente)."""

    def test_ollama_uses_v1_suffix(self, diagnostics_settings: Any) -> None:
        """Ollama usa ``<base>/v1`` e uma chave fictícia."""
        settings = diagnostics_settings.llm.model_copy(
            update={"provider": "ollama", "ollama_base_url": "http://x:11434/"}
        )
        assert _resolve_endpoint(settings) == ("http://x:11434/v1", "ollama")

    def test_openai_without_key_or_base_url_raises(
        self, diagnostics_settings: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OpenAI sem chave e sem endpoint customizado exige ``OPENAI_KEY``."""
        for name in ("OPENAI_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
            monkeypatch.delenv(name, raising=False)
        settings = diagnostics_settings.llm.model_copy(
            update={"provider": "openai", "openai_base_url": None}
        )
        with pytest.raises(MissingEnvironmentVariableError):
            _resolve_endpoint(settings)

    def test_openai_reads_key_from_environment(
        self, diagnostics_settings: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A chave vem da variável de ambiente."""
        monkeypatch.setenv("OPENAI_KEY", "segredo-de-teste")
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        settings = diagnostics_settings.llm.model_copy(
            update={"provider": "openai", "openai_base_url": None}
        )
        assert _resolve_endpoint(settings) == (None, "segredo-de-teste")

    def test_openai_compatible_endpoint_without_key_uses_placeholder(
        self, diagnostics_settings: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Endpoint compatível privado sem chave usa um valor fictício."""
        for name in ("OPENAI_KEY", "OPENAI_API_KEY"):
            monkeypatch.delenv(name, raising=False)
        settings = diagnostics_settings.llm.model_copy(
            update={"provider": "openai", "openai_base_url": "http://privado/v1"}
        )
        assert _resolve_endpoint(settings) == ("http://privado/v1", "local-no-auth")


def test_estimate_tokens_has_floor_of_one() -> None:
    """A estimativa nunca é menor que 1 token."""
    assert estimate_tokens("") == 1
    assert estimate_tokens("a" * 350) == 100


# ============================================================================
# prompt_synthesis
# ============================================================================

V1 = 'Classifique o tweet.\nResponda em JSON: {"label": "..."}\nTweet: {{TEXTO}}\nJSON:'


class TestParseRules:
    """Extração de regras da resposta do LLM."""

    def test_accepts_bullets_numbers_and_arrows(self) -> None:
        """Aceita ``->`` e ``→``, marcadores e numeração; ignora linhas sem seta."""
        completion = (
            "1. ironia elogiosa → negativo\n- palavrão como intensificador -> positivo.\n"
            "* texto vago\nregra sem classe ->\n"
        )
        assert parse_rules(completion, max_rules=10) == [
            "ironia elogiosa -> negativo",
            "palavrão como intensificador -> positivo",
        ]

    def test_dedupes_and_caps(self) -> None:
        """Remove repetidas e respeita o máximo."""
        completion = "a -> negativo\na -> negativo\nb -> neutro\nc -> positivo"
        assert parse_rules(completion, max_rules=2) == ["a -> negativo", "b -> neutro"]

    def test_garbage_gives_no_rules(self) -> None:
        """Modelo fora do formato: lista vazia (sem exceção)."""
        assert parse_rules("não sei o que fazer", max_rules=5) == []


class TestInsertRules:
    """Inserção no v1 sem alterar o resto."""

    def test_rules_go_before_the_placeholder_line(self) -> None:
        """O bloco entra antes de ``{{TEXTO}}`` e o restante do v1 fica idêntico."""
        v2 = insert_rules_into_prompt(V1, ["ironia -> negativo"])
        assert RULES_HEADER in v2
        assert v2.index("ironia -> negativo") < v2.index("{{TEXTO}}")
        assert v2.replace(RULES_HEADER + "\n- ironia -> negativo\n\n", "") == V1

    def test_no_rules_raises(self) -> None:
        """Sem regras não há v2."""
        with pytest.raises(DataValidationError, match="nenhuma regra"):
            insert_rules_into_prompt(V1, [])

    def test_missing_placeholder_raises(self) -> None:
        """V1 sem ``{{TEXTO}}`` não pode receber o bloco."""
        with pytest.raises(DataValidationError, match="marcador"):
            insert_rules_into_prompt("sem marcador", ["a -> neutro"])


class TestWritePromptV2:
    """Escrita protegida do v2 e metadados."""

    def _v1(self, tmp_path: Path) -> Path:
        path = tmp_path / "v1.txt"
        path.write_text(V1, encoding="utf-8")
        return path

    def test_writes_v2_meta_and_keeps_v1_intact(self, tmp_path: Path) -> None:
        """O v1 permanece byte a byte igual; a proveniência vai para o ``.meta.json``."""
        v1 = self._v1(tmp_path)
        before = v1.read_bytes()
        v2 = write_prompt_v2(v1, tmp_path / "v2.md", ["a -> negativo"], hypotheses=["h"], model="m")
        assert v1.read_bytes() == before
        assert "a -> negativo" in v2.read_text(encoding="utf-8")
        meta = json.loads((tmp_path / "v2.meta.json").read_text(encoding="utf-8"))
        assert meta["review_required"] is True
        assert meta["v1_sha256"] != meta["v2_sha256"]
        assert meta["hypotheses"] == ["h"]

    def test_never_overwrites_v1(self, tmp_path: Path) -> None:
        """Apontar o v2 para o próprio v1 é proibido."""
        v1 = self._v1(tmp_path)
        with pytest.raises(DataValidationError, match="não pode sobrescrever"):
            write_prompt_v2(v1, v1, ["a -> neutro"], hypotheses=[], model="m", overwrite=True)

    def test_existing_v2_needs_overwrite(self, tmp_path: Path) -> None:
        """Um v2 existente só é substituído com ``overwrite=True``."""
        v1 = self._v1(tmp_path)
        target = tmp_path / "v2.md"
        write_prompt_v2(v1, target, ["a -> neutro"], hypotheses=[], model="m")
        with pytest.raises(DataValidationError, match="já existe"):
            write_prompt_v2(v1, target, ["b -> neutro"], hypotheses=[], model="m")
        write_prompt_v2(v1, target, ["b -> neutro"], hypotheses=[], model="m", overwrite=True)
        assert "b -> neutro" in target.read_text(encoding="utf-8")


class _FakeSynthesisClient:
    """Cliente falso que devolve regras em formato livre."""

    def __init__(self) -> None:
        self.requests: list[str] = []

    async def complete(self, prompt: str, **_: Any) -> str:
        self.requests.append(prompt)
        return "1. ironia → negativo\n2. gíria positiva -> positivo"


class TestSynthesizeRules:
    """Uma única chamada ao LLM."""

    def test_request_contains_hypotheses_and_rules_are_parsed(
        self, diagnostics_settings: Any
    ) -> None:
        """O pedido lista as hipóteses; a resposta vira regras normalizadas."""
        client = _FakeSynthesisClient()
        rules = synthesize_rules(
            diagnostics_settings,
            ["h-ironia", "h-gíria"],
            max_rules=5,
            client=client,  # type: ignore[arg-type]
        )
        assert rules == ["ironia -> negativo", "gíria positiva -> positivo"]
        assert len(client.requests) == 1
        assert "h-ironia" in client.requests[0]
        assert "h-gíria" in client.requests[0]

    def test_request_text_states_the_limit(self) -> None:
        """O pedido informa o máximo de regras e proíbe inventar padrões."""
        request = build_rules_request(["x"], max_rules=3)
        assert "no máximo 3" in request
        assert "não invente" in request


# ============================================================================
# sae_runner
# ============================================================================


class TestCacheName:
    """O cache inclui o modelo de embedding."""

    def test_includes_model_slug_and_size(self) -> None:
        """Modelos diferentes geram nomes diferentes."""
        first = build_embedding_cache_name("neuralmind/bert-base-portuguese-cased", 100)
        second = build_embedding_cache_name("outro/modelo", 100)
        assert first != second
        assert first == "diagnostics_neuralmind-bert-base-portuguese-cased_100texts"


class TestAssignPartitions:
    """Partições disjuntas e deduplicação."""

    def test_partitions_are_disjoint_and_cover_all(
        self, diagnostic_corpus_large: pl.DataFrame
    ) -> None:
        """Cada tweet está em exatamente uma partição."""
        result = assign_partitions(
            diagnostic_corpus_large, holdout_size=0.2, validation_size=0.1, random_seed=42
        )
        assert result["id"].n_unique() == result.height == diagnostic_corpus_large.height
        assert set(result["partition"]) == {
            DISCOVERY_PARTITION,
            SAE_VALIDATION_PARTITION,
            HOLDOUT_PARTITION,
        }
        assert "_stratum" not in result.columns

    def test_duplicated_texts_are_removed(self, diagnostic_corpus_large: pl.DataFrame) -> None:
        """Textos repetidos (ex.: só menções) contam uma vez."""
        duplicated = pl.concat([diagnostic_corpus_large, diagnostic_corpus_large.head(10)])
        result = assign_partitions(duplicated, holdout_size=0.2, validation_size=0.1, random_seed=1)
        assert result.height == diagnostic_corpus_large.height

    def test_is_deterministic(self, diagnostic_corpus_large: pl.DataFrame) -> None:
        """Mesma semente, mesmas partições (necessário para reproduzir o holdout depois)."""
        args = {"holdout_size": 0.2, "validation_size": 0.1, "random_seed": 9}
        first = assign_partitions(diagnostic_corpus_large, **args)
        second = assign_partitions(diagnostic_corpus_large, **args)
        assert first.equals(second)

    def test_empty_corpus_raises(self, diagnostic_corpus_large: pl.DataFrame) -> None:
        """Corpus vazio é erro explícito."""
        with pytest.raises(EmptyDatasetError):
            assign_partitions(
                diagnostic_corpus_large.head(0),
                holdout_size=0.2,
                validation_size=0.1,
                random_seed=0,
            )


class TestPrepareDiscoveryData:
    """Embeddings alinhados e SAE treinado uma vez (com dublês)."""

    def test_embeddings_are_aligned_and_sae_gets_train_only(
        self,
        diagnostic_corpus_large: pl.DataFrame,
        diagnostics_settings: Any,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A matriz segue a ordem das linhas; o SAE só treina na partição de descoberta."""
        calls: dict[str, Any] = {}

        def _fake_embeddings(texts: list[str], **kwargs: Any) -> dict[str, np.ndarray]:
            calls["embedding_kwargs"] = kwargs
            return {text: np.full(4, float(index)) for index, text in enumerate(texts)}

        def _fake_train_sae(**kwargs: Any) -> str:
            calls["sae_kwargs"] = kwargs
            return "sae-falso"

        monkeypatch.setattr(sae_runner, "extract_local_embeddings", _fake_embeddings)
        monkeypatch.setattr(sae_runner, "train_sae", _fake_train_sae)
        paths = SimpleNamespace(models_checkpoints_dir=tmp_path)

        data = prepare_discovery_data(diagnostic_corpus_large, diagnostics_settings, paths)  # type: ignore[arg-type]

        n_train = int((data.partitioned["partition"] == DISCOVERY_PARTITION).sum())
        assert data.sae == "sae-falso"
        assert data.embeddings.shape == (data.partitioned.height, 4)
        assert calls["sae_kwargs"]["embeddings"].shape[0] == n_train
        assert calls["sae_kwargs"]["m_total_neurons"] == diagnostics_settings.sae.m_total_neurons
        assert calls["embedding_kwargs"]["model"] == diagnostics_settings.embedding.model_name
        assert data.cache_name in str(calls["sae_kwargs"]["checkpoint_dir"])
        assert len(data.corpus_hash) == 64
        # o embedding de cada linha corresponde ao texto da mesma linha
        first_text = data.partitioned["text_normalized"][0]
        assert data.embeddings[0][0] == float(
            data.partitioned["text_normalized"].to_list().index(first_text)
        )


# ============================================================================
# sampling
# ============================================================================


@pytest.fixture
def tweets() -> pl.DataFrame:
    """Vinte tweets sintéticos."""
    return pl.DataFrame(
        {"id": [f"id{i}" for i in range(20)], "text_normalized": [f"texto {i}" for i in range(20)]}
    )


class TestPseudonymize:
    """Pseudônimo do identificador."""

    def test_is_stable_and_salted(self) -> None:
        """Mesmo id + mesmo sal = mesmo pseudônimo; outro sal muda."""
        assert pseudonymize_id("1", "a") == pseudonymize_id("1", "a")
        assert pseudonymize_id("1", "a") != pseudonymize_id("1", "b")
        assert len(pseudonymize_id("1", "a")) == 16

    def test_salt_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """O sal vem do ambiente; ausente ou vazio é erro."""
        monkeypatch.setenv("SAL_TESTE", "abc")
        assert resolve_sample_salt("SAL_TESTE") == "abc"
        monkeypatch.setenv("SAL_TESTE", "")
        with pytest.raises(MissingEnvironmentVariableError):
            resolve_sample_salt("SAL_TESTE")
        monkeypatch.delenv("SAL_TESTE")
        with pytest.raises(MissingEnvironmentVariableError):
            resolve_sample_salt("SAL_TESTE")


class TestSampleByConcept:
    """Amostragem estratificada por conceito."""

    def test_columns_have_no_identifier(self, tweets: pl.DataFrame) -> None:
        """O CSV para rotulagem não traz ``id`` original."""
        sample, key = sample_tweets_by_concept(
            tweets, {"c": np.ones(20, dtype=int)}, per_concept=5, salt="s", random_seed=0
        )
        assert tuple(sample.columns) == SAMPLE_COLUMNS
        assert "id" not in sample.columns
        assert set(key.columns) == {"sample_id", "id", "concept"}
        assert sample.height == 5

    def test_tweets_are_not_repeated_across_concepts(self, tweets: pl.DataFrame) -> None:
        """Um tweet que atende a dois conceitos entra uma única vez."""
        annotations = {"a": np.ones(20, dtype=int), "b": np.ones(20, dtype=int)}
        sample, _ = sample_tweets_by_concept(
            tweets, annotations, per_concept=8, salt="s", random_seed=1
        )
        assert sample["sample_id"].n_unique() == sample.height == 16

    def test_only_positive_annotations_are_sampled(self, tweets: pl.DataFrame) -> None:
        """Só entram tweets em que o conceito foi anotado como 1."""
        mask = np.array([1] * 3 + [0] * 17)
        sample, _ = sample_tweets_by_concept(
            tweets, {"c": mask}, per_concept=10, salt="s", random_seed=0
        )
        assert sample.height == 3
        assert set(sample["text"]) == {"texto 0", "texto 1", "texto 2"}

    def test_rare_concepts_are_sampled_first(self, tweets: pl.DataFrame) -> None:
        """O conceito raro não fica sem tweets por causa do frequente."""
        rare = np.zeros(20, dtype=int)
        rare[:2] = 1
        common = np.ones(20, dtype=int)
        sample, _ = sample_tweets_by_concept(
            tweets, {"comum": common, "raro": rare}, per_concept=10, salt="s", random_seed=0
        )
        assert (sample["concept"] == "raro").sum() == 2

    def test_is_deterministic(self, tweets: pl.DataFrame) -> None:
        """Mesma semente, mesma amostra."""
        args = {"per_concept": 5, "salt": "s", "random_seed": 3}
        first, _ = sample_tweets_by_concept(tweets, {"c": np.ones(20, dtype=int)}, **args)
        second, _ = sample_tweets_by_concept(tweets, {"c": np.ones(20, dtype=int)}, **args)
        assert first.equals(second)

    def test_human_label_column_starts_empty(self, tweets: pl.DataFrame) -> None:
        """A coluna a preencher vem vazia."""
        sample, _ = sample_tweets_by_concept(
            tweets, {"c": np.ones(20, dtype=int)}, per_concept=3, salt="s", random_seed=0
        )
        assert sample[HUMAN_LABEL_COLUMN].null_count() == sample.height

    def test_write_files(self, tweets: pl.DataFrame, tmp_path: Path) -> None:
        """Grava o CSV (sem ids) e a chave em arquivos separados."""
        sample, key = sample_tweets_by_concept(
            tweets, {"c": np.ones(20, dtype=int)}, per_concept=3, salt="s", random_seed=0
        )
        csv_path, key_path = tmp_path / "para_rotular.csv", tmp_path / "chave.parquet"
        write_labeling_sample(sample, key, sample_csv=csv_path, key_parquet=key_path)
        header = csv_path.read_text(encoding="utf-8").splitlines()[0]
        assert "id" not in header.split(",")
        assert key_path.is_file()


# ============================================================================
# sanity
# ============================================================================


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


# ============================================================================
# settings
# ============================================================================


def _write_variant(tmp_path: Path, mutate: Any) -> Path:
    """Copia o YAML real para ``tmp_path`` aplicando ``mutate`` ao dicionário."""
    content = yaml.safe_load(DEFAULT_DIAGNOSTICS_CONFIG_FILE.read_text(encoding="utf-8"))
    mutate(content)
    target = tmp_path / "diagnostics.yaml"
    target.write_text(yaml.safe_dump(content, allow_unicode=True), encoding="utf-8")
    return target


class TestLoadDiagnosticsSettings:
    """Carga e validação do YAML."""

    def test_default_config_is_valid(self) -> None:
        """O ``configs/diagnostics.yaml`` versionado passa na validação."""
        settings = load_diagnostics_settings()
        assert settings.llm.provider in {"ollama", "openai"}
        assert settings.comparison.n_folds >= 6
        assert settings.hypotheses.selection_method == "lasso"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Arquivo inexistente levanta erro específico."""
        with pytest.raises(ConfigurationFileNotFoundError):
            load_diagnostics_settings(tmp_path / "nao_existe.yaml")

    def test_unknown_key_is_rejected(self, tmp_path: Path) -> None:
        """``extra="forbid"``: chave desconhecida falha na inicialização."""
        path = _write_variant(tmp_path, lambda cfg: cfg.update({"chave_estranha": 1}))
        with pytest.raises(InvalidConfigurationError):
            load_diagnostics_settings(path)

    def test_few_folds_are_rejected(self, tmp_path: Path) -> None:
        """Menos de 6 folds impede p < 0,05 no Wilcoxon: rejeitado."""
        path = _write_variant(tmp_path, lambda cfg: cfg["comparison"].update({"n_folds": 5}))
        with pytest.raises(InvalidConfigurationError):
            load_diagnostics_settings(path)

    def test_invalid_provider_is_rejected(self, tmp_path: Path) -> None:
        """Provedor fora de ``ollama|openai`` é inválido."""
        path = _write_variant(tmp_path, lambda cfg: cfg["llm"].update({"provider": "outro"}))
        with pytest.raises(InvalidConfigurationError):
            load_diagnostics_settings(path)

    def test_no_secret_keys_in_yaml(self) -> None:
        """Segredos não podem estar no YAML (chaves vêm do ambiente)."""
        text = DEFAULT_DIAGNOSTICS_CONFIG_FILE.read_text(encoding="utf-8").lower()
        assert "api_key" not in text
        assert "sk-" not in text


# ============================================================================
# stage
# ============================================================================


def _namespace(**kwargs: Any) -> Any:
    """Cria um objeto simples que faz as vezes de ``ProjectPaths``/``Settings`` nos testes."""
    return SimpleNamespace(**kwargs)


class TestRegistration:
    """O estágio existe, mas é opt-in."""

    def test_stage_is_registered_but_not_in_default_workflow(self) -> None:
        """``--stage all`` (lista de ``configs/config.yaml``) não executa o diagnóstico."""
        assert "diagnostics" in STAGE_REGISTRY
        assert "diagnostics" not in load_general_config().stages

    def test_cli_accepts_the_stage_and_new_flags(self) -> None:
        """Argumentos do estágio são interpretados com padrões seguros."""
        args = main.parse_arguments(["--stage", "diagnostics"])
        assert args.diagnostics_step == "hypotheses"
        assert args.diagnostics_target == "disagreement"
        assert args.dry_run is False
        assert args.diagnostics_corpus is None

    def test_cli_parses_all_diagnostics_flags(self) -> None:
        """Todas as flags novas chegam ao construtor de kwargs."""
        args = main.parse_arguments(
            [
                "--stage", "diagnostics", "--dry-run",
                "--diagnostics-step", "validation",
                "--diagnostics-target", "pseudo_label",
                "--diagnostics-model-column", "lab_a",
                "--diagnostics-label", "neutro",
                "--diagnostics-corpus", "x.parquet",
                "--diagnostics-gold", "repro",
                "--random-seed", "7",
            ]
        )  # fmt: skip
        kwargs = main._build_diagnostics_stage_kwargs(
            _namespace(),
            _namespace(),
            _namespace(),
            args,  # type: ignore[arg-type]
        )
        assert kwargs["step"] == "validation"
        assert kwargs["target_name"] == "pseudo_label"
        assert kwargs["model_column"] == "lab_a"
        assert kwargs["label"] == "neutro"
        assert kwargs["corpus_path"] == Path("x.parquet")
        assert kwargs["gold"] == "repro"
        assert kwargs["dry_run"] is True
        assert kwargs["random_seed"] == 7

    def test_invalid_target_is_rejected_by_argparse(self) -> None:
        """Alvo fora dos quatro permitidos é recusado na linha de comando."""
        with pytest.raises(SystemExit):
            main.parse_arguments(["--stage", "diagnostics", "--diagnostics-target", "outro"])

    def test_builder_is_registered_for_every_stage(self) -> None:
        """Todo estágio do registro tem construtor de kwargs em ``main.py``."""
        assert set(STAGE_REGISTRY) == set(main._STAGE_KWARGS_BUILDERS)


class TestRunDiagnosticsStage:
    """Despacho por passo, sem rede nem torch."""

    def test_unknown_step_raises(self) -> None:
        """Passo desconhecido lista os disponíveis."""
        with pytest.raises(DataValidationError, match="passo desconhecido"):
            run_diagnostics_stage(_namespace(), step="outro")  # type: ignore[arg-type]

    def test_gold_error_requires_corpus(self) -> None:
        """``gold_error`` exige as predições sobre o gold."""
        with pytest.raises(DataValidationError, match="--diagnostics-corpus"):
            run_diagnostics_stage(_namespace(), target_name="gold_error", dry_run=True)

    def test_hypotheses_dry_run_prints_report(
        self,
        diagnostic_corpus_large: pl.DataFrame,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Dry-run devolve e imprime a estimativa, sem preparar embeddings nem SAE."""
        monkeypatch.setattr(
            hypotheses_module, "load_diagnostic_corpus", lambda *_: diagnostic_corpus_large
        )

        def _must_not_run(*_: Any, **__: Any) -> None:
            raise AssertionError("dry-run não pode preparar embeddings/SAE")

        monkeypatch.setattr(sae_runner_module, "prepare_discovery_data", _must_not_run)
        report = run_diagnostics_stage(_namespace(), target_name="disagreement", dry_run=True)
        assert "DRY-RUN do alvo 'disagreement'" in report
        assert "DRY-RUN" in capsys.readouterr().out

    def test_validation_dry_run_uses_validation_report(
        self, diagnostic_corpus_large: pl.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """O passo ``validation`` estima N tweets do holdout x top-k conceitos."""
        monkeypatch.setattr(
            hypotheses_module, "load_diagnostic_corpus", lambda *_: diagnostic_corpus_large
        )
        report = run_diagnostics_stage(
            _namespace(), step="validation", target_name="uncertainty", dry_run=True
        )
        assert "DRY-RUN da validação de 'uncertainty'" in report

    def test_comparison_dry_run_delegates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """O passo ``comparison`` usa o gold correto e os modelos informados."""
        seen: dict[str, Any] = {}

        def _fake(gold_file: Path, settings: Any, models: list[str]) -> str:
            seen.update(gold_file=gold_file, models=models)
            return "RELATORIO"

        monkeypatch.setattr(comparison_module, "build_comparison_dry_run", _fake)
        paths = _namespace(tweetsentbr_file=Path("a.parquet"), repro_file=Path("b.parquet"))
        assert (
            run_diagnostics_stage(
                paths,
                step="comparison",
                gold="repro",
                models=["m1"],
                dry_run=True,
            )
            == "RELATORIO"
        )
        assert seen == {"gold_file": Path("b.parquet"), "models": ["m1"]}

    def test_hypotheses_step_wires_yaml_defaults_and_tracking(
        self, diagnostic_corpus_large: pl.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sem dry-run: prepara o SAE uma vez, usa modelo/classe do YAML e liga o MLflow."""
        calls: dict[str, Any] = {}
        monkeypatch.setattr(
            hypotheses_module, "load_diagnostic_corpus", lambda *_: diagnostic_corpus_large
        )
        monkeypatch.setattr(
            sae_runner_module, "prepare_discovery_data", lambda *a: calls.setdefault("prepared", 1)
        )

        def _fake_run(*args: Any, **kwargs: Any) -> str:
            calls["args"], calls["kwargs"] = args, kwargs
            return "resultado"

        monkeypatch.setattr(hypotheses_module, "run_target_diagnostics", _fake_run)
        result = run_diagnostics_stage(_namespace(), target_name="pseudo_label", random_seed=3)
        assert result == "resultado"
        assert calls["prepared"] == 1
        assert calls["args"][0] == "pseudo_label"
        assert calls["kwargs"]["model_column"] == "lab_huggingface"
        assert calls["kwargs"]["label"] == "negativo"
        assert calls["kwargs"]["track"] is True
        assert calls["args"][3].random_seed == 3  # settings com a semente sobrescrita

    def test_hypotheses_step_forwards_track_false(
        self, diagnostic_corpus_large: pl.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``track=False`` desliga o MLflow no passo ``hypotheses``."""
        seen: dict[str, Any] = {}
        monkeypatch.setattr(
            hypotheses_module, "load_diagnostic_corpus", lambda *_: diagnostic_corpus_large
        )
        monkeypatch.setattr(sae_runner_module, "prepare_discovery_data", lambda *a: "descoberta")
        monkeypatch.setattr(
            hypotheses_module,
            "run_target_diagnostics",
            lambda *args, **kwargs: seen.update(args=args, kwargs=kwargs),
        )
        run_diagnostics_stage(_namespace(), target_name="uncertainty", track=False)
        assert seen["kwargs"]["track"] is False
        assert seen["args"][2] == "descoberta"

    def test_validation_step_wires_yaml_defaults_and_tracking(
        self, diagnostic_corpus_large: pl.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sem dry-run, ``validation`` usa modelo/classe do YAML e repassa ``track``."""
        seen: dict[str, Any] = {}
        paths = _namespace()
        monkeypatch.setattr(
            hypotheses_module, "load_diagnostic_corpus", lambda *_: diagnostic_corpus_large
        )

        def _fake_validation(*args: Any, **kwargs: Any) -> str:
            seen.update(args=args, kwargs=kwargs)
            return "validado"

        monkeypatch.setattr(validation, "run_validation_stage", _fake_validation)
        result = run_diagnostics_stage(
            paths, step="validation", target_name="pseudo_label", track=False
        )
        assert result == "validado"
        assert seen["args"][0] == "pseudo_label"
        assert seen["args"][1] is diagnostic_corpus_large
        assert seen["args"][3] is paths
        assert seen["kwargs"] == {
            "track": False,
            "model_column": "lab_huggingface",
            "label": "negativo",
        }

    def test_comparison_step_runs_with_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Sem dry-run: gold ``tweetsentbr`` e o anotador do YAML como modelo padrão."""
        seen: dict[str, Any] = {}

        def _fake_comparison(gold_file: Path, settings: Any, paths: Any, **kwargs: Any) -> str:
            seen.update(gold_file=gold_file, paths=paths, **kwargs)
            return "comparado"

        monkeypatch.setattr(comparison_module, "run_prompt_comparison", _fake_comparison)
        paths = _namespace(tweetsentbr_file=Path("a.parquet"), repro_file=Path("b.parquet"))
        assert run_diagnostics_stage(paths, step="comparison", track=False) == "comparado"
        annotator = load_diagnostics_settings().llm.annotator_model
        assert seen == {
            "gold_file": Path("a.parquet"),
            "paths": paths,
            "models": [annotator],
            "track": False,
        }

    def test_comparison_dry_run_defaults_to_annotator_and_tweetsentbr(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No dry-run, sem ``gold``/``models``, vale o padrão do YAML e do gold TweetSentBR."""
        seen: dict[str, Any] = {}

        def _fake(gold_file: Path, settings: Any, models: list[str]) -> str:
            seen.update(gold_file=gold_file, models=models)
            return "RELATORIO"

        monkeypatch.setattr(comparison_module, "build_comparison_dry_run", _fake)
        paths = _namespace(tweetsentbr_file=Path("a.parquet"), repro_file=Path("b.parquet"))
        run_diagnostics_stage(paths, step="comparison", dry_run=True)
        assert seen["gold_file"] == Path("a.parquet")
        assert seen["models"] == [load_diagnostics_settings().llm.annotator_model]

    def test_gold_error_with_corpus_path_reaches_loader(
        self, diagnostic_corpus_large: pl.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Com ``corpus_path``, ``gold_error`` passa a validação e o caminho chega ao carregador."""
        seen: dict[str, Any] = {}

        def _fake_loader(settings: Any, paths: Any, corpus_path: Path | None) -> pl.DataFrame:
            seen["corpus_path"] = corpus_path
            return diagnostic_corpus_large

        def _fake_report(target_name: str, corpus: Any, settings: Any, **arguments: Any) -> str:
            seen.update(target_name=target_name, arguments=arguments)
            return "RELATORIO"

        monkeypatch.setattr(hypotheses_module, "load_diagnostic_corpus", _fake_loader)
        monkeypatch.setattr(hypotheses_module, "build_dry_run_report", _fake_report)
        report = run_diagnostics_stage(
            _namespace(),
            target_name="gold_error",
            corpus_path=Path("gold_predicoes.parquet"),
            model_column="lab_a",
            dry_run=True,
        )
        assert report == "RELATORIO"
        assert seen["corpus_path"] == Path("gold_predicoes.parquet")
        assert seen["target_name"] == "gold_error"
        assert seen["arguments"] == {"model_column": "lab_a", "label": None}

    def test_config_file_is_loaded_and_seed_override_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``config_file`` alternativo é lido; ``random_seed`` explícito prevalece sobre o YAML."""
        raw = yaml.safe_load(DEFAULT_DIAGNOSTICS_CONFIG_FILE.read_text(encoding="utf-8"))
        raw["random_seed"] = 99
        config_file = tmp_path / "diagnostics.yaml"
        config_file.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
        seeds: list[int] = []

        def _fake(gold_file: Path, settings: Any, models: list[str]) -> str:
            seeds.append(settings.random_seed)
            return "RELATORIO"

        monkeypatch.setattr(comparison_module, "build_comparison_dry_run", _fake)
        paths = _namespace(tweetsentbr_file=Path("a.parquet"), repro_file=Path("b.parquet"))
        run_diagnostics_stage(paths, step="comparison", dry_run=True, config_file=config_file)
        run_diagnostics_stage(
            paths, step="comparison", dry_run=True, config_file=config_file, random_seed=0
        )
        assert seeds == [99, 0]

    def test_missing_config_file_raises(self, tmp_path: Path) -> None:
        """``config_file`` inexistente falha cedo com erro de configuração."""
        with pytest.raises(ConfigurationFileNotFoundError):
            run_diagnostics_stage(_namespace(), config_file=tmp_path / "nao_existe.yaml")


class TestDiagnosticsStageHelpers:
    """Helpers privados do estágio, isolados do despacho."""

    def test_load_settings_applies_seed_only_when_given(self) -> None:
        """``random_seed=0`` é um override válido (não é confundido com ``None``)."""
        default_seed = load_diagnostics_settings().random_seed
        assert _load_settings(None, None).random_seed == default_seed
        assert _load_settings(None, 0).random_seed == 0

    def test_resolve_target_arguments_prefers_explicit_values(
        self, diagnostics_settings: Any
    ) -> None:
        """Argumento explícito vence o padrão do YAML; ausente, usa o YAML."""
        assert _resolve_target_arguments(diagnostics_settings, "pseudo_label", None, None) == {
            "model_column": "lab_huggingface",
            "label": "negativo",
        }
        assert _resolve_target_arguments(
            diagnostics_settings, "pseudo_label", "lab_x", "positivo"
        ) == {"model_column": "lab_x", "label": "positivo"}

    def test_resolve_target_arguments_without_yaml_entry(self, diagnostics_settings: Any) -> None:
        """Alvo sem entrada em ``targets`` (ex.: ``disagreement``) resulta em ``None``."""
        assert _resolve_target_arguments(diagnostics_settings, "disagreement", None, None) == {
            "model_column": None,
            "label": None,
        }


# ============================================================================
# targets
# ============================================================================


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
                "sentiment_label_openai": [None, "neutro"],
            }
        )
        adapted = adapt_labeled_corpus(raw)
        assert adapted.columns == [
            "id",
            "text_normalized",
            "agreement_score",
            "lab_huggingface",
            "lab_openai",
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
                "sentiment_label_openai": ["positivo"],
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


# ============================================================================
# tracking
# ============================================================================


class TestFlattenParams:
    """Parâmetros aninhados viram ``chave.subchave -> str``."""

    def test_nested_and_none(self) -> None:
        """Achata, ignora ``None`` e converte para texto."""
        flat = flatten_params({"a": {"b": 1, "c": None}, "d": 0.5, "e": None})
        assert flat == {"a.b": "1", "d": "0.5"}

    def test_long_values_are_truncated(self) -> None:
        """Valores longos respeitam o limite do MLflow."""
        assert len(flatten_params({"x": "y" * 1000})["x"]) <= 250

    def test_prefix(self) -> None:
        """O prefixo é aplicado a todas as chaves."""
        assert flatten_params({"a": 1}, prefix="p.") == {"p.a": "1"}


class TestKeepFiniteMetrics:
    """``nan``/``inf`` não vão para o MLflow."""

    def test_removes_non_finite(self) -> None:
        """Só valores finitos permanecem, como ``float``."""
        metrics = keep_finite_metrics({"a": 1, "b": math.nan, "c": math.inf, "d": -math.inf})
        assert metrics == {"a": 1.0}


# ============================================================================
# validation
# ============================================================================


class TestSelectTopHypotheses:
    """Escolha dos top-k conceitos da descoberta."""

    def test_sorts_by_selection_score_and_dedupes(self) -> None:
        """Ordena por força de seleção, remove repetidas e nulas."""
        table = pl.DataFrame(
            {
                "hypothesis": ["a", "b", "b", None, "c"],
                "selection_score": [0.1, 0.9, 0.8, 0.99, 0.5],
            }
        )
        assert select_top_hypotheses(table, 2) == ["b", "c"]

    def test_falls_back_to_abs_separation_score(self) -> None:
        """Sem ``selection_score``, usa ``|separation_score|``."""
        table = pl.DataFrame({"hypothesis": ["a", "b"], "separation_score": [0.1, -0.7]})
        assert select_top_hypotheses(table, 1) == ["b"]

    def test_empty_raises(self) -> None:
        """Tabela sem hipóteses é erro explícito."""
        with pytest.raises(EmptyDatasetError):
            select_top_hypotheses(pl.DataFrame({"hypothesis": [None]}), 3)


class TestStratifiedSubsample:
    """Subamostra estratificada do holdout."""

    def test_keeps_class_proportions(self) -> None:
        """A proporção de positivos é preservada (aprox.)."""
        labels = np.array([1] * 20 + [0] * 180)
        picked = stratified_subsample_indices(labels, 100, 0)
        assert 8 <= labels[picked].sum() <= 12
        assert picked.size in range(95, 106)

    def test_returns_all_when_n_exceeds_total(self) -> None:
        """Pedir mais do que existe devolve tudo."""
        assert stratified_subsample_indices(np.array([0, 1, 1]), 10, 0).tolist() == [0, 1, 2]

    def test_rare_class_is_never_dropped(self) -> None:
        """Mesmo com 1 positivo, ele entra na amostra."""
        labels = np.array([1] + [0] * 199)
        assert labels[stratified_subsample_indices(labels, 20, 0)].sum() == 1

    def test_continuous_target_uses_simple_sample(self) -> None:
        """Alvo contínuo: amostra simples do tamanho pedido."""
        labels = np.linspace(0, 1, 100)
        assert stratified_subsample_indices(labels, 30, 0).size == 30

    def test_is_deterministic_and_sorted(self) -> None:
        """Mesma semente, mesmos índices, em ordem."""
        labels = np.array([0, 1] * 50)
        first = stratified_subsample_indices(labels, 40, 3)
        assert first.tolist() == sorted(first.tolist())
        assert first.tolist() == stratified_subsample_indices(labels, 40, 3).tolist()


class TestValidateHypotheses:
    """Bonferroni sobre ``score_hypotheses``."""

    def test_marks_survivors_by_corrected_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Limiar = alfa / nº de hipóteses; só p menor sobrevive."""
        scored = pd.DataFrame(
            {
                "hypothesis": ["h1", "h2"],
                "separation_score": [0.3, 0.01],
                "separation_pval": [0.001, 0.9],
                "regression_coef": [1.0, 0.0],
                "regression_pval": [0.001, 0.5],
                "feature_prevalence": [0.2, 0.3],
            }
        )
        monkeypatch.setattr(validation, "score_hypotheses", lambda **_: ({}, scored))
        result = validate_hypotheses(
            {"h1": np.array([1, 0]), "h2": np.array([0, 1])},
            np.array([1, 0]),
            classification=True,
            corrected_pval_threshold=0.1,
        )
        assert result["bonferroni_threshold"].to_list() == [pytest.approx(0.05)] * 2
        assert result["survives"].to_list() == [True, False]


class TestHoldoutRows:
    """O holdout recalculado é disjunto da descoberta."""

    def test_holdout_is_disjoint_from_discovery(
        self, diagnostic_corpus_large: pl.DataFrame, diagnostics_settings: Any
    ) -> None:
        """Nenhum tweet do holdout está na partição de descoberta."""
        target = build_disagreement_target(diagnostic_corpus_large)
        holdout = select_holdout_rows(diagnostic_corpus_large, target, diagnostics_settings)
        partitioned = assign_partitions(
            diagnostic_corpus_large,
            holdout_size=diagnostics_settings.splits.holdout_size,
            validation_size=diagnostics_settings.splits.validation_size,
            random_seed=diagnostics_settings.random_seed,
        )
        discovery_ids = set(partitioned.filter(pl.col("partition") == DISCOVERY_PARTITION)["id"])
        assert holdout.height > 0
        assert discovery_ids.isdisjoint(set(holdout["id"]))
        held_out = set(partitioned.filter(pl.col("partition") == HOLDOUT_PARTITION)["id"])
        assert set(holdout["id"]) <= held_out


class TestDryRun:
    """Estimativa sem rede."""

    def test_report_mentions_no_network(
        self, diagnostic_corpus_large: pl.DataFrame, diagnostics_settings: Any
    ) -> None:
        """O relatório traz chamadas e afirma que não houve rede."""
        report = build_validation_dry_run_report(
            "disagreement", diagnostic_corpus_large, diagnostics_settings
        )
        assert "DRY-RUN" in report
        assert "Nenhuma chamada de rede" in report
        assert "chamadas" in report


class _AlwaysYesClient:
    """Cliente que sempre responde "Yes" (o teste troca ``score_hypotheses`` por um dublê)."""

    def __init__(self) -> None:
        from diagnostics.llm_client import CallStats

        self.stats = CallStats()

    async def complete_many(self, prompts: list[str], **_: Any) -> list[str | None]:
        self.stats.n_requests += len(prompts)
        return ["Yes."] * len(prompts)


def _fake_paths(tmp_path: Path) -> Any:
    return SimpleNamespace(
        reports_interpretability_dir=tmp_path / "interp",
        reports_tables_dir=tmp_path / "tables",
        data_interim_dir=tmp_path / "interim",
    )


class TestRunValidationStage:
    """Fluxo completo com cliente e ``score_hypotheses`` falsos."""

    def test_requires_discovery_first(
        self, diagnostic_corpus_large: pl.DataFrame, diagnostics_settings: Any, tmp_path: Path
    ) -> None:
        """Sem a descoberta do alvo, orienta a rodar ``diagnostics.hypotheses`` antes."""
        with pytest.raises(DataNotFoundError, match=r"diagnostics\.hypotheses"):
            run_validation_stage(
                "disagreement",
                diagnostic_corpus_large,
                diagnostics_settings,
                _fake_paths(tmp_path),
                track=False,
                client=_AlwaysYesClient(),  # type: ignore[arg-type]
            )

    def test_end_to_end_writes_validation_and_sample(
        self,
        diagnostic_corpus_large: pl.DataFrame,
        diagnostics_settings: Any,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Gera tabela validada, CSV para rotulagem (sem ids) e chave separada."""
        paths = _fake_paths(tmp_path)
        discovery_dir = paths.reports_interpretability_dir / diagnostics_settings.data.output_dir
        discovery_dir.mkdir(parents=True)
        pl.DataFrame(
            {"hypothesis": ["h1", "h2", "h3"], "selection_score": [0.9, 0.5, 0.1]}
        ).write_parquet(discovery_dir / "disagreement.parquet")

        scored = pd.DataFrame(
            {
                "hypothesis": ["h1", "h2", "h3"],
                "separation_score": [0.3, 0.2, 0.0],
                "separation_pval": [0.01, 0.02, 0.9],
                "regression_coef": [1.0, 0.5, 0.0],
                "regression_pval": [0.001, 0.4, 0.9],
                "feature_prevalence": [0.5, 0.5, 0.5],
            }
        )
        monkeypatch.setattr(validation, "score_hypotheses", lambda **_: ({}, scored))
        monkeypatch.setenv(diagnostics_settings.sampling.salt_env_var, "sal-de-teste")

        outcome = run_validation_stage(
            "disagreement",
            diagnostic_corpus_large,
            diagnostics_settings,
            paths,
            track=False,
            client=_AlwaysYesClient(),  # type: ignore[arg-type]
        )
        assert outcome.n_survivors == 1
        assert outcome.sample_path is not None
        header = outcome.sample_path.read_text(encoding="utf-8").splitlines()[0].split(",")
        assert "id" not in header
        assert (paths.data_interim_dir / "diagnostics" / "para_rotular_chave.parquet").is_file()
        assert (discovery_dir / "disagreement_validacao.csv").is_file()
