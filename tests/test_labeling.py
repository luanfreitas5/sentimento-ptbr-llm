"""Testes do pipeline de rotulagem semiautomática em cascata (``src/labeling``)."""

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from constants.labels import SENTIMENT_CLASSES
from exceptions.data import DataValidationError, EmptyDatasetError
from exceptions.pipeline import PipelineStageError
from labeling import llm_relabeling
from labeling.automatic import (
    LexicalHeuristicLabeler,
    _label_indexed_item,
    calculate_lexicon_sentiment_counts,
    classify_by_lexical_heuristic,
    run_cascade_labeling,
)
from labeling.confidence import (
    calculate_agreement_ratio,
    calculate_discordance_score,
    calculate_weighted_label_scores,
    flag_low_confidence_samples,
)
from labeling.consensus import aggregate_by_weighted_majority_vote, merge_consensus_into_corpus
from labeling.llm_relabeling import (
    _extract_confidence,
    _relabel_single_text,
    _run_relabel_workers,
    _validate_relabel_inputs,
    parse_relabel_response,
    relabel_low_confidence_samples,
)
from labeling.manual import (
    _bucket_confidence_level,
    apply_human_validation_labels,
    calculate_labeling_error_rate,
    select_samples_for_human_validation,
)
from labeling.validation import (
    calculate_cohen_kappa,
    calculate_krippendorff_alpha,
    evaluate_against_gold_set,
)


class _FakeInvalidLabeler:
    """Rotulador de teste que sempre retorna um rótulo fora das classes conhecidas."""

    def label(self, text: str) -> tuple[str, float]:
        """Retorna um rótulo inválido, ignorando o texto de entrada."""
        return "muito_positivo", 0.9


class _FakeRaisingLabeler:
    """Rotulador de teste que sempre levanta uma exceção genérica ao classificar."""

    def label(self, text: str) -> tuple[str, float]:
        """Levanta ``ValueError``, ignorando o texto de entrada."""
        raise ValueError("falha simulada do rotulador")


class TestCalculateLexiconSentimentCounts:
    """Testes da contagem de palavras do léxico de sentimento."""

    def test_counts_positive_words(self) -> None:
        """Palavras positivas conhecidas devem ser contadas corretamente."""
        counts = calculate_lexicon_sentiment_counts("o produto é ótimo, adorei")
        assert counts == {"positivo": 2, "negativo": 0}

    def test_counts_negative_words(self) -> None:
        """Palavras negativas conhecidas devem ser contadas corretamente."""
        counts = calculate_lexicon_sentiment_counts("péssimo atendimento, é um lixo")
        assert counts == {"positivo": 0, "negativo": 2}

    def test_returns_zero_counts_for_text_without_lexicon_words(self) -> None:
        """Um texto sem palavras do léxico deve retornar contagens zeradas."""
        assert calculate_lexicon_sentiment_counts("chegou hoje de manhã") == {
            "positivo": 0,
            "negativo": 0,
        }

    def test_is_case_insensitive(self) -> None:
        """A comparação com o léxico deve ignorar caixa."""
        assert calculate_lexicon_sentiment_counts("ADOREI o produto") == {
            "positivo": 1,
            "negativo": 0,
        }


class TestClassifyByLexicalHeuristic:
    """Testes da classificação heurística combinando léxico e emojis."""

    def test_classifies_as_positive(self) -> None:
        """Um texto com sinal positivo dominante deve ser classificado como positivo."""
        assert classify_by_lexical_heuristic("o produto é ótimo, adorei") == ("positivo", 1.0)

    def test_classifies_as_negative(self) -> None:
        """Um texto com sinal negativo dominante deve ser classificado como negativo."""
        assert classify_by_lexical_heuristic("péssimo atendimento, é um lixo") == ("negativo", 1.0)

    def test_classifies_as_neutral_without_signal(self) -> None:
        """Um texto sem nenhum sinal de sentimento deve ser neutro com confiança zero."""
        assert classify_by_lexical_heuristic("texto neutro sem sinal") == ("neutro", 0.0)

    def test_classifies_as_neutral_on_tie(self) -> None:
        """Sinais positivos e negativos empatados devem resultar em neutro com confiança 0.5."""
        label, confidence_score = classify_by_lexical_heuristic("ótimo mas péssimo ao mesmo tempo")
        assert label == "neutro"
        assert confidence_score == pytest.approx(0.5)

    def test_combines_lexicon_and_emoji_signal(self) -> None:
        """Emojis conhecidos devem somar ao sinal do léxico de palavras."""
        label, confidence_score = classify_by_lexical_heuristic("bom 😍")
        assert label == "positivo"
        assert confidence_score == pytest.approx(1.0)


class TestLexicalHeuristicLabeler:
    """Testes do rotulador heurístico como implementação de ``SentimentLabeler``."""

    def test_label_delegates_to_classify_by_lexical_heuristic(self) -> None:
        """O método ``label`` deve produzir o mesmo resultado da função pura correspondente."""
        labeler = LexicalHeuristicLabeler()
        assert labeler.label("adorei o produto") == classify_by_lexical_heuristic(
            "adorei o produto"
        )


class TestRunCascadeLabeling:
    """Testes da execução da cascata de rotuladores sobre um corpus."""

    def test_produces_one_row_per_sample_and_labeler(self) -> None:
        """Cada combinação de amostra e rotulador deve gerar uma linha de resultado."""
        df = pl.DataFrame({"id": ["1", "2"], "text": ["adorei o produto", "péssimo, é um lixo"]})
        labelers = {"heuristica_lexica": LexicalHeuristicLabeler()}
        result = run_cascade_labeling(df, labelers, show_progress=False, max_workers=1)
        assert result.height == 2
        assert set(result.columns) == {
            "id",
            "tagger",
            "sentiment_label",
            "confidence_score",
            "weight",
        }

    def test_applies_configured_weight_per_labeler(self) -> None:
        """O peso de cada rotulador deve ser repassado ao resultado, com padrão 1.0."""
        df = pl.DataFrame({"id": ["1"], "text": ["adorei o produto"]})
        labelers = {"heuristica_lexica": LexicalHeuristicLabeler()}
        result = run_cascade_labeling(
            df, labelers, weights={"heuristica_lexica": 2.0}, show_progress=False, max_workers=1
        )
        assert result["weight"].to_list() == [2.0]

    def test_defaults_weight_to_one_when_not_configured(self) -> None:
        """Um rotulador ausente de ``weights`` deve receber peso padrão 1.0."""
        df = pl.DataFrame({"id": ["1"], "text": ["adorei o produto"]})
        labelers = {"heuristica_lexica": LexicalHeuristicLabeler()}
        result = run_cascade_labeling(df, labelers, show_progress=False, max_workers=1)
        assert result["weight"].to_list() == [1.0]

    def test_raises_for_empty_dataframe(self) -> None:
        """Um DataFrame vazio deve levantar ``EmptyDatasetError``."""
        df = pl.DataFrame({"id": [], "text": []}, schema={"id": pl.Utf8, "text": pl.Utf8})
        with pytest.raises(EmptyDatasetError):
            run_cascade_labeling(
                df,
                {"heuristica_lexica": LexicalHeuristicLabeler()},
                show_progress=False,
                max_workers=1,
            )

    def test_raises_for_empty_labelers(self) -> None:
        """Um mapeamento de rotuladores vazio deve levantar ``EmptyDatasetError``."""
        df = pl.DataFrame({"id": ["1"], "text": ["adorei o produto"]})
        with pytest.raises(EmptyDatasetError):
            run_cascade_labeling(df, {}, show_progress=False, max_workers=1)

    def test_raises_data_validation_error_for_invalid_labeler_output(self) -> None:
        """Um rótulo fora das classes conhecidas deve violar o contrato de dados."""
        df = pl.DataFrame({"id": ["1"], "text": ["qualquer texto"]})
        with pytest.raises(DataValidationError):
            run_cascade_labeling(
                df,
                {"rotulador_invalido": _FakeInvalidLabeler()},
                show_progress=False,
                max_workers=1,
            )

    def test_raises_pipeline_stage_error_when_labeler_raises(self) -> None:
        """Uma exceção levantada pelo rotulador deve virar ``PipelineStageError``.

        Necessário mesmo para o rotulador heurístico-lexical atual (que só
        levanta exceções builtin, sempre picklable) porque
        ``configs/labeling.yaml`` antecipa rotuladores futuros baseados em
        LLM: uma exceção de terceiros não picklable levantada dentro de um
        worker de ``ProcessPoolExecutor`` quebraria o pool inteiro
        (``BrokenProcessPool``) em vez de isolar a falha de um único item —
        ver ``_label_row_text``.
        """
        df = pl.DataFrame({"id": ["1"], "text": ["qualquer texto"]})
        with pytest.raises(PipelineStageError):
            run_cascade_labeling(
                df,
                {"rotulador_com_erro": _FakeRaisingLabeler()},
                show_progress=False,
                max_workers=1,
            )

    def test_produces_same_result_with_and_without_parallelism(self) -> None:
        """A rotulagem paralela deve produzir exatamente o mesmo resultado que a execução padrão.

        Verifica em particular que a ordem/correspondência amostra->rótulo
        não se perde ao coletar resultados em ProcessPoolExecutor (ver
        _label_indexed_item / operator.itemgetter(0)).
        """
        df = pl.DataFrame(
            {
                "id": [str(i) for i in range(6)],
                "text": [
                    "adorei o produto",
                    "péssimo atendimento",
                    "chegou no prazo",
                    "excelente experiência",
                    "produto horrível",
                    "sem opinião formada",
                ],
            }
        )
        labelers = {"heuristica_lexica": LexicalHeuristicLabeler()}

        result_sequential = run_cascade_labeling(df, labelers, max_workers=1, show_progress=False)
        result_parallel = run_cascade_labeling(df, labelers, max_workers=4, show_progress=False)

        assert result_sequential.sort("id").to_dicts() == result_parallel.sort("id").to_dicts()


class TestLabelIndexedItem:
    """Testes diretos de ``_label_indexed_item`` (wrapper indexado usado no lote paralelo)."""

    def test_preserves_original_index(self) -> None:
        """O índice original deve passar intacto pela classificação."""
        assert _label_indexed_item((7, "adorei o produto"), labeler=LexicalHeuristicLabeler()) == (
            7,
            "positivo",
            1.0,
        )


class TestCalculateWeightedLabelScores:
    """Testes da soma de scores ponderados por amostra e rótulo candidato."""

    def test_sums_confidence_times_weight_per_label(self) -> None:
        """O score ponderado deve ser a soma de confiança x peso por rótulo."""
        df = pl.DataFrame(
            {
                "id": ["1", "1", "1"],
                "tagger": ["heuristica", "llm", "modelo"],
                "sentiment_label": ["positivo", "positivo", "negativo"],
                "confidence_score": [0.8, 0.6, 0.9],
                "weight": [1.0, 2.0, 2.0],
            }
        )
        result = calculate_weighted_label_scores(df).sort("sentiment_label")
        assert result["weighted_score"].to_list() == pytest.approx([1.8, 2.0])

    def test_raises_for_empty_dataframe(self) -> None:
        """Um DataFrame vazio deve levantar ``EmptyDatasetError``."""
        df = pl.DataFrame(
            {"id": [], "tagger": [], "sentiment_label": [], "confidence_score": [], "weight": []},
            schema={
                "id": pl.Utf8,
                "tagger": pl.Utf8,
                "sentiment_label": pl.Utf8,
                "confidence_score": pl.Float64,
                "weight": pl.Float64,
            },
        )
        with pytest.raises(EmptyDatasetError):
            calculate_weighted_label_scores(df)


class TestCalculateAgreementRatio:
    """Testes do cálculo do rótulo vencedor e da razão de concordância."""

    def test_picks_label_with_highest_weighted_score(self) -> None:
        """O rótulo com maior score ponderado deve vencer como consenso."""
        df = pl.DataFrame(
            {
                "id": ["1", "1", "1"],
                "tagger": ["heuristica", "llm", "modelo"],
                "sentiment_label": ["positivo", "positivo", "negativo"],
                "confidence_score": [0.8, 0.6, 0.9],
                "weight": [1.0, 2.0, 2.0],
            }
        )
        result = calculate_agreement_ratio(df)
        assert result["consensus_label"].to_list() == ["positivo"]
        assert result["agreement_ratio"].to_list()[0] == pytest.approx(0.5263, abs=1e-4)

    def test_computes_independently_per_sample(self) -> None:
        """A razão de concordância deve ser calculada separadamente para cada amostra."""
        df = pl.DataFrame(
            {
                "id": ["1", "1", "2", "2", "2"],
                "tagger": ["a", "b", "a", "b", "c"],
                "sentiment_label": ["positivo", "positivo", "negativo", "positivo", "positivo"],
                "confidence_score": [1.0, 1.0, 1.0, 1.0, 1.0],
                "weight": [1.0, 1.0, 1.0, 1.0, 1.0],
            }
        )
        result = calculate_agreement_ratio(df).sort("id")
        assert result["consensus_label"].to_list() == ["positivo", "positivo"]
        assert result["agreement_ratio"].to_list() == pytest.approx([1.0, 0.6667], abs=1e-4)


class TestCalculateDiscordanceScore:
    """Testes do cálculo da discordância a partir da concordância."""

    def test_is_complement_of_agreement_ratio(self) -> None:
        """A discordância deve ser o complemento (1 - concordância) da razão de concordância."""
        df = pl.DataFrame(
            {
                "id": ["1", "1", "1"],
                "tagger": ["heuristica", "llm", "modelo"],
                "sentiment_label": ["positivo", "positivo", "negativo"],
                "confidence_score": [0.8, 0.6, 0.9],
                "weight": [1.0, 2.0, 2.0],
            }
        )
        result = calculate_discordance_score(df)
        assert result["discordance_score"].to_list()[0] == pytest.approx(0.4737, abs=1e-4)


class TestFlagLowConfidenceSamples:
    """Testes da sinalização de amostras candidatas à validação humana."""

    def test_flags_sample_above_discordance_threshold(self) -> None:
        """Uma amostra com discordância acima do limiar deve ser sinalizada."""
        df = pl.DataFrame(
            {
                "id": ["1", "1", "1"],
                "tagger": ["heuristica", "llm", "modelo"],
                "sentiment_label": ["positivo", "positivo", "negativo"],
                "confidence_score": [0.8, 0.6, 0.9],
                "weight": [1.0, 2.0, 2.0],
            }
        )
        discordance = calculate_discordance_score(df)
        result = flag_low_confidence_samples(discordance)
        assert result["requires_human_validation"].to_list() == [True]

    def test_does_not_flag_sample_with_full_agreement(self) -> None:
        """Uma amostra com concordância total não deve ser sinalizada."""
        df = pl.DataFrame(
            {
                "id": ["1", "1"],
                "tagger": ["heuristica", "llm"],
                "sentiment_label": ["positivo", "positivo"],
                "confidence_score": [1.0, 1.0],
                "weight": [1.0, 1.0],
            }
        )
        discordance = calculate_discordance_score(df)
        result = flag_low_confidence_samples(discordance)
        assert result["requires_human_validation"].to_list() == [False]

    def test_respects_custom_thresholds(self) -> None:
        """Limiares customizados devem ser respeitados."""
        df = pl.DataFrame(
            {
                "id": ["1", "1", "1"],
                "tagger": ["heuristica", "llm", "modelo"],
                "sentiment_label": ["positivo", "positivo", "negativo"],
                "confidence_score": [0.8, 0.6, 0.9],
                "weight": [1.0, 2.0, 2.0],
            }
        )
        discordance = calculate_discordance_score(df)
        result = flag_low_confidence_samples(
            discordance, low_confidence_threshold=0.0, discordance_threshold=1.0
        )
        assert result["requires_human_validation"].to_list() == [False]


class TestAggregateByWeightedMajorityVote:
    """Testes da agregação de candidatos em um rótulo de consenso."""

    def test_produces_sentiment_label_and_confidence_columns(self) -> None:
        """O resultado deve conter as colunas ``sentiment_label`` e ``confidence_score``."""
        df = pl.DataFrame(
            {
                "id": ["1", "1"],
                "tagger": ["heuristica", "llm"],
                "sentiment_label": ["positivo", "positivo"],
                "confidence_score": [0.8, 0.9],
                "weight": [1.0, 2.0],
            }
        )
        result = aggregate_by_weighted_majority_vote(df)
        assert result["sentiment_label"].to_list() == ["positivo"]
        assert set(result.columns) == {"id", "sentiment_label", "confidence_score"}


class TestMergeConsensusIntoCorpus:
    """Testes da mesclagem dos rótulos de consenso ao corpus original."""

    def test_merges_matching_ids(self) -> None:
        """Amostras com consenso correspondente devem receber o rótulo mesclado."""
        corpus = pl.DataFrame({"id": ["1", "2"], "text": ["ótimo", "sem opinião"]})
        consensus = pl.DataFrame(
            {
                "id": ["1"],
                "sentiment_label": ["positivo"],
                "confidence_score": [0.9],
            }
        )
        result = merge_consensus_into_corpus(corpus, consensus).sort("id")
        assert result["sentiment_label"].to_list() == ["positivo", None]

    def test_preserves_original_row_count(self) -> None:
        """A junção à esquerda não deve alterar o número de linhas do corpus original."""
        corpus = pl.DataFrame({"id": ["1", "2"], "text": ["a", "b"]})
        consensus = pl.DataFrame(
            {
                "id": ["1"],
                "sentiment_label": ["positivo"],
                "confidence_score": [0.9],
            }
        )
        assert merge_consensus_into_corpus(corpus, consensus).height == 2


class TestBucketConfidenceLevel:
    """Testes da classificação de confiança em faixas discretas."""

    def test_classifies_low_confidence(self) -> None:
        """Valores abaixo de 0.3 devem ser classificados como baixa."""
        assert _bucket_confidence_level(0.2) == "baixa"

    def test_classifies_medium_confidence(self) -> None:
        """Valores entre 0.3 e 0.5 devem ser classificados como média."""
        assert _bucket_confidence_level(0.4) == "media"

    def test_classifies_moderate_confidence(self) -> None:
        """Valores a partir de 0.5 devem ser classificados como moderada."""
        assert _bucket_confidence_level(0.6) == "moderada"


class TestSelectSamplesForHumanValidation:
    """Testes da seleção de amostras para validação humana."""

    def test_selects_only_flagged_samples(self) -> None:
        """Apenas amostras sinalizadas para validação humana devem ser candidatas."""
        df = pl.DataFrame(
            {
                "id": ["1", "2", "3"],
                "consensus_label": ["positivo", "negativo", "neutro"],
                "agreement_ratio": [0.2, 0.9, 0.4],
                "requires_human_validation": [True, False, True],
            }
        )
        result = select_samples_for_human_validation(
            df, sample_size=10, stratify_by_confidence=False
        )
        assert result.height == 2
        assert "2" not in result["id"].to_list()

    def test_stratifies_by_confidence_bucket(self) -> None:
        """A amostragem estratificada deve manter representantes de cada faixa de confiança."""
        df = pl.DataFrame(
            {
                "id": [str(i) for i in range(6)],
                "consensus_label": ["positivo"] * 3 + ["negativo"] * 3,
                "agreement_ratio": [0.2, 0.25, 0.28, 0.35, 0.4, 0.45],
                "requires_human_validation": [True] * 6,
            }
        )
        result = select_samples_for_human_validation(df, sample_size=6)
        assert result.height == 6
        assert "confidence_bucket" not in result.columns

    def test_raises_when_no_sample_is_flagged(self) -> None:
        """Se nenhuma amostra estiver sinalizada, deve levantar ``EmptyDatasetError``."""
        df = pl.DataFrame(
            {
                "id": ["1"],
                "consensus_label": ["positivo"],
                "agreement_ratio": [0.9],
                "requires_human_validation": [False],
            }
        )
        with pytest.raises(EmptyDatasetError):
            select_samples_for_human_validation(df)


class TestApplyHumanValidationLabels:
    """Testes da sobrescrita do rótulo de consenso por rótulo humano."""

    def test_overrides_label_when_human_review_available(self) -> None:
        """O rótulo humano deve substituir o rótulo de consenso quando disponível."""
        consensus = pl.DataFrame({"id": ["1", "2"], "sentiment_label": ["positivo", "negativo"]})
        human_labels = pl.DataFrame({"id": ["1"], "sentiment_label": ["neutro"]})
        result = apply_human_validation_labels(consensus, human_labels).sort("id")
        assert result["sentiment_label"].to_list() == ["neutro", "negativo"]
        assert result["is_human_validated"].to_list() == [True, False]


class TestCalculateLabelingErrorRate:
    """Testes da estimativa da taxa de erro da rotulagem automática."""

    def test_calculates_disagreement_fraction(self) -> None:
        """A taxa de erro deve refletir a fração de discordância na amostra revisada."""
        consensus = pl.DataFrame({"id": ["1", "2"], "sentiment_label": ["positivo", "negativo"]})
        human_labels = pl.DataFrame({"id": ["1", "2"], "sentiment_label": ["positivo", "neutro"]})
        assert calculate_labeling_error_rate(consensus, human_labels) == pytest.approx(0.5)

    def test_returns_zero_when_all_agree(self) -> None:
        """Concordância total entre automático e humano deve resultar em taxa de erro zero."""
        consensus = pl.DataFrame({"id": ["1"], "sentiment_label": ["positivo"]})
        human_labels = pl.DataFrame({"id": ["1"], "sentiment_label": ["positivo"]})
        assert calculate_labeling_error_rate(consensus, human_labels) == pytest.approx(0.0)

    def test_raises_for_no_common_samples(self) -> None:
        """Sem amostras em comum, deve levantar ``EmptyDatasetError``."""
        consensus = pl.DataFrame({"id": ["1"], "sentiment_label": ["positivo"]})
        human_labels = pl.DataFrame({"id": ["2"], "sentiment_label": ["negativo"]})
        with pytest.raises(EmptyDatasetError):
            calculate_labeling_error_rate(consensus, human_labels)


class TestCalculateCohenKappa:
    """Testes do coeficiente Kappa de Cohen."""

    def test_perfect_agreement_returns_one(self) -> None:
        """Concordância perfeita e não degenerada deve resultar em kappa igual a 1.0."""
        labels = ["positivo", "negativo", "positivo"]
        assert calculate_cohen_kappa(labels, labels) == pytest.approx(1.0)

    def test_known_partial_agreement_value(self) -> None:
        """Um caso conhecido de concordância parcial deve reproduzir o valor esperado."""
        kappa = calculate_cohen_kappa(
            ["positivo", "negativo", "neutro"], ["positivo", "negativo", "positivo"]
        )
        assert kappa == pytest.approx(0.5, abs=1e-4)

    def test_raises_for_mismatched_lengths(self) -> None:
        """Sequências de tamanhos diferentes devem levantar ``ValueError``."""
        with pytest.raises(ValueError, match="mesmo tamanho"):
            calculate_cohen_kappa(["positivo"], ["positivo", "negativo"])

    def test_raises_for_empty_labels(self) -> None:
        """Sequências vazias devem levantar ``EmptyDatasetError``."""
        with pytest.raises(EmptyDatasetError):
            calculate_cohen_kappa([], [])


class TestCalculateKrippendorffAlpha:
    """Testes do Alpha de Krippendorff (métrica nominal)."""

    def test_perfect_agreement_returns_one(self) -> None:
        """Concordância perfeita e não degenerada entre avaliadores deve resultar em alpha 1.0."""
        dados = [
            ["positivo", "negativo", "positivo"],
            ["positivo", "negativo", "positivo"],
        ]
        assert calculate_krippendorff_alpha(dados) == pytest.approx(1.0)

    def test_known_partial_agreement_value(self) -> None:
        """Um caso conhecido de concordância parcial deve reproduzir o valor esperado."""
        dados = [
            ["positivo", "positivo", "negativo"],
            ["positivo", "negativo", "negativo"],
        ]
        assert calculate_krippendorff_alpha(dados) == pytest.approx(0.4444, abs=1e-4)

    def test_ignores_missing_values(self) -> None:
        """Unidades com avaliação ausente (``None``) devem ser tratadas corretamente."""
        dados = [
            ["positivo", "negativo", None],
            ["positivo", "negativo", "positivo"],
        ]
        assert calculate_krippendorff_alpha(dados) == pytest.approx(1.0)

    def test_raises_for_single_rater(self) -> None:
        """Menos de dois avaliadores deve levantar ``ValueError``."""
        with pytest.raises(ValueError, match="ao menos 2 avaliadores"):
            calculate_krippendorff_alpha([["positivo", "negativo"]])

    def test_raises_for_mismatched_row_lengths(self) -> None:
        """Avaliadores com número diferente de unidades devem levantar ``ValueError``."""
        with pytest.raises(ValueError, match="mesmo número de unidades"):
            calculate_krippendorff_alpha([["positivo", "negativo"], ["positivo"]])

    def test_raises_when_no_unit_is_pairable(self) -> None:
        """Sem nenhuma unidade com ao menos 2 avaliações, deve levantar ``EmptyDatasetError``."""
        dados = [
            ["positivo", None],
            [None, "negativo"],
        ]
        with pytest.raises(EmptyDatasetError):
            calculate_krippendorff_alpha(dados)


class TestEvaluateAgainstGoldSet:
    """Testes da validação de rótulos de consenso contra um gold set de referência."""

    def test_meets_minimum_agreement_with_high_kappa(self) -> None:
        """Concordância alta com o gold set deve atender ao limiar mínimo padrão."""
        predicted = pl.DataFrame(
            {
                "id": ["1", "2", "3"],
                "sentiment_label": ["positivo", "negativo", "positivo"],
            }
        )
        gold = pl.DataFrame(
            {
                "id": ["1", "2", "3"],
                "sentiment_label": ["positivo", "negativo", "positivo"],
            }
        )
        result = evaluate_against_gold_set(predicted, gold)
        assert result.n_samples == 3
        assert result.meets_minimum_agreement is True
        assert result.cohen_kappa == pytest.approx(1.0)

    def test_does_not_meet_minimum_agreement_with_low_kappa(self) -> None:
        """Concordância abaixo do limiar mínimo não deve ser aprovada."""
        predicted = pl.DataFrame(
            {
                "id": ["1", "2", "3"],
                "sentiment_label": ["positivo", "negativo", "positivo"],
            }
        )
        gold = pl.DataFrame(
            {
                "id": ["1", "2", "3"],
                "sentiment_label": ["positivo", "negativo", "negativo"],
            }
        )
        result = evaluate_against_gold_set(predicted, gold)
        assert result.cohen_kappa == pytest.approx(0.4, abs=1e-4)
        assert result.meets_minimum_agreement is False

    def test_raises_for_no_common_samples(self) -> None:
        """Sem amostras em comum entre predito e gold set, deve levantar ``EmptyDatasetError``."""
        predicted = pl.DataFrame({"id": ["1"], "sentiment_label": ["positivo"]})
        gold = pl.DataFrame({"id": ["2"], "sentiment_label": ["negativo"]})
        with pytest.raises(EmptyDatasetError):
            evaluate_against_gold_set(predicted, gold)


class TestParseRelabelResponse:
    """Testes de :func:`labeling.llm_relabeling.parse_relabel_response`."""

    def test_parses_label_and_confidence_fields(self) -> None:
        """Uma resposta com ``label``/``confidence`` explícitos deve ser interpretada direto."""
        assert parse_relabel_response('{"label": "positivo", "confidence": 0.9}') == (
            "positivo",
            0.9,
        )

    def test_parses_confidence_from_probs_when_present(self) -> None:
        """Quando ``probs`` está presente, a confiança deve vir de ``probs[label]``."""
        raw_response = '{"label":"negativo","probs":{"positivo":0.1,"negativo":0.8,"neutro":0.1}}'
        assert parse_relabel_response(raw_response) == ("negativo", 0.8)

    def test_defaults_confidence_to_one_when_absent(self) -> None:
        """Sem ``confidence`` nem ``probs``, a confiança deve assumir 1.0."""
        assert parse_relabel_response('{"label": "neutro"}') == ("neutro", 1.0)

    def test_returns_none_for_response_without_json(self) -> None:
        """Uma resposta sem nenhum objeto JSON deve retornar ``None``."""
        assert parse_relabel_response("resposta sem json") is None

    def test_returns_none_for_malformed_json(self) -> None:
        """Um objeto JSON malformado deve retornar ``None`` em vez de levantar exceção."""
        assert parse_relabel_response('{"label": "positivo",}') is None

    def test_returns_none_for_label_outside_allowed_classes(self) -> None:
        """Um rótulo fora de ``allowed_labels`` deve ser rejeitado."""
        assert parse_relabel_response('{"label": "muito_positivo", "confidence": 0.9}') is None

    def test_strips_reasoning_block_before_think_tag(self) -> None:
        """Conteúdo de raciocínio antes de ``</think>`` deve ser descartado."""
        raw_response = '<think>raciocinando...</think>{"label": "positivo", "confidence": 0.7}'
        assert parse_relabel_response(raw_response) == ("positivo", 0.7)

    def test_strips_markdown_code_fence(self) -> None:
        """Uma resposta envolta em cerca de código ``` json deve ser interpretada normalmente."""
        raw_response = '```json\n{"label": "positivo", "confidence": 0.8}\n```'
        assert parse_relabel_response(raw_response) == ("positivo", 0.8)

    def test_is_case_and_whitespace_insensitive_for_label(self) -> None:
        """O rótulo deve ser normalizado (minúsculas, sem espaços) antes da validação."""
        assert parse_relabel_response('{"label": " POSITIVO ", "confidence": 0.5}') == (
            "positivo",
            0.5,
        )


class TestExtractConfidence:
    """Testes de :func:`labeling.llm_relabeling._extract_confidence`."""

    def test_reads_confidence_from_probs_for_label(self) -> None:
        """A confiança deve ser lida de ``probs[label]`` quando disponível."""
        parsed = {"label": "positivo", "probs": {"positivo": 0.7, "negativo": 0.3}}
        assert _extract_confidence(parsed, "positivo") == pytest.approx(0.7)

    def test_falls_back_to_confidence_when_label_missing_from_probs(self) -> None:
        """Se ``probs`` não contiver o rótulo, deve recorrer a ``confidence``."""
        parsed = {"label": "neutro", "probs": {"positivo": 0.7}, "confidence": 0.4}
        assert _extract_confidence(parsed, "neutro") == pytest.approx(0.4)

    def test_clamps_value_above_one(self) -> None:
        """Valores de confiança acima de 1.0 devem ser limitados (clamp) a 1.0."""
        assert _extract_confidence({"confidence": 1.5}, "positivo") == pytest.approx(1.0)

    def test_clamps_negative_value_to_zero(self) -> None:
        """Valores de confiança negativos devem ser limitados (clamp) a 0.0."""
        assert _extract_confidence({"confidence": -0.5}, "positivo") == pytest.approx(0.0)

    def test_defaults_to_one_for_non_numeric_confidence(self) -> None:
        """Um valor de confiança não numérico deve resultar no padrão 1.0."""
        assert _extract_confidence({"confidence": "alta"}, "positivo") == pytest.approx(1.0)


class TestRelabelSingleText:
    """Testes de :func:`labeling.llm_relabeling._relabel_single_text`."""

    def test_returns_parsed_result_on_first_successful_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Uma primeira chamada bem-sucedida e interpretável deve retornar (rótulo, confiança)."""
        monkeypatch.setattr(
            llm_relabeling,
            "generate_completion",
            lambda **kwargs: '{"label": "positivo", "confidence": 0.8}',
        )
        result = _relabel_single_text(
            "adorei o produto",
            "Classifique: {{TEXTO}}",
            model="modelo-teste",
            temperature=0.0,
            max_retries=3,
            allowed_labels=SENTIMENT_CLASSES,
        )
        assert result == ("positivo", 0.8)

    def test_retries_until_a_parseable_response_is_returned(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Respostas não interpretáveis devem ser reprocessadas até ``max_retries``."""
        responses = iter(["resposta sem json", '{"label": "negativo", "confidence": 0.6}'])
        monkeypatch.setattr(llm_relabeling, "generate_completion", lambda **kwargs: next(responses))
        result = _relabel_single_text(
            "produto pessimo",
            "Classifique: {{TEXTO}}",
            model="modelo-teste",
            temperature=0.0,
            max_retries=3,
            allowed_labels=SENTIMENT_CLASSES,
        )
        assert result == ("negativo", 0.6)

    def test_returns_none_after_exhausting_retries_on_call_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Se todas as tentativas levantarem exceção, deve retornar ``None`` (fail-safe)."""

        def _always_raises(**kwargs: object) -> str:
            raise RuntimeError("falha simulada de rede")

        monkeypatch.setattr(llm_relabeling, "generate_completion", _always_raises)
        result = _relabel_single_text(
            "texto qualquer",
            "Classifique: {{TEXTO}}",
            model="modelo-teste",
            temperature=0.0,
            max_retries=2,
            allowed_labels=SENTIMENT_CLASSES,
        )
        assert result is None

    def test_returns_none_after_exhausting_retries_on_unparseable_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Se todas as tentativas retornarem respostas não interpretáveis, deve retornar None."""
        monkeypatch.setattr(
            llm_relabeling, "generate_completion", lambda **kwargs: "resposta sem json"
        )
        result = _relabel_single_text(
            "texto qualquer",
            "Classifique: {{TEXTO}}",
            model="modelo-teste",
            temperature=0.0,
            max_retries=2,
            allowed_labels=SENTIMENT_CLASSES,
        )
        assert result is None

    def test_substitutes_text_placeholder_in_prompt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """O placeholder ``{{TEXTO}}`` do template deve ser substituído pelo texto real."""
        captured_prompts: list[str] = []

        def _fake_generate_completion(*, prompt: str, **kwargs: object) -> str:
            captured_prompts.append(prompt)
            return '{"label": "neutro", "confidence": 1.0}'

        monkeypatch.setattr(llm_relabeling, "generate_completion", _fake_generate_completion)
        _relabel_single_text(
            "chegou no prazo",
            "Classifique o texto: {{TEXTO}} - fim",
            model="modelo-teste",
            temperature=0.0,
            max_retries=1,
            allowed_labels=SENTIMENT_CLASSES,
        )
        assert captured_prompts == ["Classifique o texto: chegou no prazo - fim"]


class TestValidateRelabelInputs:
    """Testes de :func:`labeling.llm_relabeling._validate_relabel_inputs`."""

    def test_raises_empty_dataset_error_for_empty_corpus(self) -> None:
        """Um corpus vazio deve levantar ``EmptyDatasetError``."""
        df = pl.DataFrame(
            {"id": [], "text_normalized": [], "sentiment_label": [], "confidence_score": []},
            schema={
                "id": pl.Utf8,
                "text_normalized": pl.Utf8,
                "sentiment_label": pl.Utf8,
                "confidence_score": pl.Float64,
            },
        )
        with pytest.raises(EmptyDatasetError):
            _validate_relabel_inputs(
                df,
                id_column="id",
                text_column="text_normalized",
                label_column="sentiment_label",
                confidence_column="confidence_score",
            )

    def test_raises_data_validation_error_for_missing_columns(self) -> None:
        """Um corpus sem alguma das colunas exigidas deve levantar ``DataValidationError``."""
        df = pl.DataFrame({"id": ["1"], "sentiment_label": ["positivo"]})
        with pytest.raises(DataValidationError):
            _validate_relabel_inputs(
                df,
                id_column="id",
                text_column="text_normalized",
                label_column="sentiment_label",
                confidence_column="confidence_score",
            )

    def test_does_not_raise_for_valid_corpus(self) -> None:
        """Um corpus válido não deve levantar exceção."""
        df = pl.DataFrame(
            {
                "id": ["1"],
                "text_normalized": ["ótimo"],
                "sentiment_label": ["positivo"],
                "confidence_score": [0.9],
            }
        )
        _validate_relabel_inputs(
            df,
            id_column="id",
            text_column="text_normalized",
            label_column="sentiment_label",
            confidence_column="confidence_score",
        )


class TestRunRelabelWorkers:
    """Testes de :func:`labeling.llm_relabeling._run_relabel_workers`."""

    def test_preserves_original_order_with_parallel_workers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Os resultados devem ser retornados na ordem original dos textos, mesmo em paralelo."""
        texts = ["texto-0", "texto-1", "texto-2", "texto-3"]

        def _fake_generate_completion(*, prompt: str, **kwargs: object) -> str:
            index = prompt.split("texto-")[1]
            return f'{{"label": "positivo", "confidence": 0.{index}5}}'

        monkeypatch.setattr(llm_relabeling, "generate_completion", _fake_generate_completion)
        results = _run_relabel_workers(
            texts,
            "{{TEXTO}}",
            model="modelo-teste",
            temperature=0.0,
            max_retries=1,
            allowed_labels=SENTIMENT_CLASSES,
            n_workers=4,
            show_progress=False,
        )
        assert results == [
            ("positivo", 0.05),
            ("positivo", 0.15),
            ("positivo", 0.25),
            ("positivo", 0.35),
        ]

    def test_keeps_none_for_failed_items_without_affecting_others(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Uma falha isolada em um item não deve afetar o resultado dos demais."""

        def _fake_generate_completion(*, prompt: str, **kwargs: object) -> str:
            if prompt == "texto-falho":
                raise RuntimeError("falha simulada")
            return '{"label": "negativo", "confidence": 0.9}'

        monkeypatch.setattr(llm_relabeling, "generate_completion", _fake_generate_completion)
        results = _run_relabel_workers(
            ["texto-ok", "texto-falho"],
            "{{TEXTO}}",
            model="modelo-teste",
            temperature=0.0,
            max_retries=1,
            allowed_labels=SENTIMENT_CLASSES,
            n_workers=2,
            show_progress=False,
        )
        assert results == [("negativo", 0.9), None]


class TestRelabelLowConfidenceSamples:
    """Testes de :func:`labeling.llm_relabeling.relabel_low_confidence_samples`."""

    def test_returns_unchanged_corpus_when_no_sample_is_below_threshold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sem amostras abaixo do limiar, o corpus deve ser retornado inalterado."""

        def _fail_if_called(**kwargs: object) -> str:
            raise AssertionError("generate_completion não deveria ser chamado")

        monkeypatch.setattr(llm_relabeling, "generate_completion", _fail_if_called)
        labeled_corpus = pl.DataFrame(
            {
                "id": ["1"],
                "text_normalized": ["ótimo produto"],
                "sentiment_label": ["positivo"],
                "confidence_score": [0.9],
            }
        )
        result = relabel_low_confidence_samples(
            labeled_corpus,
            score_threshold=0.5,
            prompt_name="prompt-inexistente",
            show_progress=False,
        )
        assert result.equals(labeled_corpus)

    def test_updates_label_and_confidence_for_successfully_relabeled_samples(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Amostras abaixo do limiar, re-rotuladas com sucesso, devem ter rótulo/confiança
        atualizados."""
        monkeypatch.setattr(llm_relabeling, "load_prompt_template", lambda name: "{{TEXTO}}")
        monkeypatch.setattr(
            llm_relabeling,
            "generate_completion",
            lambda **kwargs: '{"label": "negativo", "confidence": 0.85}',
        )
        labeled_corpus = pl.DataFrame(
            {
                "id": ["1", "2"],
                "text_normalized": ["texto ambíguo", "ótimo produto"],
                "sentiment_label": ["neutro", "positivo"],
                "confidence_score": [0.2, 0.9],
            }
        )
        result = relabel_low_confidence_samples(
            labeled_corpus,
            score_threshold=0.5,
            prompt_name="algum_prompt",
            show_progress=False,
        ).sort("id")
        assert result["sentiment_label"].to_list() == ["negativo", "positivo"]
        assert result["confidence_score"].to_list() == pytest.approx([0.85, 0.9])

    def test_preserves_original_label_when_relabeling_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Uma falha na re-rotulagem deve preservar o rótulo/confiança originais (fail-safe)."""
        monkeypatch.setattr(llm_relabeling, "load_prompt_template", lambda name: "{{TEXTO}}")

        def _always_raises(**kwargs: object) -> str:
            raise RuntimeError("falha simulada de rede")

        monkeypatch.setattr(llm_relabeling, "generate_completion", _always_raises)
        labeled_corpus = pl.DataFrame(
            {
                "id": ["1"],
                "text_normalized": ["texto ambíguo"],
                "sentiment_label": ["neutro"],
                "confidence_score": [0.2],
            }
        )
        result = relabel_low_confidence_samples(
            labeled_corpus,
            score_threshold=0.5,
            prompt_name="algum_prompt",
            max_retries=1,
            show_progress=False,
        )
        assert result["sentiment_label"].to_list() == ["neutro"]
        assert result["confidence_score"].to_list() == pytest.approx([0.2])

    def test_only_relabels_samples_below_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Amostras com confiança acima do limiar não devem ser reenviadas ao LLM."""
        monkeypatch.setattr(llm_relabeling, "load_prompt_template", lambda name: "{{TEXTO}}")
        called_texts: list[str] = []

        def _fake_generate_completion(*, prompt: str, **kwargs: object) -> str:
            called_texts.append(prompt)
            return '{"label": "negativo", "confidence": 0.7}'

        monkeypatch.setattr(llm_relabeling, "generate_completion", _fake_generate_completion)
        labeled_corpus = pl.DataFrame(
            {
                "id": ["1", "2"],
                "text_normalized": ["texto baixa confianca", "texto alta confianca"],
                "sentiment_label": ["neutro", "positivo"],
                "confidence_score": [0.1, 0.95],
            }
        )
        relabel_low_confidence_samples(
            labeled_corpus,
            score_threshold=0.5,
            prompt_name="algum_prompt",
            show_progress=False,
        )
        assert called_texts == ["texto baixa confianca"]

    def test_raises_for_empty_corpus(self) -> None:
        """Um corpus vazio deve levantar ``EmptyDatasetError``."""
        df = pl.DataFrame(
            {"id": [], "text_normalized": [], "sentiment_label": [], "confidence_score": []},
            schema={
                "id": pl.Utf8,
                "text_normalized": pl.Utf8,
                "sentiment_label": pl.Utf8,
                "confidence_score": pl.Float64,
            },
        )
        with pytest.raises(EmptyDatasetError):
            relabel_low_confidence_samples(
                df, score_threshold=0.5, prompt_name="algum_prompt", show_progress=False
            )

    def test_raises_for_missing_required_columns(self) -> None:
        """Um corpus sem alguma coluna exigida deve levantar ``DataValidationError``."""
        df = pl.DataFrame({"id": ["1"], "sentiment_label": ["positivo"]})
        with pytest.raises(DataValidationError):
            relabel_low_confidence_samples(
                df, score_threshold=0.5, prompt_name="algum_prompt", show_progress=False
            )


class TestLabelingProperties:
    """Testes baseados em propriedade (hypothesis) para invariantes do módulo."""

    @given(st.text(max_size=80))
    def test_classify_by_lexical_heuristic_confidence_is_in_valid_range(self, text: str) -> None:
        """A confiança retornada deve sempre estar entre 0.0 e 1.0."""
        _, confidence_score = classify_by_lexical_heuristic(text)
        assert 0.0 <= confidence_score <= 1.0

    @given(st.text(max_size=80))
    def test_classify_by_lexical_heuristic_returns_known_label(self, text: str) -> None:
        """O rótulo retornado deve sempre pertencer às classes de sentimento conhecidas."""
        label, _ = classify_by_lexical_heuristic(text)
        assert label in SENTIMENT_CLASSES

    @given(
        st.lists(
            st.tuples(
                st.sampled_from(["positivo", "negativo", "neutro"]),
                st.sampled_from(["positivo", "negativo", "neutro"]),
            ),
            min_size=1,
            max_size=15,
        )
    )
    def test_cohen_kappa_is_symmetric(self, pairs: list[tuple[str, str]]) -> None:
        """O Kappa de Cohen deve ser simétrico entre os dois avaliadores."""
        labels_a = [pair[0] for pair in pairs]
        labels_b = [pair[1] for pair in pairs]
        assert calculate_cohen_kappa(labels_a, labels_b) == pytest.approx(
            calculate_cohen_kappa(labels_b, labels_a)
        )
