"""Testes do pipeline de rotulagem semiautomática em cascata (``src/labeling``)."""

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from constants.labels import SENTIMENT_CLASSES
from exceptions.data import DataValidationError, EmptyDatasetError
from labeling.automatic import (
    LexicalHeuristicLabeler,
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
        label, confidence = classify_by_lexical_heuristic("ótimo mas péssimo ao mesmo tempo")
        assert label == "neutro"
        assert confidence == pytest.approx(0.5)

    def test_combines_lexicon_and_emoji_signal(self) -> None:
        """Emojis conhecidos devem somar ao sinal do léxico de palavras."""
        label, confidence = classify_by_lexical_heuristic("bom 😍")
        assert label == "positivo"
        assert confidence == pytest.approx(1.0)


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
        result = run_cascade_labeling(df, labelers)
        assert result.height == 2
        assert set(result.columns) == {"id", "tagger", "sentiment_label", "confidence", "weight"}

    def test_applies_configured_weight_per_labeler(self) -> None:
        """O peso de cada rotulador deve ser repassado ao resultado, com padrão 1.0."""
        df = pl.DataFrame({"id": ["1"], "text": ["adorei o produto"]})
        labelers = {"heuristica_lexica": LexicalHeuristicLabeler()}
        result = run_cascade_labeling(df, labelers, weights={"heuristica_lexica": 2.0})
        assert result["weight"].to_list() == [2.0]

    def test_defaults_weight_to_one_when_not_configured(self) -> None:
        """Um rotulador ausente de ``weights`` deve receber peso padrão 1.0."""
        df = pl.DataFrame({"id": ["1"], "text": ["adorei o produto"]})
        labelers = {"heuristica_lexica": LexicalHeuristicLabeler()}
        result = run_cascade_labeling(df, labelers)
        assert result["weight"].to_list() == [1.0]

    def test_raises_for_empty_dataframe(self) -> None:
        """Um DataFrame vazio deve levantar ``EmptyDatasetError``."""
        df = pl.DataFrame({"id": [], "text": []}, schema={"id": pl.Utf8, "text": pl.Utf8})
        with pytest.raises(EmptyDatasetError):
            run_cascade_labeling(df, {"heuristica_lexica": LexicalHeuristicLabeler()})

    def test_raises_for_empty_labelers(self) -> None:
        """Um mapeamento de rotuladores vazio deve levantar ``EmptyDatasetError``."""
        df = pl.DataFrame({"id": ["1"], "text": ["adorei o produto"]})
        with pytest.raises(EmptyDatasetError):
            run_cascade_labeling(df, {})

    def test_raises_data_validation_error_for_invalid_labeler_output(self) -> None:
        """Um rótulo fora das classes conhecidas deve violar o contrato de dados."""
        df = pl.DataFrame({"id": ["1"], "text": ["qualquer texto"]})
        with pytest.raises(DataValidationError):
            run_cascade_labeling(df, {"rotulador_invalido": _FakeInvalidLabeler()})


class TestCalculateWeightedLabelScores:
    """Testes da soma de scores ponderados por amostra e rótulo candidato."""

    def test_sums_confidence_times_weight_per_label(self) -> None:
        """O score ponderado deve ser a soma de confiança x peso por rótulo."""
        df = pl.DataFrame(
            {
                "id": ["1", "1", "1"],
                "tagger": ["heuristica", "llm", "modelo"],
                "sentiment_label": ["positivo", "positivo", "negativo"],
                "confidence": [0.8, 0.6, 0.9],
                "weight": [1.0, 2.0, 2.0],
            }
        )
        result = calculate_weighted_label_scores(df).sort("sentiment_label")
        assert result["weighted_score"].to_list() == pytest.approx([1.8, 2.0])

    def test_raises_for_empty_dataframe(self) -> None:
        """Um DataFrame vazio deve levantar ``EmptyDatasetError``."""
        df = pl.DataFrame(
            {"id": [], "tagger": [], "sentiment_label": [], "confidence": [], "weight": []},
            schema={
                "id": pl.Utf8,
                "tagger": pl.Utf8,
                "sentiment_label": pl.Utf8,
                "confidence": pl.Float64,
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
                "confidence": [0.8, 0.6, 0.9],
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
                "confidence": [1.0, 1.0, 1.0, 1.0, 1.0],
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
                "confidence": [0.8, 0.6, 0.9],
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
                "confidence": [0.8, 0.6, 0.9],
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
                "confidence": [1.0, 1.0],
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
                "confidence": [0.8, 0.6, 0.9],
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
        """O resultado deve conter as colunas ``sentiment_label`` e ``confidence``."""
        df = pl.DataFrame(
            {
                "id": ["1", "1"],
                "tagger": ["heuristica", "llm"],
                "sentiment_label": ["positivo", "positivo"],
                "confidence": [0.8, 0.9],
                "weight": [1.0, 2.0],
            }
        )
        result = aggregate_by_weighted_majority_vote(df)
        assert result["sentiment_label"].to_list() == ["positivo"]
        assert set(result.columns) == {"id", "sentiment_label", "confidence"}


class TestMergeConsensusIntoCorpus:
    """Testes da mesclagem dos rótulos de consenso ao corpus original."""

    def test_merges_matching_ids(self) -> None:
        """Amostras com consenso correspondente devem receber o rótulo mesclado."""
        corpus = pl.DataFrame({"id": ["1", "2"], "text": ["ótimo", "sem opinião"]})
        consensus = pl.DataFrame(
            {"id": ["1"], "sentiment_label": ["positivo"], "confidence": [0.9]}
        )
        result = merge_consensus_into_corpus(corpus, consensus).sort("id")
        assert result["sentiment_label"].to_list() == ["positivo", None]

    def test_preserves_original_row_count(self) -> None:
        """A junção à esquerda não deve alterar o número de linhas do corpus original."""
        corpus = pl.DataFrame({"id": ["1", "2"], "text": ["a", "b"]})
        consensus = pl.DataFrame(
            {"id": ["1"], "sentiment_label": ["positivo"], "confidence": [0.9]}
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
            {"id": ["1", "2", "3"], "sentiment_label": ["positivo", "negativo", "positivo"]}
        )
        gold = pl.DataFrame(
            {"id": ["1", "2", "3"], "sentiment_label": ["positivo", "negativo", "positivo"]}
        )
        result = evaluate_against_gold_set(predicted, gold)
        assert result.n_samples == 3
        assert result.meets_minimum_agreement is True
        assert result.cohen_kappa == pytest.approx(1.0)

    def test_does_not_meet_minimum_agreement_with_low_kappa(self) -> None:
        """Concordância abaixo do limiar mínimo não deve ser aprovada."""
        predicted = pl.DataFrame(
            {"id": ["1", "2", "3"], "sentiment_label": ["positivo", "negativo", "positivo"]}
        )
        gold = pl.DataFrame(
            {"id": ["1", "2", "3"], "sentiment_label": ["positivo", "negativo", "negativo"]}
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


class TestLabelingProperties:
    """Testes baseados em propriedade (hypothesis) para invariantes do módulo."""

    @given(st.text(max_size=80))
    def test_classify_by_lexical_heuristic_confidence_is_in_valid_range(self, text: str) -> None:
        """A confiança retornada deve sempre estar entre 0.0 e 1.0."""
        _, confidence = classify_by_lexical_heuristic(text)
        assert 0.0 <= confidence <= 1.0

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
