"""Testes dos módulos de pré-processamento e NLP (``src/preprocessing``)."""

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from constants.regex import URL_PATTERN
from exceptions.data import EmptyDatasetError
from exceptions.pipeline import PipelineStageError
from preprocessing.cleaning import (
    calculate_portuguese_stopword_ratio,
    calculate_repeated_word_ratio,
    clean_tweet_text,
    is_minimum_length_content,
    is_probable_portuguese_text,
    is_spam_like,
    remove_retweet_marker,
)
from preprocessing.emojis import calculate_emoji_sentiment_counts, normalize_emojis, remove_emojis
from preprocessing.filtering import (
    filter_by_inclusion_criteria,
    filter_by_minimum_length,
    filter_by_portuguese_language,
    filter_spam_like_rows,
    remove_duplicate_text_rows,
)
from preprocessing.pipeline import normalize_tweet_text, run_preprocessing_pipeline
from preprocessing.text import (
    normalize_hashtags,
    normalize_mentions,
    normalize_numbers,
    normalize_repeated_characters,
    normalize_urls,
)
from preprocessing.tokenization import (
    expand_contractions,
    mark_negation_scope,
    tokenize_and_normalize,
    tokenize_text,
)


class TestNormalizeUrls:
    """Testes de substituição de URLs pelo token semântico ``[URL]``."""

    def test_replaces_http_url(self) -> None:
        """Uma URL http deve ser substituída pelo token."""
        assert normalize_urls("veja em http://exemplo.com/x") == "veja em [URL]"

    def test_replaces_https_url(self) -> None:
        """Uma URL https deve ser substituída pelo token."""
        assert normalize_urls("veja em https://exemplo.com/x") == "veja em [URL]"

    def test_leaves_text_without_url_unchanged(self) -> None:
        """Um texto sem URL deve permanecer inalterado."""
        assert normalize_urls("sem link aqui") == "sem link aqui"


class TestNormalizeMentions:
    """Testes de substituição de menções pelo token semântico ``[MENCAO]``."""

    def test_replaces_mention(self) -> None:
        """Uma menção deve ser substituída pelo token."""
        assert normalize_mentions("oi @usuario tudo bem?") == "oi [MENCAO] tudo bem?"

    def test_leaves_text_without_mention_unchanged(self) -> None:
        """Um texto sem menção deve permanecer inalterado."""
        assert normalize_mentions("sem mencao aqui") == "sem mencao aqui"


class TestNormalizeHashtags:
    """Testes de normalização de hashtags."""

    def test_keeps_word_by_default(self) -> None:
        """Por padrão, apenas o caractere ``#`` deve ser removido."""
        assert normalize_hashtags("adorei #promocao") == "adorei promocao"

    def test_replaces_with_token_when_keep_word_false(self) -> None:
        """Com ``keep_word=False``, a hashtag inteira vira o token."""
        assert normalize_hashtags("adorei #promocao", keep_word=False) == "adorei [HASHTAG]"


class TestNormalizeNumbers:
    """Testes de substituição de sequências numéricas pelo token ``[NUMERO]``."""

    def test_replaces_digit_sequences(self) -> None:
        """Cada sequência de dígitos deve virar o token, independentemente."""
        assert normalize_numbers("nota 10 de 10") == "nota [NUMERO] de [NUMERO]"


class TestNormalizeRepeatedCharacters:
    """Testes de redução de caracteres repetidos."""

    def test_collapses_three_or_more_repeats(self) -> None:
        """Três ou mais repetições do mesmo caractere devem virar uma."""
        assert normalize_repeated_characters("muitooo bom") == "muito bom"

    def test_leaves_two_repeats_unchanged(self) -> None:
        """Duas repetições não devem ser alteradas."""
        assert normalize_repeated_characters("carro bom") == "carro bom"


class TestNormalizeEmojis:
    """Testes de mapeamento de emojis para tokens semânticos de polaridade."""

    def test_maps_known_positive_emoji_to_semantic_token(self) -> None:
        """Um emoji positivo conhecido deve virar o token de polaridade positiva."""
        assert normalize_emojis("adorei 😍") == "adorei [EMOJI_POSITIVO]"

    def test_maps_known_negative_emoji_to_semantic_token(self) -> None:
        """Um emoji negativo conhecido deve virar o token de polaridade negativa."""
        assert normalize_emojis("péssimo 😢") == "péssimo [EMOJI_NEGATIVO]"

    def test_maps_unknown_emoji_to_generic_token(self) -> None:
        """Um emoji reconhecido, mas fora do conjunto curado, vira o token genérico."""
        assert normalize_emojis("🚀 lançamento") == "[EMOJI] lançamento"


class TestRemoveEmojis:
    """Testes de remoção completa de emojis."""

    def test_removes_all_emojis(self) -> None:
        """Todos os emojis reconhecidos devem ser removidos, sem substituto."""
        assert remove_emojis("bom demais 😍🔥") == "bom demais "


class TestCalculateEmojiSentimentCounts:
    """Testes de contagem de emojis por polaridade."""

    def test_counts_by_polarity(self) -> None:
        """Cada emoji deve ser contado na polaridade correta."""
        counts = calculate_emoji_sentiment_counts("bom 😊 mas ruim 😢😢 e 😐")
        assert counts == {"positivo": 1, "negativo": 2, "neutro": 1}

    def test_returns_zero_counts_for_text_without_emoji(self) -> None:
        """Um texto sem emojis deve retornar contagens zeradas."""
        counts = calculate_emoji_sentiment_counts("sem emoji aqui")
        assert counts == {"positivo": 0, "negativo": 0, "neutro": 0}


class TestTokenizeText:
    """Testes do tokenizador baseado em expressão regular."""

    def test_splits_words_punctuation_and_special_tokens(self) -> None:
        """Palavras, pontuação e tokens especiais devem virar tokens distintos."""
        assert tokenize_text("não gostei, [URL]") == ["não", "gostei", ",", "[URL]"]

    def test_returns_empty_list_for_empty_text(self) -> None:
        """Um texto vazio não deve gerar nenhum token."""
        assert tokenize_text("") == []


class TestExpandContractions:
    """Testes de expansão de contrações e gírias."""

    def test_expands_known_slang(self) -> None:
        """Gírias conhecidas devem ser expandidas para a forma normalizada."""
        assert expand_contractions(["vc", "eh", "mto", "gente"]) == [
            "você",
            "é",
            "muito",
            "gente",
        ]

    def test_is_case_insensitive(self) -> None:
        """A comparação deve ignorar caixa."""
        assert expand_contractions(["VC"]) == ["você"]

    def test_leaves_unknown_tokens_unchanged(self) -> None:
        """Tokens sem expansão conhecida devem ser preservados."""
        assert expand_contractions(["produto", "!"]) == ["produto", "!"]


class TestMarkNegationScope:
    """Testes de marcação do escopo de negação."""

    def test_marks_tokens_until_punctuation(self) -> None:
        """Tokens após negação devem receber o sufixo até a pontuação."""
        tokens = ["não", "gostei", "nada", ",", "mas", "voltaria"]
        assert mark_negation_scope(tokens) == [
            "não",
            "gostei_NEG",
            "nada_NEG",
            ",",
            "mas",
            "voltaria",
        ]

    def test_does_not_mark_when_no_negation_word_present(self) -> None:
        """Sem palavra de negação, nenhum token deve ser marcado."""
        tokens = ["gostei", "muito"]
        assert mark_negation_scope(tokens) == ["gostei", "muito"]


class TestTokenizeAndNormalize:
    """Testes da composição de tokenização, expansão e marcação de negação."""

    def test_composes_tokenization_expansion_and_negation(self) -> None:
        """As três etapas devem ser aplicadas em conjunto, por padrão."""
        assert tokenize_and_normalize("não gostei vc entendeu") == [
            "não",
            "gostei_NEG",
            "você_NEG",
            "entendeu_NEG",
        ]

    def test_can_disable_slang_expansion_and_negation_marking(self) -> None:
        """As etapas opcionais devem poder ser desabilitadas individualmente."""
        result = tokenize_and_normalize(
            "não gostei vc", expand_slang=False, apply_negation_marking=False
        )
        assert result == ["não", "gostei", "vc"]


class TestRemoveRetweetMarker:
    """Testes de remoção do marcador de retweet."""

    def test_removes_marker_at_start(self) -> None:
        """O marcador ``RT @usuario:`` deve ser removido do início do texto."""
        assert remove_retweet_marker("RT @exemplo: ótimo produto") == "ótimo produto"

    def test_leaves_text_without_marker_unchanged(self) -> None:
        """Um texto sem marcador de retweet deve permanecer inalterado."""
        assert remove_retweet_marker("ótimo produto") == "ótimo produto"


class TestCleanTweetText:
    """Testes da limpeza mínima (retweet + espaçamento)."""

    def test_removes_marker_and_normalizes_whitespace(self) -> None:
        """Marcador de retweet e espaços excedentes devem ser removidos juntos."""
        assert clean_tweet_text("RT @exemplo:   ótimo   produto  ") == "ótimo produto"


class TestIsMinimumLengthContent:
    """Testes do predicado de conteúdo mínimo."""

    def test_rejects_short_text(self) -> None:
        """Um texto abaixo dos limiares padrão deve ser rejeitado."""
        assert is_minimum_length_content("bom") is False

    def test_accepts_text_meeting_thresholds(self) -> None:
        """Um texto que atende aos limiares padrão deve ser aceito."""
        assert is_minimum_length_content("muito bom mesmo") is True

    def test_respects_custom_thresholds(self) -> None:
        """Limiares customizados devem ser respeitados."""
        assert is_minimum_length_content("ok bom", minimum_characters=3, minimum_words=2) is True


class TestCalculateRepeatedWordRatio:
    """Testes do cálculo da proporção da palavra mais frequente."""

    def test_calculates_ratio_of_most_frequent_word(self) -> None:
        """A proporção deve refletir a contagem da palavra mais repetida."""
        assert calculate_repeated_word_ratio("promo promo promo compre agora") == 0.6

    def test_returns_zero_for_empty_text(self) -> None:
        """Um texto sem palavras deve retornar proporção zero."""
        assert calculate_repeated_word_ratio("") == 0.0


class TestIsSpamLike:
    """Testes do predicado heurístico de spam."""

    def test_flags_text_dominated_by_one_word(self) -> None:
        """Um texto dominado por uma palavra deve ser sinalizado como spam."""
        assert is_spam_like("compre compre compre compre agora") is True

    def test_does_not_flag_diverse_text(self) -> None:
        """Um texto com vocabulário diverso não deve ser sinalizado como spam."""
        assert is_spam_like("muito bom o atendimento") is False


class TestCalculatePortugueseStopwordRatio:
    """Testes do cálculo da proporção de stopwords em português."""

    def test_calculates_stopword_ratio(self) -> None:
        """A proporção deve refletir a contagem de stopwords conhecidas."""
        ratio = calculate_portuguese_stopword_ratio("o produto que eu comprei é muito bom")
        assert ratio == pytest.approx(0.625)

    def test_returns_zero_for_empty_text(self) -> None:
        """Um texto sem palavras deve retornar proporção zero."""
        assert calculate_portuguese_stopword_ratio("") == 0.0


class TestIsProbablePortugueseText:
    """Testes do predicado heurístico de idioma português."""

    def test_accepts_text_with_enough_stopwords(self) -> None:
        """Um texto com stopwords suficientes deve ser aceito."""
        assert is_probable_portuguese_text("o produto que eu comprei é muito bom") is True

    def test_rejects_text_without_portuguese_stopwords(self) -> None:
        """Um texto sem stopwords em português deve ser rejeitado."""
        assert is_probable_portuguese_text("the product I bought is great") is False


class TestFilterByMinimumLength:
    """Testes do filtro de conteúdo mínimo em nível de DataFrame."""

    def test_keeps_only_rows_meeting_threshold(self) -> None:
        """Apenas as linhas que atendem ao limiar devem ser mantidas."""
        df = pl.DataFrame({"text": ["bom", "muito bom mesmo"]})
        result = filter_by_minimum_length(df, text_column="text")
        assert result["text"].to_list() == ["muito bom mesmo"]


class TestFilterByPortugueseLanguage:
    """Testes do filtro heurístico de idioma em nível de DataFrame."""

    def test_keeps_only_probable_portuguese_rows(self) -> None:
        """Apenas as linhas provavelmente em português devem ser mantidas."""
        df = pl.DataFrame({"text": ["o produto que eu comprei é muito bom", "great product"]})
        result = filter_by_portuguese_language(df, text_column="text")
        assert result["text"].to_list() == ["o produto que eu comprei é muito bom"]


class TestFilterSpamLikeRows:
    """Testes do filtro de spam em nível de DataFrame."""

    def test_removes_spam_like_rows(self) -> None:
        """Linhas identificadas como spam devem ser removidas."""
        df = pl.DataFrame({"text": ["compre compre compre compre agora", "muito bom o produto"]})
        result = filter_spam_like_rows(df, text_column="text")
        assert result["text"].to_list() == ["muito bom o produto"]


class TestRemoveDuplicateTextRows:
    """Testes de remoção de linhas com texto duplicado."""

    def test_keeps_first_occurrence_only(self) -> None:
        """Apenas a primeira ocorrência de um texto duplicado deve ser mantida."""
        df = pl.DataFrame({"text": ["ótimo", "ótimo", "péssimo"]})
        result = remove_duplicate_text_rows(df, text_column="text")
        assert result["text"].to_list() == ["ótimo", "péssimo"]


class TestFilterByInclusionCriteria:
    """Testes da composição de todos os critérios de inclusão."""

    def test_applies_all_criteria_in_sequence(self) -> None:
        """Apenas as linhas que atendem a todos os critérios devem sobrar."""
        df = pl.DataFrame(
            {
                "text": [
                    "muito bom o produto",
                    "ok",
                    "compre compre compre compre agora",
                ]
            }
        )
        result = filter_by_inclusion_criteria(df, text_column="text")
        assert result["text"].to_list() == ["muito bom o produto"]

    def test_removes_duplicate_text_by_default(self) -> None:
        """Por padrão, textos duplicados devem ser removidos."""
        df = pl.DataFrame({"text": ["muito bom o produto", "muito bom o produto"]})
        result = filter_by_inclusion_criteria(df, text_column="text")
        assert result.height == 1

    def test_can_keep_duplicates_when_disabled(self) -> None:
        """Com ``drop_duplicate_text=False``, duplicatas devem ser mantidas."""
        df = pl.DataFrame({"text": ["muito bom o produto", "muito bom o produto"]})
        result = filter_by_inclusion_criteria(df, text_column="text", drop_duplicate_text=False)
        assert result.height == 2


class TestNormalizeTweetText:
    """Testes da composição completa de normalização de um único tweet."""

    def test_applies_full_normalization_sequence(self) -> None:
        """Todas as etapas devem ser aplicadas na ordem correta."""
        text = "RT @exemplo: amei o produto!! 😍 #recomendo https://exemplo.com"
        assert normalize_tweet_text(text) == "amei o produto!! [EMOJI_POSITIVO] recomendo [URL]"

    def test_replaces_hashtag_with_token_when_keep_word_false(self) -> None:
        """A opção ``keep_hashtag_word=False`` deve ser repassada corretamente."""
        result = normalize_tweet_text("adorei #promocao", keep_hashtag_word=False)
        assert result == "adorei [HASHTAG]"


class TestRunPreprocessingPipeline:
    """Testes do pipeline reprodutível de pré-processamento sobre um corpus."""

    def test_adds_normalized_column_and_filters_by_default(self) -> None:
        """A coluna normalizada deve ser criada e os critérios de inclusão aplicados."""
        df = pl.DataFrame({"id": ["1", "2"], "text": ["RT @a: muito bom!! 😍", "RT @b: oi"]})
        result = run_preprocessing_pipeline(df)
        assert result["text_normalized"].to_list() == ["muito bom!! [EMOJI_POSITIVO]"]

    def test_can_disable_inclusion_filters(self) -> None:
        """Com ``apply_inclusion_filters=False``, nenhuma linha deve ser removida."""
        df = pl.DataFrame({"id": ["1", "2"], "text": ["RT @a: oi", "RT @b: oi"]})
        result = run_preprocessing_pipeline(df, apply_inclusion_filters=False)
        assert result.height == 2

    def test_adds_tokens_column_when_requested(self) -> None:
        """A coluna de tokens só deve ser criada quando ``tokens_column`` é informado."""
        df = pl.DataFrame({"id": ["1"], "text": ["RT @a: muito bom!! 😍"]})
        result = run_preprocessing_pipeline(df, tokens_column="tokens")
        assert result["tokens"].to_list() == [["muito", "bom", "!", "!", "[EMOJI_POSITIVO]"]]

    def test_raises_for_empty_dataframe(self) -> None:
        """Um DataFrame vazio deve levantar ``EmptyDatasetError``."""
        df = pl.DataFrame({"id": [], "text": []}, schema={"id": pl.Utf8, "text": pl.Utf8})
        with pytest.raises(EmptyDatasetError):
            run_preprocessing_pipeline(df)

    def test_raises_pipeline_stage_error_when_normalization_fails(self) -> None:
        """Uma falha de normalização em uma linha deve virar ``PipelineStageError``."""
        df = pl.DataFrame({"id": ["1"], "text": [None]}, schema={"id": pl.Utf8, "text": pl.Utf8})
        with pytest.raises(PipelineStageError):
            run_preprocessing_pipeline(df)


class TestPreprocessingProperties:
    """Testes baseados em propriedade (hypothesis) para invariantes das transformações."""

    @given(st.text(max_size=80))
    def test_normalize_urls_never_leaves_url_pattern_behind(self, text: str) -> None:
        """Após a normalização, nenhum trecho do texto deve casar com o padrão de URL."""
        assert URL_PATTERN.search(normalize_urls(text)) is None

    @given(st.lists(st.text(min_size=1, max_size=10), max_size=10))
    def test_mark_negation_scope_preserves_token_count(self, tokens: list[str]) -> None:
        """A marcação de negação nunca deve alterar a quantidade de tokens."""
        assert len(mark_negation_scope(tokens)) == len(tokens)
