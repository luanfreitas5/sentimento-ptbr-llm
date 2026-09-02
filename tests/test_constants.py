"""Testes das constantes globais do projeto."""

import pytest

from constants.columns import (
    LABELED_CORPUS_REQUIRED_COLUMNS,
    RAW_CORPUS_REQUIRED_COLUMNS,
    TARGET_COLUMN,
)
from constants.defaults import (
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TEST_SIZE,
    DEFAULT_VALIDATION_SIZE,
)
from constants.labels import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    SENTIMENT_CLASSES,
    transform_id_to_label,
    transform_label_to_id,
    validate_label,
)
from constants.metrics import ALL_METRICS, PRIMARY_METRIC, validate_metric_name
from constants.regex import (
    EMOJI_PATTERN,
    HASHTAG_PATTERN,
    MENTION_PATTERN,
    RETWEET_PATTERN,
    URL_PATTERN,
)
from constants.tokens import SPECIAL_TOKENS, URL_TOKEN
from exceptions.data import DataValidationError


class TestColumns:
    """Testes das constantes de nomes de coluna."""

    def test_target_column_matches_config(self) -> None:
        """A coluna-alvo deve ser 'sentimento', conforme configs/config.yaml."""
        assert TARGET_COLUMN == "sentimento"

    def test_required_columns_contain_target(self) -> None:
        """O conjunto de colunas obrigatórias do corpus rotulado deve incluir a coluna-alvo."""
        assert TARGET_COLUMN in LABELED_CORPUS_REQUIRED_COLUMNS

    def test_raw_and_labeled_required_columns_are_disjoint_enough(self) -> None:
        """Os dois conjuntos de colunas obrigatórias devem compartilhar ao menos 'id'."""
        assert "id" in RAW_CORPUS_REQUIRED_COLUMNS
        assert "id" in LABELED_CORPUS_REQUIRED_COLUMNS


class TestLabels:
    """Testes das classes de sentimento e conversões de rótulo."""

    def test_sentiment_classes_has_three_classes(self) -> None:
        """Deve haver exatamente três classes de sentimento."""
        assert len(SENTIMENT_CLASSES) == 3

    def test_label_to_id_and_back_are_consistent(self) -> None:
        """Converter um rótulo para id e de volta deve retornar o rótulo original."""
        for label in SENTIMENT_CLASSES:
            assert ID_TO_LABEL[LABEL_TO_ID[label]] == label

    @pytest.mark.parametrize("label", ["negativo", "neutro", "positivo"])
    def test_validate_label_accepts_known_labels(self, label: str) -> None:
        """validate_label deve aceitar todas as classes conhecidas."""
        assert validate_label(label) == label

    def test_validate_label_rejects_unknown_label(self) -> None:
        """validate_label deve levantar DataValidationError para rótulo desconhecido."""
        with pytest.raises(DataValidationError):
            validate_label("muito_positivo")

    def test_transform_label_to_id_roundtrip(self) -> None:
        """transform_label_to_id e transform_id_to_label devem ser inversas."""
        for label in SENTIMENT_CLASSES:
            assert transform_id_to_label(transform_label_to_id(label)) == label

    def test_transform_id_to_label_rejects_unknown_id(self) -> None:
        """transform_id_to_label deve levantar DataValidationError para id desconhecido."""
        with pytest.raises(DataValidationError):
            transform_id_to_label(999)


class TestMetrics:
    """Testes dos nomes de métricas reconhecidas."""

    def test_primary_metric_is_f1_macro(self) -> None:
        """A métrica principal deve ser f1_macro, conforme configs/evaluation.yaml."""
        assert PRIMARY_METRIC == "f1_macro"

    def test_validate_metric_name_accepts_known_metric(self) -> None:
        """validate_metric_name deve aceitar uma métrica conhecida."""
        assert validate_metric_name("mcc") == "mcc"

    def test_validate_metric_name_rejects_unknown_metric(self) -> None:
        """validate_metric_name deve levantar DataValidationError para métrica desconhecida."""
        with pytest.raises(DataValidationError):
            validate_metric_name("metrica_inexistente")

    def test_all_metrics_contains_primary(self) -> None:
        """ALL_METRICS deve incluir a métrica principal."""
        assert PRIMARY_METRIC in ALL_METRICS


class TestRegexPatterns:
    """Testes dos padrões de expressão regular de limpeza de texto."""

    def test_url_pattern_matches_http_url(self) -> None:
        """URL_PATTERN deve casar com uma URL http/https típica."""
        assert URL_PATTERN.search("veja em https://exemplo.com/pagina") is not None

    def test_mention_pattern_matches_at_username(self) -> None:
        """MENTION_PATTERN deve casar com uma menção @usuario."""
        assert MENTION_PATTERN.findall("olá @usuario_exemplo, tudo bem?") == ["@usuario_exemplo"]

    def test_hashtag_pattern_matches_hashtag(self) -> None:
        """HASHTAG_PATTERN deve casar com uma hashtag."""
        assert HASHTAG_PATTERN.findall("adorei o produto #recomendo") == ["#recomendo"]

    def test_retweet_pattern_matches_prefix_only(self) -> None:
        """RETWEET_PATTERN deve casar apenas o prefixo 'RT @usuario:' no início do texto."""
        texto = "RT @usuario: ótimo produto"
        match_result = RETWEET_PATTERN.match(texto)
        assert match_result is not None
        assert texto[match_result.end() :] == "ótimo produto"

    def test_emoji_pattern_matches_emoji(self) -> None:
        """EMOJI_PATTERN deve casar com um emoji comum."""
        assert EMOJI_PATTERN.search("adorei 😀 o produto") is not None

    def test_url_pattern_does_not_match_plain_text(self) -> None:
        """URL_PATTERN não deve casar com um texto sem URLs."""
        assert URL_PATTERN.search("nenhuma url aqui") is None


class TestTokens:
    """Testes dos tokens especiais de normalização textual."""

    def test_url_token_is_in_special_tokens(self) -> None:
        """URL_TOKEN deve estar presente em SPECIAL_TOKENS."""
        assert URL_TOKEN in SPECIAL_TOKENS

    def test_special_tokens_are_unique(self) -> None:
        """Não deve haver tokens especiais duplicados."""
        assert len(SPECIAL_TOKENS) == len(set(SPECIAL_TOKENS))


class TestDefaults:
    """Testes dos valores padrão de hiperparâmetros e limiares."""

    def test_default_random_seed_matches_config(self) -> None:
        """A semente padrão deve ser 42, conforme configs/config.yaml."""
        assert DEFAULT_RANDOM_SEED == 42

    def test_split_sizes_are_valid_fractions(self) -> None:
        """Os tamanhos de partição devem estar estritamente entre 0 e 1."""
        assert 0 < DEFAULT_TEST_SIZE < 1
        assert 0 < DEFAULT_VALIDATION_SIZE < 1

    def test_confidence_level_is_valid_probability(self) -> None:
        """O nível de confiança padrão deve estar entre 0 e 1."""
        assert 0 < DEFAULT_CONFIDENCE_LEVEL < 1
