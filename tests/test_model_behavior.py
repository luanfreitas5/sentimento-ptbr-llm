"""Testes comportamentais e de regressão de métrica do classificador de sentimento.

Usam um modelo TF-IDF + Regressão Logística treinado em um corpus sintético
determinístico com estrutura conhecida (léxico positivo, negativo e neutro), de
modo que rodam em qualquer ambiente de CI sem artefatos baixados nem GPU.
"""

import random

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from models.logistic_regression import build_logistic_regression_classifier
from preprocessing.cleaning import clean_tweet_text

pytestmark = pytest.mark.ml

RANDOM_SEED = 42
MACRO_F1_FLOOR = 0.90  # piso acordado para o corpus sintético (o modelo atinge ~1,0)

_LEXICON = {
    "positivo": ["ótimo", "excelente", "adorei", "maravilhoso", "perfeito", "amei"],
    "negativo": ["péssimo", "horrível", "odiei", "terrível", "ruim", "decepcionante"],
    "neutro": ["chegou", "segunda", "reunião", "pacote", "horário", "endereço"],
}
_FILLERS = ["o produto", "o atendimento", "a entrega", "o serviço", "a loja"]


def _generate_corpus(n_per_class: int, seed: int) -> tuple[list[str], list[str]]:
    """Gera tweets sintéticos: um termo do léxico da classe mais um complemento neutro."""
    rng = random.Random(seed)
    texts: list[str] = []
    labels: list[str] = []
    for label, words in _LEXICON.items():
        for _ in range(n_per_class):
            texts.append(f"{rng.choice(_FILLERS)} {rng.choice(words)}")
            labels.append(label)
    return texts, labels


@pytest.fixture(scope="module")
def sentiment_model() -> Pipeline:
    """Pipeline TF-IDF + Regressão Logística treinado no corpus sintético."""
    texts, labels = _generate_corpus(n_per_class=120, seed=RANDOM_SEED)
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
            ("clf", build_logistic_regression_classifier(random_state=RANDOM_SEED)),
        ]
    )
    return pipeline.fit(texts, labels)


class TestMinimumFunctionality:
    """Casos óbvios que qualquer classificador funcional deve acertar."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("o produto ótimo", "positivo"),
            ("o atendimento péssimo", "negativo"),
            ("a entrega chegou", "neutro"),
        ],
    )
    def test_predicts_obvious_cases(
        self, sentiment_model: Pipeline, text: str, expected: str
    ) -> None:
        """Casos inequívocos recebem o rótulo esperado."""
        assert sentiment_model.predict([text])[0] == expected


class TestInvariance:
    """Mudanças irrelevantes ao sentimento não podem alterar a predição."""

    @pytest.mark.parametrize("text", ["o produto adorei", "a loja horrível", "o serviço segunda"])
    def test_case_does_not_change_prediction(self, sentiment_model: Pipeline, text: str) -> None:
        """Maiúsculas e minúsculas produzem a mesma predição."""
        original = sentiment_model.predict([text])[0]
        assert sentiment_model.predict([text.upper()])[0] == original
        assert sentiment_model.predict([text.title()])[0] == original

    @pytest.mark.parametrize("text", ["o produto perfeito", "o atendimento terrível"])
    def test_mention_and_url_do_not_change_prediction(
        self, sentiment_model: Pipeline, text: str
    ) -> None:
        """@menção e URL, depois da limpeza do pipeline, não mudam a predição."""
        original = sentiment_model.predict([clean_tweet_text(text)])[0]
        noisy = f"@usuario {text} https://t.co/abc123"
        assert sentiment_model.predict([clean_tweet_text(noisy)])[0] == original


class TestDirectional:
    """Relações direcionais esperadas entre entrada e probabilidade."""

    def test_positive_term_raises_positive_probability(self, sentiment_model: Pipeline) -> None:
        """Trocar um termo neutro por um positivo aumenta P(positivo)."""
        classes = list(sentiment_model.classes_)
        idx = classes.index("positivo")
        neutral = sentiment_model.predict_proba(["o produto chegou"])[0, idx]
        positive = sentiment_model.predict_proba(["o produto ótimo"])[0, idx]
        assert positive > neutral

    def test_probabilities_are_valid_distribution(self, sentiment_model: Pipeline) -> None:
        """Probabilidades somam 1 e ficam em [0, 1]."""
        proba = sentiment_model.predict_proba(["o produto ótimo", "a loja ruim", "reunião segunda"])
        assert np.all((proba >= 0) & (proba <= 1))
        assert np.allclose(proba.sum(axis=1), 1.0)


def test_macro_f1_does_not_regress(sentiment_model: Pipeline) -> None:
    """Regressão de métrica: o macro-F1 no conjunto de teste congelado não cai do piso."""
    texts, labels = _generate_corpus(n_per_class=60, seed=RANDOM_SEED + 1)
    macro_f1 = f1_score(labels, sentiment_model.predict(texts), average="macro")
    assert macro_f1 >= MACRO_F1_FLOOR, f"Regressão de métrica detectada: macro-F1={macro_f1:.4f}"
