"""Testes do módulo de modelos de sentimento (``src/models``)."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

from exceptions.data import DataNotFoundError, EmptyDatasetError
from exceptions.model import ModelNotFittedError, ModelPersistenceError, UnsupportedModelError
from models.autoencoder import AutoencoderFeatureReducer, build_autoencoder_reducer
from models.base import (
    SentimentClassifier,
    TransformerSentimentClassifier,
    build_token_vocabulary,
    encode_token_sequences,
)
from models.bertimbau import build_bertimbau_classifier
from models.cnn import CNNSentimentClassifier, build_cnn_classifier
from models.distilbert import build_distilbert_classifier
from models.factory import create_classifier, list_available_models
from models.gradient_boosting import build_gradient_boosting_classifier
from models.llm import (
    LLMSentimentClassifier,
    build_sentiment_prompt,
    parse_llm_sentiment_output,
    select_balanced_few_shot_examples,
)
from models.logistic_regression import build_logistic_regression_classifier
from models.lstm import LSTMSentimentClassifier, build_lstm_classifier
from models.naive_bayes import build_naive_bayes_classifier
from models.persistence import load_classifier, log_classifier_to_mlflow, save_classifier
from models.random_forest import build_random_forest_classifier
from models.roberta import build_roberta_classifier
from models.svm import build_svm_classifier


class _FakeLLMBackend:
    """Backend LLM de teste: retorna respostas pré-programadas, na ordem das chamadas."""

    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        """Registra a chamada e retorna a próxima resposta pré-programada (repete a última)."""
        self.calls.append(prompt)
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[index]


class TestSentimentClassifierProtocol:
    """Testes da interface comum ``SentimentClassifier``."""

    def test_sklearn_estimator_satisfies_protocol(self) -> None:
        """Um estimador scikit-learn com fit/predict/predict_proba satisfaz o Protocol."""
        assert isinstance(build_naive_bayes_classifier(), SentimentClassifier)

    def test_llm_classifier_satisfies_protocol(self) -> None:
        """O classificador LLM também satisfaz o Protocol por duck typing."""
        assert isinstance(LLMSentimentClassifier(_FakeLLMBackend(["{}"])), SentimentClassifier)


class TestBuildTokenVocabulary:
    """Testes da construção de vocabulário termo -> índice."""

    def test_reserves_pad_and_unk_indices(self) -> None:
        """PAD deve ocupar o índice 0 e UNK o índice 1."""
        vocabulary = build_token_vocabulary([["bom", "dia"], ["bom", "produto"]])
        assert vocabulary["<pad>"] == 0
        assert vocabulary["<unk>"] == 1

    def test_orders_terms_by_descending_document_frequency(self) -> None:
        """Termos mais frequentes devem receber índices menores que termos raros."""
        vocabulary = build_token_vocabulary([["a", "b"], ["a", "c"], ["a", "d"]])
        assert vocabulary["a"] < vocabulary["b"]

    def test_respects_max_vocabulary_size(self) -> None:
        """O vocabulário não deve exceder PAD/UNK + ``max_vocabulary_size`` termos."""
        vocabulary = build_token_vocabulary([["a", "b", "c"]], max_vocabulary_size=1)
        assert len(vocabulary) == 3

    def test_filters_by_minimum_document_frequency(self) -> None:
        """Termos com frequência de documento abaixo do limiar são descartados."""
        vocabulary = build_token_vocabulary(
            [["frequente"], ["frequente"], ["raro"]], minimum_document_frequency=2
        )
        assert "frequente" in vocabulary
        assert "raro" not in vocabulary

    def test_raises_for_empty_documents(self) -> None:
        """Uma lista de documentos vazia deve levantar ``EmptyDatasetError``."""
        with pytest.raises(EmptyDatasetError):
            build_token_vocabulary([])


class TestEncodeTokenSequences:
    """Testes da codificação de documentos tokenizados em sequências de índices."""

    def test_pads_short_documents(self) -> None:
        """Documentos mais curtos que ``max_sequence_length`` devem ser preenchidos com PAD."""
        vocabulary = {"<pad>": 0, "<unk>": 1, "bom": 2, "dia": 3}
        encoded = encode_token_sequences([["bom"]], vocabulary, max_sequence_length=3)
        assert encoded.tolist() == [[2, 0, 0]]

    def test_truncates_long_documents(self) -> None:
        """Documentos mais longos que ``max_sequence_length`` devem ser truncados."""
        vocabulary = {"<pad>": 0, "<unk>": 1, "bom": 2, "dia": 3}
        encoded = encode_token_sequences(
            [["bom", "dia", "bom", "dia"]], vocabulary, max_sequence_length=2
        )
        assert encoded.tolist() == [[2, 3]]

    def test_maps_unknown_terms_to_unk_index(self) -> None:
        """Termos fora do vocabulário devem ser mapeados para UNK."""
        vocabulary = {"<pad>": 0, "<unk>": 1, "bom": 2}
        encoded = encode_token_sequences([["desconhecido"]], vocabulary, max_sequence_length=1)
        assert encoded.tolist() == [[1]]

    def test_raises_for_empty_documents(self) -> None:
        """Uma lista de documentos vazia deve levantar ``EmptyDatasetError``."""
        with pytest.raises(EmptyDatasetError):
            encode_token_sequences([], {"<pad>": 0, "<unk>": 1}, max_sequence_length=1)


class TestBuildNaiveBayesClassifier:
    """Testes da fábrica do classificador Naive Bayes."""

    def test_returns_configured_multinomial_nb(self) -> None:
        """O ``alpha`` informado deve ser repassado ao estimador."""
        classifier = build_naive_bayes_classifier(alpha=0.5)
        assert isinstance(classifier, MultinomialNB)
        assert classifier.alpha == 0.5

    def test_uses_default_alpha(self) -> None:
        """O padrão de ``alpha`` deve ser 1.0."""
        assert build_naive_bayes_classifier().alpha == 1.0


class TestBuildLogisticRegressionClassifier:
    """Testes da fábrica do classificador de Regressão Logística."""

    def test_returns_configured_logistic_regression(self) -> None:
        """O ``C`` informado deve ser repassado ao estimador."""
        classifier = build_logistic_regression_classifier(C=0.1)
        assert isinstance(classifier, LogisticRegression)
        assert classifier.C == 0.1

    def test_uses_defaults_from_model_params(self) -> None:
        """Os padrões devem espelhar ``configs/model_params.yaml``."""
        classifier = build_logistic_regression_classifier()
        assert classifier.C == 1.0
        assert classifier.penalty == "l2"
        assert classifier.solver == "lbfgs"
        assert classifier.max_iter == 1000
        assert classifier.class_weight == "balanced"
        assert classifier.random_state == 42


class TestBuildSvmClassifier:
    """Testes da fábrica do classificador SVM."""

    def test_returns_configured_svc(self) -> None:
        """O ``kernel`` informado deve ser repassado ao estimador."""
        classifier = build_svm_classifier(kernel="rbf")
        assert isinstance(classifier, SVC)
        assert classifier.kernel == "rbf"

    def test_uses_defaults_from_model_params(self) -> None:
        """Os padrões devem espelhar ``configs/model_params.yaml``."""
        classifier = build_svm_classifier()
        assert classifier.kernel == "linear"
        assert classifier.C == 1.0
        assert classifier.probability is True
        assert classifier.class_weight == "balanced"


class TestBuildRandomForestClassifier:
    """Testes da fábrica do classificador Random Forest."""

    def test_returns_configured_random_forest(self) -> None:
        """O ``n_estimators`` informado deve ser repassado ao estimador."""
        classifier = build_random_forest_classifier(n_estimators=50)
        assert isinstance(classifier, RandomForestClassifier)
        assert classifier.n_estimators == 50

    def test_uses_defaults_from_model_params(self) -> None:
        """Os padrões devem espelhar ``configs/model_params.yaml``."""
        classifier = build_random_forest_classifier()
        assert classifier.max_depth == 20
        assert classifier.min_samples_leaf == 2
        assert classifier.n_jobs == -1
        assert classifier.class_weight == "balanced"


class TestBuildGradientBoostingClassifier:
    """Testes da fábrica do classificador Gradient Boosting (XGBoost)."""

    def test_returns_configured_xgb_classifier(self) -> None:
        """O ``n_estimators`` informado deve ser repassado ao estimador."""
        classifier = build_gradient_boosting_classifier(n_estimators=50)
        assert isinstance(classifier, XGBClassifier)
        assert classifier.n_estimators == 50

    def test_uses_defaults_from_model_params(self) -> None:
        """Os padrões devem espelhar ``configs/model_params.yaml``."""
        classifier = build_gradient_boosting_classifier()
        assert classifier.max_depth == 6
        assert classifier.learning_rate == 0.05
        assert classifier.subsample == 0.8
        assert classifier.objective == "multi:softprob"


class TestLSTMSentimentClassifier:
    """Testes do classificador LSTM/BiLSTM (requerem ``torch`` instalado)."""

    def test_fit_predict_returns_known_labels(self) -> None:
        """As predições devem pertencer às classes vistas no treino."""
        pytest.importorskip("torch")
        X = [
            ["ótimo", "produto", "adorei"],
            ["péssimo", "atendimento", "horrível"],
            ["ótimo", "excelente", "adorei"],
            ["péssimo", "ruim", "horrível"],
        ]
        y = ["positivo", "negativo", "positivo", "negativo"]
        classifier = LSTMSentimentClassifier(
            embedding_dim=8,
            hidden_dim=4,
            num_layers=1,
            batch_size=2,
            epochs=2,
            early_stopping_patience=2,
            max_sequence_length=5,
        ).fit(X, y)
        predictions = classifier.predict(X)
        assert predictions.shape == (4,)
        assert set(predictions).issubset({"positivo", "negativo"})

    def test_predict_proba_rows_sum_to_one(self) -> None:
        """Cada linha da matriz de probabilidades deve somar (aproximadamente) 1."""
        pytest.importorskip("torch")
        X = [["bom"], ["ruim"], ["bom"], ["ruim"]]
        y = ["positivo", "negativo", "positivo", "negativo"]
        classifier = LSTMSentimentClassifier(
            embedding_dim=4,
            hidden_dim=4,
            num_layers=1,
            batch_size=2,
            epochs=1,
            max_sequence_length=2,
        ).fit(X, y)
        probabilities = classifier.predict_proba(X)
        assert probabilities.shape == (4, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4)

    def test_raises_for_empty_training_data(self) -> None:
        """Um conjunto de treino vazio deve levantar ``EmptyDatasetError``."""
        with pytest.raises(EmptyDatasetError):
            LSTMSentimentClassifier().fit([], [])

    def test_predict_raises_when_not_fitted(self) -> None:
        """Prever antes de treinar deve levantar ``ModelNotFittedError``."""
        with pytest.raises(ModelNotFittedError):
            LSTMSentimentClassifier().predict([["bom"]])


class TestBuildLstmClassifier:
    """Testes da fábrica do classificador LSTM."""

    def test_returns_configured_instance(self) -> None:
        """O ``hidden_dim`` informado deve ser repassado ao construtor."""
        classifier = build_lstm_classifier(hidden_dim=64)
        assert isinstance(classifier, LSTMSentimentClassifier)
        assert classifier.hidden_dim == 64


class TestCNNSentimentClassifier:
    """Testes do classificador TextCNN (requerem ``torch`` instalado)."""

    def test_fit_predict_returns_known_labels(self) -> None:
        """As predições devem pertencer às classes vistas no treino."""
        pytest.importorskip("torch")
        X = [
            ["ótimo", "produto", "adorei"],
            ["péssimo", "atendimento", "horrível"],
            ["ótimo", "excelente", "adorei"],
            ["péssimo", "ruim", "horrível"],
        ]
        y = ["positivo", "negativo", "positivo", "negativo"]
        classifier = CNNSentimentClassifier(
            embedding_dim=8,
            num_filters=4,
            filter_sizes=(2,),
            batch_size=2,
            epochs=2,
            early_stopping_patience=2,
            max_sequence_length=3,
        ).fit(X, y)
        predictions = classifier.predict(X)
        assert predictions.shape == (4,)
        assert set(predictions).issubset({"positivo", "negativo"})

    def test_predict_proba_rows_sum_to_one(self) -> None:
        """Cada linha da matriz de probabilidades deve somar (aproximadamente) 1."""
        pytest.importorskip("torch")
        X = [["bom", "dia"], ["ruim", "dia"], ["bom", "dia"], ["ruim", "dia"]]
        y = ["positivo", "negativo", "positivo", "negativo"]
        classifier = CNNSentimentClassifier(
            embedding_dim=4,
            num_filters=4,
            filter_sizes=(2,),
            batch_size=2,
            epochs=1,
            max_sequence_length=2,
        ).fit(X, y)
        probabilities = classifier.predict_proba(X)
        assert probabilities.shape == (4, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4)

    def test_raises_for_empty_training_data(self) -> None:
        """Um conjunto de treino vazio deve levantar ``EmptyDatasetError``."""
        with pytest.raises(EmptyDatasetError):
            CNNSentimentClassifier().fit([], [])

    def test_predict_raises_when_not_fitted(self) -> None:
        """Prever antes de treinar deve levantar ``ModelNotFittedError``."""
        with pytest.raises(ModelNotFittedError):
            CNNSentimentClassifier().predict([["bom"]])


class TestBuildCnnClassifier:
    """Testes da fábrica do classificador TextCNN."""

    def test_returns_configured_instance(self) -> None:
        """O ``num_filters`` informado deve ser repassado ao construtor."""
        classifier = build_cnn_classifier(num_filters=64)
        assert isinstance(classifier, CNNSentimentClassifier)
        assert classifier.num_filters == 64


class TestAutoencoderFeatureReducer:
    """Testes do adaptador fit/transform do autoencoder (requerem ``torch`` instalado)."""

    def test_fit_transform_reduces_dimensionality(self) -> None:
        """O espaço latente projetado deve ter a dimensão configurada em ``latent_dim``."""
        pytest.importorskip("torch")
        embeddings = np.random.default_rng(42).normal(size=(20, 16)).astype(np.float32)
        reducer = AutoencoderFeatureReducer(
            input_dim=16,
            latent_dim=4,
            hidden_layers=(8,),
            batch_size=4,
            epochs=2,
            early_stopping_patience=2,
        )
        reduced = reducer.fit_transform(embeddings)
        assert reduced.shape == (20, 4)

    def test_transform_raises_when_not_fitted(self) -> None:
        """Projetar antes de treinar deve levantar ``ModelNotFittedError``."""
        with pytest.raises(ModelNotFittedError):
            AutoencoderFeatureReducer().transform(np.zeros((2, 768)))

    def test_score_reconstruction_error_raises_when_not_fitted(self) -> None:
        """Calcular o erro de reconstrução antes de treinar deve levantar ``ModelNotFittedError``."""
        with pytest.raises(ModelNotFittedError):
            AutoencoderFeatureReducer().score_reconstruction_error(np.zeros((2, 768)))

    def test_score_reconstruction_error_returns_one_value_per_sample(self) -> None:
        """Deve haver um valor de erro de reconstrução por amostra de entrada."""
        pytest.importorskip("torch")
        embeddings = np.random.default_rng(42).normal(size=(20, 16)).astype(np.float32)
        reducer = AutoencoderFeatureReducer(
            input_dim=16,
            latent_dim=4,
            hidden_layers=(8,),
            batch_size=4,
            epochs=2,
            early_stopping_patience=2,
        ).fit(embeddings)
        errors = reducer.score_reconstruction_error(embeddings)
        assert errors.shape == (20,)


class TestBuildAutoencoderReducer:
    """Testes da fábrica do redutor de dimensionalidade via autoencoder."""

    def test_returns_configured_instance(self) -> None:
        """O ``latent_dim`` informado deve ser repassado ao construtor."""
        reducer = build_autoencoder_reducer(latent_dim=64)
        assert isinstance(reducer, AutoencoderFeatureReducer)
        assert reducer.latent_dim == 64


class TestBuildBertimbauClassifier:
    """Testes da fábrica do classificador BERTimbau."""

    def test_uses_bertimbau_defaults(self) -> None:
        """Os padrões devem espelhar ``configs/model_params.yaml -> transformers.bertimbau``."""
        classifier = build_bertimbau_classifier()
        assert isinstance(classifier, TransformerSentimentClassifier)
        assert classifier.model_name == "neuralmind/bert-base-portuguese-cased"
        assert classifier.batch_size == 16
        assert classifier.epochs == 4

    def test_allows_overriding_model_name(self) -> None:
        """Um ``model_name`` customizado deve sobrescrever o padrão do BERTimbau."""
        classifier = build_bertimbau_classifier(model_name="outro-modelo")
        assert classifier.model_name == "outro-modelo"


class TestBuildRobertaClassifier:
    """Testes da fábrica do classificador RoBERTa pt-BR."""

    def test_uses_roberta_defaults(self) -> None:
        """Os padrões devem espelhar ``configs/model_params.yaml -> transformers.roberta``."""
        classifier = build_roberta_classifier()
        assert classifier.model_name == "rdenadai/BR_BERTo"
        assert classifier.learning_rate == 0.00002


class TestBuildDistilbertClassifier:
    """Testes da fábrica do classificador DistilBERT pt-BR."""

    def test_uses_distilbert_defaults(self) -> None:
        """Os padrões devem espelhar ``configs/model_params.yaml -> transformers.distilbert``."""
        classifier = build_distilbert_classifier()
        assert classifier.model_name == "adalbertojunior/distilbert-portuguese-cased"
        assert classifier.batch_size == 32
        assert classifier.learning_rate == 0.00003


class TestTransformerSentimentClassifierNotFitted:
    """Testes do estado não treinado do motor genérico de fine-tuning."""

    def test_predict_proba_raises_when_not_fitted(self) -> None:
        """Prever antes de treinar deve levantar ``ModelNotFittedError``."""
        with pytest.raises(ModelNotFittedError):
            build_bertimbau_classifier().predict_proba(["texto"])


class TestBuildSentimentPrompt:
    """Testes da montagem do prompt de classificação de sentimento."""

    def test_includes_text_and_ends_with_resposta_marker(self) -> None:
        """O prompt deve terminar com o marcador ``Resposta:`` após o texto de entrada."""
        prompt = build_sentiment_prompt("ótimo produto", allowed_labels=("positivo", "negativo"))
        assert '"ótimo produto"' in prompt
        assert prompt.endswith('Texto: "ótimo produto"\nResposta:')

    def test_includes_few_shot_examples(self) -> None:
        """Exemplos few-shot informados devem aparecer no prompt final."""
        prompt = build_sentiment_prompt(
            "texto",
            few_shot_examples=[("exemplo bom", "positivo")],
            allowed_labels=("positivo", "negativo"),
        )
        assert "exemplo bom" in prompt
        assert '"sentimento": "positivo"' in prompt


class TestParseLlmSentimentOutput:
    """Testes da extração de rótulo/confiança da resposta de um LLM."""

    def test_extracts_label_and_confidence_from_valid_json(self) -> None:
        """Uma resposta JSON válida deve ter rótulo e confiança extraídos corretamente."""
        label, confidence = parse_llm_sentiment_output(
            '{"sentimento": "positivo", "confianca": 0.9, "justificativa": "..."}'
        )
        assert label == "positivo"
        assert confidence == 0.9

    def test_returns_none_for_text_without_json(self) -> None:
        """Uma resposta sem objeto JSON deve retornar ``(None, 0.0)``."""
        assert parse_llm_sentiment_output("não há json aqui") == (None, 0.0)

    def test_returns_none_for_label_outside_allowed(self) -> None:
        """Um rótulo fora das classes conhecidas deve retornar ``(None, 0.0)``."""
        label, confidence = parse_llm_sentiment_output(
            '{"sentimento": "desconhecido", "confianca": 0.9}'
        )
        assert label is None
        assert confidence == 0.0

    def test_clips_confidence_to_unit_interval(self) -> None:
        """Uma confiança fora de ``[0, 1]`` deve ser recortada para o intervalo."""
        _, confidence = parse_llm_sentiment_output('{"sentimento": "positivo", "confianca": 5.0}')
        assert confidence == 1.0

    def test_defaults_confidence_to_zero_when_missing(self) -> None:
        """A ausência da chave ``confianca`` deve resultar em confiança 0.0."""
        label, confidence = parse_llm_sentiment_output('{"sentimento": "neutro"}')
        assert label == "neutro"
        assert confidence == 0.0


class TestSelectBalancedFewShotExamples:
    """Testes da seleção balanceada de exemplos few-shot."""

    def test_selects_up_to_n_examples_per_class(self) -> None:
        """Deve selecionar até ``n_examples_per_class`` exemplos de cada classe presente."""
        texts = ["a1", "a2", "b1", "c1"]
        labels = ["positivo", "positivo", "negativo", "neutro"]
        examples = select_balanced_few_shot_examples(texts, labels, n_examples_per_class=1)
        assert len(examples) == 3
        assert {label for _, label in examples} == {"positivo", "negativo", "neutro"}

    def test_raises_for_empty_texts(self) -> None:
        """Uma lista de textos vazia deve levantar ``EmptyDatasetError``."""
        with pytest.raises(EmptyDatasetError):
            select_balanced_few_shot_examples([], [])


class TestLLMSentimentClassifier:
    """Testes do classificador LLM zero-shot/few-shot, com um backend de teste."""

    def test_predict_returns_parsed_labels(self) -> None:
        """O rótulo retornado deve ser o extraído da resposta do backend."""
        backend = _FakeLLMBackend(['{"sentimento": "positivo", "confianca": 0.8}'])
        classifier = LLMSentimentClassifier(backend, few_shot=False)
        predictions = classifier.predict(["ótimo produto"])
        assert predictions.tolist() == ["positivo"]

    def test_predict_proba_assigns_confidence_to_predicted_class(self) -> None:
        """A confiança relatada pelo LLM deve ser atribuída à classe predita."""
        backend = _FakeLLMBackend(['{"sentimento": "positivo", "confianca": 0.7}'])
        classifier = LLMSentimentClassifier(
            backend, few_shot=False, allowed_labels=("positivo", "negativo", "neutro")
        )
        probabilities = classifier.predict_proba(["ótimo produto"])
        assert probabilities.shape == (1, 3)
        assert np.isclose(probabilities[0].sum(), 1.0)
        positive_index = classifier.allowed_labels.index("positivo")
        assert np.isclose(probabilities[0, positive_index], 0.7)

    def test_falls_back_after_exhausting_retries(self) -> None:
        """Respostas não interpretáveis em todas as tentativas devem usar o rótulo de fallback."""
        backend = _FakeLLMBackend(["resposta inválida"])
        classifier = LLMSentimentClassifier(
            backend, few_shot=False, max_retries=2, fallback_label="neutro"
        )
        predictions = classifier.predict(["texto ambíguo"])
        assert predictions.tolist() == ["neutro"]
        assert len(backend.calls) == 2

    def test_fit_selects_few_shot_examples_when_enabled(self) -> None:
        """Com ``few_shot=True``, ``fit`` deve selecionar exemplos balanceados."""
        backend = _FakeLLMBackend(['{"sentimento": "positivo", "confianca": 1.0}'])
        classifier = LLMSentimentClassifier(backend, few_shot=True, n_examples_per_class=1)
        classifier.fit(["bom", "ruim", "ok"], ["positivo", "negativo", "neutro"])
        assert len(classifier._few_shot_examples) == 3

    def test_fit_is_noop_when_few_shot_disabled(self) -> None:
        """Com ``few_shot=False``, ``fit`` não deve selecionar exemplos."""
        backend = _FakeLLMBackend(["{}"])
        classifier = LLMSentimentClassifier(backend, few_shot=False)
        classifier.fit(["bom"], ["positivo"])
        assert classifier._few_shot_examples == []


class TestListAvailableModels:
    """Testes da listagem de modelos suportados pela fábrica."""

    def test_includes_all_expected_model_names(self) -> None:
        """Todos os modelos implementados devem constar na listagem."""
        available = list_available_models()
        for name in (
            "naive_bayes",
            "logistic_regression",
            "svm",
            "random_forest",
            "gradient_boosting",
            "lstm",
            "cnn",
            "autoencoder",
            "bertimbau",
            "roberta",
            "distilbert",
            "llm",
        ):
            assert name in available

    def test_returns_sorted_tuple(self) -> None:
        """A listagem deve estar em ordem alfabética."""
        available = list_available_models()
        assert available == tuple(sorted(available))


class TestCreateClassifier:
    """Testes da fábrica única de classificadores."""

    def test_creates_naive_bayes_with_overrides(self) -> None:
        """Overrides informados devem ser repassados ao construtor do modelo escolhido."""
        classifier = create_classifier("naive_bayes", alpha=0.3)
        assert isinstance(classifier, MultinomialNB)
        assert classifier.alpha == 0.3

    def test_creates_bertimbau_classifier(self) -> None:
        """O nome ``"bertimbau"`` deve produzir um ``TransformerSentimentClassifier``."""
        classifier = create_classifier("bertimbau")
        assert isinstance(classifier, TransformerSentimentClassifier)

    def test_raises_for_unsupported_model_name(self) -> None:
        """Um nome de modelo desconhecido deve levantar ``UnsupportedModelError``."""
        with pytest.raises(UnsupportedModelError):
            create_classifier("modelo_inexistente")


class TestSaveAndLoadClassifier:
    """Testes da persistência de modelos em disco."""

    def test_round_trips_a_joblib_model(self, tmp_path: Path) -> None:
        """Um modelo salvo e recarregado deve preservar seus hiperparâmetros."""
        model = build_naive_bayes_classifier(alpha=0.7)
        file_path = tmp_path / "modelo.joblib"
        save_classifier(model, file_path)
        loaded_model = load_classifier(file_path)
        assert loaded_model.alpha == 0.7

    def test_save_creates_parent_directories(self, tmp_path: Path) -> None:
        """Diretórios pais ausentes devem ser criados automaticamente."""
        model = build_naive_bayes_classifier()
        file_path = tmp_path / "aninhado" / "modelo.joblib"
        save_classifier(model, file_path)
        assert file_path.is_file()

    def test_save_raises_for_unsupported_backend(self, tmp_path: Path) -> None:
        """Um backend desconhecido deve levantar ``ModelPersistenceError``."""
        with pytest.raises(ModelPersistenceError):
            save_classifier(
                build_naive_bayes_classifier(),
                tmp_path / "modelo.bin",
                backend="invalido",  # type: ignore[arg-type]
            )

    def test_load_raises_for_missing_file(self, tmp_path: Path) -> None:
        """Um caminho inexistente deve levantar ``DataNotFoundError``."""
        with pytest.raises(DataNotFoundError):
            load_classifier(tmp_path / "inexistente.joblib")

    def test_load_raises_for_unsupported_backend(self, tmp_path: Path) -> None:
        """Um backend desconhecido deve levantar ``ModelPersistenceError``."""
        file_path = tmp_path / "modelo.joblib"
        save_classifier(build_naive_bayes_classifier(), file_path)
        with pytest.raises(ModelPersistenceError):
            load_classifier(file_path, backend="invalido")  # type: ignore[arg-type]


class TestLogClassifierToMlflow:
    """Testes do registro de modelos no MLflow."""

    def test_raises_for_unsupported_backend(self) -> None:
        """Um backend desconhecido deve levantar ``ModelPersistenceError``."""
        with pytest.raises(ModelPersistenceError):
            log_classifier_to_mlflow(
                build_naive_bayes_classifier(),
                "modelo",
                backend="invalido",  # type: ignore[arg-type]
            )
