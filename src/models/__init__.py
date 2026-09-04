"""Modelos de classificação de sentimento em português brasileiro.

Implementa a Fase 9 do plano de elaboração (``PLANO-ELABORACAO.md``) e as
Seções 4.4-4.6 do documento mestre (``projeto-mestrado-analise-sentimentos-ptbr.md``):
os quatro paradigmas de modelo do projeto (ML clássico, deep learning,
fine-tuning de Transformer e LLM zero/few-shot), atrás de uma interface
comum (:class:`models.base.SentimentClassifier`) e de uma fábrica única
(``src/models/factory.py``) consumida por ``src/training`` e
``src/inference``.

Modules
-------
base
    Interface comum ``fit``/``predict``/``predict_proba``, o motor de
    fine-tuning de Transformer reaproveitado pelos modelos concretos e os
    utilitários de vocabulário/padding usados pela LSTM/CNN.
naive_bayes, logistic_regression, svm, random_forest, gradient_boosting
    Classificadores clássicos (scikit-learn/XGBoost) sobre TF-IDF ou
    embeddings.
lstm, cnn
    Classificadores de deep learning (PyTorch) sobre texto tokenizado, com
    embedding treinado do zero.
autoencoder
    Adaptador ``fit``/``transform`` do autoencoder de redução de
    dimensionalidade (``src/features/reduction.py``) à interface de modelo.
bertimbau, roberta, distilbert
    Fábricas de fine-tuning de encoders Transformer pré-treinados em
    português brasileiro.
llm
    Classificador zero-shot/few-shot baseado em LLM, via backend Ollama.
factory
    Fábrica única de modelos a partir de ``configs/model_params.yaml``.
persistence
    Salvamento/carregamento de modelos (joblib/PyTorch) e registro no
    MLflow Model Registry.
"""

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
    LLMBackend,
    LLMSentimentClassifier,
    build_sentiment_prompt,
    load_ollama_backend,
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

__all__: list[str] = [
    "AutoencoderFeatureReducer",
    "CNNSentimentClassifier",
    "LLMBackend",
    "LLMSentimentClassifier",
    "LSTMSentimentClassifier",
    "SentimentClassifier",
    "TransformerSentimentClassifier",
    "build_autoencoder_reducer",
    "build_bertimbau_classifier",
    "build_cnn_classifier",
    "build_distilbert_classifier",
    "build_gradient_boosting_classifier",
    "build_logistic_regression_classifier",
    "build_lstm_classifier",
    "build_naive_bayes_classifier",
    "build_random_forest_classifier",
    "build_roberta_classifier",
    "build_sentiment_prompt",
    "build_svm_classifier",
    "build_token_vocabulary",
    "create_classifier",
    "encode_token_sequences",
    "list_available_models",
    "load_classifier",
    "load_ollama_backend",
    "log_classifier_to_mlflow",
    "parse_llm_sentiment_output",
    "save_classifier",
    "select_balanced_few_shot_examples",
]
