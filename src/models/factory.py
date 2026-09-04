"""Fábrica de classificadores de sentimento a partir de ``configs/model_params.yaml``.

Implementa a Fase 9 do plano de elaboração: centraliza a criação de qualquer
modelo do projeto (clássico, profundo, Transformer ou LLM) por nome,
evitando que os módulos consumidores (``src/training``, ``src/pipelines``)
precisem conhecer o construtor específico de cada família de modelo.
"""

import logging
from collections.abc import Callable, Mapping
from typing import Any

from exceptions.model import UnsupportedModelError
from models.autoencoder import build_autoencoder_reducer
from models.bertimbau import build_bertimbau_classifier
from models.cnn import build_cnn_classifier
from models.distilbert import build_distilbert_classifier
from models.gradient_boosting import build_gradient_boosting_classifier
from models.llm import LLMSentimentClassifier, load_ollama_backend
from models.logistic_regression import build_logistic_regression_classifier
from models.lstm import build_lstm_classifier
from models.naive_bayes import build_naive_bayes_classifier
from models.random_forest import build_random_forest_classifier
from models.roberta import build_roberta_classifier
from models.svm import build_svm_classifier

logger = logging.getLogger(__name__)

_LLM_BACKEND_PARAMETER_NAMES = (
    "model_name",
    "base_url",
    "temperature",
    "top_p",
    "max_new_tokens",
    "seed",
)


def _build_llm_classifier(**overrides: Any) -> LLMSentimentClassifier:
    """Constrói o classificador LLM com um backend Ollama recém-carregado.

    Separa os parâmetros de configuração do backend (``model_name``,
    ``base_url`` etc.) dos parâmetros do classificador propriamente dito
    (``few_shot``, ``max_retries`` etc.) a partir dos ``overrides``
    recebidos por :func:`create_classifier`.

    Parameters
    ----------
    **overrides : Any
        Hiperparâmetros do backend Ollama e/ou de
        :class:`models.llm.LLMSentimentClassifier`.

    Returns
    -------
    LLMSentimentClassifier
        Classificador LLM pronto para ``fit``/``predict``.
    """
    backend_kwargs = {
        key: overrides.pop(key) for key in _LLM_BACKEND_PARAMETER_NAMES if key in overrides
    }
    backend = load_ollama_backend(**backend_kwargs)
    return LLMSentimentClassifier(backend, **overrides)


_MODEL_BUILDERS: Mapping[str, Callable[..., Any]] = {
    "autoencoder": build_autoencoder_reducer,
    "bertimbau": build_bertimbau_classifier,
    "cnn": build_cnn_classifier,
    "distilbert": build_distilbert_classifier,
    "gradient_boosting": build_gradient_boosting_classifier,
    "llm": _build_llm_classifier,
    "logistic_regression": build_logistic_regression_classifier,
    "lstm": build_lstm_classifier,
    "naive_bayes": build_naive_bayes_classifier,
    "random_forest": build_random_forest_classifier,
    "roberta": build_roberta_classifier,
    "svm": build_svm_classifier,
}


def list_available_models() -> tuple[str, ...]:
    """Lista os nomes de modelo aceitos por :func:`create_classifier`.

    Returns
    -------
    tuple[str, ...]
        Nomes de modelo suportados, em ordem alfabética.

    Examples
    --------
    >>> "logistic_regression" in list_available_models()
    True
    """
    return tuple(sorted(_MODEL_BUILDERS))


def create_classifier(model_name: str, **overrides: Any) -> Any:
    """Constrói um classificador de sentimento a partir do nome do modelo.

    Parameters
    ----------
    model_name : str
        Nome do modelo, uma das chaves retornadas por
        :func:`list_available_models` (ex.: ``"logistic_regression"``,
        ``"bertimbau"``, ``"llm"``).
    **overrides : Any
        Hiperparâmetros que sobrescrevem os padrões de
        ``configs/model_params.yaml`` para o modelo escolhido.

    Returns
    -------
    Any
        Instância do classificador (ou, para ``"autoencoder"``, do
        transformador de redução de dimensionalidade), satisfazendo
        :class:`models.base.SentimentClassifier` quando aplicável.

    Raises
    ------
    UnsupportedModelError
        Se ``model_name`` não for um dos modelos suportados.

    Examples
    --------
    >>> create_classifier("naive_bayes", alpha=0.5).alpha
    0.5
    """
    builder = _MODEL_BUILDERS.get(model_name)
    if builder is None:
        raise UnsupportedModelError(model_name, list(list_available_models()))
    logger.info("Criando classificador '%s' (overrides=%s).", model_name, overrides)
    return builder(**overrides)
