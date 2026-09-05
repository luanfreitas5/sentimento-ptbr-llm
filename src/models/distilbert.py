"""Fábrica do classificador DistilBERT pt-BR (fine-tuning) para sentimento.

Fábrica fina sobre :class:`models.base.TransformerSentimentClassifier`,
pré-configurada com um encoder DistilBERT treinado em português brasileiro
(baseline leve/rápido) e os hiperparâmetros de
``configs/model_params.yaml -> transformers.distilbert``.
"""

import logging
from typing import Any

from models.base import TransformerSentimentClassifier

logger = logging.getLogger(__name__)

DEFAULT_DISTILBERT_MODEL_NAME = "adalbertojunior/distilbert-portuguese-cased"

_DEFAULT_PARAMETERS: dict[str, Any] = {
    "model_name": DEFAULT_DISTILBERT_MODEL_NAME,
    "max_length": 128,
    "batch_size": 32,
    "learning_rate": 0.00003,
    "epochs": 4,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "early_stopping_patience": 2,
    "random_state": 42,
}


def build_distilbert_classifier(**overrides: Any) -> TransformerSentimentClassifier:
    """Constrói o classificador de fine-tuning do DistilBERT pt-BR.

    Parameters
    ----------
    **overrides : Any
        Hiperparâmetros que sobrescrevem os padrões de
        ``configs/model_params.yaml -> transformers.distilbert`` (ex.:
        ``model_name`` para usar uma variante diferente).

    Returns
    -------
    TransformerSentimentClassifier
        Classificador não treinado, pronto para ``fit``.

    Examples
    --------
    >>> build_distilbert_classifier().model_name
    'adalbertojunior/distilbert-portuguese-cased'
    """
    parameters = _DEFAULT_PARAMETERS | overrides
    logger.info("Construindo classificador DistilBERT '%s'.", parameters["model_name"])
    return TransformerSentimentClassifier(**parameters)
