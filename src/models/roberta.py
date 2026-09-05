"""Fábrica do classificador RoBERTa pt-BR (fine-tuning) para sentimento.

Fábrica fina sobre :class:`models.base.TransformerSentimentClassifier`,
pré-configurada com um encoder RoBERTa treinado em português brasileiro e os
hiperparâmetros de ``configs/model_params.yaml -> transformers.roberta``.
"""

import logging
from typing import Any

from models.base import TransformerSentimentClassifier

logger = logging.getLogger(__name__)

DEFAULT_ROBERTA_MODEL_NAME = "rdenadai/BR_BERTo"

_DEFAULT_PARAMETERS: dict[str, Any] = {
    "model_name": DEFAULT_ROBERTA_MODEL_NAME,
    "max_length": 128,
    "batch_size": 16,
    "learning_rate": 0.00002,
    "epochs": 4,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "early_stopping_patience": 2,
    "random_state": 42,
}


def build_roberta_classifier(**overrides: Any) -> TransformerSentimentClassifier:
    """Constrói o classificador de fine-tuning do RoBERTa pt-BR.

    Parameters
    ----------
    **overrides : Any
        Hiperparâmetros que sobrescrevem os padrões de
        ``configs/model_params.yaml -> transformers.roberta`` (ex.:
        ``model_name`` para usar uma variante diferente).

    Returns
    -------
    TransformerSentimentClassifier
        Classificador não treinado, pronto para ``fit``.

    Examples
    --------
    >>> build_roberta_classifier().model_name
    'rdenadai/BR_BERTo'
    """
    parameters = _DEFAULT_PARAMETERS | overrides
    logger.info("Construindo classificador RoBERTa '%s'.", parameters["model_name"])
    return TransformerSentimentClassifier(**parameters)
