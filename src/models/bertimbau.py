"""Fábrica do classificador BERTimbau (fine-tuning) para sentimento em pt-BR.

Fábrica fina sobre :class:`models.base.TransformerSentimentClassifier`,
pré-configurada com o encoder BERTimbau base e os hiperparâmetros de
``configs/model_params.yaml -> transformers.bertimbau``.
"""

import logging
from typing import Any

from models.base import TransformerSentimentClassifier

logger = logging.getLogger(__name__)

DEFAULT_BERTIMBAU_MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

_DEFAULT_PARAMETERS: dict[str, Any] = {
    "model_name": DEFAULT_BERTIMBAU_MODEL_NAME,
    "max_length": 128,
    "batch_size": 16,
    "learning_rate": 0.00002,
    "epochs": 4,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "early_stopping_patience": 2,
    "random_state": 42,
}


def build_bertimbau_classifier(**overrides: Any) -> TransformerSentimentClassifier:
    """Constrói o classificador de fine-tuning do BERTimbau.

    Parameters
    ----------
    **overrides : Any
        Hiperparâmetros que sobrescrevem os padrões de
        ``configs/model_params.yaml -> transformers.bertimbau`` (ex.:
        ``model_name`` para usar uma variante diferente do BERTimbau).

    Returns
    -------
    TransformerSentimentClassifier
        Classificador não treinado, pronto para ``fit``.

    Examples
    --------
    >>> build_bertimbau_classifier().model_name
    'neuralmind/bert-base-portuguese-cased'
    """
    parameters = {**_DEFAULT_PARAMETERS, **overrides}
    logger.info("Construindo classificador BERTimbau '%s'.", parameters["model_name"])
    return TransformerSentimentClassifier(**parameters)
