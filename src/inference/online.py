"""Inferência ponto a ponto, para uso interativo ou por uma futura API/dashboard.

Implementa a Fase 12: uma casca fina sobre
:class:`inference.predictor.Predictor` especializada em uma única amostra
por chamada, adequada a um endpoint de API (FastAPI) ou a um dashboard
interativo (Streamlit) que classifique um texto de cada vez — diferente de
``src/inference/batch.py``/``src/inference/llm_batch.py``, que assumem um
conjunto de amostras conhecido de antemão.
"""

import logging
from typing import Any

from inference.predictor import Predictor

logger = logging.getLogger(__name__)


class OnlinePredictor:
    """Classifica uma amostra por vez, para uso interativo ou por uma API.

    Parameters
    ----------
    predictor : inference.predictor.Predictor
        Interface de inferência sobre um modelo já treinado.
    """

    def __init__(self, predictor: Predictor) -> None:
        self.predictor = predictor

    def predict(self, sample: Any) -> dict[str, Any]:
        """Classifica uma única amostra.

        Parameters
        ----------
        sample : Any
            Amostra de entrada isolada (não em lote), no formato esperado
            pelo modelo (ex.: um texto, para modelos baseados em texto
            cru, ou um vetor de features).

        Returns
        -------
        dict[str, Any]
            Registro com as chaves ``"sentiment_label"``, ``"confidence"``
            e ``"probabilities"`` (ver
            :meth:`inference.predictor.Predictor._build_record`).

        Raises
        ------
        ValueError
            Se ``sample`` for ``None`` ou uma string vazia.
        ModelNotFittedError
            Se o modelo subjacente ainda não tiver sido treinado.

        Examples
        --------
        >>> online_predictor.predict("ótimo produto")  # doctest: +SKIP
        """
        if sample is None or sample == "":
            raise ValueError("A amostra a classificar não pode ser vazia.")
        record = self.predictor.predict_one_from_features([sample])
        logger.debug("Predição online: %r -> %s", sample, record["sentiment_label"])
        return record
