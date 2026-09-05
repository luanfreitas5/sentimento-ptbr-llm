"""Interface única de inferência por modelo, para os quatro paradigmas do projeto.

Implementa a Fase 12: envolve qualquer modelo que satisfaça
:class:`models.base.SentimentClassifier` (clássico, DL, Transformer ou LLM)
sob uma única API de predição (:class:`Predictor`), delegando a
normalização de saída a ``src/inference/postprocessing.py`` — os módulos
consumidores (``src/inference/batch.py``, ``src/inference/online.py``) não
precisam conhecer a implementação concreta do modelo.
"""

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl

from constants.labels import SENTIMENT_CLASSES
from exceptions.model import ModelNotFittedError
from inference.postprocessing import build_prediction_dataframe, standardize_prediction_output
from models.base import SentimentClassifier
from utils.validation import validate_not_empty_collection

logger = logging.getLogger(__name__)


def _resolve_model_classes(
    model: SentimentClassifier, allowed_labels: Sequence[str]
) -> tuple[str, ...]:
    """Resolve a ordem das classes de um modelo, para associar cada probabilidade ao seu rótulo.

    Parameters
    ----------
    model : SentimentClassifier
        Modelo treinado, tipicamente expondo o atributo ``classes_``
        (convenção scikit-learn, seguida por todos os modelos deste
        projeto).
    allowed_labels : Sequence[str]
        Classes de sentimento aceitas, usadas como alternativa quando o
        modelo não expõe ``classes_``.

    Returns
    -------
    tuple[str, ...]
        Rótulos de sentimento, na ordem correspondente às colunas de
        ``predict_proba``.
    """
    classes = getattr(model, "classes_", None)
    if classes is None:
        return tuple(allowed_labels)
    return tuple(str(sentiment_class) for sentiment_class in classes)


class Predictor:
    """Interface única de inferência sobre um classificador de sentimento treinado.

    Parameters
    ----------
    model : SentimentClassifier
        Modelo treinado (qualquer implementação de
        ``src/models/``/``src/llm/``), satisfazendo
        :class:`models.base.SentimentClassifier`.
    allowed_labels : Sequence[str], optional
        Classes de sentimento aceitas, by default
        :data:`constants.labels.SENTIMENT_CLASSES`.

    Examples
    --------
    >>> from models.naive_bayes import build_naive_bayes_classifier
    >>> model = build_naive_bayes_classifier().fit([[1, 0], [0, 1]], ["positivo", "negativo"])
    >>> predictor = Predictor(model, allowed_labels=("positivo", "negativo"))
    >>> predictor.predict_one_from_features([[1, 0]])["sentiment_label"]
    'positivo'
    """

    def __init__(
        self, model: SentimentClassifier, *, allowed_labels: Sequence[str] = SENTIMENT_CLASSES
    ) -> None:
        self.model = model
        self.allowed_labels = tuple(allowed_labels)
        self._classes = _resolve_model_classes(model, self.allowed_labels)

    def _build_record(self, label: str, probabilities: np.ndarray) -> dict[str, Any]:
        """Monta o registro de predição padronizado para uma única amostra.

        Parameters
        ----------
        label : str
            Rótulo bruto predito pelo modelo.
        probabilities : np.ndarray
            Vetor de probabilidades por classe, na ordem de
            :attr:`_classes`.

        Returns
        -------
        dict[str, Any]
            Dicionário com as chaves ``"sentiment_label"``,
            ``"confidence"`` e ``"probabilities"``.
        """
        class_index = (
            self._classes.index(label) if label in self._classes else int(np.argmax(probabilities))
        )
        confidence = float(probabilities[class_index])
        normalized_label, normalized_confidence = standardize_prediction_output(
            label, confidence, allowed_labels=self.allowed_labels
        )
        return {
            "sentiment_label": normalized_label,
            "confidence": normalized_confidence,
            "probabilities": dict(
                zip(self._classes, (float(p) for p in probabilities), strict=True)
            ),
        }

    def predict_one_from_features(self, features: Any) -> dict[str, Any]:
        """Classifica uma única amostra, já em lote de tamanho 1 (formato do modelo).

        Parameters
        ----------
        features : Any
            Amostra de entrada em lote de tamanho 1, no formato esperado
            pelo modelo (ex.: ``[[valor1, valor2]]`` para matriz de
            features, ``["um texto"]`` para modelos baseados em texto
            cru).

        Returns
        -------
        dict[str, Any]
            Registro padronizado (ver :meth:`_build_record`).

        Raises
        ------
        ModelNotFittedError
            Se o modelo ainda não tiver sido treinado.
        """
        try:
            predicted_label = str(self.model.predict(features)[0])
            probabilities = np.asarray(self.model.predict_proba(features))[0]
        except AttributeError as exception:
            raise ModelNotFittedError(type(self.model).__name__) from exception
        return self._build_record(predicted_label, probabilities)

    def predict(self, texts: Sequence[Any], *, ids: Sequence[str] | None = None) -> pl.DataFrame:
        """Classifica um lote de amostras, retornando um DataFrame de predições validado.

        Parameters
        ----------
        texts : Sequence[Any]
            Amostras de entrada, no formato esperado pelo modelo.
        ids : Sequence[str] | None, optional
            Identificadores das amostras, mesmo tamanho de ``texts``, by
            default None (gera identificadores sequenciais ``"0"``,
            ``"1"``, ...).

        Returns
        -------
        pl.DataFrame
            DataFrame validado contra :class:`schemas.prediction.PredictionSchema`.

        Raises
        ------
        EmptyDatasetError
            Se ``texts`` estiver vazio.
        """
        validate_not_empty_collection(texts, collection_name="texts")
        resolved_ids = list(ids) if ids is not None else [str(index) for index in range(len(texts))]

        predicted_labels = [str(label) for label in self.model.predict(texts)]
        probabilities_matrix = np.asarray(self.model.predict_proba(texts))
        confidences = [
            self._build_record(label, row)["confidence"]
            for label, row in zip(predicted_labels, probabilities_matrix, strict=True)
        ]
        return build_prediction_dataframe(
            resolved_ids, [str(text) for text in texts], predicted_labels, confidences
        )

    def predict_proba(self, texts: Sequence[Any]) -> np.ndarray:
        """Estima a distribuição de probabilidade por classe para um lote de amostras.

        Parameters
        ----------
        texts : Sequence[Any]
            Amostras de entrada, no formato esperado pelo modelo.

        Returns
        -------
        np.ndarray
            Matriz ``(len(texts), n_classes)`` de probabilidades, na ordem
            de :attr:`_classes`.
        """
        return np.asarray(self.model.predict_proba(texts))
