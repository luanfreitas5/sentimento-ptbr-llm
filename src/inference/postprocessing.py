"""Padronização e parsing determinístico das saídas de inferência.

Implementa a Fase 12: normaliza rótulos/confianças brutos de qualquer
paradigma de modelo (ML clássico, DL, Transformer ou LLM) para o formato
comum validado por :mod:`schemas.prediction`, evitando que cada consumidor
de ``src/inference/`` reimplemente sua própria normalização.
"""

import logging
from collections.abc import Mapping, Sequence

import polars as pl
from itertools import starmap
from constants.labels import SENTIMENT_CLASSES
from schemas.prediction import validate_prediction
from utils.validation import validate_not_empty_collection

logger = logging.getLogger(__name__)


def standardize_prediction_output(
    label: str, confidence: float, *, allowed_labels: Sequence[str] = SENTIMENT_CLASSES
) -> tuple[str, float]:
    """Normaliza um par rótulo/confiança bruto para o formato padrão do projeto.

    Normaliza o rótulo para minúsculas sem espaços nas bordas e restringe a
    confiança ao intervalo ``[0.0, 1.0]``.

    Parameters
    ----------
    label : str
        Rótulo de sentimento bruto, possivelmente com variação de caixa
        e/ou espaços.
    confidence : float
        Confiança bruta associada ao rótulo.
    allowed_labels : Sequence[str], optional
        Classes de sentimento aceitas, by default
        :data:`constants.labels.SENTIMENT_CLASSES`.

    Returns
    -------
    tuple[str, float]
        Par ``(rótulo_normalizado, confiança_normalizada)``.

    Raises
    ------
    ValueError
        Se o rótulo normalizado não pertencer a ``allowed_labels``.

    Examples
    --------
    >>> standardize_prediction_output(" Positivo ", 1.5)
    ('positivo', 1.0)
    >>> standardize_prediction_output("NEGATIVO", -0.1)
    ('negativo', 0.0)
    """
    normalized_label = label.strip().lower()
    if normalized_label not in allowed_labels:
        raise ValueError(
            f"Rótulo '{normalized_label}' não pertence às classes conhecidas {allowed_labels}."
        )
    normalized_confidence = min(max(confidence, 0.0), 1.0)
    return normalized_label, normalized_confidence


def build_prediction_dataframe(
    ids: Sequence[str],
    texts: Sequence[str],
    labels: Sequence[str],
    confidences: Sequence[float],
    *,
    extra_columns: Mapping[str, Sequence[object]] | None = None,
) -> pl.DataFrame:
    """Monta e valida um DataFrame de predições no formato de :mod:`schemas.prediction`.

    Parameters
    ----------
    ids : Sequence[str]
        Identificadores das amostras.
    texts : Sequence[str]
        Textos classificados, mesmo tamanho de ``ids``.
    labels : Sequence[str]
        Rótulos de sentimento preditos (brutos ou já normalizados), mesmo
        tamanho de ``ids``.
    confidences : Sequence[float]
        Confianças associadas a cada predição, mesmo tamanho de ``ids``.
    extra_columns : Mapping[str, Sequence[object]] | None, optional
        Colunas adicionais (ex.: probabilidades por classe), mesmo tamanho
        de ``ids``, by default None.

    Returns
    -------
    pl.DataFrame
        DataFrame validado contra :class:`schemas.prediction.PredictionSchema`.

    Raises
    ------
    EmptyDatasetError
        Se ``ids`` estiver vazio.
    DataValidationError
        Se o resultado montado violar o contrato de dados.

    Examples
    --------
    >>> df = build_prediction_dataframe(["1"], ["ótimo produto"], ["positivo"], [0.9])
    >>> df["sentiment_label"].to_list()
    ['positivo']
    """
    validate_not_empty_collection(ids, collection_name="ids")

    normalized = list(starmap(standardize_prediction_output, zip(labels, confidences, strict=True)))
    normalized_labels, normalized_confidences = (
        zip(*normalized, strict=True) if normalized else ((), ())
    )

    data: dict[str, Sequence[object]] = {
        "id": ids,
        "text": texts,
        "sentiment_label": list(normalized_labels),
        "confidence": list(normalized_confidences),
    }
    if extra_columns:
        data.update(extra_columns)

    dataframe = pl.DataFrame(data)
    logger.debug("DataFrame de predições montado: %d linha(s).", dataframe.height)
    return validate_prediction(dataframe)
