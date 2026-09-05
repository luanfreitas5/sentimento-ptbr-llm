"""Inferência em lote concorrente para classificadores LLM locais.

Implementa a Fase 12: diferente de ``src/inference/batch.py`` (blocos
sequenciais, adequados a modelos vetorizados em CPU/GPU), a geração de
texto por um LLM local (Ollama/Hugging Face, via ``src/llm/``) é ligada a
I/O — cada chamada aguarda o backend responder. Este módulo distribui essas
chamadas entre múltiplas threads via
``src/parallel/inference.py``, isolando a falha de um texto sem
interromper o restante do lote.
"""

import logging
import operator
from collections.abc import Sequence
from typing import Any

import polars as pl

from inference.postprocessing import build_prediction_dataframe
from inference.predictor import Predictor
from parallel.inference import run_parallel_predictions
from utils.validation import validate_not_empty_collection

logger = logging.getLogger(__name__)

_IndexedItem = tuple[int, tuple[str, str]]
_IndexedPrediction = tuple[int, str, dict[str, Any]]


def _predict_indexed_item(predictor: Predictor, item: _IndexedItem) -> _IndexedPrediction:
    """Classifica um item indexado, preservando sua posição original e identificador.

    Parameters
    ----------
    predictor : inference.predictor.Predictor
        Interface de inferência sobre o classificador LLM.
    item : tuple[int, tuple[str, str]]
        Par ``(índice_original, (identificador, texto))``.

    Returns
    -------
    tuple[int, str, dict[str, Any]]
        Tripla ``(índice_original, identificador, registro_de_predição)``.
    """
    original_index, (item_id, text) = item
    return original_index, item_id, predictor.predict_one_from_features([text])


def run_llm_batch_inference(
    predictor: Predictor,
    texts: Sequence[str],
    *,
    ids: Sequence[str] | None = None,
    max_workers: int | None = None,
    show_progress: bool = True,
) -> pl.DataFrame:
    """Executa a inferência LLM em lote, distribuindo as chamadas entre múltiplas threads.

    Falhas em textos individuais (ex.: uma resposta de LLM que esgota as
    retentativas de parsing, uma falha transitória de rede ao servidor
    Ollama) são isoladas e registradas em log, sem interromper o
    processamento dos demais textos do lote (ver
    ``src/parallel/core.py``).

    Parameters
    ----------
    predictor : inference.predictor.Predictor
        Interface de inferência sobre um classificador LLM já ajustado
        (``src/llm/classifier.py`` ou ``src/models/llm.py``).
    texts : Sequence[str]
        Textos a classificar. Não vazio.
    ids : Sequence[str] | None, optional
        Identificadores dos textos, mesmo tamanho de ``texts``, by default
        None (gera identificadores sequenciais).
    max_workers : int | None, optional
        Número máximo de threads usadas, by default None (o executor
        escolhe automaticamente).
    show_progress : bool, optional
        Se ``True``, exibe uma barra de progresso no console, by default
        True.

    Returns
    -------
    pl.DataFrame
        DataFrame de predições validado contra
        :class:`schemas.prediction.PredictionSchema`, contendo apenas os
        textos classificados com sucesso, na ordem original de ``texts``.

    Raises
    ------
    EmptyDatasetError
        Se ``texts`` estiver vazio.

    Examples
    --------
    >>> run_llm_batch_inference(predictor, ["ótimo", "péssimo"])  # doctest: +SKIP
    """
    validate_not_empty_collection(texts, collection_name="texts")
    resolved_ids = list(ids) if ids is not None else [str(index) for index in range(len(texts))]
    indexed_items: list[_IndexedItem] = list(enumerate(zip(resolved_ids, texts, strict=True)))

    result = run_parallel_predictions(
        lambda item: _predict_indexed_item(predictor, item),
        indexed_items,
        max_workers=max_workers,
        show_progress=show_progress,
    )
    if result.failures:
        logger.warning(
            "%d/%d texto(s) falharam durante a inferência LLM em lote.",
            len(result.failures),
            len(texts),
        )

    ordered_successes = sorted(result.successes, key=operator.itemgetter(0))
    output_ids = [item_id for _, item_id, _ in ordered_successes]
    output_texts = [texts[original_index] for original_index, _, _ in ordered_successes]
    output_labels = [record["sentiment_label"] for _, _, record in ordered_successes]
    output_confidences = [record["confidence"] for _, _, record in ordered_successes]

    return build_prediction_dataframe(output_ids, output_texts, output_labels, output_confidences)
