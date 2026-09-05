"""Inferência em lote sobre classificadores treinados (ML clássico, DL, Transformer).

Implementa a Fase 12: percorre um conjunto de amostras em blocos
(``batch_size``), delegando a predição de cada bloco a
:class:`inference.predictor.Predictor`, com uma barra de progresso
``rich`` (ver ``CLAUDE.md``, "Progress Bars"). Voltado a modelos cuja
inferência é rápida o bastante para não precisar de paralelismo (ver
``src/inference/llm_batch.py`` para inferência concorrente de LLMs, cuja
geração é ligada a I/O).
"""

import logging
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any

import polars as pl
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from inference.predictor import Predictor
from utils.validation import validate_not_empty_collection

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 32


def _build_progress_bar() -> Progress:
    """Monta a barra de progresso padrão do projeto para a inferência em lote.

    Returns
    -------
    Progress
        Instância de ``rich.progress.Progress`` configurada com as colunas
        padrão definidas em ``CLAUDE.md`` (spinner, descrição, barra,
        contagem, percentual e tempos decorrido/restante).
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )


def run_batch_inference(
    predictor: Predictor,
    texts: Sequence[Any],
    *,
    ids: Sequence[str] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    show_progress: bool = True,
) -> pl.DataFrame:
    """Executa a inferência em lote sobre um conjunto de amostras, em blocos.

    Parameters
    ----------
    predictor : inference.predictor.Predictor
        Interface de inferência sobre um modelo já treinado.
    texts : Sequence[Any]
        Amostras de entrada, no formato esperado pelo modelo. Não vazio.
    ids : Sequence[str] | None, optional
        Identificadores das amostras, mesmo tamanho de ``texts``, by
        default None (gera identificadores sequenciais).
    batch_size : int, optional
        Número de amostras processadas por bloco, by default
        :data:`DEFAULT_BATCH_SIZE`.
    show_progress : bool, optional
        Se ``True``, exibe uma barra de progresso no console, by default
        True.

    Returns
    -------
    pl.DataFrame
        DataFrame de predições validado contra
        :class:`schemas.prediction.PredictionSchema`, com uma linha por
        amostra de ``texts``.

    Raises
    ------
    EmptyDatasetError
        Se ``texts`` estiver vazio.
    ValueError
        Se ``batch_size`` for menor que 1.

    Examples
    --------
    >>> run_batch_inference(predictor, ["ótimo", "péssimo"])  # doctest: +SKIP
    """
    validate_not_empty_collection(texts, collection_name="texts")
    if batch_size < 1:
        raise ValueError(f"batch_size deve ser >= 1, recebido: {batch_size}")

    resolved_ids = list(ids) if ids is not None else [str(index) for index in range(len(texts))]

    progress = _build_progress_bar() if show_progress else None
    progress_context = progress if progress is not None else nullcontext()

    frames: list[pl.DataFrame] = []
    with progress_context:
        task_id = progress.add_task("Inferência em lote", total=len(texts)) if progress else None
        for start in range(0, len(texts), batch_size):
            end = start + batch_size
            batch_texts = texts[start:end]
            batch_ids = resolved_ids[start:end]
            frames.append(predictor.predict(batch_texts, ids=batch_ids))
            if progress is not None and task_id is not None:
                progress.update(task_id, advance=len(batch_texts))

    result = pl.concat(frames)
    logger.info("Inferência em lote concluída: %d amostra(s).", result.height)
    return result
