"""Execução incremental e retomável da rotulagem de um corpus por LLM.

Independente do provedor: recebe uma função que classifica um lote de textos
(:data:`BatchClassifier`) — implementada pelo rotulador OpenAI
(``src/labeling/openai_labeler.py``) e pelo do Hugging Face
(``src/labeling/huggingface.py``) — e cuida de tudo que é comum: pular
tweets já rotulados (checkpoint), gravar cada lote assim que concluído,
exibir progresso e falhar de forma explícita se sobrarem tweets sem rótulo.
"""

import logging
from collections.abc import Callable, Sequence
from pathlib import Path

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

from exceptions.data import DataValidationError
from exceptions.pipeline import IncompleteLabelingError
from labeling.checkpoint import append_labeling_checkpoint, read_labeling_checkpoint
from utils.validation import validate_not_empty_collection

logger = logging.getLogger(__name__)

LabelPrediction = tuple[str, float]
BatchClassifier = Callable[[Sequence[str]], Sequence[LabelPrediction | None]]


def _build_progress(*, disable: bool) -> Progress:
    """Barra de progresso padrão do projeto (ver CLAUDE.md, "Progress Bars")."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        disable=disable,
    )


def _validate_inputs(corpus: pl.DataFrame, *, id_column: str, batch_size: int) -> None:
    """Valida corpus não vazio, ``batch_size`` positivo e ids únicos (chave do checkpoint)."""
    validate_not_empty_collection(corpus, collection_name="corpus")
    if batch_size < 1:
        raise DataValidationError(
            schema_name="labeling_batch_size",
            detail=f"batch_size deve ser >= 1, recebido {batch_size}",
        )
    if corpus[id_column].n_unique() != corpus.height:
        raise DataValidationError(
            schema_name="labeling_corpus",
            detail=f"a coluna '{id_column}' possui valores duplicados",
        )


def _classify_and_store_batch(
    texts: Sequence[str],
    batch_ids: Sequence[str],
    classify_batch: BatchClassifier,
    checkpoint_path: Path,
) -> dict[str, LabelPrediction]:
    """Classifica um lote e grava no checkpoint os tweets com rótulo válido.

    Returns
    -------
    dict[str, LabelPrediction]
        Rótulos válidos do lote (``id -> (rótulo, confiança)``); os que falharam ficam de fora.

    Raises
    ------
    DataValidationError
        Se o classificador devolver um número de resultados diferente do lote.
    """
    predictions = classify_batch(texts)
    if len(predictions) != len(texts):
        raise DataValidationError(
            schema_name="labeling_batch_size",
            detail=(
                f"o classificador devolveu {len(predictions)} resultado(s) para "
                f"{len(texts)} texto(s)"
            ),
        )
    batch_results = {
        tweet_id: prediction
        for tweet_id, prediction in zip(batch_ids, predictions, strict=True)
        if prediction is not None
    }
    append_labeling_checkpoint(checkpoint_path, batch_results)
    return batch_results


def _label_pending_batches(
    pending: Sequence[int],
    ids: Sequence[str],
    texts: Sequence[str],
    labeled: dict[str, LabelPrediction],
    classify_batch: BatchClassifier,
    *,
    checkpoint_path: Path,
    batch_size: int,
    progress_description: str,
    show_progress: bool,
) -> int:
    """Classifica, em lotes, os tweets pendentes, atualizando ``labeled`` e o checkpoint.

    Returns
    -------
    int
        Quantidade de tweets que continuam sem rótulo válido.
    """
    n_failed = 0
    with _build_progress(disable=not show_progress or not pending) as progress:
        task_id = progress.add_task(progress_description, total=len(pending))
        for start in range(0, len(pending), batch_size):
            batch_indices = pending[start : start + batch_size]
            batch_results = _classify_and_store_batch(
                [texts[index] for index in batch_indices],
                [ids[index] for index in batch_indices],
                classify_batch,
                checkpoint_path,
            )
            labeled.update(batch_results)
            n_failed += len(batch_indices) - len(batch_results)
            progress.advance(task_id, len(batch_indices))
    return n_failed


def run_incremental_labeling(
    corpus: pl.DataFrame,
    classify_batch: BatchClassifier,
    *,
    source_name: str,
    checkpoint_path: Path,
    batch_size: int,
    id_column: str = "id",
    text_column: str = "text_normalized",
    show_progress: bool = True,
) -> pl.DataFrame:
    """Rotula todos os tweets do corpus, retomando de onde parou.

    Parameters
    ----------
    corpus : pl.DataFrame
        Corpus a rotular, com ``id_column`` (único) e ``text_column``. Não vazio.
    classify_batch : BatchClassifier
        Recebe um lote de textos e devolve, na mesma ordem, ``(rótulo, confiança)``
        ou ``None`` para o tweet que falhou após todas as tentativas.
    source_name : str
        Nome da fonte (``huggingface``/``openai``), usado em logs e erros.
    checkpoint_path : Path
        Checkpoint JSON Lines (ver :mod:`labeling.checkpoint`).
    batch_size : int
        Tweets por lote; cada lote é gravado no checkpoint ao terminar.
    id_column : str, optional
        Coluna identificadora, by default "id".
    text_column : str, optional
        Coluna do texto classificado, by default "text_normalized".
    show_progress : bool, optional
        Se ``True``, exibe barra de progresso, by default True.

    Returns
    -------
    pl.DataFrame
        Colunas ``id_column``, ``sentiment_label`` e ``confidence_score``, uma linha por
        tweet de ``corpus``, na mesma ordem.

    Raises
    ------
    EmptyDatasetError
        Se ``corpus`` estiver vazio.
    DataValidationError
        Se ``id_column`` tiver valores duplicados, ``batch_size`` for inválido ou o
        classificador devolver um número de resultados diferente do lote.
    IncompleteLabelingError
        Se restarem tweets sem rótulo válido; os já rotulados ficam no checkpoint.

    Examples
    --------
    >>> corpus = pl.DataFrame({"id": ["1"], "text_normalized": ["adorei"]})
    >>> run_incremental_labeling(  # doctest: +SKIP
    ...     corpus,
    ...     lambda textos: [("positivo", 0.9)] * len(textos),
    ...     source_name="teste",
    ...     checkpoint_path=Path("ckpt.jsonl"),
    ...     batch_size=8,
    ... )
    """
    _validate_inputs(corpus, id_column=id_column, batch_size=batch_size)

    ids = [str(value) for value in corpus[id_column].to_list()]
    texts = corpus[text_column].to_list()
    labeled = read_labeling_checkpoint(checkpoint_path)
    pending = [index for index, tweet_id in enumerate(ids) if tweet_id not in labeled]
    logger.info(
        "Rotulagem '%s': %d tweet(s) no corpus, %d já rotulado(s) no checkpoint, %d pendente(s).",
        source_name,
        len(ids),
        len(ids) - len(pending),
        len(pending),
    )

    n_failed = _label_pending_batches(
        pending,
        ids,
        texts,
        labeled,
        classify_batch,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        progress_description=f"Rotulando ({source_name})",
        show_progress=show_progress,
    )

    if n_failed:
        logger.error(
            "Rotulagem '%s': %d tweet(s) sem rótulo válido após as tentativas.",
            source_name,
            n_failed,
        )
        raise IncompleteLabelingError(source_name, n_failed, str(checkpoint_path))

    logger.info("Rotulagem '%s' concluída: %d tweet(s).", source_name, len(ids))
    return pl.DataFrame(
        {
            id_column: ids,
            "sentiment_label": [labeled[tweet_id][0] for tweet_id in ids],
            "confidence_score": [round(labeled[tweet_id][1], 4) for tweet_id in ids],
        }
    )
