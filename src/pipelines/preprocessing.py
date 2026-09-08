"""Normalização e limpeza do corpus bruto de tweets.

Implementa o estágio ``preprocessing`` de ``configs/config.yaml -> stages``:
carrega em paralelo o lote de tweets brutos coletados por usuário
(``data/raw/*.parquet`` — ver ``src/data/loader.py``), filtra por metadados
da coleta (retweet/idioma — ``src/preprocessing/filtering.py``), aplica o
pipeline de normalização/limpeza de ``src/preprocessing/pipeline.py`` e
grava o resultado no corpus normalizado (``paths.normalized_corpus_file``).
"""

import logging
from pathlib import Path
from typing import Any

from config.paths import ProjectPaths
from data.loader import load_raw_tweet_batch
from data.writer import write_dataset
from preprocessing.filtering import filter_by_raw_metadata
from preprocessing.pipeline import run_preprocessing_pipeline

logger = logging.getLogger(__name__)


def run_preprocessing_stage(
    paths: ProjectPaths,
    *,
    max_workers: int | None = None,
    show_progress: bool = True,
    exclude_retweets: bool = True,
    required_language: str | None = "pt",
    **preprocessing_overrides: Any,
) -> Path:
    """Executa a etapa de pré-processamento sobre o lote de tweets brutos.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto (``configs/paths.yaml``).
    max_workers : int | None, optional
        Repassado ao carregamento em lote
        (:func:`data.loader.load_raw_tweet_batch`) e à normalização
        paralela (:func:`preprocessing.pipeline.run_preprocessing_pipeline`),
        by default None (o executor escolhe automaticamente).
    show_progress : bool, optional
        Se ``True``, exibe barras de progresso no console, by default True.
    exclude_retweets : bool, optional
        Repassado a :func:`preprocessing.filtering.filter_by_raw_metadata`,
        by default True.
    required_language : str | None, optional
        Repassado a :func:`preprocessing.filtering.filter_by_raw_metadata`,
        by default "pt".
    **preprocessing_overrides : Any
        Hiperparâmetros repassados a
        :func:`preprocessing.pipeline.run_preprocessing_pipeline` (ex.:
        ``apply_inclusion_filters``, ``tokens_column``).

    Returns
    -------
    Path
        Caminho do corpus normalizado escrito (``paths.normalized_corpus_file``).

    Raises
    ------
    EmptyDatasetError
        Se ``data/raw/`` não contiver arquivos, ou se o corpus ficar vazio
        após os filtros.
    PipelineStageError
        Se a normalização de algum texto do corpus falhar.

    Examples
    --------
    >>> run_preprocessing_stage(paths)  # doctest: +SKIP
    """
    raw_batch = load_raw_tweet_batch(
        paths.data_raw_dir, max_workers=max_workers, show_progress=show_progress
    )
    filtered_batch = filter_by_raw_metadata(
        raw_batch, exclude_retweets=exclude_retweets, required_language=required_language
    )
    normalized_corpus = run_preprocessing_pipeline(
        filtered_batch,
        max_workers=max_workers,
        show_progress=show_progress,
        **preprocessing_overrides,
    )
    write_dataset(normalized_corpus, paths.normalized_corpus_file)

    logger.info(
        "Etapa de pré-processamento concluída: %d/%d linha(s) mantida(s) (de %d brutas).",
        normalized_corpus.height,
        filtered_batch.height,
        raw_batch.height,
    )
    return paths.normalized_corpus_file
