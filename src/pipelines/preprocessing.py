"""Normalização e limpeza do corpus bruto de tweets.

Implementa o estágio ``preprocessing`` de ``configs/config.yaml -> stages``:
carrega o corpus bruto (``paths.raw_tweets_file``), aplica o pipeline de
normalização/limpeza de ``src/preprocessing/pipeline.py`` e grava o
resultado no corpus normalizado (``paths.normalized_corpus_file``).
"""

import logging
from pathlib import Path
from typing import Any

from config.paths import ProjectPaths
from data.loader import load_raw_tweet_dataset
from data.writer import write_dataset
from preprocessing.pipeline import run_preprocessing_pipeline

logger = logging.getLogger(__name__)


def run_preprocessing_stage(paths: ProjectPaths, **preprocessing_overrides: Any) -> Path:
    """Executa a etapa de pré-processamento sobre o corpus bruto de tweets.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto (``configs/paths.yaml``).
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
        Se o corpus bruto estiver vazio.
    PipelineStageError
        Se a normalização de algum texto do corpus falhar.

    Examples
    --------
    >>> run_preprocessing_stage(paths)  # doctest: +SKIP
    """
    raw_corpus = load_raw_tweet_dataset(paths.raw_tweets_file)
    normalized_corpus = run_preprocessing_pipeline(raw_corpus, **preprocessing_overrides)
    write_dataset(normalized_corpus, paths.normalized_corpus_file)

    logger.info(
        "Etapa de pré-processamento concluída: %d/%d linha(s) mantida(s).",
        normalized_corpus.height,
        raw_corpus.height,
    )
    return paths.normalized_corpus_file
