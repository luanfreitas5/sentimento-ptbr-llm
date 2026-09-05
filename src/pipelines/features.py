"""Split estratificado e extração de features do corpus rotulado.

Implementa o estágio ``features`` de ``configs/config.yaml -> stages``:
particiona o corpus rotulado em treino/validação/teste
(``src/data/splitter.py``), grava os três conjuntos validados contra
:class:`schemas.training.TrainingExampleSchema` e calcula a matriz de
features TF-IDF (``src/features/lexical.py``) do conjunto de treino.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from config.paths import ProjectPaths
from constants.defaults import DEFAULT_RANDOM_SEED, DEFAULT_TEST_SIZE, DEFAULT_VALIDATION_SIZE
from data.loader import load_labeled_corpus
from data.splitter import create_stratified_split
from data.writer import write_dataset, write_training_example_dataset
from features.lexical import compute_tfidf_features

logger = logging.getLogger(__name__)

_TRAINING_EXAMPLE_COLUMNS: tuple[str, ...] = ("id", "text", "sentiment_label", "split")
_TFIDF_FEATURES_FILE_NAME = "tfidf_features.parquet"


@dataclass(frozen=True)
class FeatureArtifacts:
    """Caminhos dos artefatos produzidos pela etapa de extração de features.

    Parameters
    ----------
    training_corpus_path : Path
        Caminho do conjunto de treino particionado.
    validation_corpus_path : Path
        Caminho do conjunto de validação particionado.
    test_corpus_path : Path
        Caminho do conjunto de teste particionado.
    tfidf_features_path : Path
        Caminho da matriz de features TF-IDF (formato longo) do conjunto de
        treino.
    """

    training_corpus_path: Path
    validation_corpus_path: Path
    test_corpus_path: Path
    tfidf_features_path: Path


def run_features_stage(
    paths: ProjectPaths,
    *,
    label_column: str = "sentiment_label",
    text_column: str = "text",
    tfidf_overrides: dict[str, Any] | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
    validation_size: float = DEFAULT_VALIDATION_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> FeatureArtifacts:
    """Executa o split estratificado do corpus rotulado e a extração de features TF-IDF.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto (``configs/paths.yaml``).
    label_column : str, optional
        Coluna usada para estratificação do split, repassada a
        :func:`data.splitter.create_stratified_split`, by default
        "sentiment_label".
    text_column : str, optional
        Coluna de texto usada no cálculo do TF-IDF (deve estar tokenizada,
        com termos separados por espaço), repassada a
        :func:`features.lexical.compute_tfidf_features`, by default "text".
    tfidf_overrides : dict[str, Any] | None, optional
        Hiperparâmetros adicionais repassados a
        :func:`features.lexical.compute_tfidf_features` (ex.:
        ``ngram_range``, ``max_features``), by default None.
    test_size : float, optional
        Repassado a :func:`data.splitter.create_stratified_split`, by
        default :data:`constants.defaults.DEFAULT_TEST_SIZE`.
    validation_size : float, optional
        Repassado a :func:`data.splitter.create_stratified_split`, by
        default :data:`constants.defaults.DEFAULT_VALIDATION_SIZE`.
    random_seed : int, optional
        Repassado a :func:`data.splitter.create_stratified_split`, by
        default :data:`constants.defaults.DEFAULT_RANDOM_SEED`.

    Returns
    -------
    FeatureArtifacts
        Caminhos dos três conjuntos particionados e da matriz de features
        TF-IDF do conjunto de treino.

    Raises
    ------
    EmptyDatasetError
        Se o corpus rotulado, ou algum dos conjuntos particionados,
        estiver vazio.

    Examples
    --------
    >>> run_features_stage(paths)  # doctest: +SKIP
    """
    labeled_corpus = load_labeled_corpus(paths.labeled_corpus_file)
    split_corpus = create_stratified_split(
        labeled_corpus,
        label_column=label_column,
        test_size=test_size,
        validation_size=validation_size,
        random_seed=random_seed,
    ).select(list(_TRAINING_EXAMPLE_COLUMNS))

    training_split = split_corpus.filter(pl.col("split") == "treino")
    validation_split = split_corpus.filter(pl.col("split") == "validacao")
    test_split = split_corpus.filter(pl.col("split") == "teste")

    write_training_example_dataset(training_split, paths.training_corpus_file)
    write_training_example_dataset(validation_split, paths.validation_corpus_file)
    write_training_example_dataset(test_split, paths.test_corpus_file)

    tfidf_features = compute_tfidf_features(
        training_split, text_column=text_column, **(tfidf_overrides or {})
    )
    tfidf_features_path = paths.data_processed_dir / _TFIDF_FEATURES_FILE_NAME
    write_dataset(tfidf_features, tfidf_features_path)

    logger.info(
        "Etapa de features concluída: %d treino, %d validação, %d teste, %d peso(s) TF-IDF.",
        training_split.height,
        validation_split.height,
        test_split.height,
        tfidf_features.height,
    )
    return FeatureArtifacts(
        training_corpus_path=paths.training_corpus_file,
        validation_corpus_path=paths.validation_corpus_file,
        test_corpus_path=paths.test_corpus_file,
        tfidf_features_path=tfidf_features_path,
    )
