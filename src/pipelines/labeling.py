"""Rotulagem semiautomática em cascata do corpus normalizado.

Implementa o estágio ``labeling`` de ``configs/config.yaml -> stages``:
executa a cascata de rotuladores (``src/labeling/automatic.py``) sobre o
corpus normalizado, agrega os candidatos por votação majoritária ponderada
(``src/labeling/consensus.py``), sinaliza e amostra candidatos à validação
humana (``src/labeling/manual.py``), incorpora rótulos humanos e/ou valida
contra um gold set de referência (``src/labeling/validation.py``) quando
informados, e grava o corpus rotulado final (``paths.labeled_corpus_file``).
"""

import logging
from collections.abc import Mapping
from pathlib import Path

import polars as pl

from config.paths import ProjectPaths
from data.loader import read_dataset_file
from data.writer import write_labeled_corpus
from io_utils.csv import write_csv
from labeling.automatic import SentimentLabeler, run_cascade_labeling
from labeling.confidence import calculate_discordance_score, flag_low_confidence_samples
from labeling.consensus import aggregate_by_weighted_majority_vote, merge_consensus_into_corpus
from labeling.manual import apply_human_validation_labels, select_samples_for_human_validation
from labeling.validation import evaluate_against_gold_set

logger = logging.getLogger(__name__)

_HUMAN_VALIDATION_SAMPLE_FILE_NAME = "human_validation_sample.csv"


def run_labeling_stage(
    paths: ProjectPaths,
    labelers: Mapping[str, SentimentLabeler],
    *,
    text_column: str = "text_normalized",
    weights: Mapping[str, float] | None = None,
    select_for_human_validation: bool = True,
    human_validation_sample_size: int = 500,
    human_validation_labels: pl.DataFrame | None = None,
    gold_set: pl.DataFrame | None = None,
    minimum_kappa: float = 0.6,
) -> Path:
    """Executa a etapa de rotulagem em cascata sobre o corpus normalizado.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto (``configs/paths.yaml``).
    labelers : Mapping[str, SentimentLabeler]
        Rotuladores da cascata, nomeados pela chave (ver
        ``configs/labeling.yaml -> cascade.labelers``), repassados a
        :func:`labeling.automatic.run_cascade_labeling`.
    text_column : str, optional
        Coluna de texto classificada pelos rotuladores, by default
        "text_normalized" (produzida por
        ``src/pipelines/preprocessing.py``).
    weights : Mapping[str, float] | None, optional
        Peso de cada rotulador na agregação, by default None (peso 1.0
        para todos).
    select_for_human_validation : bool, optional
        Se ``True``, seleciona e grava uma amostra de candidatos à
        validação humana em ``paths.reports_tables_dir``, by default True.
    human_validation_sample_size : int, optional
        Repassado a
        :func:`labeling.manual.select_samples_for_human_validation`, by
        default 500.
    human_validation_labels : pl.DataFrame | None, optional
        Rótulos já revisados por humanos, incorporados via
        :func:`labeling.manual.apply_human_validation_labels`, by default
        None (nenhuma incorporação).
    gold_set : pl.DataFrame | None, optional
        Gold set de referência (TweetSentBR/RePro) para validação via
        :func:`labeling.validation.evaluate_against_gold_set`, by default
        None (nenhuma validação).
    minimum_kappa : float, optional
        Repassado a :func:`labeling.validation.evaluate_against_gold_set`,
        by default 0.6.

    Returns
    -------
    Path
        Caminho do corpus rotulado escrito (``paths.labeled_corpus_file``).

    Raises
    ------
    EmptyDatasetError
        Se o corpus normalizado ou ``labelers`` estiverem vazios.
    DataValidationError
        Se o corpus rotulado final violar o contrato de dados.

    Examples
    --------
    >>> from labeling.automatic import LexicalHeuristicLabeler
    >>> run_labeling_stage(
    ...     paths, {"heuristica_lexica": LexicalHeuristicLabeler()}
    ... )  # doctest: +SKIP
    """
    normalized_corpus = read_dataset_file(paths.normalized_corpus_file)

    labeling_results = run_cascade_labeling(
        normalized_corpus, labelers, text_column=text_column, weights=weights
    )
    consensus = aggregate_by_weighted_majority_vote(labeling_results)
    labeled_corpus = merge_consensus_into_corpus(normalized_corpus, consensus)

    if select_for_human_validation:
        flagged = flag_low_confidence_samples(calculate_discordance_score(labeling_results))
        if flagged.filter(pl.col("requires_human_validation")).height > 0:
            human_validation_sample = select_samples_for_human_validation(
                flagged, sample_size=human_validation_sample_size
            )
            write_csv(
                human_validation_sample,
                paths.reports_tables_dir / _HUMAN_VALIDATION_SAMPLE_FILE_NAME,
            )
        else:
            logger.info("Nenhuma amostra sinalizada para validação humana nesta execução.")

    if human_validation_labels is not None:
        labeled_corpus = apply_human_validation_labels(labeled_corpus, human_validation_labels)

    if gold_set is not None:
        validation_result = evaluate_against_gold_set(
            labeled_corpus, gold_set, minimum_kappa=minimum_kappa
        )
        if not validation_result.meets_minimum_agreement:
            logger.warning(
                "Concordância com o gold set (kappa=%.4f, n=%d) abaixo do limiar mínimo (%.2f).",
                validation_result.cohen_kappa,
                validation_result.n_samples,
                minimum_kappa,
            )

    write_labeled_corpus(labeled_corpus, paths.labeled_corpus_file)
    logger.info("Etapa de rotulagem concluída: %d amostra(s) rotulada(s).", labeled_corpus.height)
    return paths.labeled_corpus_file
