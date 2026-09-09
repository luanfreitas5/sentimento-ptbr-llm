"""Descoberta de padrões e identificação de inconsistências via HypotheSAEs.

Implementa o estágio ``hypothesaes_analysis`` de ``configs/config.yaml ->
stages``: treina um Sparse Autoencoder sobre embeddings do corpus rotulado
(``src/hypothesaes/``), interpreta uma amostra de neurônios para checagem de
sanidade (DESCOBERTA DE PADRÕES,
:func:`hypothesaes.quickstart.interpret_sae`) e seleciona/interpreta os
neurônios mais preditivos de rótulos de BAIXA confiança — a coluna
``confidence_score`` produzida pela etapa ``labeling``
(:func:`labeling.consensus.aggregate_by_weighted_majority_vote`) — como
candidatos a INCONSISTÊNCIA DE ROTULAGEM
(:func:`hypothesaes.quickstart.generate_hypotheses`). Ao final, consolida os
resultados em tabelas e um gráfico de barras divergente
(``evaluation.hypothesaes_report``/``visualization.hypothesaes``), salvos em
``paths.reports_interpretability_dir``/``paths.reports_figures_dir``.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from config.paths import ProjectPaths
from constants.defaults import DEFAULT_RANDOM_SEED
from data.splitter import create_stratified_split
from evaluation.hypothesaes_report import build_top_hypotheses_table, save_top_hypotheses_table
from exceptions.data import DataValidationError, EmptyDatasetError
from hypothesaes.embedding import extract_local_embeddings
from hypothesaes.quickstart import (
    evaluate_hypotheses,
    generate_hypotheses,
    interpret_sae,
    train_sae,
)
from io_utils.csv import write_csv
from io_utils.json import write_json
from utils.timing import measure_execution_time
from utils.validation import validate_not_empty_collection
from visualization.hypothesaes import plot_hypotheses_bars
from visualization.theme import save_figure

logger = logging.getLogger(__name__)

_ID_COLUMN = "id"
_TEXT_COLUMN = "text"
_LABEL_COLUMN = "sentiment_label"
_CONFIDENCE_COLUMN = "confidence_score"
_SPLIT_COLUMN = "split"
_LOW_CONFIDENCE_COLUMN = "low_confidence"

_LOW_CONFIDENCE_TWEETS_FILE_NAME = "hypothesaes_tweets_baixa_confianca.csv"
_PATTERNS_FILE_NAME = "hypothesaes_descoberta_padroes.csv"
_HYPOTHESES_FILE_NAME = "hypothesaes_hipoteses_inconsistencia.csv"
_TOP_HYPOTHESES_FILE_NAME = "hypothesaes_top_hipoteses.csv"
_HOLDOUT_EVALUATION_FILE_NAME = "hypothesaes_avaliacao_holdout.csv"
_HOLDOUT_METRICS_FILE_NAME = "hypothesaes_avaliacao_metricas.json"
_SUMMARY_FILE_NAME = "hypothesaes_resumo_execucao.json"
_FIGURE_NAME = "hypothesaes_hipoteses"
_CHECKPOINTS_SUBDIR = "hypothesaes"


@dataclass(frozen=True)
class HypothesaesArtifacts:
    """Resultado consolidado do estágio ``hypothesaes_analysis``.

    Parameters
    ----------
    low_confidence_tweets : pl.DataFrame
        Tweets com ``confidence_score`` abaixo do limiar configurado, ordenados
        do menor para o maior score.
    patterns : pl.DataFrame
        Neurônios interpretados na descoberta de padrões (ver
        :func:`hypothesaes.quickstart.interpret_sae`).
    hypotheses : pl.DataFrame
        Hipóteses de inconsistência geradas (ver
        :func:`hypothesaes.quickstart.generate_hypotheses`), ordenadas por
        poder preditivo.
    top_hypotheses_table : pl.DataFrame
        Versão limpa/classificada de ``hypotheses`` (ver
        :func:`evaluation.hypothesaes_report.build_top_hypotheses_table`),
        usada no gráfico salvo.
    holdout_metrics : dict[str, Any] | None
        Métricas agregadas da avaliação em holdout
        (:func:`hypothesaes.quickstart.evaluate_hypotheses`), quando
        habilitada; ``None`` caso contrário.
    figure_paths : tuple[Path, Path]
        Caminhos ``.png``/``.svg`` do gráfico de barras salvo (ver
        :func:`visualization.theme.save_figure`).
    summary_path : Path
        Caminho do resumo JSON da execução.
    """

    low_confidence_tweets: pl.DataFrame
    patterns: pl.DataFrame
    hypotheses: pl.DataFrame
    top_hypotheses_table: pl.DataFrame
    holdout_metrics: dict[str, Any] | None
    figure_paths: tuple[Path, Path]
    summary_path: Path


def _prepare_corpus(
    labeled_corpus: pl.DataFrame,
    *,
    text_column: str,
    confidence_column: str,
    score_threshold: float,
) -> pl.DataFrame:
    """Deduplica por texto, remove textos vazios e sinaliza rótulos de baixa confiança."""
    if confidence_column not in labeled_corpus.columns:
        raise DataValidationError(
            schema_name="labeled_corpus",
            detail=(
                f"coluna de confiança '{confidence_column}' ausente; esperada em "
                "paths.labeled_corpus_file, produzida pela etapa 'labeling' "
                "(ver labeling.consensus.aggregate_by_weighted_majority_vote)"
            ),
        )

    return (
        labeled_corpus.filter(pl.col(text_column).str.strip_chars() != "")
        .unique(subset=[text_column], keep="first")
        .with_columns(
            (pl.col(confidence_column) < score_threshold)
            .cast(pl.Int8)
            .alias(_LOW_CONFIDENCE_COLUMN)
        )
    )


def _build_cache_name(
    *,
    selection_method: str,
    n_train: int,
    n_validation: int,
    m_total_neurons: int,
    k_active_neurons: int,
) -> str:
    """Monta um nome de cache/checkpoint estável, dependente do tamanho dos dados e do SAE."""
    return (
        f"hypothesaes_{selection_method}_{n_train}train_{n_validation}val_"
        f"{m_total_neurons}M_{k_active_neurons}K"
    )


def _build_dataset_summary(
    corpus: pl.DataFrame, *, label_column: str, score_threshold: float
) -> dict[str, Any]:
    """Resume a distribuição de rótulos de baixa confiança no corpus preparado."""
    low_confidence_by_label = (
        corpus.group_by(label_column)
        .agg(pl.col(_LOW_CONFIDENCE_COLUMN).mean().round(4).alias("frac_baixa_confianca"))
        .sort(label_column)
        .to_dicts()
    )
    n_low_confidence = int(corpus[_LOW_CONFIDENCE_COLUMN].sum())
    return {
        "n_tweets": corpus.height,
        "score_threshold": score_threshold,
        "n_baixa_confianca": n_low_confidence,
        "frac_baixa_confianca": round(n_low_confidence / corpus.height, 4),
        "baixa_confianca_por_sentimento": low_confidence_by_label,
        "contagem_por_sentimento": corpus[label_column].value_counts().to_dicts(),
    }


def run_hypothesaes_analysis_stage(
    labeled_corpus: pl.DataFrame,
    paths: ProjectPaths,
    *,
    id_column: str = _ID_COLUMN,
    text_column: str = _TEXT_COLUMN,
    label_column: str = _LABEL_COLUMN,
    confidence_column: str = _CONFIDENCE_COLUMN,
    score_threshold: float = 0.5,
    embedder_model_name: str = "neuralmind/bert-base-portuguese-cased",
    embedding_batch_size: int = 128,
    m_total_neurons: int = 256,
    k_active_neurons: int = 8,
    matryoshka_prefix_lengths: list[int] | None = None,
    n_random_neurons: int = 8,
    selection_method: str = "separation_score",
    n_selected_neurons: int = 15,
    n_scoring_examples: int = 100,
    interpreter_model: str = "llama3.1:latest",
    annotator_model: str = "llama3.1:latest",
    n_examples_for_interpretation: int = 12,
    max_words_per_example: int = 60,
    max_interpretation_tokens: int | None = 200,
    task_specific_instructions: str | None = None,
    n_workers: int = 8,
    evaluate_on_holdout: bool = False,
    holdout_size: float = 0.1,
    validation_size: float = 0.1,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> HypothesaesArtifacts:
    """Executa o fluxo completo do HypotheSAEs sobre o corpus rotulado.

    Parameters
    ----------
    labeled_corpus : pl.DataFrame
        Corpus rotulado (``paths.labeled_corpus_file``), contendo ao menos
        ``id_column``, ``text_column``, ``label_column`` e
        ``confidence_column``.
    paths : ProjectPaths
        Caminhos resolvidos do projeto (``configs/paths.yaml``).
    id_column : str, optional
        Coluna identificadora de cada tweet, by default "id".
    text_column : str, optional
        Coluna de texto, by default "text".
    label_column : str, optional
        Coluna de rótulo de sentimento (usada apenas para estratificar o
        split interno), by default "sentiment_label".
    confidence_column : str, optional
        Coluna de confiança do rótulo de consenso (ver
        :func:`labeling.consensus.aggregate_by_weighted_majority_vote`), by
        default "confidence_score".
    score_threshold : float, optional
        Limiar de baixa confiança (``confidence_column < score_threshold``),
        by default 0.5.
    embedder_model_name : str, optional
        Modelo ``sentence-transformers`` usado para calcular embeddings, by
        default "neuralmind/bert-base-portuguese-cased" (BERTimbau).
    embedding_batch_size : int, optional
        Tamanho do lote de codificação dos embeddings, by default 128.
    m_total_neurons : int, optional
        Número total de neurônios do SAE, by default 256.
    k_active_neurons : int, optional
        Número de neurônios ativos (top-K) por exemplo, by default 8.
    matryoshka_prefix_lengths : list[int] | None, optional
        Prefixos para a perda Matryoshka do SAE, by default None.
    n_random_neurons : int, optional
        Número de neurônios amostrados na descoberta de padrões, by
        default 8.
    selection_method : {"separation_score", "correlation", "lasso"}, optional
        Método de seleção dos neurônios mais preditivos de baixa confiança,
        by default "separation_score".
    n_selected_neurons : int, optional
        Número de neurônios selecionados e interpretados como hipóteses, by
        default 15.
    n_scoring_examples : int, optional
        Exemplos usados no fidelity-scoring das hipóteses (0 desliga), by
        default 100.
    interpreter_model : str, optional
        LLM usado para interpretar neurônios, by default "llama3.1:latest".
    annotator_model : str, optional
        LLM usado para pontuar/anotar hipóteses, by default
        "llama3.1:latest".
    n_examples_for_interpretation : int, optional
        Exemplos por prompt de interpretação, by default 12.
    max_words_per_example : int, optional
        Máximo de palavras por exemplo enviado ao LLM, by default 60.
    max_interpretation_tokens : int | None, optional
        Teto de tokens da interpretação gerada, by default 200.
    task_specific_instructions : str | None, optional
        Instruções específicas da tarefa injetadas no prompt de
        interpretação (``configs/hypothesaes.yaml -> llm.task_specific_instructions``),
        by default None.
    n_workers : int, optional
        Threads paralelas para chamadas ao LLM (interpretação/anotação), by
        default 8.
    evaluate_on_holdout : bool, optional
        Se avalia as hipóteses geradas em um conjunto de holdout via LLM
        (lento), by default False.
    holdout_size : float, optional
        Fração do corpus reservada para o holdout, quando
        ``evaluate_on_holdout`` é ``True``, by default 0.1.
    validation_size : float, optional
        Fração do corpus reservada para validação (early stopping do SAE),
        by default 0.1.
    random_seed : int, optional
        Semente do split treino/validação/holdout, by default
        :data:`constants.defaults.DEFAULT_RANDOM_SEED`.

    Returns
    -------
    HypothesaesArtifacts
        Tweets de baixa confiança, padrões descobertos, hipóteses geradas,
        tabela/figura consolidadas e (se habilitada) métricas de holdout.

    Raises
    ------
    EmptyDatasetError
        Se ``labeled_corpus`` estiver vazio.
    DataValidationError
        Se ``confidence_column`` não existir em ``labeled_corpus``.

    Examples
    --------
    >>> run_hypothesaes_analysis_stage(labeled_corpus, paths)  # doctest: +SKIP
    """
    validate_not_empty_collection(labeled_corpus, collection_name="labeled_corpus")
    corpus = _prepare_corpus(
        labeled_corpus,
        text_column=text_column,
        confidence_column=confidence_column,
        score_threshold=score_threshold,
    )

    low_confidence_tweets = (
        corpus.filter(pl.col(_LOW_CONFIDENCE_COLUMN) == 1)
        .select([id_column, text_column, label_column, confidence_column])
        .sort(confidence_column)
    )
    write_csv(
        low_confidence_tweets, paths.reports_interpretability_dir / _LOW_CONFIDENCE_TWEETS_FILE_NAME
    )
    logger.info(
        "Tweets de baixa confiança (confidence_score < %.2f): %d/%d.",
        score_threshold,
        low_confidence_tweets.height,
        corpus.height,
    )

    with measure_execution_time() as execution_timing:
        split_corpus = create_stratified_split(
            corpus,
            label_column=label_column,
            split_column=_SPLIT_COLUMN,
            test_size=holdout_size if evaluate_on_holdout else 0.0,
            validation_size=validation_size,
            random_seed=random_seed,
        )
        train_split = split_corpus.filter(pl.col(_SPLIT_COLUMN) == "treino")
        validation_split = split_corpus.filter(pl.col(_SPLIT_COLUMN) == "validacao")
        holdout_split = split_corpus.filter(pl.col(_SPLIT_COLUMN) == "teste")

        texts = train_split[text_column].to_list()
        low_confidence_labels = train_split[_LOW_CONFIDENCE_COLUMN].to_list()
        validation_texts = validation_split[text_column].to_list()
        holdout_texts = holdout_split[text_column].to_list() if evaluate_on_holdout else []

        cache_name = _build_cache_name(
            selection_method=selection_method,
            n_train=len(texts),
            n_validation=len(validation_texts),
            m_total_neurons=m_total_neurons,
            k_active_neurons=k_active_neurons,
        )
        logger.info(
            "Split -> treino: %d | validação: %d%s",
            len(texts),
            len(validation_texts),
            f" | holdout: {len(holdout_texts)}" if evaluate_on_holdout else "",
        )

        logger.info("Calculando embeddings ('%s')...", embedder_model_name)
        text_to_embedding = extract_local_embeddings(
            texts + validation_texts + holdout_texts,
            model=embedder_model_name,
            batch_size=embedding_batch_size,
            cache_name=cache_name,
        )
        train_embeddings = [text_to_embedding[text] for text in texts]
        validation_embeddings = [text_to_embedding[text] for text in validation_texts]

        logger.info("Treinando/carregando o Sparse Autoencoder...")
        sae = train_sae(
            embeddings=train_embeddings,
            m_total_neurons=m_total_neurons,
            k_active_neurons=k_active_neurons,
            matryoshka_prefix_lengths=matryoshka_prefix_lengths,
            val_embeddings=validation_embeddings,
            checkpoint_dir=paths.models_checkpoints_dir / _CHECKPOINTS_SUBDIR / cache_name,
        )

        logger.info("Descoberta de padrões: interpretando neurônios do SAE...")
        patterns = pl.from_pandas(
            interpret_sae(
                texts=texts,
                embeddings=train_embeddings,
                sae=sae,
                n_random_neurons=n_random_neurons,
                interpreter_model=interpreter_model,
                n_examples_for_interpretation=n_examples_for_interpretation,
                max_words_per_example=max_words_per_example,
                max_interpretation_tokens=max_interpretation_tokens,
                task_specific_instructions=task_specific_instructions,
            )
        )
        write_csv(patterns, paths.reports_interpretability_dir / _PATTERNS_FILE_NAME)

        logger.info("Identificação de inconsistências: gerando hipóteses...")
        target_column = f"target_{selection_method}"
        hypotheses = pl.from_pandas(
            generate_hypotheses(
                texts=texts,
                labels=low_confidence_labels,
                embeddings=train_embeddings,
                sae=sae,
                cache_name=cache_name,
                classification=True,
                selection_method=selection_method,
                n_selected_neurons=n_selected_neurons,
                interpreter_model=interpreter_model,
                annotator_model=annotator_model,
                n_examples_for_interpretation=n_examples_for_interpretation,
                max_words_per_example=max_words_per_example,
                max_interpretation_tokens=max_interpretation_tokens,
                n_scoring_examples=n_scoring_examples,
                n_workers_interpretation=n_workers,
                n_workers_annotation=n_workers,
                task_specific_instructions=task_specific_instructions,
            ).sort_values(by=target_column, ascending=False)
        )
        write_csv(hypotheses, paths.reports_interpretability_dir / _HYPOTHESES_FILE_NAME)

        top_hypotheses_table = build_top_hypotheses_table(hypotheses, target_column=target_column)
        save_top_hypotheses_table(
            top_hypotheses_table, paths.reports_interpretability_dir / _TOP_HYPOTHESES_FILE_NAME
        )
        figure = plot_hypotheses_bars(
            top_hypotheses_table,
            target_column=target_column,
            title=f"Hipóteses x baixa confiança — método '{selection_method}'",
        )
        figure_paths = save_figure(figure, _FIGURE_NAME, directory=paths.reports_figures_dir)

        holdout_metrics: dict[str, Any] | None = None
        if evaluate_on_holdout:
            if holdout_split.height == 0:
                raise EmptyDatasetError("holdout_split")
            logger.info("Avaliando hipóteses no holdout (via LLM)...")
            metrics, evaluation_df = evaluate_hypotheses(
                hypotheses_df=hypotheses.to_pandas(),
                texts=holdout_texts,
                labels=holdout_split[_LOW_CONFIDENCE_COLUMN].to_list(),
                cache_name=cache_name,
                annotator_model=annotator_model,
                classification=True,
                n_workers_annotation=n_workers,
            )
            write_csv(
                pl.from_pandas(evaluation_df),
                paths.reports_interpretability_dir / _HOLDOUT_EVALUATION_FILE_NAME,
            )
            holdout_metrics = {
                key: (list(value) if isinstance(value, tuple) else value)
                for key, value in metrics.items()
            }
            write_json(
                holdout_metrics, paths.reports_interpretability_dir / _HOLDOUT_METRICS_FILE_NAME
            )

    summary = _build_dataset_summary(
        corpus, label_column=label_column, score_threshold=score_threshold
    )
    summary["parametros"] = {
        "selection_method": selection_method,
        "m_total_neurons": m_total_neurons,
        "k_active_neurons": k_active_neurons,
        "embedder_model_name": embedder_model_name,
        "interpreter_model": interpreter_model,
        "annotator_model": annotator_model,
        "evaluate_on_holdout": evaluate_on_holdout,
    }
    summary["elapsed_seconds"] = round(execution_timing.elapsed_seconds, 1)
    summary_path = paths.reports_interpretability_dir / _SUMMARY_FILE_NAME
    write_json(summary, summary_path)

    logger.info(
        "Etapa 'hypothesaes_analysis' concluída em %.1fs: %d padrão(ões), %d hipótese(s).",
        execution_timing.elapsed_seconds,
        patterns.height,
        hypotheses.height,
    )
    return HypothesaesArtifacts(
        low_confidence_tweets=low_confidence_tweets,
        patterns=patterns,
        hypotheses=hypotheses,
        top_hypotheses_table=top_hypotheses_table,
        holdout_metrics=holdout_metrics,
        figure_paths=figure_paths,
        summary_path=summary_path,
    )
