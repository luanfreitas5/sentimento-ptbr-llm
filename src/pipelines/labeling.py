"""Rotulagem de sentimento do corpus normalizado via pipeline Hugging Face.

Implementa o estágio ``labeling`` de ``configs/config.yaml -> stages``:
classifica o corpus normalizado em lote com um único modelo do Hugging Face
Hub (``src/labeling/huggingface.py``), sinaliza e amostra candidatos de
baixa confiança à validação humana (``src/labeling/manual.py``), incorpora
rótulos humanos e/ou valida contra um gold set de referência
(``src/labeling/validation.py``) quando informados, re-rotula via LLM as
amostras de baixa confiança remanescentes
(``src/labeling/llm_relabeling.py``), e grava o corpus rotulado final
(``paths.labeled_corpus_file``).

O corpus rotulado final preserva, em colunas próprias, o rótulo e a
confiança de cada fonte da cascata — nenhuma etapa sobrescreve a saída de
outra: ``sentiment_label_huggingface``/``confidence_score_huggingface``
(modelo Hugging Face, ver ``src/labeling/consensus.py``),
``sentiment_label_llm_relabel``/``confidence_score_llm_relabel``
(re-rotulagem via LLM, nulas fora das amostras candidatas) e
``sentiment_label_manual``/``confidence_score_manual`` (validação humana,
nulas fora da amostra revisada). ``sentiment_label``/``confidence_score``
continuam sendo a coluna de trabalho — o rótulo final consumido pelas
etapas seguintes (``features``, ``hypothesaes_analysis``) —, atualizada em
cascata (Hugging Face -> LLM -> manual, sempre a fonte mais confiável
disponível por amostra).
"""

import logging
from collections.abc import Mapping
from pathlib import Path

import polars as pl

from config.paths import ProjectPaths
from data.loader import read_dataset_file
from data.writer import write_labeled_corpus
from hypothesaes.llm_api import DEFAULT_OLLAMA_BASE_URL, LLMProvider
from io_utils.csv import write_csv
from labeling.confidence import flag_low_confidence_predictions
from labeling.consensus import merge_consensus_into_corpus
from labeling.huggingface import SentimentPipeline, label_corpus_with_huggingface_pipeline
from labeling.llm_relabeling import relabel_low_confidence_samples
from labeling.manual import apply_human_validation_labels, select_samples_for_human_validation
from labeling.validation import evaluate_against_gold_set

logger = logging.getLogger(__name__)

_HUMAN_VALIDATION_SAMPLE_FILE_NAME = "human_validation_sample.csv"


def _select_and_write_human_validation_sample(
    labeled_corpus: pl.DataFrame,
    paths: ProjectPaths,
    *,
    sample_size: int,
    low_confidence_threshold: float,
) -> None:
    """Sinaliza candidatos de baixa confiança e grava a amostra de validação humana, se houver.

    Parameters
    ----------
    labeled_corpus : pl.DataFrame
        Corpus já rotulado, contendo ``confidence_score`` (saída de
        :func:`labeling.huggingface.label_corpus_with_huggingface_pipeline`,
        mesclada ao corpus original).
    paths : ProjectPaths
        Caminhos resolvidos do projeto (``configs/paths.yaml``).
    sample_size : int
        Repassado a
        :func:`labeling.manual.select_samples_for_human_validation`.
    low_confidence_threshold : float
        Repassado a
        :func:`labeling.confidence.flag_low_confidence_predictions`.
    """
    flagged = flag_low_confidence_predictions(
        labeled_corpus, low_confidence_threshold=low_confidence_threshold
    )
    if flagged.filter(pl.col("requires_human_validation")).height > 0:
        human_validation_sample = select_samples_for_human_validation(
            flagged, confidence_column="confidence_score", sample_size=sample_size
        )
        write_csv(
            human_validation_sample,
            paths.reports_tables_dir / _HUMAN_VALIDATION_SAMPLE_FILE_NAME,
        )
    else:
        logger.info("Nenhuma amostra sinalizada para validação humana nesta execução.")


def _validate_against_gold_set_if_provided(
    labeled_corpus: pl.DataFrame, gold_set: pl.DataFrame | None, *, minimum_kappa: float
) -> None:
    """Avalia a concordância com o gold set e alerta se abaixo do limiar mínimo.

    Parameters
    ----------
    labeled_corpus : pl.DataFrame
        Corpus rotulado (após consenso e eventual validação humana).
    gold_set : pl.DataFrame | None
        Gold set de referência; ``None`` desativa a validação.
    minimum_kappa : float
        Repassado a :func:`labeling.validation.evaluate_against_gold_set`.
    """
    if gold_set is None:
        return
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


def run_labeling_stage(
    paths: ProjectPaths,
    pipeline: SentimentPipeline,
    *,
    text_column: str = "text_normalized",
    label_mapping: Mapping[str, str] | None = None,
    huggingface_batch_size: int = 32,
    select_for_human_validation: bool = True,
    human_validation_sample_size: int = 500,
    low_confidence_threshold: float = 0.5,
    human_validation_labels: pl.DataFrame | None = None,
    gold_set: pl.DataFrame | None = None,
    minimum_kappa: float = 0.6,
    llm_relabeling_enabled: bool = False,
    llm_relabeling_score_threshold: float = 0.5,
    llm_relabeling_prompt_name: str | None = None,
    llm_relabeling_model: str | None = None,
    llm_relabeling_temperature: float = 0.0,
    llm_relabeling_max_retries: int = 3,
    llm_relabeling_n_workers: int = 8,
    llm_relabeling_provider: LLMProvider = "openai",
    llm_relabeling_ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
    llm_relabeling_request_interval_seconds: float = 0.0,
    show_progress: bool = True,
) -> Path:
    """Executa a etapa de rotulagem via pipeline Hugging Face sobre o corpus normalizado.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto (``configs/paths.yaml``).
    pipeline : SentimentPipeline
        Pipeline de classificação de sentimento (ver
        ``configs/labeling.yaml -> huggingface``), via
        :func:`labeling.huggingface.load_huggingface_sentiment_pipeline`,
        repassado a
        :func:`labeling.huggingface.label_corpus_with_huggingface_pipeline`.
    text_column : str, optional
        Coluna de texto classificada pelo pipeline, by default
        "text_normalized" (produzida por
        ``src/pipelines/preprocessing.py``).
    label_mapping : Mapping[str, str] | None, optional
        Mapeamento do rótulo bruto do modelo para as classes de sentimento
        em pt-BR, by default None (usa
        :data:`labeling.huggingface.DEFAULT_LABEL_MAPPING`).
    huggingface_batch_size : int, optional
        Repassado como ``batch_size`` a
        :func:`labeling.huggingface.label_corpus_with_huggingface_pipeline`,
        by default 32.
    select_for_human_validation : bool, optional
        Se ``True``, seleciona e grava uma amostra de candidatos à
        validação humana em ``paths.reports_tables_dir``, by default True.
    human_validation_sample_size : int, optional
        Repassado a
        :func:`labeling.manual.select_samples_for_human_validation`, by
        default 500.
    low_confidence_threshold : float, optional
        Repassado a
        :func:`labeling.confidence.flag_low_confidence_predictions`, by
        default 0.5 (``configs/labeling.yaml ->
        confidence.low_confidence_threshold``).
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
    llm_relabeling_enabled : bool, optional
        Se ``True``, re-rotula via LLM as amostras com ``confidence_score``
        abaixo de ``llm_relabeling_score_threshold`` (ver
        :func:`labeling.llm_relabeling.relabel_low_confidence_samples`),
        antes da incorporação de ``human_validation_labels`` — que, quando
        informado, sempre prevalece sobre o rótulo do LLM, by default
        False (``configs/labeling.yaml -> llm_relabeling.enabled``).
    llm_relabeling_score_threshold : float, optional
        Repassado como ``score_threshold``, by default 0.5.
    llm_relabeling_prompt_name : str | None, optional
        Nome do template de prompt em ``prompts/`` (sem ``.txt``),
        repassado como ``prompt_name``; obrigatório quando
        ``llm_relabeling_enabled=True``, by default None.
    llm_relabeling_model : str | None, optional
        Repassado como ``model``, by default None (resolvido conforme
        ``llm_relabeling_provider`` — ver
        :func:`labeling.llm_relabeling.relabel_low_confidence_samples`).
    llm_relabeling_temperature : float, optional
        Repassado como ``temperature``, by default 0.0.
    llm_relabeling_max_retries : int, optional
        Repassado como ``max_retries``, by default 3.
    llm_relabeling_n_workers : int, optional
        Repassado como ``n_workers``, by default 8.
    llm_relabeling_provider : {"openai", "ollama"}, optional
        Repassado como ``provider`` (``configs/llm.yaml ->
        active_provider``), by default "openai".
    llm_relabeling_ollama_base_url : str, optional
        Repassado como ``ollama_base_url``, usado apenas quando
        ``llm_relabeling_provider="ollama"``, by default
        :data:`hypothesaes.llm_api.DEFAULT_OLLAMA_BASE_URL`
        (``configs/llm.yaml -> backends.ollama.base_url``).
    llm_relabeling_request_interval_seconds : float, optional
        Repassado como ``request_interval_seconds`` — pausa antes de cada
        chamada/tentativa ao LLM, para reduzir a taxa de requisições e
        evitar bloqueios por limite de taxa (HTTP 429), by default 0.0.
    show_progress : bool, optional
        Se ``True``, exibe uma barra de progresso no console, by default
        True.

    Returns
    -------
    Path
        Caminho do corpus rotulado escrito (``paths.labeled_corpus_file``).

    Raises
    ------
    EmptyDatasetError
        Se o corpus normalizado estiver vazio.
    DataValidationError
        Se o pipeline devolver um rótulo bruto fora de ``label_mapping``,
        ou se o corpus rotulado final violar o contrato de dados.
    ValueError
        Se ``llm_relabeling_enabled=True`` e ``llm_relabeling_prompt_name``
        não for informado.

    Examples
    --------
    >>> from labeling.huggingface import load_huggingface_sentiment_pipeline
    >>> run_labeling_stage(paths, load_huggingface_sentiment_pipeline())  # doctest: +SKIP
    """
    normalized_corpus = read_dataset_file(paths.normalized_corpus_file)

    labeling_results = label_corpus_with_huggingface_pipeline(
        normalized_corpus,
        pipeline,
        text_column=text_column,
        label_mapping=label_mapping,
        batch_size=huggingface_batch_size,
        show_progress=show_progress,
    )
    labeled_corpus = merge_consensus_into_corpus(normalized_corpus, labeling_results)

    if select_for_human_validation:
        _select_and_write_human_validation_sample(
            labeled_corpus,
            paths,
            sample_size=human_validation_sample_size,
            low_confidence_threshold=low_confidence_threshold,
        )

    if llm_relabeling_enabled:
        if not llm_relabeling_prompt_name:
            raise ValueError(
                "llm_relabeling_prompt_name é obrigatório quando llm_relabeling_enabled=True "
                "(ver configs/labeling.yaml -> llm_relabeling.prompt_name)"
            )
        labeled_corpus = relabel_low_confidence_samples(
            labeled_corpus,
            text_column=text_column,
            score_threshold=llm_relabeling_score_threshold,
            prompt_name=llm_relabeling_prompt_name,
            model=llm_relabeling_model,
            temperature=llm_relabeling_temperature,
            max_retries=llm_relabeling_max_retries,
            n_workers=llm_relabeling_n_workers,
            provider=llm_relabeling_provider,
            ollama_base_url=llm_relabeling_ollama_base_url,
            request_interval_seconds=llm_relabeling_request_interval_seconds,
            show_progress=show_progress,
        )

    if human_validation_labels is not None:
        labeled_corpus = apply_human_validation_labels(labeled_corpus, human_validation_labels)

    _validate_against_gold_set_if_provided(labeled_corpus, gold_set, minimum_kappa=minimum_kappa)

    write_labeled_corpus(labeled_corpus, paths.labeled_corpus_file)
    logger.info("Etapa de rotulagem concluída: %d amostra(s) rotulada(s).", labeled_corpus.height)
    return paths.labeled_corpus_file
