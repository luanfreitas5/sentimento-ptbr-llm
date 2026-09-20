"""Rotulagem de sentimento do corpus normalizado em duas bases independentes.

Implementa o estágio ``labeling`` de ``configs/config.yaml -> stages``:
classifica **todos** os tweets do corpus normalizado com dois LLMs
independentes — um do Hugging Face (``src/labeling/huggingface.py``) e outro
via API OpenAI-compatível (``src/labeling/openai_labeler.py``) — e grava uma
base por fonte, com os mesmos tweets e o mesmo ``id``, prontas para a
comparação da etapa ``comparative_evaluation``:

* ``tweets_data_huggingface`` (``paths.huggingface_labeled_file``);
* ``tweets_data_openai`` (``paths.openai_labeled_file``).

Cada base contém ``id``, ``text`` (texto original), ``text_normalized``
(texto após o pré-processamento), ``sentiment_label`` (classe atribuída pelo
modelo) e ``confidence_score`` (confiança da classificação), validadas por
:class:`schemas.labeling.LabeledSourceSchema`; modelo, prompt e hash dos
dados ficam num arquivo ``.meta.json`` ao lado da base.

A rotulagem é incremental e retomável (``src/labeling/incremental.py``):
tweets já rotulados são lidos do checkpoint, e uma falha deixa o progresso
salvo. Depois de rotular, o estágio grava o corpus consumido pelas etapas
seguintes (``features``, ``hypothesaes_analysis``;
``paths.labeled_corpus_file``): ``sentiment_label``/``confidence_score`` vêm
da fonte ``downstream_source``, e as colunas
``sentiment_label_<fonte>``/``confidence_score_<fonte>`` preservam o rótulo
de cada base disponível.
"""

import logging
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from config.paths import ProjectPaths
from data.loader import read_dataset_file
from data.writer import write_dataset, write_labeled_corpus
from exceptions.configuration import InvalidConfigurationError
from io_utils.csv import write_csv
from io_utils.json import write_json
from labeling.checkpoint import build_checkpoint_path
from labeling.confidence import flag_low_confidence_predictions
from labeling.incremental import BatchClassifier, run_incremental_labeling
from labeling.manual import apply_human_validation_labels, select_samples_for_human_validation
from labeling.validation import evaluate_against_gold_set
from schemas.labeling import validate_labeled_source
from utils.hashing import calculate_file_hash, calculate_text_hash

logger = logging.getLogger(__name__)

LABEL_SOURCE_NAMES: tuple[str, ...] = ("huggingface", "openai")

_HUMAN_VALIDATION_SAMPLE_FILE_NAME = "human_validation_sample.csv"


@dataclass(frozen=True)
class LabelingSource:
    """Uma fonte de rotulagem (LLM) e tudo que é preciso para reproduzir sua base.

    Parameters
    ----------
    name : str
        ``huggingface`` ou ``openai`` (um de :data:`LABEL_SOURCE_NAMES`).
    model_name : str
        Modelo usado, registrado nos metadados da base.
    prompt_name : str
        Nome do template em ``prompts/`` (sem ``.txt``).
    prompt_template : str
        Conteúdo do template; entra no hash do checkpoint e nos metadados.
    temperature : float
        Temperatura de geração (0.0 = determinístico).
    batch_size : int
        Tweets por lote (e por gravação no checkpoint).
    open_classifier : Callable[[], AbstractContextManager[BatchClassifier]]
        Abre o classificador dentro de um ``with`` (o do Hugging Face carrega o
        modelo na entrada e libera a GPU na saída).
    """

    name: str
    model_name: str
    prompt_name: str
    prompt_template: str
    temperature: float
    batch_size: int
    open_classifier: Callable[[], AbstractContextManager[BatchClassifier]]


def resolve_labeled_source_path(paths: ProjectPaths, source_name: str) -> Path:
    """Resolve o arquivo da base ``tweets_data_<fonte>``.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto.
    source_name : str
        ``huggingface`` ou ``openai``.

    Returns
    -------
    Path
        ``paths.huggingface_labeled_file`` ou ``paths.openai_labeled_file``.

    Raises
    ------
    InvalidConfigurationError
        Se ``source_name`` não for uma fonte conhecida.

    Examples
    --------
    >>> resolve_labeled_source_path(paths, "openai").name  # doctest: +SKIP
    'tweets_data_openai.parquet'
    """
    if source_name == "huggingface":
        return paths.huggingface_labeled_file
    if source_name == "openai":
        return paths.openai_labeled_file
    raise InvalidConfigurationError(
        f"fonte de rotulagem desconhecida: '{source_name}'. Válidas: {list(LABEL_SOURCE_NAMES)}"
    )


def _build_metadata_path(labeled_source_path: Path) -> Path:
    """Caminho do arquivo de metadados ao lado da base (``<base>.meta.json``)."""
    return labeled_source_path.with_suffix(".meta.json")


def _write_source_metadata(
    source: LabelingSource, output_path: Path, *, n_tweets: int, normalized_corpus_hash: str
) -> None:
    """Grava modelo, prompt e hash dos dados de entrada ao lado da base rotulada."""
    write_json(
        {
            "source": source.name,
            "model": source.model_name,
            "prompt_name": source.prompt_name,
            "prompt_sha256": calculate_text_hash(source.prompt_template),
            "temperature": source.temperature,
            "n_tweets": n_tweets,
            "normalized_corpus_sha256": normalized_corpus_hash,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        _build_metadata_path(output_path),
    )


def label_source(
    paths: ProjectPaths,
    source: LabelingSource,
    normalized_corpus: pl.DataFrame,
    *,
    text_column: str = "text_normalized",
    show_progress: bool = True,
) -> Path:
    """Rotula todo o corpus com uma fonte e grava a base ``tweets_data_<fonte>``.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto.
    source : LabelingSource
        Fonte a executar.
    normalized_corpus : pl.DataFrame
        Corpus normalizado (``id``, ``text``, ``text_normalized`` ...).
    text_column : str, optional
        Coluna enviada ao LLM (sanitizada: sem menções/URLs), by default "text_normalized".
    show_progress : bool, optional
        Exibe barra de progresso, by default True.

    Returns
    -------
    Path
        Caminho da base gravada.

    Raises
    ------
    IncompleteLabelingError
        Se restarem tweets sem rótulo válido (o checkpoint preserva o progresso).
    DataValidationError
        Se a base violar o contrato de dados.

    Examples
    --------
    >>> label_source(paths, source, normalized_corpus)  # doctest: +SKIP
    """
    output_path = resolve_labeled_source_path(paths, source.name)
    checkpoint_path = build_checkpoint_path(
        paths.labeling_checkpoints_dir,
        source.name,
        model_name=source.model_name,
        prompt_template=source.prompt_template,
        temperature=source.temperature,
    )
    with source.open_classifier() as classify_batch:
        results = run_incremental_labeling(
            normalized_corpus,
            classify_batch,
            source_name=source.name,
            checkpoint_path=checkpoint_path,
            batch_size=source.batch_size,
            text_column=text_column,
            show_progress=show_progress,
        )

    labeled_source = validate_labeled_source(
        normalized_corpus.select(
            pl.col("id").cast(pl.String), "text", pl.col(text_column).alias("text_normalized")
        ).join(results, on="id", how="left")
    )
    write_dataset(labeled_source, output_path)
    _write_source_metadata(
        source,
        output_path,
        n_tweets=labeled_source.height,
        normalized_corpus_hash=calculate_file_hash(paths.normalized_corpus_file),
    )
    logger.info("Base '%s' gravada em: %s", source.name, output_path)
    return output_path


def build_labeled_corpus(
    normalized_corpus: pl.DataFrame, paths: ProjectPaths, downstream_source: str
) -> pl.DataFrame | None:
    """Monta o corpus rotulado das etapas seguintes a partir das bases disponíveis.

    ``sentiment_label``/``confidence_score`` vêm da base ``downstream_source``; as
    colunas ``sentiment_label_<fonte>``/``confidence_score_<fonte>`` preservam o
    rótulo de cada base já gravada.

    Parameters
    ----------
    normalized_corpus : pl.DataFrame
        Corpus normalizado original.
    paths : ProjectPaths
        Caminhos resolvidos do projeto.
    downstream_source : str
        Fonte cujo rótulo vira ``sentiment_label`` (um de :data:`LABEL_SOURCE_NAMES`).

    Returns
    -------
    pl.DataFrame | None
        Corpus rotulado, ou ``None`` se a base de ``downstream_source`` ainda não existir.

    Raises
    ------
    InvalidConfigurationError
        Se ``downstream_source`` não for uma fonte conhecida.

    Examples
    --------
    >>> build_labeled_corpus(normalized_corpus, paths, "huggingface")  # doctest: +SKIP
    """
    resolve_labeled_source_path(paths, downstream_source)  # valida o nome
    labeled_corpus = normalized_corpus.with_columns(pl.col("id").cast(pl.String))
    downstream_path = resolve_labeled_source_path(paths, downstream_source)
    if not downstream_path.is_file():
        return None

    for source_name in LABEL_SOURCE_NAMES:
        source_path = resolve_labeled_source_path(paths, source_name)
        if not source_path.is_file():
            continue
        source_labels = read_dataset_file(source_path).select(
            "id",
            pl.col("sentiment_label").alias(f"sentiment_label_{source_name}"),
            pl.col("confidence_score").alias(f"confidence_score_{source_name}"),
        )
        labeled_corpus = labeled_corpus.join(source_labels, on="id", how="left")
    return labeled_corpus.with_columns(
        pl.col(f"sentiment_label_{downstream_source}").alias("sentiment_label"),
        pl.col(f"confidence_score_{downstream_source}").alias("confidence_score"),
    )


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
        Corpus rotulado, contendo ``confidence_score``.
    paths : ProjectPaths
        Caminhos resolvidos do projeto (``configs/paths.yaml``).
    sample_size : int
        Repassado a :func:`labeling.manual.select_samples_for_human_validation`.
    low_confidence_threshold : float
        Repassado a :func:`labeling.confidence.flag_low_confidence_predictions`.
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
        Corpus rotulado (após eventual validação humana).
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
    sources: Sequence[LabelingSource],
    *,
    downstream_source: str = "huggingface",
    text_column: str = "text_normalized",
    select_for_human_validation: bool = True,
    human_validation_sample_size: int = 500,
    low_confidence_threshold: float = 0.5,
    human_validation_labels: pl.DataFrame | None = None,
    gold_set: pl.DataFrame | None = None,
    minimum_kappa: float = 0.6,
    show_progress: bool = True,
) -> dict[str, Path]:
    """Executa a etapa de rotulagem: uma base por fonte + o corpus das etapas seguintes.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto (``configs/paths.yaml``).
    sources : Sequence[LabelingSource]
        Fontes a rotular nesta execução (uma ou as duas de :data:`LABEL_SOURCE_NAMES`).
    downstream_source : str, optional
        Fonte cujo rótulo alimenta ``features``/``hypothesaes_analysis``
        (``configs/labeling.yaml -> downstream_source``), by default "huggingface".
    text_column : str, optional
        Coluna do texto enviado ao LLM, by default "text_normalized".
    select_for_human_validation : bool, optional
        Se ``True``, grava uma amostra de candidatos à validação humana em
        ``paths.reports_tables_dir``, by default True.
    human_validation_sample_size : int, optional
        Repassado a :func:`labeling.manual.select_samples_for_human_validation`, by default 500.
    low_confidence_threshold : float, optional
        Repassado a :func:`labeling.confidence.flag_low_confidence_predictions`, by default 0.5.
    human_validation_labels : pl.DataFrame | None, optional
        Rótulos revisados por humanos, incorporados ao corpus das etapas seguintes,
        by default None.
    gold_set : pl.DataFrame | None, optional
        Gold set de referência para validação por Kappa, by default None.
    minimum_kappa : float, optional
        Repassado a :func:`labeling.validation.evaluate_against_gold_set`, by default 0.6.
    show_progress : bool, optional
        Exibe barra de progresso, by default True.

    Returns
    -------
    dict[str, Path]
        Arquivos gravados: uma chave por fonte rotulada e, quando a base de
        ``downstream_source`` existe, ``"labeled_corpus"``.

    Raises
    ------
    EmptyDatasetError
        Se o corpus normalizado estiver vazio.
    IncompleteLabelingError
        Se alguma fonte terminar com tweets sem rótulo (rode de novo para retomar).
    DataValidationError
        Se alguma base ou o corpus final violar o contrato de dados.
    InvalidConfigurationError
        Se uma fonte (ou ``downstream_source``) for desconhecida.

    Examples
    --------
    >>> run_labeling_stage(paths, [huggingface_source, openai_source])  # doctest: +SKIP
    """
    normalized_corpus = read_dataset_file(paths.normalized_corpus_file)

    written: dict[str, Path] = {}
    for source in sources:
        written[source.name] = label_source(
            paths, source, normalized_corpus, text_column=text_column, show_progress=show_progress
        )

    labeled_corpus = build_labeled_corpus(normalized_corpus, paths, downstream_source)
    if labeled_corpus is None:
        logger.warning(
            "Base '%s' ainda não existe: o corpus rotulado das etapas seguintes não foi gerado. "
            "Rotule essa fonte para gerá-lo.",
            downstream_source,
        )
        return written

    if select_for_human_validation:
        _select_and_write_human_validation_sample(
            labeled_corpus,
            paths,
            sample_size=human_validation_sample_size,
            low_confidence_threshold=low_confidence_threshold,
        )
    if human_validation_labels is not None:
        labeled_corpus = apply_human_validation_labels(labeled_corpus, human_validation_labels)
    _validate_against_gold_set_if_provided(labeled_corpus, gold_set, minimum_kappa=minimum_kappa)

    write_labeled_corpus(labeled_corpus, paths.labeled_corpus_file)
    written["labeled_corpus"] = paths.labeled_corpus_file
    logger.info(
        "Etapa de rotulagem concluída: %d tweet(s); corpus das etapas seguintes usa '%s'.",
        labeled_corpus.height,
        downstream_source,
    )
    return written
