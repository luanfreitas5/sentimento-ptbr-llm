"""Validação das hipóteses no holdout (partição disjunta) com Bonferroni.

Pega os top-k conceitos gerados na descoberta, anota uma **subamostra** do
holdout (``teste``) com eles via LLM (assíncrono, com cache) e valida cada
hipótese com ``score_hypotheses``: uma hipótese "sobrevive" se o p-valor da
regressão for menor que ``alfa / nº de hipóteses`` (Bonferroni, alfa = 0,1).
Também gera a amostra estratificada por conceito para rotulagem humana
(``para_rotular.csv``, sem identificadores).

Uso::

    PYTHONPATH=src python -m diagnostics.validation --target disagreement --dry-run
    PYTHONPATH=src python -m diagnostics.validation --target disagreement

Hipóteses sobre pseudo-rótulo descrevem o comportamento do modelo, não a
verdade: só o gold set mede acerto.
"""

import argparse
import logging
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl

from config.environment import configure_environment_variables
from config.logging import configure_logging
from config.paths import ProjectPaths, load_project_paths
from diagnostics.annotation import annotate_concepts
from diagnostics.cost import estimate_annotation_cost, format_cost_report
from diagnostics.hypotheses import (
    build_target_slug,
    load_diagnostic_corpus,
    resolve_target_arguments,
    score_hypotheses,
)
from diagnostics.llm_client import AsyncLLMClient
from diagnostics.sae_runner import HOLDOUT_PARTITION, PARTITION_COLUMN, assign_partitions
from diagnostics.sampling import (
    resolve_sample_salt,
    sample_tweets_by_concept,
    write_labeling_sample,
)
from diagnostics.settings import (
    DEFAULT_DIAGNOSTICS_CONFIG_FILE,
    DiagnosticsSettings,
    load_diagnostics_settings,
)
from diagnostics.targets import TARGET_NAMES, TargetName, build_target
from diagnostics.tracking import (
    configure_mlflow,
    log_diagnostics_artifact,
    log_diagnostics_metrics,
    track_diagnostics_run,
)
from exceptions.data import DataNotFoundError, EmptyDatasetError
from io_utils.csv import write_csv
from io_utils.parquet import read_parquet, write_parquet

logger = logging.getLogger(__name__)

SAMPLE_FILE_NAME = "para_rotular.csv"
SAMPLE_KEY_FILE_NAME = "para_rotular_chave.parquet"


@dataclass(frozen=True)
class ValidationOutcome:
    """Resultado da validação de um alvo.

    Attributes
    ----------
    validated : pl.DataFrame
        Uma linha por hipótese, com ``survives`` (Bonferroni).
    n_survivors : int
        Hipóteses que sobrevivem.
    sample_path : Path | None
        ``para_rotular.csv`` gerado (``None`` se não houve conceito com tweets).
    """

    validated: pl.DataFrame
    n_survivors: int
    sample_path: Path | None


def select_top_hypotheses(discovery_table: pl.DataFrame, top_k: int) -> list[str]:
    """Escolhe as ``top_k`` hipóteses da descoberta, únicas, por força de seleção.

    Ordena por ``selection_score`` (ou, se ausente, por ``|separation_score|``),
    decrescente.

    Parameters
    ----------
    discovery_table : pl.DataFrame
        Saída de :mod:`diagnostics.hypotheses` (coluna ``hypothesis``).
    top_k : int
        Nº de hipóteses.

    Returns
    -------
    list[str]
        Textos das hipóteses, do mais para o menos forte.

    Raises
    ------
    EmptyDatasetError
        Se a tabela não tiver hipóteses.

    Examples
    --------
    >>> t = pl.DataFrame({"hypothesis": ["a", "b"], "selection_score": [0.1, 0.9]})
    >>> select_top_hypotheses(t, 1)
    ['b']
    """
    table = discovery_table.filter(pl.col("hypothesis").is_not_null())
    if table.is_empty():
        raise EmptyDatasetError("tabela de hipóteses da descoberta")
    key = (
        pl.col("selection_score")
        if "selection_score" in table.columns
        else pl.col("separation_score").abs()
    )
    ordered = table.sort(key, descending=True, nulls_last=True)
    return (
        ordered.unique(subset=["hypothesis"], keep="first", maintain_order=True)["hypothesis"]
        .head(top_k)
        .to_list()
    )


def stratified_subsample_indices(labels: np.ndarray, n_rows: int, random_seed: int) -> np.ndarray:
    """Sorteia ``n_rows`` índices preservando a proporção das classes (alvo binário).

    Para alvos contínuos (mais de 2 valores distintos) faz uma amostra aleatória simples.

    Parameters
    ----------
    labels : np.ndarray
        Valores do alvo.
    n_rows : int
        Tamanho desejado (limitado ao total).
    random_seed : int
        Semente.

    Returns
    -------
    np.ndarray
        Índices ordenados.

    Examples
    --------
    >>> stratified_subsample_indices(np.array([0, 0, 0, 1]), 2, 0).size
    2
    """
    rng = np.random.default_rng(random_seed)
    total = labels.size
    if n_rows >= total:
        return np.arange(total)
    classes = np.unique(labels)
    if classes.size > 2:
        return np.sort(rng.choice(total, size=n_rows, replace=False))
    chosen: list[np.ndarray] = []
    for value in classes:
        members = np.flatnonzero(labels == value)
        size = min(members.size, max(1, round(n_rows * members.size / total)))
        chosen.append(rng.choice(members, size=size, replace=False))
    return np.sort(np.concatenate(chosen))


def select_holdout_rows(
    corpus: pl.DataFrame, target: pl.DataFrame, settings: DiagnosticsSettings
) -> pl.DataFrame:
    """Devolve as linhas do alvo que caem na partição ``teste`` (mesmas da descoberta).

    Recalcula as partições de forma determinística (mesmo corpus + semente),
    de modo que o holdout é disjunto do treino usado para descobrir hipóteses.

    Parameters
    ----------
    corpus : pl.DataFrame
        Corpus no contrato de diagnóstico (o mesmo usado na descoberta).
    target : pl.DataFrame
        Alvo (``id``/``target``).
    settings : DiagnosticsSettings
        Configuração validada.

    Returns
    -------
    pl.DataFrame
        Colunas ``id``, ``text_normalized`` e ``target`` do holdout.

    Raises
    ------
    EmptyDatasetError
        Se nenhuma linha do alvo estiver no holdout.
    """
    holdout = (
        assign_partitions(
            corpus,
            holdout_size=settings.splits.holdout_size,
            validation_size=settings.splits.validation_size,
            random_seed=settings.random_seed,
        )
        .filter(pl.col(PARTITION_COLUMN) == HOLDOUT_PARTITION)
        .select("id", "text_normalized")
        .join(target, on="id", how="inner")
    )
    if holdout.is_empty():
        raise EmptyDatasetError("holdout sem linhas do alvo")
    return holdout


def validate_hypotheses(
    annotations: dict[str, np.ndarray],
    y_true: np.ndarray,
    *,
    classification: bool,
    corrected_pval_threshold: float,
) -> pl.DataFrame:
    """Valida hipóteses anotadas no holdout, marcando as que sobrevivem ao Bonferroni.

    Parameters
    ----------
    annotations : dict[str, np.ndarray]
        Hipótese -> vetor 0/1 no holdout.
    y_true : np.ndarray
        Alvo observado no holdout.
    classification : bool
        ``True`` para alvo binário.
    corrected_pval_threshold : float
        Alfa antes da correção (0,1).

    Returns
    -------
    pl.DataFrame
        ``hypothesis``, ``separation_score``, ``separation_pval``, ``regression_coef``,
        ``regression_pval``, ``feature_prevalence``, ``bonferroni_threshold`` e ``survives``.

    Examples
    --------
    >>> validate_hypotheses(annotations, y, classification=True, corrected_pval_threshold=0.1)
    ... # doctest: +SKIP
    """
    _, scored = score_hypotheses(
        hypothesis_annotations=annotations,
        y_true=y_true,
        classification=classification,
        corrected_pval_threshold=corrected_pval_threshold,
    )
    threshold = corrected_pval_threshold / len(annotations)
    return pl.from_pandas(scored).with_columns(
        pl.lit(threshold).alias("bonferroni_threshold"),
        (pl.col("regression_pval") < threshold).alias("survives"),
    )


def _discovery_table_path(paths: ProjectPaths, settings: DiagnosticsSettings, slug: str) -> Path:
    return paths.reports_interpretability_dir / settings.data.output_dir / f"{slug}.parquet"


def build_dry_run_report(
    target_name: str, corpus: pl.DataFrame, settings: DiagnosticsSettings, **target_arguments: Any
) -> str:
    """Estima as chamadas da validação (N tweets do holdout × top-k conceitos), sem rede.

    Parameters
    ----------
    target_name : str
        Alvo escolhido.
    corpus : pl.DataFrame
        Corpus no contrato de diagnóstico.
    settings : DiagnosticsSettings
        Configuração validada.
    **target_arguments : Any
        ``model_column``/``model_columns``/``label``.

    Returns
    -------
    str
        Relatório multilinha em pt-BR.
    """
    target = build_target(corpus, cast(TargetName, target_name), **target_arguments)
    holdout = select_holdout_rows(corpus, target, settings)
    n_tweets = min(settings.validation.max_tweets_per_run, holdout.height)
    estimate = estimate_annotation_cost(
        settings, n_tweets=n_tweets, n_concepts=settings.validation.top_k_concepts
    )
    header = (
        f"DRY-RUN da validação de '{target_name}': {n_tweets} tweets do holdout x "
        f"{settings.validation.top_k_concepts} conceitos. Nenhuma chamada de rede foi feita."
    )
    return (
        header
        + "\n"
        + format_cost_report({"anotacao_holdout": estimate}, provider=settings.llm.provider)
    )


def run_validation_stage(
    target_name: str,
    corpus: pl.DataFrame,
    settings: DiagnosticsSettings,
    paths: ProjectPaths,
    *,
    model_column: str | None = None,
    model_columns: Sequence[str] | None = None,
    label: str | None = None,
    track: bool = True,
    client: AsyncLLMClient | None = None,
) -> ValidationOutcome:
    """Valida os top-k conceitos no holdout e gera a amostra para rotulagem humana.

    Parameters
    ----------
    target_name : str
        Um de :data:`diagnostics.targets.TARGET_NAMES`.
    corpus : pl.DataFrame
        Corpus no contrato (o mesmo da descoberta).
    settings : DiagnosticsSettings
        Configuração validada.
    paths : ProjectPaths
        Caminhos do projeto.
    model_column, model_columns, label : optional
        Argumentos do alvo (ver :func:`diagnostics.targets.build_target`).
    track : bool, optional
        Se registra no MLflow, by default True.
    client : AsyncLLMClient | None, optional
        Cliente LLM (testes), by default None.

    Returns
    -------
    ValidationOutcome
        Tabela validada, nº de hipóteses que sobrevivem e caminho da amostra.

    Raises
    ------
    DataNotFoundError
        Se a descoberta do alvo ainda não foi executada.
    PipelineStageError
        Se o orçamento de chamadas for excedido ou o anotador não seguir o formato.
    MissingEnvironmentVariableError
        Se o sal da amostra (``sampling.salt_env_var``) não estiver definido.
    """
    slug = build_target_slug(target_name, model_column=model_column, label=label)
    discovery_path = _discovery_table_path(paths, settings, slug)
    if not discovery_path.is_file():
        raise DataNotFoundError(f"{discovery_path} — rode diagnostics.hypotheses antes")
    target = build_target(
        corpus,
        cast(TargetName, target_name),
        model_column=model_column,
        model_columns=model_columns,
        label=label,
    )
    concepts = select_top_hypotheses(
        read_parquet(discovery_path), settings.validation.top_k_concepts
    )
    holdout = select_holdout_rows(corpus, target, settings)
    picked = stratified_subsample_indices(
        holdout["target"].to_numpy(), settings.validation.max_tweets_per_run, settings.random_seed
    )
    subsample = holdout.select(pl.all().gather(picked.tolist()))

    context: Any = nullcontext()
    if track:
        configure_mlflow(paths)
        context = track_diagnostics_run(
            f"validacao-{slug}",
            params={
                "target": slug,
                "top_k": len(concepts),
                "n_tweets": subsample.height,
                "annotator_model": settings.llm.annotator_model,
                "bonferroni_alpha": settings.validation.corrected_pval_threshold,
            },
        )
    with context:
        result, stats = annotate_concepts(
            settings, subsample["text_normalized"].to_list(), concepts, client=client
        )
        validated = validate_hypotheses(
            result.annotations,
            subsample["target"].to_numpy(),
            classification=target_name != "uncertainty",
            corrected_pval_threshold=settings.validation.corrected_pval_threshold,
        ).with_columns(pl.lit(slug).alias("target"), pl.lit("holdout").alias("partition"))
        output_dir = paths.reports_interpretability_dir / settings.data.output_dir
        write_parquet(validated, output_dir / f"{slug}_validacao.parquet")
        write_csv(validated, output_dir / f"{slug}_validacao.csv")
        n_survivors = int(validated["survives"].sum())
        logger.info("%d/%d hipóteses sobrevivem ao Bonferroni.", n_survivors, validated.height)
        sample_path = _write_sample(subsample, result.annotations, validated, settings, paths)
        if track:
            log_diagnostics_metrics(
                {
                    "n_hypotheses_tested": validated.height,
                    "n_survivors": n_survivors,
                    "annotation_requests": stats.n_requests,
                    "annotation_cache_hits": stats.n_cache_hits,
                    "annotation_failure_rate": result.failure_rate,
                }
            )
            log_diagnostics_artifact(output_dir / f"{slug}_validacao.csv")
    return ValidationOutcome(validated, n_survivors, sample_path)


def _write_sample(
    subsample: pl.DataFrame,
    annotations: dict[str, np.ndarray],
    validated: pl.DataFrame,
    settings: DiagnosticsSettings,
    paths: ProjectPaths,
) -> Path | None:
    """Gera ``para_rotular.csv`` com os conceitos sobreviventes (ou todos, se não houver)."""
    survivors = validated.filter(pl.col("survives"))["hypothesis"].to_list()
    concepts = survivors or list(annotations)
    if not survivors:
        logger.warning("Nenhuma hipótese sobrevive; a amostra usa todos os conceitos testados.")
    sample, key = sample_tweets_by_concept(
        subsample,
        {concept: annotations[concept] for concept in concepts},
        per_concept=settings.sampling.tweets_per_concept,
        salt=resolve_sample_salt(settings.sampling.salt_env_var),
        random_seed=settings.random_seed,
    )
    if sample.is_empty():
        return None
    sample_path = paths.reports_tables_dir / SAMPLE_FILE_NAME
    write_labeling_sample(
        sample,
        key,
        sample_csv=sample_path,
        key_parquet=paths.data_interim_dir / "diagnostics" / SAMPLE_KEY_FILE_NAME,
    )
    return sample_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Interpreta os argumentos de linha de comando.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Argumentos (``None`` usa ``sys.argv``), by default None.

    Returns
    -------
    argparse.Namespace
        Argumentos validados.

    Examples
    --------
    >>> parse_arguments(["--target", "uncertainty"]).target
    'uncertainty'
    """
    parser = argparse.ArgumentParser(
        prog="python -m diagnostics.validation",
        description="Valida hipóteses no holdout (Bonferroni) e gera a amostra para rotulagem.",
    )
    parser.add_argument("--target", required=True, choices=TARGET_NAMES)
    parser.add_argument("--model-column")
    parser.add_argument("--model-columns", nargs="+")
    parser.add_argument("--label")
    parser.add_argument("--corpus", type=Path, help="Parquet já no contrato de diagnóstico.")
    parser.add_argument("--config", type=Path, default=DEFAULT_DIAGNOSTICS_CONFIG_FILE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Ponto de entrada da CLI.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Argumentos (``None`` usa ``sys.argv``), by default None.

    Returns
    -------
    int
        Código de saída (0 = sucesso).

    Examples
    --------
    >>> main(["--target", "uncertainty", "--dry-run"])  # doctest: +SKIP
    0
    """
    args = parse_arguments(argv)
    configure_environment_variables()
    configure_logging()
    settings = load_diagnostics_settings(args.config)
    paths = load_project_paths()
    corpus = load_diagnostic_corpus(settings, paths, args.corpus)
    target_arguments = resolve_target_arguments(args.target, args, settings)
    if args.dry_run:
        print(build_dry_run_report(args.target, corpus, settings, **target_arguments))
        return 0
    run_validation_stage(
        args.target, corpus, settings, paths, track=not args.no_mlflow, **target_arguments
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
