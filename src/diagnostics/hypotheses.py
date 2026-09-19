"""Geração de hipóteses do HypotheSAEs por alvo de diagnóstico, com CLI.

Fluxo por alvo (ver ``docs/guides/diagnostico-hypothesaes.md``):

1. constrói o alvo (:mod:`diagnostics.targets`) e o alinha às partições;
2. **gate de sanidade** (:mod:`diagnostics.sanity`): se os embeddings não
   preveem o alvo acima do acaso, aborta e registra o motivo no MLflow;
3. ``generate_hypotheses`` na partição de **descoberta** (``treino``);
4. estatísticas dos neurônios (``separation_score``, ``regression_pval``,
   ``feature_prevalence``) calculadas com ``score_hypotheses`` sobre as
   ativações do SAE, **sem chamadas de LLM**;
5. grava parquet/CSV e registra o experimento no MLflow.

Uso::

    PYTHONPATH=src python -m diagnostics.hypotheses --target disagreement --dry-run
    PYTHONPATH=src python -m diagnostics.hypotheses --target disagreement

A validação em partição disjunta (holdout) é feita em :mod:`diagnostics.validation`.
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

from config.environment import configure_environment_variables, configure_reproducibility
from config.logging import configure_logging
from config.paths import ProjectPaths, load_project_paths
from config.settings import load_general_config
from data.loader import read_dataset_file
from diagnostics.cost import estimate_hypothesis_generation_cost, format_cost_report
from diagnostics.sae_runner import (
    DISCOVERY_PARTITION,
    HOLDOUT_PARTITION,
    PARTITION_COLUMN,
    DiscoveryData,
    prepare_discovery_data,
)
from diagnostics.sanity import SanityGateResult, assert_sanity_gate_passed, evaluate_sanity_gate
from diagnostics.settings import (
    DEFAULT_DIAGNOSTICS_CONFIG_FILE,
    DiagnosticsSettings,
    load_diagnostics_settings,
)
from diagnostics.targets import TARGET_NAMES, TargetName, adapt_labeled_corpus, build_target
from diagnostics.tracking import (
    configure_mlflow,
    log_diagnostics_artifact,
    log_diagnostics_metrics,
    track_diagnostics_run,
)
from exceptions.data import DataValidationError, EmptyDatasetError
from io_utils.csv import write_csv
from io_utils.json import write_json
from io_utils.parquet import read_parquet, write_parquet
from schemas.diagnostics import validate_diagnostic_corpus

logger = logging.getLogger(__name__)

_TEXT_COLUMN = "text_normalized"
OUTPUT_COLUMNS: tuple[str, ...] = (
    "target",
    "partition",
    "hypothesis",
    "neuron_idx",
    "selection_score",
    "fidelity_score",
    "separation_score",
    "regression_pval",
    "feature_prevalence",
    "n_tweets",
)


def generate_hypotheses(**kwargs: Any) -> Any:
    """Import tardio de :func:`hypothesaes.quickstart.generate_hypotheses`."""
    from hypothesaes.quickstart import generate_hypotheses as _generate

    return _generate(**kwargs)


def score_hypotheses(**kwargs: Any) -> tuple[dict[str, Any], Any]:
    """Import tardio de :func:`hypothesaes.evaluation.score_hypotheses`."""
    from hypothesaes.evaluation import score_hypotheses as _score

    return _score(**kwargs)


@dataclass(frozen=True)
class PartitionSlice:
    """Textos, alvo e embeddings de uma partição, já restritos às linhas do alvo.

    Attributes
    ----------
    texts : list[str]
        Textos sanitizados.
    labels : np.ndarray
        Valores do alvo (0/1 ou contínuo).
    embeddings : np.ndarray
        Embeddings correspondentes (n, dim).
    """

    texts: list[str]
    labels: np.ndarray
    embeddings: np.ndarray


@dataclass(frozen=True)
class TargetRunResult:
    """Resultado da execução de um alvo.

    Attributes
    ----------
    target_slug : str
        Identificador do alvo (inclui modelo/rótulo quando aplicável).
    gate : SanityGateResult
        Resultado do gate de sanidade.
    hypotheses : pl.DataFrame | None
        Tabela de hipóteses (``None`` se o gate reprovou).
    output_path : Path | None
        Parquet gravado (``None`` se abortou).
    """

    target_slug: str
    gate: SanityGateResult
    hypotheses: pl.DataFrame | None
    output_path: Path | None


def build_target_slug(
    target_name: str, *, model_column: str | None = None, label: str | None = None
) -> str:
    """Nome estável do alvo para arquivos, cache e MLflow.

    Parameters
    ----------
    target_name : str
        Um de :data:`diagnostics.targets.TARGET_NAMES`.
    model_column : str | None, optional
        Coluna ``lab_<modelo>`` do alvo, se aplicável, by default None.
    label : str | None, optional
        Classe do one-vs-rest, se aplicável, by default None.

    Returns
    -------
    str
        Ex.: ``pseudo_label_lab_huggingface_negativo``.

    Examples
    --------
    >>> build_target_slug("pseudo_label", model_column="lab_a", label="negativo")
    'pseudo_label_lab_a_negativo'
    >>> build_target_slug("disagreement")
    'disagreement'
    """
    return "_".join(part for part in (target_name, model_column, label) if part)


def build_partition_slice(
    target: pl.DataFrame, discovery: DiscoveryData, partition: str
) -> PartitionSlice:
    """Restringe uma partição às linhas que têm alvo e devolve textos/alvo/embeddings.

    Parameters
    ----------
    target : pl.DataFrame
        Alvo (``id``/``target``).
    discovery : DiscoveryData
        Corpus particionado com embeddings alinhados.
    partition : str
        Partição desejada (``treino``/``validacao``/``teste``).

    Returns
    -------
    PartitionSlice
        Dados alinhados da partição.

    Raises
    ------
    EmptyDatasetError
        Se nenhuma linha do alvo cair na partição.
    """
    joined = (
        discovery.partitioned.with_row_index("_row")
        .filter(pl.col(PARTITION_COLUMN) == partition)
        .join(target, on="id", how="inner")
    )
    if joined.is_empty():
        raise EmptyDatasetError(f"partição '{partition}' sem linhas do alvo")
    rows = joined["_row"].to_numpy()
    return PartitionSlice(
        texts=joined[_TEXT_COLUMN].to_list(),
        labels=joined["target"].to_numpy(),
        embeddings=discovery.embeddings[rows],
    )


def build_generation_kwargs(
    settings: DiagnosticsSettings, *, cache_name: str, classification: bool
) -> dict[str, Any]:
    """Monta os argumentos de ``generate_hypotheses`` a partir da configuração.

    Parameters
    ----------
    settings : DiagnosticsSettings
        Configuração validada.
    cache_name : str
        Prefixo do cache de anotações.
    classification : bool
        ``True`` para alvo binário.

    Returns
    -------
    dict[str, Any]
        Argumentos nomeados (sem ``texts``/``labels``/``embeddings``/``sae``).
    """
    hyp, llm = settings.hypotheses, settings.llm
    llm_kwargs = {"provider": llm.provider, "ollama_base_url": llm.ollama_base_url}
    return {
        "cache_name": cache_name,
        "classification": classification,
        "selection_method": hyp.selection_method,
        "n_selected_neurons": hyp.n_selected_neurons,
        "interpreter_model": llm.interpreter_model,
        "annotator_model": llm.annotator_model,
        "n_examples_for_interpretation": hyp.n_examples_for_interpretation,
        "max_words_per_example": hyp.max_words_per_example,
        "max_interpretation_tokens": hyp.max_interpretation_tokens,
        "n_candidate_interpretations": hyp.n_candidate_interpretations,
        "n_scoring_examples": hyp.n_scoring_examples,
        "scoring_metric": hyp.scoring_metric,
        "n_workers_interpretation": hyp.n_workers,
        "n_workers_annotation": min(hyp.n_workers, llm.max_concurrency),
        "task_specific_instructions": hyp.task_specific_instructions,
        "interpret_llm_kwargs": llm_kwargs,
        "annotation_llm_kwargs": llm_kwargs,
    }


def compute_neuron_statistics(
    hypotheses: pl.DataFrame,
    sae: Any,
    slice_: PartitionSlice,
    *,
    settings: DiagnosticsSettings,
    classification: bool,
    target_slug: str,
    partition_name: str,
) -> pl.DataFrame:
    """Calcula ``separation_score``/``regression_pval``/``feature_prevalence`` por hipótese.

    A anotação de cada hipótese é a ativação do neurônio correspondente
    (``> 0``) nos tweets da partição — ou seja, sem chamadas de LLM. É uma
    medida do comportamento do SAE na partição, não da fidelidade da frase.

    Parameters
    ----------
    hypotheses : pl.DataFrame
        Saída de ``generate_hypotheses`` (colunas ``neuron_idx``, ``interpretation``).
    sae : Any
        SAE treinado.
    slice_ : PartitionSlice
        Dados da partição.
    settings : DiagnosticsSettings
        Configuração validada.
    classification : bool
        ``True`` para alvo binário.
    target_slug : str
        Identificador do alvo.
    partition_name : str
        Rótulo da partição (ex.: ``discovery``).

    Returns
    -------
    pl.DataFrame
        Tabela com as colunas de :data:`OUTPUT_COLUMNS` (vazia se não houver hipóteses).
    """
    valid = hypotheses.filter(pl.col("interpretation").is_not_null()).unique(
        subset=["interpretation"], keep="first", maintain_order=True
    )
    if valid.is_empty():
        return pl.DataFrame(schema=dict.fromkeys(OUTPUT_COLUMNS, pl.String))

    activations = np.asarray(sae.compute_activations(slice_.embeddings, show_progress=False))
    annotations = {
        row["interpretation"]: (activations[:, int(row["neuron_idx"])] > 0).astype(int)
        for row in valid.iter_rows(named=True)
    }
    _, scored = score_hypotheses(
        hypothesis_annotations=annotations,
        y_true=slice_.labels,
        classification=classification,
        corrected_pval_threshold=settings.validation.corrected_pval_threshold,
    )
    renamed = valid.rename(
        {old: new for old, new in _column_renames(settings).items() if old in valid.columns}
    )
    meta = renamed.select(
        [
            column
            for column in ("hypothesis", "neuron_idx", "selection_score", "fidelity_score")
            if column in renamed.columns
        ]
    )
    table = (
        pl.from_pandas(scored)
        .select(["hypothesis", "separation_score", "regression_pval", "feature_prevalence"])
        .join(meta, on="hypothesis", how="left")
        .with_columns(
            pl.lit(target_slug).alias("target"),
            pl.lit(partition_name).alias("partition"),
            pl.lit(len(slice_.texts)).alias("n_tweets"),
        )
    )
    return table.select([c for c in OUTPUT_COLUMNS if c in table.columns])


def _column_renames(settings: DiagnosticsSettings) -> dict[str, str]:
    """Mapa das colunas de ``generate_hypotheses`` para o esquema de saída."""
    return {
        "interpretation": "hypothesis",
        f"target_{settings.hypotheses.selection_method}": "selection_score",
        f"{settings.hypotheses.scoring_metric}_fidelity_score": "fidelity_score",
    }


def _write_outputs(table: pl.DataFrame, directory: Path, slug: str) -> Path:
    """Grava a tabela em parquet e CSV e devolve o caminho do parquet."""
    parquet_path = directory / f"{slug}.parquet"
    write_parquet(table, parquet_path)
    write_csv(table, directory / f"{slug}.csv")
    return parquet_path


def _execute_target(
    target: pl.DataFrame,
    discovery: DiscoveryData,
    settings: DiagnosticsSettings,
    paths: ProjectPaths,
    *,
    target_slug: str,
    classification: bool,
    generate_fn: Any,
) -> TargetRunResult:
    """Corpo do fluxo por alvo: gate, hipóteses, estatísticas e gravação (sem MLflow)."""
    discovery_slice = build_partition_slice(target, discovery, DISCOVERY_PARTITION)
    holdout_slice = build_partition_slice(target, discovery, HOLDOUT_PARTITION)
    sanity = settings.sanity
    gate = evaluate_sanity_gate(
        discovery_slice.embeddings,
        discovery_slice.labels,
        holdout_slice.embeddings,
        holdout_slice.labels,
        target_name=target_slug,
        classification=classification,
        ridge_alpha=sanity.ridge_alpha,
        n_bootstrap=sanity.n_bootstrap,
        n_permutations=sanity.n_permutations,
        confidence_level=sanity.confidence_level,
        significance_alpha=sanity.significance_alpha,
        min_effect=sanity.min_effect,
        random_seed=settings.random_seed,
    )
    output_dir = paths.reports_interpretability_dir / settings.data.output_dir
    if not gate.go:
        write_json(gate.to_dict(), output_dir / f"{target_slug}_gate_reprovado.json")
        return TargetRunResult(target_slug, gate, None, None)

    logger.info(
        "Gerando hipóteses para '%s' (%d tweets de descoberta)...",
        target_slug,
        len(discovery_slice.texts),
    )
    generated = generate_fn(
        texts=discovery_slice.texts,
        labels=discovery_slice.labels,
        embeddings=discovery_slice.embeddings,
        sae=discovery.sae,
        **build_generation_kwargs(
            settings,
            cache_name=f"{discovery.cache_name}_{target_slug}",
            classification=classification,
        ),
    )
    table = compute_neuron_statistics(
        pl.from_pandas(generated),
        discovery.sae,
        discovery_slice,
        settings=settings,
        classification=classification,
        target_slug=target_slug,
        partition_name="discovery",
    )
    output_path = _write_outputs(table, output_dir, target_slug)
    return TargetRunResult(target_slug, gate, table, output_path)


def run_target_diagnostics(
    target_name: str,
    corpus: pl.DataFrame,
    discovery: DiscoveryData,
    settings: DiagnosticsSettings,
    paths: ProjectPaths,
    *,
    model_column: str | None = None,
    model_columns: Sequence[str] | None = None,
    label: str | None = None,
    track: bool = True,
    generate_fn: Any = None,
) -> TargetRunResult:
    """Executa o diagnóstico de um alvo: gate, hipóteses, estatísticas e MLflow.

    Parameters
    ----------
    target_name : str
        Um de :data:`diagnostics.targets.TARGET_NAMES`.
    corpus : pl.DataFrame
        Corpus no contrato de diagnóstico.
    discovery : DiscoveryData
        Partições, embeddings e SAE (:func:`diagnostics.sae_runner.prepare_discovery_data`).
    settings : DiagnosticsSettings
        Configuração validada.
    paths : ProjectPaths
        Caminhos do projeto.
    model_column : str | None, optional
        Modelo analisado (``pseudo_label``/``gold_error``), by default None.
    model_columns : Sequence[str] | None, optional
        Modelos comparados (``disagreement``), by default None.
    label : str | None, optional
        Classe positiva (``pseudo_label``), by default None.
    track : bool, optional
        Se registra no MLflow, by default True.
    generate_fn : Any, optional
        Substituto de :func:`generate_hypotheses` (testes), by default None.

    Returns
    -------
    TargetRunResult
        Gate, tabela de hipóteses e caminho de saída.

    Raises
    ------
    SanityGateFailedError
        Se o gate reprovar o alvo (após registrar o motivo no MLflow).

    Examples
    --------
    >>> run_target_diagnostics("disagreement", corpus, discovery, settings, paths)  # doctest: +SKIP
    """
    target = build_target(
        corpus,
        cast(TargetName, target_name),
        model_column=model_column,
        model_columns=model_columns,
        label=label,
    )
    slug = build_target_slug(target_name, model_column=model_column, label=label)
    classification = target_name != "uncertainty"
    params = {
        "target": slug,
        "m_total_neurons": settings.sae.m_total_neurons,
        "k_active_neurons": settings.sae.k_active_neurons,
        "selection_method": settings.hypotheses.selection_method,
        "embedding_model": settings.embedding.model_name,
        "interpreter_model": settings.llm.interpreter_model,
        "annotator_model": settings.llm.annotator_model,
        "llm_provider": settings.llm.provider,
        "prompt_version": settings.prompts.v1,
        "corpus_hash": discovery.corpus_hash,
        "random_seed": settings.random_seed,
    }
    context: Any = nullcontext()
    if track:
        configure_mlflow(paths)
        context = track_diagnostics_run(f"hipoteses-{slug}", params=params)
    with context:
        result = _execute_target(
            target,
            discovery,
            settings,
            paths,
            target_slug=slug,
            classification=classification,
            generate_fn=generate_fn or generate_hypotheses,
        )
        if track:
            _log_result(result)
    assert_sanity_gate_passed(result.gate)
    return result


def _log_result(result: TargetRunResult) -> None:
    """Registra métricas do gate e, se houver, a tabela de hipóteses no MLflow."""
    gate = result.gate
    log_diagnostics_metrics(
        {
            f"gate_{gate.metric_name}": gate.holdout_score,
            "gate_ci_lower": gate.ci_lower,
            "gate_ci_upper": gate.ci_upper,
            "gate_permutation_pvalue": gate.permutation_pvalue,
            "gate_go": float(gate.go),
            "n_discovery": gate.n_train,
            "n_holdout": gate.n_holdout,
            "n_hypotheses": 0 if result.hypotheses is None else result.hypotheses.height,
        }
    )
    if result.output_path is not None:
        log_diagnostics_artifact(result.output_path)
    else:
        import mlflow

        mlflow.set_tag("aborted", f"sanity_gate: {gate.reason}")


def resolve_target_arguments(
    target_name: str, args: argparse.Namespace, settings: DiagnosticsSettings
) -> dict[str, Any]:
    """Combina argumentos da CLI com os padrões de ``configs/diagnostics.yaml -> targets``.

    Parameters
    ----------
    target_name : str
        Alvo escolhido.
    args : argparse.Namespace
        Argumentos da CLI (``model_column``, ``model_columns``, ``label``).
    settings : DiagnosticsSettings
        Configuração validada.

    Returns
    -------
    dict[str, Any]
        Argumentos para :func:`run_target_diagnostics`.

    Examples
    --------
    >>> resolve_target_arguments("uncertainty", args, settings)  # doctest: +SKIP
    {'model_column': None, 'model_columns': None, 'label': None}
    """
    defaults = settings.targets.get(target_name, {})
    return {
        "model_column": args.model_column or defaults.get("model_column"),
        "model_columns": args.model_columns,
        "label": args.label or defaults.get("label"),
    }


def load_diagnostic_corpus(
    settings: DiagnosticsSettings, paths: ProjectPaths, corpus_path: Path | None
) -> pl.DataFrame:
    """Carrega o corpus de diagnóstico.

    Sem ``corpus_path``, adapta ``paths.labeled_corpus_file`` (texto sanitizado,
    sem ``user_id``). Com ``corpus_path``, lê um parquet **já no contrato**
    (ex.: predições sobre o gold, para o alvo ``gold_error``).

    Parameters
    ----------
    settings : DiagnosticsSettings
        Configuração validada.
    paths : ProjectPaths
        Caminhos do projeto.
    corpus_path : Path | None
        Parquet no contrato de diagnóstico, ou ``None`` para o corpus rotulado padrão.

    Returns
    -------
    pl.DataFrame
        Corpus validado.
    """
    if corpus_path is not None:
        return validate_diagnostic_corpus(read_parquet(corpus_path))
    return adapt_labeled_corpus(
        read_dataset_file(paths.labeled_corpus_file),
        model_columns=settings.data.model_columns,
        agreement_column=settings.data.agreement_column,
        text_column=settings.data.text_column,
    )


def build_dry_run_report(
    target_name: str, corpus: pl.DataFrame, settings: DiagnosticsSettings, **target_arguments: Any
) -> str:
    """Estima chamadas, tokens e custo do alvo, **sem** rede, embeddings ou SAE.

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
    train_fraction = 1.0 - settings.splits.holdout_size - settings.splits.validation_size
    n_discovery = max(1, round(target.height * train_fraction))
    estimate = estimate_hypothesis_generation_cost(settings, n_discovery_tweets=n_discovery)
    header = (
        f"DRY-RUN do alvo '{target_name}': {target.height} tweets elegíveis "
        f"(~{n_discovery} na descoberta). Nenhuma chamada de rede foi feita."
    )
    return (
        header
        + "\n"
        + format_cost_report({"generate_hypotheses": estimate}, provider=settings.llm.provider)
    )


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
    >>> parse_arguments(["--target", "disagreement", "--dry-run"]).dry_run
    True
    """
    parser = argparse.ArgumentParser(
        prog="python -m diagnostics.hypotheses",
        description="Gera hipóteses HypotheSAEs para um alvo de diagnóstico da rotulagem.",
    )
    parser.add_argument("--target", required=True, choices=TARGET_NAMES)
    parser.add_argument("--model-column", help="Coluna lab_<modelo> (pseudo_label/gold_error).")
    parser.add_argument("--model-columns", nargs="+", help="Colunas lab_<modelo> (disagreement).")
    parser.add_argument("--label", help="Classe positiva do one-vs-rest (pseudo_label).")
    parser.add_argument("--corpus", type=Path, help="Parquet já no contrato de diagnóstico.")
    parser.add_argument("--config", type=Path, default=DEFAULT_DIAGNOSTICS_CONFIG_FILE)
    parser.add_argument("--dry-run", action="store_true", help="Só estima chamadas e custo.")
    parser.add_argument("--no-mlflow", action="store_true", help="Não registra no MLflow.")
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

    Raises
    ------
    DataValidationError
        Se ``--target gold_error`` for usado sem ``--corpus``.
    SanityGateFailedError
        Se o gate de sanidade reprovar o alvo.

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
    if args.target == "gold_error" and args.corpus is None:
        raise DataValidationError(
            schema_name="DiagnosticCorpusSchema",
            detail="o alvo 'gold_error' exige --corpus (predições sobre o gold no contrato)",
        )
    corpus = load_diagnostic_corpus(settings, paths, args.corpus)
    target_arguments = resolve_target_arguments(args.target, args, settings)

    if args.dry_run:
        report = build_dry_run_report(args.target, corpus, settings, **target_arguments)
        print(report)
        return 0

    configure_reproducibility(
        settings.random_seed,
        deterministic_algorithms=load_general_config().reproducibility.deterministic_algorithms,
    )
    discovery = prepare_discovery_data(corpus, settings, paths)
    run_target_diagnostics(
        args.target,
        corpus,
        discovery,
        settings,
        paths,
        track=not args.no_mlflow,
        **target_arguments,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
