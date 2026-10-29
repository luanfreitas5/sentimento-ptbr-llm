"""Estágio ``hypotheses``: geração de hipóteses com o HypotheSAEs.

Reúne, sob um único estágio, toda a geração de hipóteses em linguagem natural
do projeto. O HypotheSAEs não classifica: treina um SAE sobre embeddings e
descreve padrões associados a um alvo derivado. Hipóteses sobre pseudo-rótulos
descrevem o comportamento dos modelos, nunca a verdade; só o gold set mede
acerto. Três modos, selecionados por ``mode``:

* ``disagreement`` (padrão): alvos de discordância/incerteza entre cada modelo do
  Hugging Face (primário, secundário e adicionais) e cada gold set
  (``gold_openai``/``gold_unb``)
  (:func:`diagnostics.model_disagreement.run_disagreement_hypotheses`);
* ``patterns``: descoberta de padrões e candidatos a inconsistência de
  rotulagem nos rótulos de baixa confiança
  (:func:`pipelines.hypothesaes_analysis.run_hypothesaes_analysis_stage`);
* ``diagnostics``: camada de diagnóstico completa — gate de sanidade,
  validação no holdout e comparação de prompts
  (:func:`pipelines.diagnostics_analysis.run_diagnostics_stage`).

Exige ``make install-hypothesaes`` e o LLM configurado em
``configs/diagnostics.yaml``/``configs/hypothesaes.yaml``.
"""

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from config.paths import ProjectPaths
from exceptions.data import DataNotFoundError, DataValidationError
from io_utils.json import write_json
from labeling.extra_sources import load_extra_model_names
from labeling.model_names import load_source_file_names
from pipelines.comparative_evaluation import (
    build_comparison_subdir,
    list_available_gold_names,
    list_available_model_names,
    load_comparison_frame,
)
from pipelines.diagnostics_analysis import run_diagnostics_stage
from pipelines.hypothesaes_analysis import run_hypothesaes_analysis_stage

logger = logging.getLogger(__name__)

HYPOTHESES_MODES: tuple[str, ...] = ("disagreement", "patterns", "diagnostics")


def _summarize_outcome(outcome: Any) -> dict[str, Any]:
    """Resumo serializável do resultado de um alvo do modo ``disagreement``."""
    return {
        "target": outcome.target,
        "status": outcome.status,
        "detail": outcome.detail,
        "n_hypotheses": outcome.n_hypotheses,
        "hypotheses_path": str(outcome.hypotheses_path or ""),
        "evidence_path": str(outcome.evidence_path or ""),
    }


def _run_patterns_mode(patterns_kwargs: Mapping[str, Any] | None) -> Any:
    """Delega ao estágio de padrões/inconsistências; exige os argumentos já montados."""
    if patterns_kwargs is None:
        raise DataValidationError(
            schema_name="HypothesesStage", detail="o modo 'patterns' exige patterns_kwargs"
        )
    return run_hypothesaes_analysis_stage(**patterns_kwargs)


def _run_disagreement_mode(
    paths: ProjectPaths,
    *,
    output_subdir: str,
    targets: Sequence[str],
    top_tweets_per_hypothesis: int,
    config_file: Path | None,
    random_seed: int,
    track: bool,
) -> list[dict[str, Any]]:
    """Gera hipóteses sobre a discordância/incerteza de cada modelo do Hugging Face e gold set."""
    from diagnostics.model_disagreement import run_disagreement_hypotheses

    file_names = {**load_extra_model_names(paths), **load_source_file_names()}
    available = list_available_gold_names(paths)
    if not available:
        raise DataNotFoundError(
            "nenhum gold set encontrado; gere-os com `make label-gold-openai` e/ou "
            "`make label-gold-unb`"
        )
    summary: list[dict[str, Any]] = []
    for model_name in list_available_model_names(paths):
        for gold_name in available:
            pair_subdir = build_comparison_subdir(output_subdir, file_names, gold_name, model_name)
            outcomes = run_disagreement_hypotheses(
                load_comparison_frame(paths, gold_name, model_name),
                paths,
                output_dir=paths.reports_tables_dir / pair_subdir,
                targets=targets,
                top_tweets_per_hypothesis=top_tweets_per_hypothesis,
                config_file=config_file,
                random_seed=random_seed,
                track=track,
            )
            pair_summary = [
                {"model": model_name, "gold": gold_name} | _summarize_outcome(o) for o in outcomes
            ]
            write_json(pair_summary, paths.reports_metrics_dir / f"hipoteses_{pair_subdir}.json")
            for item in pair_summary:
                logger.info(
                    "Hipóteses '%s' (%s × %s): %s — %s.",
                    item["target"],
                    model_name,
                    gold_name,
                    item["status"],
                    item["detail"],
                )
            summary.extend(pair_summary)
    return summary


def run_hypotheses_stage(
    paths: ProjectPaths,
    *,
    mode: str = "disagreement",
    skip: bool = False,
    output_subdir: str = "comparativo",
    targets: Sequence[str] = ("disagreement", "uncertainty"),
    top_tweets_per_hypothesis: int = 10,
    diagnostics_config_file: Path | None = None,
    random_seed: int = 42,
    track_with_mlflow: bool = True,
    patterns_kwargs: Mapping[str, Any] | None = None,
    diagnostics_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Executa a geração de hipóteses do HypotheSAEs no modo escolhido.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto.
    mode : {"disagreement", "patterns", "diagnostics"}, optional
        Modo de geração, by default "disagreement".
    skip : bool, optional
        Se ``True``, não executa nada (útil em ``--stage all`` sem LLM/GPU), by default False.
    output_subdir : str, optional
        Prefixo das subpastas de ``reports/tables`` (modo ``disagreement``), igual ao da
        etapa ``comparative_evaluation`` (uma subpasta por gold set), by default "comparativo".
    targets : Sequence[str], optional
        ``disagreement`` e/ou ``uncertainty`` (modo ``disagreement``), by default os dois.
    top_tweets_per_hypothesis : int, optional
        Tweets de evidência por hipótese (modo ``disagreement``), by default 10.
    diagnostics_config_file : Path | None, optional
        ``configs/diagnostics.yaml`` alternativo (modo ``disagreement``), by default None.
    random_seed : int, optional
        Semente do modo ``disagreement``, by default 42.
    track_with_mlflow : bool, optional
        Se registra cada alvo no MLflow (modo ``disagreement``), by default True.
    patterns_kwargs : Mapping[str, Any] | None, optional
        Argumentos de :func:`pipelines.hypothesaes_analysis.run_hypothesaes_analysis_stage`
        (modo ``patterns``), by default None.
    diagnostics_kwargs : Mapping[str, Any] | None, optional
        Argumentos de :func:`pipelines.diagnostics_analysis.run_diagnostics_stage`, exceto
        ``paths`` (modo ``diagnostics``), by default None.

    Returns
    -------
    Any
        ``None`` se ``skip``; no modo ``disagreement``, o resumo por alvo (``list[dict]``); nos
        demais, o resultado do estágio delegado.

    Raises
    ------
    DataValidationError
        Se ``mode`` for desconhecido ou faltarem os argumentos do modo escolhido.
    DataNotFoundError
        Se o Hugging Face ou nenhum gold set (modo ``disagreement``) foi gerado.

    Examples
    --------
    >>> run_hypotheses_stage(paths, skip=True) is None
    True
    """
    if mode not in HYPOTHESES_MODES:
        raise DataValidationError(
            schema_name="HypothesesStage",
            detail=f"modo desconhecido '{mode}'; disponíveis: {list(HYPOTHESES_MODES)}",
        )
    if skip:
        logger.info("Geração de hipóteses dispensada (--skip-hypotheses).")
        return None

    if mode == "patterns":
        return _run_patterns_mode(patterns_kwargs)

    if mode == "diagnostics":
        return run_diagnostics_stage(paths, **(diagnostics_kwargs or {}))

    return _run_disagreement_mode(
        paths,
        output_subdir=output_subdir,
        targets=targets,
        top_tweets_per_hypothesis=top_tweets_per_hypothesis,
        config_file=diagnostics_config_file,
        random_seed=random_seed,
        track=track_with_mlflow,
    )
