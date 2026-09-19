"""Estágio ``diagnostics``: camada de diagnóstico HypotheSAEs pós-rotulagem (opt-in).

Encapsula, sob um único nome de estágio de ``src/main.py``, os três passos de
:mod:`diagnostics` (ver ``docs/guides/diagnostico-hypothesaes.md``):

* ``hypotheses``: gate de sanidade + hipóteses da partição de descoberta
  (:func:`diagnostics.hypotheses.run_target_diagnostics`);
* ``validation``: validação no holdout com Bonferroni e amostra para
  rotulagem humana (:func:`diagnostics.validation.run_validation_stage`);
* ``comparison``: prompts v1 vs v2 no gold set
  (:func:`diagnostics.comparison.run_prompt_comparison`).

O estágio NÃO faz parte de ``configs/config.yaml -> stages``: ``--stage all``
não o executa. Os módulos de ``diagnostics`` são importados dentro da função
para que a camada permaneça opt-in (nenhum custo de import nem dependência de
``torch``/``openai`` para os demais estágios).
"""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from config.paths import ProjectPaths
from exceptions.data import DataValidationError

if TYPE_CHECKING:
    from diagnostics.settings import DiagnosticsSettings

logger = logging.getLogger(__name__)

DiagnosticsStep = Literal["hypotheses", "validation", "comparison"]
DIAGNOSTICS_STEPS: tuple[str, ...] = ("hypotheses", "validation", "comparison")
DIAGNOSTICS_GOLD_CHOICES: tuple[str, ...] = ("tweetsentbr", "repro")


def _emit_report(report: str) -> str:
    """Registra e imprime o relatório de dry-run (a saída é o objetivo do comando)."""
    logger.info("\n%s", report)
    print(report)
    return report


def _load_settings(config_file: Path | None, random_seed: int | None) -> "DiagnosticsSettings":
    """Carrega ``configs/diagnostics.yaml`` e aplica o override opcional da semente."""
    from diagnostics.settings import DEFAULT_DIAGNOSTICS_CONFIG_FILE, load_diagnostics_settings

    settings = load_diagnostics_settings(config_file or DEFAULT_DIAGNOSTICS_CONFIG_FILE)
    if random_seed is not None:
        settings = settings.model_copy(update={"random_seed": random_seed})
    return settings


def _resolve_target_arguments(
    settings: "DiagnosticsSettings",
    target_name: str,
    model_column: str | None,
    label: str | None,
) -> dict[str, Any]:
    """Resolve ``model_column``/``label`` do alvo: argumento explícito ou padrão do YAML."""
    defaults = settings.targets.get(target_name, {})
    return {
        "model_column": model_column or defaults.get("model_column"),
        "label": label or defaults.get("label"),
    }


def _run_comparison_step(
    paths: ProjectPaths,
    settings: "DiagnosticsSettings",
    *,
    gold: str,
    models: Sequence[str] | None,
    dry_run: bool,
    track: bool,
) -> Any:
    """Executa (ou estima, em ``dry_run``) a comparação de prompts v1 vs v2 no gold set."""
    from diagnostics.comparison import build_comparison_dry_run, run_prompt_comparison

    gold_file = paths.tweetsentbr_file if gold == "tweetsentbr" else paths.repro_file
    chosen_models = list(models or [settings.llm.annotator_model])
    if dry_run:
        return _emit_report(build_comparison_dry_run(gold_file, settings, chosen_models))
    return run_prompt_comparison(gold_file, settings, paths, models=chosen_models, track=track)


def _run_validation_step(
    target_name: str,
    corpus: Any,
    settings: "DiagnosticsSettings",
    paths: ProjectPaths,
    target_arguments: dict[str, Any],
    *,
    dry_run: bool,
    track: bool,
) -> Any:
    """Executa (ou estima, em ``dry_run``) a validação das hipóteses no holdout."""
    from diagnostics.validation import build_dry_run_report, run_validation_stage

    if dry_run:
        return _emit_report(build_dry_run_report(target_name, corpus, settings, **target_arguments))
    return run_validation_stage(
        target_name, corpus, settings, paths, track=track, **target_arguments
    )


def _run_hypotheses_step(
    target_name: str,
    corpus: Any,
    settings: "DiagnosticsSettings",
    paths: ProjectPaths,
    target_arguments: dict[str, Any],
    *,
    dry_run: bool,
    track: bool,
) -> Any:
    """Executa (ou estima, em ``dry_run``) o gate de sanidade e a geração de hipóteses."""
    from diagnostics.hypotheses import build_dry_run_report, run_target_diagnostics
    from diagnostics.sae_runner import prepare_discovery_data

    if dry_run:
        return _emit_report(build_dry_run_report(target_name, corpus, settings, **target_arguments))
    discovery = prepare_discovery_data(corpus, settings, paths)
    return run_target_diagnostics(
        target_name, corpus, discovery, settings, paths, track=track, **target_arguments
    )


def run_diagnostics_stage(
    paths: ProjectPaths,
    *,
    step: DiagnosticsStep = "hypotheses",
    target_name: str = "disagreement",
    model_column: str | None = None,
    label: str | None = None,
    corpus_path: Path | None = None,
    gold: str = "tweetsentbr",
    models: Sequence[str] | None = None,
    dry_run: bool = False,
    track: bool = True,
    config_file: Path | None = None,
    random_seed: int | None = None,
) -> Any:
    """Executa um passo da camada de diagnóstico HypotheSAEs.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto.
    step : {"hypotheses", "validation", "comparison"}, optional
        Passo a executar, by default "hypotheses".
    target_name : str, optional
        Alvo (``disagreement``/``uncertainty``/``pseudo_label``/``gold_error``);
        usado em ``hypotheses`` e ``validation``, by default "disagreement".
    model_column : str | None, optional
        Coluna ``lab_<modelo>`` (``pseudo_label``/``gold_error``); por padrão, a de
        ``configs/diagnostics.yaml -> targets``, by default None.
    label : str | None, optional
        Classe do one-vs-rest (``pseudo_label``); por padrão, a do YAML, by default None.
    corpus_path : Path | None, optional
        Parquet já no contrato de diagnóstico (obrigatório para ``gold_error``); por padrão
        adapta ``paths.labeled_corpus_file``, by default None.
    gold : {"tweetsentbr", "repro"}, optional
        Gold set do passo ``comparison``, by default "tweetsentbr".
    models : Sequence[str] | None, optional
        Modelos classificadores do passo ``comparison``; por padrão o anotador do YAML,
        by default None.
    dry_run : bool, optional
        Só estima chamadas e custo (sem rede, embeddings, SAE ou MLflow), by default False.
    track : bool, optional
        Se registra no MLflow, by default True.
    config_file : Path | None, optional
        ``configs/diagnostics.yaml`` alternativo, by default None.
    random_seed : int | None, optional
        Sobrescreve ``random_seed`` do YAML, by default None.

    Returns
    -------
    Any
        Em ``dry_run``, o relatório (``str``); senão, o resultado do passo
        (``TargetRunResult``, ``ValidationOutcome`` ou ``pl.DataFrame``).

    Raises
    ------
    DataValidationError
        Se o passo for desconhecido, ou ``gold_error`` for usado sem ``corpus_path``.
    SanityGateFailedError
        Se o gate de sanidade reprovar o alvo (passo ``hypotheses``).

    Examples
    --------
    >>> run_diagnostics_stage(paths, target_name="uncertainty", dry_run=True)  # doctest: +SKIP
    """
    from diagnostics.hypotheses import load_diagnostic_corpus

    if step not in DIAGNOSTICS_STEPS:
        raise DataValidationError(
            schema_name="DiagnosticsStage",
            detail=f"passo desconhecido '{step}'; disponíveis: {list(DIAGNOSTICS_STEPS)}",
        )
    settings = _load_settings(config_file, random_seed)

    if step == "comparison":
        return _run_comparison_step(
            paths, settings, gold=gold, models=models, dry_run=dry_run, track=track
        )

    if target_name == "gold_error" and corpus_path is None:
        raise DataValidationError(
            schema_name="DiagnosticCorpusSchema",
            detail="o alvo 'gold_error' exige --diagnostics-corpus (predições sobre o gold)",
        )
    target_arguments = _resolve_target_arguments(settings, target_name, model_column, label)
    corpus = load_diagnostic_corpus(settings, paths, corpus_path)

    if step == "validation":
        return _run_validation_step(
            target_name, corpus, settings, paths, target_arguments, dry_run=dry_run, track=track
        )
    return _run_hypotheses_step(
        target_name, corpus, settings, paths, target_arguments, dry_run=dry_run, track=track
    )
