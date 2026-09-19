"""Comparação pareada entre as versões de prompt v1 e v2 no gold set.

Reclassifica ``G_eval`` (parte do gold que NUNCA entra na descoberta de
hipóteses) com v1 e v2 e reporta, por modelo:

* MCC e macro-F1 de cada versão, com IC por bootstrap;
* o **ganho** (v2 − v1) com IC por bootstrap pareado;
* McNemar exato por tweet (acertos/erros pareados);
* Wilcoxon entre folds pareados de ``G_eval`` (com ``temperature=0`` as seeds
  seriam idênticas, então a variação vem de reamostrar tweets, sem multiplicar
  as chamadas por seed);
* taxa de discordância e agreement entre modelos, quando há mais de um.

Também mede se as hipóteses de discordância **enfraquecem** após o v2
(:func:`summarize_hypothesis_weakening`): rode o HypotheSAEs sobre ``G_disc``
(disjunto de ``G_eval``) com as predições v1 e v2 e compare as duas tabelas.

Uso::

    PYTHONPATH=src python -m diagnostics.comparison --gold tweetsentbr --dry-run
"""

import argparse
import asyncio
import logging
import math
import re
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from config.environment import configure_environment_variables
from config.logging import configure_logging
from config.paths import ProjectPaths, load_project_paths, resolve_project_path
from data.gold import load_gold_set, split_gold_discovery_eval
from diagnostics.cost import (
    estimate_classification_cost,
    estimate_prompt_tokens,
    format_cost_report,
)
from diagnostics.gold_eval import calculate_scores_with_ci
from diagnostics.llm_client import AsyncLLMClient, CallStats, DiskCompletionCache
from diagnostics.settings import (
    DEFAULT_DIAGNOSTICS_CONFIG_FILE,
    DiagnosticsSettings,
    load_diagnostics_settings,
)
from diagnostics.tracking import (
    configure_mlflow,
    log_diagnostics_artifact,
    log_diagnostics_metrics,
    track_diagnostics_run,
)
from evaluation.significance import run_mcnemar_test, run_wilcoxon_signed_rank_test
from exceptions.data import DataValidationError
from io_utils.csv import write_csv
from io_utils.parquet import write_parquet
from labeling.llm_relabeling import parse_relabel_response
from metrics.classification import calculate_matthews_correlation_coefficient

logger = logging.getLogger(__name__)

TEXT_PLACEHOLDER = "{{TEXTO}}"
CLASSIFICATION_MAX_TOKENS = 200


def slugify_model(model_name: str) -> str:
    """Converte um nome de modelo em sufixo seguro para nome de coluna.

    Parameters
    ----------
    model_name : str
        Ex.: ``gemma2:9b``.

    Returns
    -------
    str
        Ex.: ``gemma2_9b``.

    Examples
    --------
    >>> slugify_model("gemma2:9b")
    'gemma2_9b'
    """
    return re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_")


def render_prompt(template: str, text: str) -> str:
    """Substitui ``{{TEXTO}}`` pelo tweet (``str.replace``: o template tem JSON com chaves).

    Parameters
    ----------
    template : str
        Prompt de rotulagem (v1 ou v2).
    text : str
        Tweet sanitizado.

    Returns
    -------
    str
        Prompt pronto.

    Examples
    --------
    >>> render_prompt("Tweet: {{TEXTO}}", "oi")
    'Tweet: oi'
    """
    return template.replace(TEXT_PLACEHOLDER, text)


async def classify_texts_async(
    client: AsyncLLMClient, texts: Sequence[str], template: str, *, model: str, namespace: str
) -> list[str | None]:
    """Classifica tweets com um template de prompt, concorrentemente.

    Parameters
    ----------
    client : AsyncLLMClient
        Cliente LLM.
    texts : Sequence[str]
        Tweets sanitizados.
    template : str
        Prompt com ``{{TEXTO}}``.
    model : str
        Modelo classificador.
    namespace : str
        Separador de cache (versão do prompt).

    Returns
    -------
    list[str | None]
        Rótulos na ordem de entrada; ``None`` onde a resposta não pôde ser interpretada.
    """
    completions = await client.complete_many(
        [render_prompt(template, text) for text in texts],
        model=model,
        max_tokens=CLASSIFICATION_MAX_TOKENS,
        namespace=namespace,
    )
    labels: list[str | None] = []
    for completion in completions:
        parsed = parse_relabel_response(completion) if completion else None
        labels.append(parsed[0] if parsed else None)
    return labels


def classify_texts(
    settings: DiagnosticsSettings,
    texts: Sequence[str],
    template: str,
    *,
    model: str,
    namespace: str,
    client: AsyncLLMClient | None = None,
) -> tuple[list[str | None], CallStats]:
    """Versão síncrona de :func:`classify_texts_async` (com cache em disco).

    Parameters
    ----------
    settings : DiagnosticsSettings
        Configuração validada.
    texts : Sequence[str]
        Tweets sanitizados.
    template : str
        Prompt com ``{{TEXTO}}``.
    model : str
        Modelo classificador.
    namespace : str
        Separador de cache (versão do prompt).
    client : AsyncLLMClient | None, optional
        Cliente pronto (testes), by default None.

    Returns
    -------
    tuple[list[str | None], CallStats]
        Rótulos e contadores.
    """

    async def _run() -> tuple[list[str | None], CallStats]:
        active = client or AsyncLLMClient(
            settings.llm, cache=DiskCompletionCache(resolve_project_path(settings.llm.cache_dir))
        )
        return await classify_texts_async(
            active, texts, template, model=model, namespace=namespace
        ), active.stats

    return asyncio.run(_run())


@dataclass(frozen=True)
class PromptComparison:
    """Resultado pareado v1 vs v2 para um modelo.

    Attributes
    ----------
    n_paired : int
        Tweets com predição válida nas duas versões.
    n_unparsed : int
        Tweets descartados por resposta não interpretável em alguma versão.
    scores_v1, scores_v2 : dict[str, float]
        MCC e macro-F1 com ICs (ver :func:`diagnostics.gold_eval.calculate_scores_with_ci`).
    delta_mcc, delta_f1_macro : float
        Ganho (v2 − v1).
    delta_mcc_ci, delta_f1_macro_ci : tuple[float, float]
        IC do ganho por bootstrap pareado.
    mcnemar_pvalue : float
        p-valor do McNemar exato por tweet.
    wilcoxon_pvalue : float
        p-valor do Wilcoxon entre folds pareados.
    """

    n_paired: int
    n_unparsed: int
    scores_v1: dict[str, float]
    scores_v2: dict[str, float]
    delta_mcc: float
    delta_f1_macro: float
    delta_mcc_ci: tuple[float, float]
    delta_f1_macro_ci: tuple[float, float]
    mcnemar_pvalue: float
    wilcoxon_pvalue: float

    def to_row(self, model: str) -> dict[str, Any]:
        """Achata o resultado em uma linha de tabela (uma por modelo)."""
        row: dict[str, Any] = {
            "model": model,
            "n_paired": self.n_paired,
            "n_unparsed": self.n_unparsed,
        }
        row.update({f"v1_{key}": value for key, value in self.scores_v1.items()})
        row.update({f"v2_{key}": value for key, value in self.scores_v2.items()})
        row.update(
            delta_mcc=self.delta_mcc,
            delta_mcc_ci_low=self.delta_mcc_ci[0],
            delta_mcc_ci_high=self.delta_mcc_ci[1],
            delta_f1_macro=self.delta_f1_macro,
            delta_f1_macro_ci_low=self.delta_f1_macro_ci[0],
            delta_f1_macro_ci_high=self.delta_f1_macro_ci[1],
            mcnemar_pvalue=self.mcnemar_pvalue,
            wilcoxon_pvalue=self.wilcoxon_pvalue,
        )
        return row


def _macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    """Macro-F1 sobre as classes presentes (sklearn, sem avisos de classe ausente)."""
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))  # type: ignore[reportArgumentType]


def _paired_delta_interval(
    y_true: list[str],
    pred_v1: list[str],
    pred_v2: list[str],
    *,
    n_bootstrap: int,
    confidence_level: float,
    seed: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """IC bootstrap pareado do ganho (v2 − v1) em MCC e macro-F1."""
    rng = np.random.default_rng(seed)
    truth, first, second = np.array(y_true), np.array(pred_v1), np.array(pred_v2)
    mcc_deltas: list[float] = []
    f1_deltas: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, truth.size, truth.size)
        t, a, b = truth[idx].tolist(), first[idx].tolist(), second[idx].tolist()
        mcc_deltas.append(
            calculate_matthews_correlation_coefficient(t, b)
            - calculate_matthews_correlation_coefficient(t, a)
        )
        f1_deltas.append(_macro_f1(t, b) - _macro_f1(t, a))
    tail = (1.0 - confidence_level) / 2.0 * 100.0
    mcc_low, mcc_high = np.percentile(mcc_deltas, [tail, 100.0 - tail])
    f1_low, f1_high = np.percentile(f1_deltas, [tail, 100.0 - tail])
    return (float(mcc_low), float(mcc_high)), (float(f1_low), float(f1_high))


def _fold_wilcoxon_pvalue(
    y_true: list[str], pred_v1: list[str], pred_v2: list[str], *, n_folds: int, seed: int
) -> float:
    """Wilcoxon sobre o MCC por fold pareado; 1.0 se todas as diferenças forem nulas."""
    order = np.random.default_rng(seed).permutation(len(y_true))
    scores_v1: list[float] = []
    scores_v2: list[float] = []
    for fold in np.array_split(order, n_folds):
        truth = [y_true[i] for i in fold]
        scores_v1.append(
            calculate_matthews_correlation_coefficient(truth, [pred_v1[i] for i in fold])
        )
        scores_v2.append(
            calculate_matthews_correlation_coefficient(truth, [pred_v2[i] for i in fold])
        )
    if all(math.isclose(a, b) for a, b in zip(scores_v1, scores_v2, strict=True)):
        return 1.0
    return run_wilcoxon_signed_rank_test(scores_v1, scores_v2)["p_value"]


def compare_prompt_versions(
    y_true: Sequence[str],
    pred_v1: Sequence[str | None],
    pred_v2: Sequence[str | None],
    *,
    n_folds: int,
    n_bootstrap: int,
    confidence_level: float,
    random_seed: int,
) -> PromptComparison:
    """Compara v1 e v2 de forma pareada por tweet.

    Parameters
    ----------
    y_true : Sequence[str]
        Rótulos gold.
    pred_v1, pred_v2 : Sequence[str | None]
        Predições de cada versão (``None`` = resposta não interpretável).
    n_folds : int
        Folds pareados do Wilcoxon (>= 6 para que p < 0,05 seja alcançável).
    n_bootstrap : int
        Reamostragens dos ICs.
    confidence_level : float
        Nível dos ICs.
    random_seed : int
        Semente.

    Returns
    -------
    PromptComparison
        Métricas, ganho com IC, McNemar e Wilcoxon.

    Raises
    ------
    DataValidationError
        Se os vetores tiverem tamanhos diferentes ou restarem menos de ``n_folds`` pares.

    Examples
    --------
    >>> compare_prompt_versions(
    ...     y, p1, p2, n_folds=10, n_bootstrap=200, confidence_level=0.95, random_seed=0
    ... )  # doctest: +SKIP
    """
    if not len(y_true) == len(pred_v1) == len(pred_v2):
        raise DataValidationError(schema_name="PromptComparison", detail="tamanhos diferentes")
    pairs = [
        (t, a, b)
        for t, a, b in zip(y_true, pred_v1, pred_v2, strict=True)
        if a is not None and b is not None
    ]
    if len(pairs) < n_folds:
        raise DataValidationError(
            schema_name="PromptComparison",
            detail=f"apenas {len(pairs)} pares válidos (< {n_folds} folds)",
        )
    truth, first, second = (list(column) for column in zip(*pairs, strict=True))
    options = {
        "n_bootstrap": n_bootstrap,
        "confidence_level": confidence_level,
        "seed": random_seed,
    }
    scores_v1 = calculate_scores_with_ci(truth, first, **options)
    scores_v2 = calculate_scores_with_ci(truth, second, **options)
    mcc_ci, f1_ci = _paired_delta_interval(
        truth,
        first,
        second,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=random_seed,
    )
    return PromptComparison(
        n_paired=len(pairs),
        n_unparsed=len(y_true) - len(pairs),
        scores_v1=scores_v1,
        scores_v2=scores_v2,
        delta_mcc=scores_v2["mcc"] - scores_v1["mcc"],
        delta_f1_macro=scores_v2["f1_macro"] - scores_v1["f1_macro"],
        delta_mcc_ci=mcc_ci,
        delta_f1_macro_ci=f1_ci,
        mcnemar_pvalue=run_mcnemar_test(truth, first, second)["p_value"],
        wilcoxon_pvalue=_fold_wilcoxon_pvalue(
            truth, first, second, n_folds=n_folds, seed=random_seed
        ),
    )


def calculate_disagreement_and_agreement(
    predictions: Mapping[str, Sequence[str | None]],
) -> dict[str, float]:
    """Calcula a taxa de discordância e o agreement médio entre modelos.

    Considera só tweets em que todos os modelos têm predição válida. O agreement
    de um tweet é a fração de modelos que votam no rótulo mais frequente.

    Parameters
    ----------
    predictions : dict[str, Sequence[str | None]]
        Modelo -> predições alinhadas (>= 2 modelos).

    Returns
    -------
    dict[str, float]
        ``disagreement_rate``, ``mean_agreement`` e ``n_tweets`` (``nan`` se não houver
        tweets comparáveis).

    Examples
    --------
    >>> calculate_disagreement_and_agreement({"a": ["x", "y"], "b": ["x", "x"]})
    {'disagreement_rate': 0.5, 'mean_agreement': 0.75, 'n_tweets': 2.0}
    """
    columns = list(predictions.values())
    rows = [row for row in zip(*columns, strict=True) if all(label is not None for label in row)]
    if not rows:
        return {"disagreement_rate": math.nan, "mean_agreement": math.nan, "n_tweets": 0.0}
    disagreements = [len(set(row)) > 1 for row in rows]
    agreements = [max(row.count(label) for label in set(row)) / len(row) for row in rows]
    return {
        "disagreement_rate": float(np.mean(disagreements)),
        "mean_agreement": float(np.mean(agreements)),
        "n_tweets": float(len(rows)),
    }


def summarize_hypothesis_weakening(
    before: pl.DataFrame, after: pl.DataFrame, *, top_k: int = 10, alpha: float = 0.1
) -> dict[str, float | bool]:
    """Compara as hipóteses mais fortes antes (v1) e depois (v2) do novo prompt.

    Espera-se que, se o v2 corrigiu o padrão de discordância, as principais
    hipóteses enfraqueçam: menor ``|separation_score|`` médio e menos hipóteses
    significativas (Bonferroni).

    Parameters
    ----------
    before, after : pl.DataFrame
        Tabelas de :mod:`diagnostics.hypotheses` sobre ``G_disc`` com predições v1 e v2
        (colunas ``separation_score`` e ``regression_pval``).
    top_k : int, optional
        Nº de hipóteses mais fortes comparadas, by default 10.
    alpha : float, optional
        Alfa antes de Bonferroni, by default 0.1.

    Returns
    -------
    dict[str, float | bool]
        Médias de ``|separation_score|``, nº de significativas antes/depois e ``weakened``.

    Examples
    --------
    >>> t = pl.DataFrame({"separation_score": [0.3], "regression_pval": [0.001]})
    >>> summarize_hypothesis_weakening(t, t)["weakened"]
    False
    """

    def _summary(table: pl.DataFrame) -> tuple[float, int]:
        top = (
            table.with_columns(pl.col("separation_score").abs().alias("_abs"))
            .sort("_abs", descending=True)
            .head(top_k)
        )
        if top.is_empty():
            return 0.0, 0
        threshold = alpha / max(1, table.height)
        mean_abs = float(top.select(pl.col("_abs").mean()).item())
        return mean_abs, int((top["regression_pval"] < threshold).sum())

    mean_before, sig_before = _summary(before)
    mean_after, sig_after = _summary(after)
    return {
        "mean_abs_separation_before": mean_before,
        "mean_abs_separation_after": mean_after,
        "n_significant_before": float(sig_before),
        "n_significant_after": float(sig_after),
        "weakened": bool(mean_after < mean_before and sig_after <= sig_before),
    }


def _read_prompt(settings: DiagnosticsSettings, file_name: str) -> str:
    """Lê um prompt de rotulagem da pasta configurada (v1 ou v2)."""
    path = resolve_project_path(Path(settings.prompts.directory) / file_name)
    if not path.is_file():
        raise DataValidationError(schema_name="PromptFile", detail=f"prompt não encontrado: {path}")
    return path.read_text(encoding="utf-8")


def _contract_frame(
    gold: pl.DataFrame, predictions: dict[str, list[str | None]], agreement: list[float]
) -> pl.DataFrame:
    """Monta o parquet no contrato de diagnóstico (``lab_*`` + ``gold_label``)."""
    columns: dict[str, Any] = {
        "id": gold["id"],
        "text_normalized": gold["text_normalized"],
        "agreement_score": agreement,
    }
    columns.update({f"lab_{slugify_model(model)}": labels for model, labels in predictions.items()})
    columns["gold_label"] = gold["gold_label"]
    return pl.DataFrame(columns)


def _classify_all(
    settings: DiagnosticsSettings,
    gold: pl.DataFrame,
    template: str,
    models: Sequence[str],
    version: str,
) -> dict[str, list[str | None]]:
    """Classifica o gold com cada modelo usando uma versão de prompt."""
    texts = gold["text_normalized"].to_list()
    return {
        model: classify_texts(
            settings, texts, template, model=model, namespace=f"labeling_{version}"
        )[0]
        for model in models
    }


def build_dry_run_report(
    settings: DiagnosticsSettings, *, n_gold: int, models: Sequence[str], prompt_tokens: int
) -> str:
    """Estima a reclassificação (gold × 2 versões × modelos) sem rede.

    Parameters
    ----------
    settings : DiagnosticsSettings
        Configuração validada.
    n_gold : int
        Tweets do gold (G_disc + G_eval, já limitados por ``gold_max_tweets``).
    models : Sequence[str]
        Modelos classificadores.
    prompt_tokens : int
        Tokens estimados do prompt v1.

    Returns
    -------
    str
        Relatório multilinha em pt-BR.
    """
    estimate = estimate_classification_cost(
        settings, n_tweets=n_gold, n_prompts=2 * len(models), prompt_tokens=prompt_tokens
    )
    return (
        f"DRY-RUN da comparação v1 x v2: {n_gold} tweets do gold x 2 versões x "
        f"{len(models)} modelo(s). Nenhuma chamada de rede foi feita.\n"
        + format_cost_report({"reclassificacao_gold": estimate}, provider=settings.llm.provider)
    )


def build_comparison_dry_run(
    gold_file: Path, settings: DiagnosticsSettings, models: Sequence[str]
) -> str:
    """Lê o gold (limitado por ``gold_max_tweets``) e estima a reclassificação, sem rede.

    Parameters
    ----------
    gold_file : Path
        Gold set (ex.: ``paths.tweetsentbr_file``).
    settings : DiagnosticsSettings
        Configuração validada.
    models : Sequence[str]
        Modelos classificadores.

    Returns
    -------
    str
        Relatório multilinha em pt-BR (ver :func:`build_dry_run_report`).

    Examples
    --------
    >>> build_comparison_dry_run(gold_file, settings, ["gemma2:9b"])  # doctest: +SKIP
    """
    gold = load_gold_set(
        gold_file,
        max_tweets=settings.comparison.gold_max_tweets,
        random_seed=settings.random_seed,
    )
    v1_tokens = estimate_prompt_tokens(_read_prompt(settings, settings.prompts.v1))
    return build_dry_run_report(
        settings, n_gold=gold.height, models=models, prompt_tokens=v1_tokens
    )


def run_prompt_comparison(
    gold_file: Path,
    settings: DiagnosticsSettings,
    paths: ProjectPaths,
    *,
    models: Sequence[str] | None = None,
    track: bool = True,
) -> pl.DataFrame:
    """Reclassifica o gold com v1 e v2 e compara em ``G_eval``; grava ``G_disc`` para o rerun.

    Parameters
    ----------
    gold_file : Path
        Gold set (ex.: ``paths.tweetsentbr_file``).
    settings : DiagnosticsSettings
        Configuração validada.
    paths : ProjectPaths
        Caminhos do projeto.
    models : Sequence[str] | None, optional
        Modelos classificadores; por padrão ``[llm.annotator_model]``, by default None.
    track : bool, optional
        Se registra no MLflow, by default True.

    Returns
    -------
    pl.DataFrame
        Uma linha por modelo com métricas, ganho v2−v1 (IC), McNemar e Wilcoxon.

    Raises
    ------
    DataValidationError
        Se o v1/v2 não existirem ou houver poucos pares válidos.
    """
    cmp = settings.comparison
    chosen_models = list(models or [settings.llm.annotator_model])
    gold = load_gold_set(
        gold_file, max_tweets=cmp.gold_max_tweets, random_seed=settings.random_seed
    )
    g_disc, g_eval = split_gold_discovery_eval(
        gold, eval_fraction=cmp.gold_eval_fraction, random_seed=settings.random_seed
    )
    templates = {
        "v1": _read_prompt(settings, settings.prompts.v1),
        "v2": _read_prompt(settings, settings.prompts.v2),
    }
    context: Any = nullcontext()
    if track:
        configure_mlflow(paths)
        context = track_diagnostics_run(
            "comparacao-prompt-v1-v2",
            params={
                "gold_file": gold_file.name,
                "models": chosen_models,
                "n_eval": g_eval.height,
                "n_disc": g_disc.height,
                "prompt_v1": settings.prompts.v1,
                "prompt_v2": settings.prompts.v2,
            },
        )
    with context:
        eval_preds: dict[str, dict[str, list[str | None]]] = {}
        agreements: dict[str, dict[str, float] | None] = {}
        for version, template in templates.items():
            eval_preds[version] = _classify_all(settings, g_eval, template, chosen_models, version)
            disc_preds = _classify_all(settings, g_disc, template, chosen_models, version)
            agreements[version] = (
                calculate_disagreement_and_agreement(disc_preds) if len(disc_preds) > 1 else None
            )
            write_parquet(
                _contract_frame(g_disc, disc_preds, _per_tweet_agreement(disc_preds)),
                paths.data_interim_dir / "diagnostics" / f"gold_disc_{version}.parquet",
            )
        result = pl.DataFrame(
            [
                compare_prompt_versions(
                    g_eval["gold_label"].to_list(),
                    eval_preds["v1"][model],
                    eval_preds["v2"][model],
                    n_folds=cmp.n_folds,
                    n_bootstrap=cmp.n_bootstrap,
                    confidence_level=cmp.confidence_level,
                    random_seed=settings.random_seed,
                ).to_row(model)
                for model in chosen_models
            ]
        )
        table_path = paths.reports_metrics_dir / "diagnostics_comparacao_v1_v2.csv"
        write_csv(result, table_path)
        if track:
            _log_comparison(result, agreements["v1"], agreements["v2"], table_path)
    return result


def _per_tweet_agreement(predictions: dict[str, list[str | None]]) -> list[float]:
    """Agreement por tweet: fração de votos do rótulo mais frequente (0 se ninguém votou)."""
    rows = zip(*predictions.values(), strict=True)
    result: list[float] = []
    for row in rows:
        votes = [label for label in row if label is not None]
        result.append(
            max(votes.count(label) for label in set(votes)) / len(votes) if votes else 0.0
        )
    return result


def _log_comparison(
    result: pl.DataFrame,
    v1_agreement: dict[str, float] | None,
    v2_agreement: dict[str, float] | None,
    table_path: Path,
) -> None:
    """Registra ganhos, p-valores e discordância/agreement no MLflow."""
    metrics: dict[str, float] = {}
    for row in result.iter_rows(named=True):
        prefix = slugify_model(str(row["model"]))
        for key in (
            "delta_mcc",
            "delta_mcc_ci_low",
            "delta_mcc_ci_high",
            "delta_f1_macro",
            "mcnemar_pvalue",
            "wilcoxon_pvalue",
            "v1_mcc",
            "v2_mcc",
        ):
            metrics[f"{prefix}_{key}"] = float(row[key])
    for version, summary in (("v1", v1_agreement), ("v2", v2_agreement)):
        if summary:
            metrics[f"{version}_disagreement_rate"] = summary["disagreement_rate"]
            metrics[f"{version}_mean_agreement"] = summary["mean_agreement"]
    log_diagnostics_metrics(metrics)
    log_diagnostics_artifact(table_path)


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
    >>> parse_arguments(["--gold", "repro"]).gold
    'repro'
    """
    parser = argparse.ArgumentParser(
        prog="python -m diagnostics.comparison",
        description="Compara os prompts v1 e v2 no gold set (McNemar, Wilcoxon, IC bootstrap).",
    )
    parser.add_argument("--gold", required=True, choices=("tweetsentbr", "repro"))
    parser.add_argument("--models", nargs="+", help="Modelos classificadores.")
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
    >>> main(["--gold", "tweetsentbr", "--dry-run"])  # doctest: +SKIP
    0
    """
    args = parse_arguments(argv)
    configure_environment_variables()
    configure_logging()
    settings = load_diagnostics_settings(args.config)
    paths = load_project_paths()
    gold_file = paths.tweetsentbr_file if args.gold == "tweetsentbr" else paths.repro_file
    models = args.models or [settings.llm.annotator_model]
    if args.dry_run:
        print(build_comparison_dry_run(gold_file, settings, models))
        return 0
    result = run_prompt_comparison(
        gold_file, settings, paths, models=models, track=not args.no_mlflow
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
