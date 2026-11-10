"""Resumos em texto (pt-BR) dos resultados de cada etapa do diagnóstico.

Complementa as tabelas ``.parquet``/``.csv``: o mesmo conteúdo que a etapa
mostra na tela fica também em ``resumo_<etapa>_<alvo>.txt``, ao lado das tabelas.

Functions
---------
format_hypotheses_summary
    Gate de sanidade + principais hipóteses de um alvo.
format_validation_summary
    Hipóteses testadas, sobreviventes à correção e amostra para rotulagem.
format_comparison_summary
    Comparação de prompts v1 vs v2 por modelo.
save_summary_report
    Grava o texto em ``resumo_<name>.txt``.
"""

import logging
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)

TOP_HYPOTHESES_IN_SUMMARY = 10


def _format_value(value: Any) -> str:
    """Formata um valor de célula: floats com 4 algarismos significativos, demais com ``str``."""
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def format_hypotheses_summary(result: Any, *, top_n: int = TOP_HYPOTHESES_IN_SUMMARY) -> str:
    """Resume o resultado de um alvo: decisão do gate e principais hipóteses.

    Parameters
    ----------
    result : Any
        ``TargetRunResult`` (``target_slug``, ``gate``, ``hypotheses``, ``output_path``).
    top_n : int, optional
        Quantidade de hipóteses listadas, by default 10.

    Returns
    -------
    str
        Texto multilinha em pt-BR.

    Examples
    --------
    >>> print(format_hypotheses_summary(result))  # doctest: +SKIP
    """
    gate = result.gate
    lines = [
        f"RESUMO das hipóteses do alvo '{result.target_slug}'",
        f"Gate de sanidade: {'APROVADO' if gate.go else 'REPROVADO'} — {gate.reason}",
        f"  métrica={gate.metric_name} holdout={_format_value(float(gate.holdout_score))} "
        f"IC=[{_format_value(float(gate.ci_lower))}; {_format_value(float(gate.ci_upper))}] "
        f"p_perm={_format_value(float(gate.permutation_pvalue))} "
        f"n_descoberta={gate.n_train} n_holdout={gate.n_holdout}",
    ]
    table: pl.DataFrame | None = result.hypotheses
    if table is None:
        lines.append("Nenhuma hipótese gerada (gate reprovado).")
        return "\n".join(lines)
    lines.append(f"Hipóteses geradas: {table.height} (tabela: {result.output_path})")
    if "separation_score" in table.columns:
        table = table.sort("separation_score", descending=True, nulls_last=True)
    for rank, row in enumerate(table.head(top_n).iter_rows(named=True), start=1):
        details = ", ".join(
            f"{key}={_format_value(row[key])}"
            for key in ("separation_score", "regression_pval", "feature_prevalence")
            if key in row and row[key] is not None
        )
        lines.append(f"  {rank:>2}. {row.get('hypothesis')} [{details}]")
    return "\n".join(lines)


def format_validation_summary(
    target_slug: str, outcome: Any, *, correction: str, top_n: int = TOP_HYPOTHESES_IN_SUMMARY
) -> str:
    """Resume a validação no holdout: testadas, sobreviventes e amostra gerada.

    Parameters
    ----------
    target_slug : str
        Identificador do alvo.
    outcome : Any
        ``ValidationOutcome`` (``validated``, ``n_survivors``, ``sample_path``).
    correction : str
        Correção de múltiplos testes aplicada.
    top_n : int, optional
        Hipóteses listadas (primeiro as sobreviventes), by default 10.

    Returns
    -------
    str
        Texto multilinha em pt-BR.

    Examples
    --------
    >>> print(format_validation_summary("uncertainty", outcome, correction="bonferroni"))
    ... # doctest: +SKIP
    """
    validated: pl.DataFrame = outcome.validated
    lines = [
        f"RESUMO da validação do alvo '{target_slug}'",
        f"{outcome.n_survivors}/{validated.height} hipóteses sobrevivem à correção '{correction}'.",
        f"Amostra para rotulagem humana: {outcome.sample_path or 'não gerada'}",
    ]
    ordered = validated.sort(
        ["survives", "regression_pval_adjusted"], descending=[True, False], nulls_last=True
    )
    for rank, row in enumerate(ordered.head(top_n).iter_rows(named=True), start=1):
        mark = "SOBREVIVE" if row["survives"] else "descartada"
        lines.append(
            f"  {rank:>2}. [{mark}] {row['hypothesis']} "
            f"(p_ajustado={_format_value(row['regression_pval_adjusted'])})"
        )
    return "\n".join(lines)


def format_comparison_summary(
    result: pl.DataFrame,
    v1_agreement: Mapping[str, float] | None,
    v2_agreement: Mapping[str, float] | None,
    *,
    gold_name: str | None = None,
) -> str:
    """Resume a comparação de prompts v1 vs v2 (uma seção por modelo).

    Parameters
    ----------
    result : pl.DataFrame
        Tabela de resultados (uma linha por modelo).
    v1_agreement, v2_agreement : Mapping[str, float] | None
        Discordância/agreement médios na descoberta (``None`` com um só modelo).
    gold_name : str | None, optional
        Nome do gold set avaliado, by default None.

    Returns
    -------
    str
        Texto multilinha em pt-BR.

    Examples
    --------
    >>> print(format_comparison_summary(result, None, None))  # doctest: +SKIP
    """
    title = "RESUMO da comparação de prompts v1 x v2"
    lines = [f"{title} (gold: {gold_name})" if gold_name else title]
    for row in result.iter_rows(named=True):
        lines.append(f"Modelo: {row['model']}")
        lines.extend(
            f"  {key}: {_format_value(value)}" for key, value in row.items() if key != "model"
        )
    for version, summary in (("v1", v1_agreement), ("v2", v2_agreement)):
        if summary:
            lines.append(
                f"Prompt {version}: discordância={_format_value(summary['disagreement_rate'])} "
                f"agreement médio={_format_value(summary['mean_agreement'])}"
            )
    return "\n".join(lines)


def save_summary_report(report: str, output_dir: Path, name: str) -> Path:
    """Grava o resumo em ``resumo_<name>.txt`` (sobrescreve a cada execução).

    Parameters
    ----------
    report : str
        Texto do resumo.
    output_dir : Path
        Pasta de destino (criada se não existir).
    name : str
        Identificador (ex.: ``hipoteses_disagreement``).

    Returns
    -------
    Path
        Caminho do arquivo gravado.

    Examples
    --------
    >>> save_summary_report("texto", Path("reports/x"), "validacao_uncertainty")  # doctest: +SKIP
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"resumo_{name}.txt"
    header = f"# Gerado em {datetime.now().astimezone().isoformat(timespec='seconds')}\n"
    file_path.write_text(header + report.rstrip("\n") + "\n", encoding="utf-8")
    logger.info("Resumo salvo em %s", file_path)
    return file_path
