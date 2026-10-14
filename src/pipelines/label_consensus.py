"""Consenso de rotulagem entre os modelos do Hugging Face e os gold sets.

Implementa o estágio ``label_consensus`` de ``configs/config.yaml -> stages``. A etapa lê as
bases rotuladas já geradas pela etapa ``labeling`` — ``tweets_data_huggingface`` (restrita à
amostra), ``tweets_data_huggingface_cardiff``, ``gold_openai`` e ``gold_unb`` — e, sobre os
MESMOS tweets:

1. une as bases pelo ``id`` e valida os contratos de dados;
2. conta os votos por classe de cada tweet e define a classe e o nível de consenso
   (unanimidade, maioria, pluralidade ou empate);
3. mede a concordância par a par (Kappa de Cohen com IC por bootstrap) e a global (Kappa de
   Fleiss e Alpha de Krippendorff);
4. grava tabelas (``reports/tables/<subpasta>/``), figuras (``reports/figures/<subpasta>/``,
   PNG 300 dpi + SVG), o resumo em JSON (``reports/metrics/<subpasta>.json``) e em Markdown.

Os gold sets são rotulados por LLMs, não por humanos: o consenso mede concordância entre
rotuladores automáticos, não "acerto" absoluto. Roda com os rotuladores disponíveis (mínimo
de dois); os ausentes são avisados no log.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import polars as pl

from config.paths import ProjectPaths
from constants.labels import HUGGINGFACE_CARDIFF_SOURCE, HUGGINGFACE_SOURCE
from data.loader import read_dataset_file
from evaluation.label_consensus import (
    build_consensus_frame,
    calculate_multi_rater_agreement,
    calculate_pairwise_agreement,
    find_disputed_tweets,
    summarize_consensus_by_class,
    summarize_consensus_levels,
)
from exceptions.data import DataNotFoundError, DataValidationError
from io_utils.csv import write_csv
from io_utils.json import write_json
from visualization.consensus import plot_consensus_by_class, plot_pairwise_kappa_heatmap
from visualization.theme import apply_project_theme, save_figure

logger = logging.getLogger(__name__)

_SUMMARY_MARKDOWN_FILE_NAME = "resumo_consenso.md"
_MIN_RATERS = 2


@dataclass(frozen=True)
class LabelConsensusResult:
    """Resultado da análise de consenso de rotulagem.

    Attributes
    ----------
    consensus_frame : pl.DataFrame
        Um tweet por linha, com os rótulos, os votos e o nível de consenso.
    summary : dict[str, Any]
        Métricas consolidadas, o mesmo conteúdo do JSON gravado em ``metrics_path``.
    tables : dict[str, Path]
        Tabelas gravadas, indexadas pelo nome.
    figures : list[Path]
        Arquivos de figura gravados (PNG e SVG).
    metrics_path : Path
        JSON com o resumo consolidado.
    """

    consensus_frame: pl.DataFrame
    summary: dict[str, Any]
    tables: dict[str, Path]
    figures: list[Path]
    metrics_path: Path


def _resolve_source_paths(paths: ProjectPaths) -> dict[str, Path]:
    """Arquivo de cada rotulador, na ordem em que aparecem nas tabelas."""
    return {
        HUGGINGFACE_SOURCE: paths.huggingface_labeled_file,
        HUGGINGFACE_CARDIFF_SOURCE: paths.huggingface_cardiff_file,
        "gold_openai": paths.gold_openai_file,
        "gold_unb": paths.gold_unb_file,
    }


def load_consensus_sources(paths: ProjectPaths) -> dict[str, pl.DataFrame]:
    """Lê as bases rotuladas disponíveis; o Hugging Face é restrito à amostra das demais.

    A base ``huggingface`` cobre todo o corpus, enquanto as outras três cobrem só a amostra
    dos gold sets: ela é filtrada para os ``id`` da primeira base de amostra encontrada.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto.

    Returns
    -------
    dict[str, pl.DataFrame]
        Base de cada rotulador disponível (ao menos dois), indexada pelo nome da fonte.

    Raises
    ------
    DataNotFoundError
        Se houver menos de dois rotuladores gerados ou nenhuma base de amostra.
    DataValidationError
        Se a base de amostra tiver tweets ausentes na base ``huggingface``.

    Examples
    --------
    >>> sorted(load_consensus_sources(paths))  # doctest: +SKIP
    ['gold_openai', 'gold_unb', 'huggingface', 'huggingface_cardiff']
    """
    source_paths = _resolve_source_paths(paths)
    missing = [name for name, path in source_paths.items() if not path.is_file()]
    for name in missing:
        logger.warning("Rotulador '%s' ainda não gerado: fora da análise de consenso.", name)
    available = {
        name: read_dataset_file(path) for name, path in source_paths.items() if name not in missing
    }
    sample_names = [name for name in available if name != HUGGINGFACE_SOURCE]
    if len(available) < _MIN_RATERS or not sample_names:
        raise DataNotFoundError(
            "o consenso exige ao menos dois rotuladores, um deles de amostra "
            "(`huggingface_cardiff`, `gold_openai` ou `gold_unb`); gere-os com "
            "`make label-hf-cardiff`, `make label-gold-openai` e `make label-gold-unb`. "
            f"Disponíveis: {list(available)}"
        )
    if HUGGINGFACE_SOURCE in available:
        sample_ids = available[sample_names[0]]["id"].to_list()
        available[HUGGINGFACE_SOURCE] = available[HUGGINGFACE_SOURCE].filter(
            pl.col("id").is_in(sample_ids)
        )
        n_missing = len(sample_ids) - available[HUGGINGFACE_SOURCE].height
        if n_missing:
            raise DataValidationError(
                schema_name="label_consensus",
                detail=(
                    f"a base '{sample_names[0]}' tem {n_missing} tweet(s) ausente(s) em "
                    f"'{HUGGINGFACE_SOURCE}'"
                ),
            )
    return available


def _write_tables(tables: dict[str, pl.DataFrame], directory: Path) -> dict[str, Path]:
    """Grava cada tabela em CSV e devolve os caminhos, indexados pelo nome."""
    written: dict[str, Path] = {}
    for name, table in tables.items():
        written[name] = directory / f"{name}.csv"
        write_csv(table, written[name])
    return written


def _format_optional(value: float | None) -> str:
    """Número com 3 casas, ou ``n/d`` quando indefinido."""
    return "n/d" if value is None else f"{value:.3f}"


def _format_rate(value: float | None) -> str:
    """Percentual com 1 casa, ou ``n/d`` quando indefinido."""
    return "n/d" if value is None else f"{value:.1%}"


def _build_markdown_summary(
    summary: dict[str, Any], levels: pl.DataFrame, by_class: pl.DataFrame
) -> str:
    """Resumo legível dos números principais do consenso."""
    multi = summary["multi_rater"]
    interval = multi["fleiss_kappa_ci"]
    fleiss_ci = f" (IC: {interval[0]:.3f}–{interval[1]:.3f})" if interval else ""
    lines = [
        "# Consenso de rotulagem",
        "",
        f"- Rotuladores: **{', '.join(summary['sources'])}**",
        f"- Tweets analisados: **{multi['n_tweets']}**",
        f"- Unanimidade: **{multi['unanimity_rate']:.1%}**",
        f"- Kappa de Fleiss: **{_format_optional(multi['fleiss_kappa'])}**{fleiss_ci}",
        f"- Alpha de Krippendorff: **{multi['krippendorff_alpha']:.3f}**",
        "",
        "## Nível de consenso",
        "",
        "| Nível | Tweets | Proporção |",
        "|---|---|---|",
        *(
            f"| {row['consensus_level']} | {row['n_tweets']} | {row['proportion']:.1%} |"
            for row in levels.iter_rows(named=True)
        ),
        "",
        "## Classe de consenso",
        "",
        "| Classe | Tweets | Taxa de unanimidade |",
        "|---|---|---|",
        *(
            f"| {row['consensus_label']} | {row['n_tweets']} | "
            f"{_format_rate(row['unanimity_rate'])} |"
            for row in by_class.iter_rows(named=True)
        ),
        "",
        "Os gold sets são rotulados por LLMs (não por humanos): o consenso mede a concordância",
        "entre rotuladores automáticos, não o acerto absoluto.",
    ]
    return "\n".join(lines) + "\n"


def run_label_consensus_stage(
    paths: ProjectPaths,
    *,
    output_subdir: str = "consenso_rotulagem",
    random_seed: int = 42,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    top_n_disputed: int = 50,
) -> LabelConsensusResult:
    """Executa a análise de consenso entre os rotuladores disponíveis.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto.
    output_subdir : str, optional
        Subpasta de ``reports/tables`` e ``reports/figures`` e nome do JSON em
        ``reports/metrics``, by default "consenso_rotulagem".
    random_seed : int, optional
        Semente do bootstrap, by default 42.
    n_bootstrap : int, optional
        Reamostragens do bootstrap, by default 1000.
    confidence_level : float, optional
        Nível de confiança dos intervalos, by default 0.95.
    top_n_disputed : int, optional
        Tweets em disputa exportados para inspeção, by default 50.

    Returns
    -------
    LabelConsensusResult
        Quadro de consenso, resumo e caminhos de tudo o que foi gravado.

    Raises
    ------
    DataNotFoundError
        Se houver menos de dois rotuladores gerados pela etapa ``labeling``.
    DataValidationError
        Se alguma base violar o contrato ou as bases tiverem tweets diferentes.

    Examples
    --------
    >>> run_label_consensus_stage(paths)  # doctest: +SKIP
    """
    sources = load_consensus_sources(paths)
    source_names = list(sources)
    frame = build_consensus_frame(sources)
    logger.info(
        "Consenso entre %d rotuladores (%s) sobre %d tweet(s).",
        len(source_names),
        ", ".join(source_names),
        frame.height,
    )

    levels = summarize_consensus_levels(frame, confidence_level=confidence_level)
    by_class = summarize_consensus_by_class(frame)
    pairwise = calculate_pairwise_agreement(
        frame,
        source_names,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_seed=random_seed,
    )
    multi_rater = calculate_multi_rater_agreement(
        frame,
        source_names,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_seed=random_seed,
    )

    tables_dir = paths.reports_tables_dir / output_subdir
    figures_dir = paths.reports_figures_dir / output_subdir
    metrics_path = paths.reports_metrics_dir / f"{output_subdir}.json"
    tables = _write_tables(
        {
            "consenso_por_tweet": frame,
            "niveis_de_consenso": levels,
            "consenso_por_classe": by_class,
            "concordancia_par_a_par": pairwise,
            "tweets_em_disputa": find_disputed_tweets(frame, top_n=top_n_disputed),
        },
        tables_dir,
    )

    apply_project_theme()
    figures: list[Path] = []
    for name, figure in {
        "kappa_par_a_par": plot_pairwise_kappa_heatmap(pairwise, source_names),
        "consenso_por_classe": plot_consensus_by_class(by_class),
    }.items():
        figures.extend(save_figure(figure, name, directory=figures_dir))

    summary: dict[str, Any] = {
        "sources": source_names,
        "multi_rater": multi_rater,
        "levels": levels.to_dicts(),
        "by_class": by_class.to_dicts(),
        "pairwise": pairwise.to_dicts(),
        "polarity_conflict_rate": float(cast("float", frame["polarity_conflict"].mean())),
    }
    write_json(summary, metrics_path)
    (tables_dir / _SUMMARY_MARKDOWN_FILE_NAME).write_text(
        _build_markdown_summary(summary, levels, by_class), encoding="utf-8"
    )
    logger.info(
        "Consenso concluído: unanimidade %.1f%%, kappa de Fleiss %s; resultados em %s.",
        100 * multi_rater["unanimity_rate"],
        _format_optional(multi_rater["fleiss_kappa"]),
        tables_dir,
    )
    return LabelConsensusResult(
        consensus_frame=frame,
        summary=summary,
        tables=tables,
        figures=figures,
        metrics_path=metrics_path,
    )
