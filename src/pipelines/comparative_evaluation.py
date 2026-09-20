"""Avaliação comparativa entre as bases rotuladas pelo Hugging Face e pela OpenAI.

Implementa o estágio ``comparative_evaluation`` de ``configs/config.yaml ->
stages``. Executada diretamente (``make pipeline-comparative-evaluation``), a
etapa faz sozinha todo o fluxo, sem passos manuais intermediários:

1. carrega ``tweets_data_huggingface`` e ``tweets_data_openai``;
2. valida os contratos de dados e une as bases pelo ``id`` do tweet (as duas
   precisam conter exatamente os mesmos tweets);
3. calcula concordância (com IC por bootstrap), Kappa de Cohen (simples e
   ponderado), homogeneidade marginal e comparação de confiança;
4. grava as tabelas (``reports/tables/<subpasta>/``);
5. gera os gráficos (``reports/figures/<subpasta>/``, PNG 300 dpi + SVG);
6. analisa divergências, conflitos de confiança, tamanho do texto, casos
   ambíguos e transições de classe;
7. opcionalmente, aplica o HypotheSAEs à divergência/incerteza entre os modelos
   (``src/diagnostics/model_disagreement.py``), como última etapa — as tabelas
   e os gráficos já estão em disco quando ele roda;
8. grava o resumo consolidado (``reports/metrics/<subpasta>.json``) e um
   resumo em Markdown.

Sem gold set, a comparação mede concordância entre os modelos e descreve o
comportamento de cada um; nunca "acerto". Todas as tabelas com texto usam o
texto normalizado (sem menções/URLs).
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from config.paths import ProjectPaths
from constants.labels import SENTIMENT_CLASSES
from data.loader import read_dataset_file
from evaluation.llm_comparison import (
    analyze_by_text_length,
    build_agreement_matrix,
    build_comparison_frame,
    calculate_agreement_summary,
    calculate_class_distribution,
    compare_confidence_scores,
    find_ambiguous_cases,
    find_confidence_conflicts,
    find_top_divergences,
    summarize_classification_differences,
)
from exceptions.data import DataNotFoundError
from io_utils.csv import write_csv
from io_utils.json import write_json
from visualization.comparison import (
    plot_agreement_by_text_length,
    plot_class_distribution_by_model,
    plot_confidence_comparison,
    plot_divergence_transitions,
)
from visualization.confusion_matrix import plot_confusion_matrix_heatmap
from visualization.theme import apply_project_theme, save_figure

logger = logging.getLogger(__name__)

_SUMMARY_MARKDOWN_FILE_NAME = "resumo_comparativo.md"


@dataclass(frozen=True)
class ComparativeEvaluationResult:
    """Resultado consolidado da avaliação comparativa Hugging Face vs. OpenAI.

    Attributes
    ----------
    comparison_frame : pl.DataFrame
        Bases unidas por ``id`` (ver :func:`evaluation.llm_comparison.build_comparison_frame`).
    summary : dict[str, Any]
        Métricas consolidadas (concordância, Kappa, confiança, hipóteses), o mesmo conteúdo do
        JSON gravado em ``metrics_path``.
    tables : dict[str, Path]
        Tabelas gravadas, indexadas pelo nome.
    figures : list[Path]
        Arquivos de figura gravados (PNG e SVG).
    metrics_path : Path
        JSON com o resumo consolidado.
    """

    comparison_frame: pl.DataFrame
    summary: dict[str, Any]
    tables: dict[str, Path]
    figures: list[Path]
    metrics_path: Path


def _load_labeled_source(path: Path, source_name: str) -> pl.DataFrame:
    """Lê uma base rotulada, com mensagem orientada quando ainda não foi gerada."""
    if not path.is_file():
        raise DataNotFoundError(
            f"{path} — base '{source_name}' ausente; gere-a com "
            f"`make pipeline-labeling-{source_name}`"
        )
    return read_dataset_file(path)


def _write_tables(tables: dict[str, pl.DataFrame], directory: Path) -> dict[str, Path]:
    """Grava cada tabela em CSV e devolve os caminhos, indexados pelo nome."""
    written: dict[str, Path] = {}
    for name, table in tables.items():
        written[name] = directory / f"{name}.csv"
        write_csv(table, written[name])
    return written


def _build_agreement_matrix_table(matrix: np.ndarray) -> pl.DataFrame:
    """Matriz de concordância como tabela: uma linha por classe do Hugging Face."""
    return pl.DataFrame(
        {"label_huggingface": list(SENTIMENT_CLASSES)}
        | {
            f"openai_{label}": matrix[:, index].astype(int).tolist()
            for index, label in enumerate(SENTIMENT_CLASSES)
        }
    )


def _save_figures(
    frame: pl.DataFrame,
    distribution: pl.DataFrame,
    matrix: np.ndarray,
    length_analysis: pl.DataFrame,
    transitions: pl.DataFrame,
    directory: Path,
) -> list[Path]:
    """Gera e salva os gráficos da comparação (PNG 300 dpi + SVG)."""
    apply_project_theme()
    x_label = "Classe atribuída — API OpenAI"
    y_label = "Classe atribuída — LLM Hugging Face"
    figures = {
        "distribuicao_classes": plot_class_distribution_by_model(distribution),
        "matriz_concordancia": plot_confusion_matrix_heatmap(
            matrix, title="Concordância entre os modelos (tweets)", x_label=x_label, y_label=y_label
        ),
        "matriz_concordancia_normalizada": plot_confusion_matrix_heatmap(
            matrix,
            normalize=True,
            title="Concordância entre os modelos (proporção por linha)",
            x_label=x_label,
            y_label=y_label,
        ),
        "comparacao_confianca": plot_confidence_comparison(frame),
        "concordancia_por_tamanho": plot_agreement_by_text_length(length_analysis),
    }
    if not transitions.is_empty():
        figures["transicoes_divergencia"] = plot_divergence_transitions(transitions)
    saved: list[Path] = []
    for name, figure in figures.items():
        saved.extend(save_figure(figure, name, directory=directory))
    return saved


def _build_markdown_summary(summary: dict[str, Any]) -> str:
    """Resumo legível dos números principais, para leitura rápida do relatório."""
    agreement = summary["agreement"]
    confidence = summary["confidence"]
    lines = [
        "# Comparação Hugging Face × OpenAI",
        "",
        f"- Tweets comparados: **{agreement['n_tweets']}**",
        (
            f"- Concordância: **{agreement['agreement_rate']:.1%}** "
            f"(IC {agreement['confidence_level']:.0%}: "
            f"{agreement['agreement_rate_ci'][0]:.1%}–{agreement['agreement_rate_ci'][1]:.1%})"
        ),
        f"- Divergência: **{agreement['divergence_rate']:.1%}**",
        (
            f"- Kappa de Cohen: **{agreement['cohen_kappa']:.3f}** "
            f"(IC: {agreement['cohen_kappa_ci'][0]:.3f}–{agreement['cohen_kappa_ci'][1]:.3f}); "
            f"ponderado (quadrático): {agreement['weighted_kappa_quadratic']:.3f}"
        ),
        f"- Divergências de polaridade oposta: {agreement['opposite_polarity_rate']:.1%}",
        (f"- Correlação de Spearman entre as confianças: {confidence['spearman_rho']:.3f}"),
        "",
        "Sem gold set, estes números medem concordância entre os modelos, não acerto.",
    ]
    hypotheses = summary.get("hypotheses")
    if hypotheses:
        lines += ["", "## HypotheSAEs"]
        lines += [
            f"- `{item['target']}`: {item['status']} — {item['detail']}" for item in hypotheses
        ]
    return "\n".join(lines) + "\n"


def run_comparative_evaluation_stage(
    paths: ProjectPaths,
    *,
    output_subdir: str = "comparativo_hf_openai",
    random_seed: int = 42,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    top_n_divergences: int = 50,
    high_confidence_threshold: float = 0.8,
    low_confidence_threshold: float = 0.5,
    n_length_bins: int = 5,
    examples_per_transition: int = 3,
    run_hypotheses: bool = True,
    hypotheses_targets: Sequence[str] = ("disagreement", "uncertainty"),
    top_tweets_per_hypothesis: int = 10,
    diagnostics_config_file: Path | None = None,
    track_with_mlflow: bool = True,
) -> ComparativeEvaluationResult:
    """Executa toda a avaliação comparativa entre as bases do Hugging Face e da OpenAI.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto.
    output_subdir : str, optional
        Subpasta de ``reports/tables``, ``reports/figures`` e nome do JSON em
        ``reports/metrics``, by default "comparativo_hf_openai".
    random_seed : int, optional
        Semente do bootstrap e do HypotheSAEs, by default 42.
    n_bootstrap : int, optional
        Reamostragens do bootstrap, by default 1000.
    confidence_level : float, optional
        Nível de confiança dos intervalos, by default 0.95.
    top_n_divergences : int, optional
        Tweets divergentes mais graves exportados, by default 50.
    high_confidence_threshold : float, optional
        Confiança considerada alta (conflito de confiança), by default 0.8.
    low_confidence_threshold : float, optional
        Confiança considerada baixa (conflito de confiança e ambiguidade), by default 0.5.
    n_length_bins : int, optional
        Faixas de tamanho do texto (quantis de palavras), by default 5.
    examples_per_transition : int, optional
        Exemplos por par de classes divergentes, by default 3.
    run_hypotheses : bool, optional
        Se ``True``, aplica o HypotheSAEs ao final (exige ``make install-hypothesaes`` e o LLM de
        ``configs/diagnostics.yaml``), by default True.
    hypotheses_targets : Sequence[str], optional
        ``disagreement`` e/ou ``uncertainty``, by default os dois.
    top_tweets_per_hypothesis : int, optional
        Tweets de evidência por hipótese, by default 10.
    diagnostics_config_file : Path | None, optional
        ``configs/diagnostics.yaml`` alternativo, by default None.
    track_with_mlflow : bool, optional
        Se registra a etapa de hipóteses no MLflow, by default True.

    Returns
    -------
    ComparativeEvaluationResult
        Bases unidas, resumo consolidado e caminhos de tudo o que foi gravado.

    Raises
    ------
    DataNotFoundError
        Se alguma base ainda não foi gerada pela etapa ``labeling``.
    DataValidationError
        Se alguma base violar o contrato ou as duas tiverem tweets diferentes.

    Examples
    --------
    >>> run_comparative_evaluation_stage(paths)  # doctest: +SKIP
    """
    hf_base = _load_labeled_source(paths.huggingface_labeled_file, "huggingface")
    oa_base = _load_labeled_source(paths.openai_labeled_file, "openai")
    frame = build_comparison_frame(hf_base, oa_base)
    logger.info("Comparando %d tweet(s) das bases huggingface e openai.", frame.height)

    tables_dir = paths.reports_tables_dir / output_subdir
    figures_dir = paths.reports_figures_dir / output_subdir
    metrics_path = paths.reports_metrics_dir / f"{output_subdir}.json"

    distribution = calculate_class_distribution(frame)
    matrix = build_agreement_matrix(frame)
    agreement = calculate_agreement_summary(
        frame, n_bootstrap=n_bootstrap, confidence_level=confidence_level, random_seed=random_seed
    )
    confidence_table, confidence_summary = compare_confidence_scores(frame)
    length_analysis = analyze_by_text_length(
        frame, n_bins=n_length_bins, confidence_level=confidence_level
    )
    ambiguous_cases, ambiguity_summary = find_ambiguous_cases(
        frame, low_threshold=low_confidence_threshold
    )
    transitions, transition_examples = summarize_classification_differences(
        frame, examples_per_transition=examples_per_transition
    )
    tables = _write_tables(
        {
            "distribuicao_classes": distribution,
            "matriz_concordancia": _build_agreement_matrix_table(matrix),
            "confianca_por_modelo": confidence_table,
            "maiores_divergencias": find_top_divergences(frame, top_n=top_n_divergences),
            "conflitos_de_confianca": find_confidence_conflicts(
                frame,
                high_threshold=high_confidence_threshold,
                low_threshold=low_confidence_threshold,
            ),
            "concordancia_por_tamanho": length_analysis,
            "casos_ambiguos": ambiguous_cases,
            "resumo_ambiguidade": ambiguity_summary,
            "transicoes_divergencia": transitions,
            "exemplos_transicoes": transition_examples,
        },
        tables_dir,
    )
    figures = _save_figures(frame, distribution, matrix, length_analysis, transitions, figures_dir)

    summary: dict[str, Any] = {
        "agreement": agreement,
        "confidence": confidence_summary,
        "thresholds": {
            "high_confidence": high_confidence_threshold,
            "low_confidence": low_confidence_threshold,
        },
        "ambiguity": ambiguity_summary.to_dicts(),
        "hypotheses": None,
    }
    write_json(summary, metrics_path)

    if run_hypotheses:
        from diagnostics.model_disagreement import run_disagreement_hypotheses

        outcomes = run_disagreement_hypotheses(
            frame,
            paths,
            output_dir=tables_dir,
            targets=hypotheses_targets,
            top_tweets_per_hypothesis=top_tweets_per_hypothesis,
            config_file=diagnostics_config_file,
            random_seed=random_seed,
            track=track_with_mlflow,
        )
        summary["hypotheses"] = [
            {
                "target": outcome.target,
                "status": outcome.status,
                "detail": outcome.detail,
                "n_hypotheses": outcome.n_hypotheses,
                "hypotheses_path": str(outcome.hypotheses_path or ""),
                "evidence_path": str(outcome.evidence_path or ""),
            }
            for outcome in outcomes
        ]
        write_json(summary, metrics_path)

    (tables_dir / _SUMMARY_MARKDOWN_FILE_NAME).write_text(
        _build_markdown_summary(summary), encoding="utf-8"
    )
    logger.info(
        "Avaliação comparativa concluída: concordância %.1f%%, kappa %.3f; resultados em %s.",
        100 * agreement["agreement_rate"],
        agreement["cohen_kappa"],
        tables_dir,
    )
    return ComparativeEvaluationResult(
        comparison_frame=frame,
        summary=summary,
        tables=tables,
        figures=figures,
        metrics_path=metrics_path,
    )
