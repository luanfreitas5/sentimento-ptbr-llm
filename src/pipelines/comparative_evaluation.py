"""Avaliação comparativa entre a base do Hugging Face e as demais bases rotuladas.

Implementa o estágio ``comparative_evaluation`` de ``configs/config.yaml ->
stages``. Assim como em ``label_consensus``, a etapa roda com o que estiver
disponível: compara ``tweets_data_huggingface`` (modelo primário) com cada base
de referência já gerada — o segundo modelo do Hugging Face
(``huggingface_secondary``) e/ou os gold sets (``gold_openai`` e ``gold_unb``) —,
com ou sem gold set. Executada diretamente (``make compare``), faz sozinha, para
cada referência, todo o fluxo, sem passos manuais intermediários:

1. carrega ``tweets_data_huggingface`` e a referência; havendo gold set, as bases
   do Hugging Face são restritas aos tweets da amostra do gold set, e sem gold
   set a comparação entre os dois modelos do Hugging Face cobre o corpus inteiro;
2. valida os contratos de dados e une as bases pelo ``id`` do tweet (as duas
   precisam conter exatamente os mesmos tweets);
3. calcula concordância (com IC por bootstrap), Kappa de Cohen (simples e
   ponderado), homogeneidade marginal e comparação de confiança;
4. grava as tabelas (``reports/tables/<subpasta>/``);
5. gera os gráficos (``reports/figures/<subpasta>/``, PNG 300 dpi + SVG);
6. analisa divergências, conflitos de confiança, tamanho do texto, casos
   ambíguos e transições de classe;
7. grava o resumo consolidado (``reports/metrics/<subpasta>.json``) e um
   resumo em Markdown.

A geração de hipóteses (HypotheSAEs) sobre a divergência/incerteza entre os
modelos NÃO faz parte desta etapa: tem estágio próprio (``hypotheses``,
``src/pipelines/hypotheses.py``), que reutiliza :func:`load_comparison_frame`.

Os gold sets são rotulados por LLMs, não por humanos: a comparação mede a
concordância do modelo do Hugging Face com cada rotulador de referência, não
"acerto" absoluto. Todas as tabelas com texto usam o texto normalizado (sem
menções/URLs).
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from config.paths import ProjectPaths
from constants.labels import (
    GOLD_SOURCE_NAMES,
    HUGGINGFACE_SECONDARY_SOURCE,
    HUGGINGFACE_SOURCE,
    SENTIMENT_CLASSES,
)
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
    DEFAULT_REFERENCE_LABEL,
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
    """Resultado consolidado da avaliação comparativa Hugging Face vs. uma base de referência.

    Attributes
    ----------
    comparison_frame : pl.DataFrame
        Bases unidas por ``id`` (ver :func:`evaluation.llm_comparison.build_comparison_frame`).
    summary : dict[str, Any]
        Métricas consolidadas (concordância, Kappa, confiança, ambiguidade), o mesmo conteúdo do
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


_MAKE_TARGETS: dict[str, str] = {
    HUGGINGFACE_SOURCE: "label-hf",
    HUGGINGFACE_SECONDARY_SOURCE: "label-hf-secondary",
    "gold_openai": "label-gold-openai",
    "gold_unb": "label-gold-unb",
}
REFERENCE_SOURCE_NAMES: tuple[str, ...] = (HUGGINGFACE_SECONDARY_SOURCE, *GOLD_SOURCE_NAMES)
_REFERENCE_LABELS: dict[str, str] = {
    HUGGINGFACE_SECONDARY_SOURCE: "LLM Hugging Face (secundário)",
}


def load_labeled_source(path: Path, source_name: str) -> pl.DataFrame:
    """Lê uma base rotulada, com mensagem orientada quando ainda não foi gerada."""
    if not path.is_file():
        make_target = _MAKE_TARGETS.get(source_name, "label")
        raise DataNotFoundError(
            f"{path} — base '{source_name}' ausente; gere-a com `make {make_target}`"
        )
    return read_dataset_file(path)


def resolve_gold_path(paths: ProjectPaths, gold_name: str) -> Path:
    """Arquivo do gold set ``gold_openai`` ou ``gold_unb``."""
    gold_paths = {"gold_openai": paths.gold_openai_file, "gold_unb": paths.gold_unb_file}
    if gold_name not in gold_paths:
        raise DataNotFoundError(
            f"gold set desconhecido '{gold_name}'; disponíveis: {list(GOLD_SOURCE_NAMES)}"
        )
    return gold_paths[gold_name]


def resolve_reference_path(paths: ProjectPaths, reference_name: str) -> Path:
    """Arquivo de uma base de referência: ``huggingface_secondary`` ou um gold set."""
    if reference_name == HUGGINGFACE_SECONDARY_SOURCE:
        return paths.huggingface_secondary_file
    return resolve_gold_path(paths, reference_name)


def list_available_gold_names(paths: ProjectPaths) -> list[str]:
    """Gold sets já gerados pela etapa ``labeling`` (arquivos existentes)."""
    return [name for name in GOLD_SOURCE_NAMES if resolve_gold_path(paths, name).is_file()]


def list_available_reference_names(paths: ProjectPaths) -> list[str]:
    """Bases de referência já geradas: segundo modelo do Hugging Face e gold sets."""
    return [
        name for name in REFERENCE_SOURCE_NAMES if resolve_reference_path(paths, name).is_file()
    ]


def _resolve_sample_ids(
    paths: ProjectPaths, reference_name: str, reference_base: pl.DataFrame
) -> list[str] | None:
    """``id`` da amostra gold que restringe a comparação, ou ``None`` para o corpus inteiro."""
    if reference_name in GOLD_SOURCE_NAMES:
        return reference_base["id"].to_list()
    gold_names = list_available_gold_names(paths)
    if not gold_names:
        return None
    return read_dataset_file(resolve_gold_path(paths, gold_names[0]))["id"].to_list()


def load_comparison_frame(paths: ProjectPaths, reference_name: str) -> pl.DataFrame:
    """Une o Hugging Face primário e uma base de referência pelo ``id``.

    Um gold set rotula só uma amostra do corpus; o Hugging Face (todos os tweets) é filtrado
    para essa amostra antes da união. Para ``huggingface_secondary`` (também sobre todo o
    corpus), havendo gold set as duas bases do Hugging Face são restritas à amostra do primeiro
    gold encontrado, para avaliar os mesmos tweets dos demais rotuladores; sem gold set, a
    comparação cobre o corpus inteiro.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto.
    reference_name : str
        ``huggingface_secondary``, ``gold_openai`` ou ``gold_unb``.

    Returns
    -------
    pl.DataFrame
        Quadro de comparação (ver :func:`evaluation.llm_comparison.build_comparison_frame`).

    Raises
    ------
    DataNotFoundError
        Se alguma das bases ainda não foi gerada.
    DataValidationError
        Se alguma base violar o contrato ou a amostra tiver tweets ausentes no Hugging Face.

    Examples
    --------
    >>> load_comparison_frame(paths, "gold_unb")  # doctest: +SKIP
    """
    reference_base = load_labeled_source(
        resolve_reference_path(paths, reference_name), reference_name
    )
    hf_base = load_labeled_source(paths.huggingface_labeled_file, HUGGINGFACE_SOURCE)
    sample_ids = _resolve_sample_ids(paths, reference_name, reference_base)
    if sample_ids is not None:
        hf_base = hf_base.filter(pl.col("id").is_in(sample_ids))
        if reference_name == HUGGINGFACE_SECONDARY_SOURCE:
            reference_base = reference_base.filter(pl.col("id").is_in(sample_ids))
    return build_comparison_frame(hf_base, reference_base)


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
            f"reference_{label}": matrix[:, index].astype(int).tolist()
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
    reference_label: str,
) -> list[Path]:
    """Gera e salva os gráficos da comparação (PNG 300 dpi + SVG)."""
    apply_project_theme()
    x_label = f"Classe atribuída — {reference_label}"
    y_label = "Classe atribuída — modelo Hugging Face"
    figures = {
        "distribuicao_classes": plot_class_distribution_by_model(
            distribution, reference_label=reference_label
        ),
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
        "comparacao_confianca": plot_confidence_comparison(frame, reference_label=reference_label),
        "concordancia_por_tamanho": plot_agreement_by_text_length(length_analysis),
    }
    if not transitions.is_empty():
        figures["transicoes_divergencia"] = plot_divergence_transitions(
            transitions, reference_label=reference_label
        )
    saved: list[Path] = []
    for name, figure in figures.items():
        saved.extend(save_figure(figure, name, directory=directory))
    return saved


def _build_markdown_disclaimer(reference_name: str) -> list[str]:
    """Ressalva de interpretação: nenhuma das bases é rotulada por humanos."""
    if reference_name == HUGGINGFACE_SECONDARY_SOURCE:
        return [
            "Ambos são modelos do Hugging Face (não há rotulagem humana): estes números medem a",
            "concordância entre os dois modelos, não o acerto absoluto.",
        ]
    return [
        "O gold set é rotulado por um LLM (não por humanos): estes números medem a concordância",
        "com esse LLM, não o acerto absoluto.",
    ]


def _build_markdown_summary(summary: dict[str, Any], reference_name: str) -> str:
    """Resumo legível dos números principais, para leitura rápida do relatório."""
    agreement = summary["agreement"]
    confidence = summary["confidence"]
    lines = [
        f"# Comparação Hugging Face × {reference_name}",
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
        *_build_markdown_disclaimer(reference_name),
    ]
    return "\n".join(lines) + "\n"


def run_reference_comparison(
    paths: ProjectPaths,
    reference_name: str,
    *,
    output_subdir: str,
    random_seed: int = 42,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    top_n_divergences: int = 50,
    high_confidence_threshold: float = 0.8,
    low_confidence_threshold: float = 0.5,
    n_length_bins: int = 5,
    examples_per_transition: int = 3,
) -> ComparativeEvaluationResult:
    """Executa toda a avaliação comparativa entre o Hugging Face e uma base de referência.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto.
    reference_name : str
        ``huggingface_secondary``, ``gold_openai`` ou ``gold_unb``.
    output_subdir : str
        Subpasta de ``reports/tables``, ``reports/figures`` e nome do JSON em
        ``reports/metrics``.
    random_seed : int, optional
        Semente do bootstrap, by default 42.
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
    >>> run_reference_comparison(
    ...     paths, "gold_unb", output_subdir="comparativo_hf_gold_unb"
    ... )  # doctest: +SKIP
    """
    frame = load_comparison_frame(paths, reference_name)
    reference_label = _REFERENCE_LABELS.get(reference_name, DEFAULT_REFERENCE_LABEL)
    logger.info("Comparando %d tweet(s) do Hugging Face com '%s'.", frame.height, reference_name)

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
    figures = _save_figures(
        frame, distribution, matrix, length_analysis, transitions, figures_dir, reference_label
    )

    summary: dict[str, Any] = {
        "agreement": agreement,
        "confidence": confidence_summary,
        "thresholds": {
            "high_confidence": high_confidence_threshold,
            "low_confidence": low_confidence_threshold,
        },
        "ambiguity": ambiguity_summary.to_dicts(),
    }
    write_json(summary, metrics_path)
    (tables_dir / _SUMMARY_MARKDOWN_FILE_NAME).write_text(
        _build_markdown_summary(summary, reference_name), encoding="utf-8"
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


def run_comparative_evaluation_stage(
    paths: ProjectPaths,
    *,
    output_subdir: str = "comparativo_hf",
    **comparison_kwargs: Any,
) -> dict[str, ComparativeEvaluationResult]:
    """Compara o Hugging Face primário com cada base de referência já gerada.

    As referências são o segundo modelo do Hugging Face (``huggingface_secondary``) e os gold
    sets (``gold_openai`` e ``gold_unb``); as ausentes são avisadas no log e dispensadas, como
    em ``label_consensus``. A comparação entre os modelos do Hugging Face roda mesmo sem gold set.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto.
    output_subdir : str, optional
        Prefixo das subpastas de saída; cada referência grava em
        ``<output_subdir>_<referência>`` (ex.: ``comparativo_hf_gold_unb`` ou
        ``comparativo_hf_huggingface_secondary``), by default "comparativo_hf".
    **comparison_kwargs : Any
        Demais argumentos de :func:`run_reference_comparison` (semente, bootstrap, limiares).

    Returns
    -------
    dict[str, ComparativeEvaluationResult]
        Resultado por base de referência comparada.

    Raises
    ------
    DataNotFoundError
        Se nenhuma base de referência foi gerada ou o Hugging Face ainda não foi rotulado.

    Examples
    --------
    >>> run_comparative_evaluation_stage(paths)  # doctest: +SKIP
    """
    available = list_available_reference_names(paths)
    if not available:
        raise DataNotFoundError(
            "nenhuma base de referência encontrada; gere-as com `make label-hf-secondary`, "
            "`make label-gold-openai` e/ou `make label-gold-unb`"
        )
    for missing in set(REFERENCE_SOURCE_NAMES) - set(available):
        logger.warning("Base '%s' ainda não gerada: comparação dispensada.", missing)
    return {
        reference_name: run_reference_comparison(
            paths,
            reference_name,
            output_subdir=f"{output_subdir}_{reference_name}",
            **comparison_kwargs,
        )
        for reference_name in available
    }
