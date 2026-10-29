"""Avaliação comparativa entre a base do Hugging Face e as demais bases rotuladas.

Implementa o estágio ``comparative_evaluation`` de ``configs/config.yaml ->
stages``. Assim como em ``label_consensus``, a etapa roda com o que estiver
disponível, tratando os modelos do Hugging Face como uma lista: o primário
(``tweets_data_huggingface``), o secundário (``huggingface_secondary``) e os modelos
adicionais (``tweets_data_<modelo>.parquet`` em ``data/gold``, ver
:mod:`labeling.extra_sources`). Cada modelo é comparado com cada gold set
(``gold_openai`` e ``gold_unb``) e o primário também com cada um dos demais modelos do
Hugging Face, com ou sem gold set. Executada diretamente (``make compare``), faz
sozinha, para cada par, todo o fluxo, sem passos manuais intermediários:

1. carrega as duas bases do par; havendo gold set, as bases do Hugging Face são
   restritas aos tweets da amostra do gold set, e sem gold set a comparação entre
   modelos do Hugging Face cobre o corpus inteiro;
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
from collections.abc import Mapping
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
    MODEL_NAMES,
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
from evaluation.reference_agreement import (
    analyze_agreement_by_confidence,
    analyze_agreement_by_confidence_pair,
    calculate_per_class_metrics,
    compare_confidence_by_agreement,
)
from exceptions.data import DataNotFoundError
from io_utils.csv import write_csv
from io_utils.json import write_json
from labeling.extra_sources import (
    is_extra_source,
    load_extra_model_names,
    map_extra_source_paths,
)
from labeling.model_names import (
    build_file_slug,
    load_source_file_names,
    load_source_model_names,
)
from visualization.comparison import (
    plot_agreement_by_text_length,
    plot_class_distribution_by_model,
    plot_confidence_comparison,
    plot_divergence_transitions,
)
from visualization.confusion_matrix import plot_confusion_matrix_heatmap
from visualization.reference_agreement import (
    plot_agreement_by_confidence,
    plot_agreement_confidence_heatmap,
    plot_per_class_metrics,
)
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
_INTERNAL_MODEL_SUFFIXES: tuple[str, str] = MODEL_NAMES


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
    """Arquivo de uma base de referência: modelo do Hugging Face (não primário) ou gold set."""
    if reference_name == HUGGINGFACE_SECONDARY_SOURCE:
        return paths.huggingface_secondary_file
    if is_extra_source(reference_name):
        extra_paths = map_extra_source_paths(paths)
        if reference_name not in extra_paths:
            raise DataNotFoundError(
                f"base adicional desconhecida '{reference_name}'; disponíveis: {list(extra_paths)}"
            )
        return extra_paths[reference_name]
    return resolve_gold_path(paths, reference_name)


def list_available_gold_names(paths: ProjectPaths) -> list[str]:
    """Gold sets já gerados pela etapa ``labeling`` (arquivos existentes)."""
    return [name for name in GOLD_SOURCE_NAMES if resolve_gold_path(paths, name).is_file()]


def list_available_reference_names(paths: ProjectPaths) -> list[str]:
    """Bases de referência já geradas: demais modelos do Hugging Face e gold sets."""
    names = [*REFERENCE_SOURCE_NAMES, *map_extra_source_paths(paths)]
    return [name for name in names if resolve_reference_path(paths, name).is_file()]


def list_available_model_names(paths: ProjectPaths) -> list[str]:
    """Modelos do Hugging Face já rotulados: primário, secundário e adicionais.

    O primário sempre consta (a ausência da base é avisada ao carregá-la); os demais só
    constam quando o arquivo existe.
    """
    others = [
        name
        for name in (HUGGINGFACE_SECONDARY_SOURCE, *map_extra_source_paths(paths))
        if resolve_reference_path(paths, name).is_file()
    ]
    return [HUGGINGFACE_SOURCE, *others]


def plan_comparison_pairs(paths: ProjectPaths) -> list[tuple[str, str]]:
    """Pares ``(modelo do Hugging Face, referência)`` a comparar.

    Cada modelo do Hugging Face é comparado com cada gold set já gerado; o primário também é
    comparado com cada um dos demais modelos do Hugging Face.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto.

    Returns
    -------
    list[tuple[str, str]]
        Pares na ordem: primário contra os demais modelos, depois cada modelo contra cada gold.

    Examples
    --------
    >>> plan_comparison_pairs(paths)  # doctest: +SKIP
    [('huggingface', 'huggingface_secondary'), ('huggingface', 'gold_openai'), ...]
    """
    models = list_available_model_names(paths)
    golds = list_available_gold_names(paths)
    return [
        *((HUGGINGFACE_SOURCE, other) for other in models[1:]),
        *((model, gold) for model in models for gold in golds),
    ]


def resolve_model_path(paths: ProjectPaths, model_name: str) -> Path:
    """Arquivo da base de um modelo do Hugging Face (primário, secundário ou adicional)."""
    if model_name == HUGGINGFACE_SOURCE:
        return paths.huggingface_labeled_file
    return resolve_reference_path(paths, model_name)


def build_pair_key(model_name: str, reference_name: str) -> str:
    """Chave de um par no resultado da etapa: a referência, se o modelo é o primário."""
    if model_name == HUGGINGFACE_SOURCE:
        return reference_name
    return f"{model_name}_vs_{reference_name}"


def _merge_extra_names(names: Mapping[str, str], paths: ProjectPaths) -> dict[str, str]:
    """Acrescenta aos nomes configurados os dos modelos adicionais do Hugging Face."""
    return {**load_extra_model_names(paths), **names}


def build_comparison_subdir(
    prefix: str,
    model_names: Mapping[str, str],
    reference_name: str,
    model_name: str = HUGGINGFACE_SOURCE,
) -> str:
    """Nome da subpasta/JSON de uma comparação, com os nomes dos modelos comparados.

    Parameters
    ----------
    prefix : str
        Prefixo configurado (``configs/evaluation.yaml -> llm_comparison.output_subdir``).
    model_names : Mapping[str, str]
        Nome do modelo de cada fonte para pastas e arquivos, sem apelido
        (ver :func:`labeling.model_names.load_source_file_names`).
    reference_name : str
        ``huggingface_secondary``, ``gold_openai``, ``gold_unb`` ou uma base adicional.
    model_name : str, optional
        Fonte do modelo do Hugging Face comparado, by default ``huggingface`` (o primário).

    Returns
    -------
    str
        ``<prefixo>_<modelo do Hugging Face>_vs_<modelo de referência>``.

    Examples
    --------
    >>> names = {"huggingface": "modelo-a", "gold_unb": "Modelo-B"}
    >>> build_comparison_subdir("comparativo", names, "gold_unb")
    'comparativo_modelo-a_vs_Modelo-B'
    """
    primary_slug = build_file_slug(model_names[model_name])
    reference_slug = build_file_slug(model_names[reference_name])
    return f"{prefix}_{primary_slug}_vs_{reference_slug}"


def _apply_model_names(table: pl.DataFrame, primary: str, reference: str) -> pl.DataFrame:
    """Troca os identificadores internos pelos nomes dos modelos nas colunas e valores.

    Colunas ``*_huggingface``/``*_openai`` viram ``*_<modelo>``; os valores das colunas
    ``model`` e ``more_confident_model`` são trocados da mesma forma.
    """
    suffix_names = dict(zip(_INTERNAL_MODEL_SUFFIXES, (primary, reference), strict=True))
    renames = {
        column: f"{column[: -len(suffix)]}{name}"
        for column in table.columns
        for suffix, name in suffix_names.items()
        if column.endswith(f"_{suffix}")
    }
    renamed = table.rename(renames)
    value_columns = [c for c in ("model", "more_confident_model") if c in renamed.columns]
    return renamed.with_columns(pl.col(c).replace(suffix_names) for c in value_columns)


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


def load_comparison_frame(
    paths: ProjectPaths, reference_name: str, model_name: str = HUGGINGFACE_SOURCE
) -> pl.DataFrame:
    """Une um modelo do Hugging Face (por padrão, o primário) e uma referência pelo ``id``.

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
        ``huggingface_secondary``, ``gold_openai``, ``gold_unb`` ou uma base adicional.
    model_name : str, optional
        Fonte do modelo do Hugging Face comparado (``huggingface``, ``huggingface_secondary``
        ou uma base adicional), by default ``huggingface``.

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
    hf_base = load_labeled_source(resolve_model_path(paths, model_name), model_name)
    sample_ids = _resolve_sample_ids(paths, reference_name, reference_base)
    if sample_ids is not None:
        hf_base = hf_base.filter(pl.col("id").is_in(sample_ids))
        if reference_name not in GOLD_SOURCE_NAMES:
            reference_base = reference_base.filter(pl.col("id").is_in(sample_ids))
    return build_comparison_frame(hf_base, reference_base)


def _write_tables(
    tables: dict[str, pl.DataFrame], directory: Path, primary: str, reference: str
) -> dict[str, Path]:
    """Grava cada tabela em CSV, com os nomes dos modelos, e devolve os caminhos."""
    written: dict[str, Path] = {}
    for name, table in tables.items():
        written[name] = directory / f"{name}.csv"
        write_csv(_apply_model_names(table, primary, reference), written[name])
    return written


def _build_agreement_matrix_table(matrix: np.ndarray, reference: str) -> pl.DataFrame:
    """Matriz de concordância como tabela: uma linha por classe do modelo primário."""
    return pl.DataFrame(
        {f"label_{MODEL_NAMES[0]}": list(SENTIMENT_CLASSES)}
        | {
            f"{reference}_{label}": matrix[:, index].astype(int).tolist()
            for index, label in enumerate(SENTIMENT_CLASSES)
        }
    )


def _save_figures(
    frame: pl.DataFrame,
    distribution: pl.DataFrame,
    matrix: np.ndarray,
    length_analysis: pl.DataFrame,
    transitions: pl.DataFrame,
    per_class_metrics: pl.DataFrame,
    agreement_by_confidence: pl.DataFrame,
    agreement_pair: pl.DataFrame,
    directory: Path,
    primary_label: str,
    reference_label: str,
) -> list[Path]:
    """Gera e salva os gráficos da comparação (PNG 300 dpi + SVG)."""
    apply_project_theme()
    x_label = f"Classe atribuída — {reference_label}"
    y_label = f"Classe atribuída — {primary_label}"
    figures = {
        "distribuicao_classes": plot_class_distribution_by_model(
            distribution, primary_label=primary_label, reference_label=reference_label
        ),
        "matriz_concordancia": plot_confusion_matrix_heatmap(
            matrix,
            title=f"Concordância: {primary_label} × {reference_label} (tweets)",
            x_label=x_label,
            y_label=y_label,
        ),
        "matriz_concordancia_normalizada": plot_confusion_matrix_heatmap(
            matrix,
            normalize=True,
            title=f"Concordância: {primary_label} × {reference_label} (proporção por linha)",
            x_label=x_label,
            y_label=y_label,
        ),
        "comparacao_confianca": plot_confidence_comparison(
            frame, primary_label=primary_label, reference_label=reference_label
        ),
        "concordancia_por_tamanho": plot_agreement_by_text_length(length_analysis),
        "metricas_por_classe": plot_per_class_metrics(
            per_class_metrics, primary_label=primary_label, reference_label=reference_label
        ),
        "concordancia_por_confianca": plot_agreement_by_confidence(
            agreement_by_confidence, primary_label=primary_label, reference_label=reference_label
        ),
        "concordancia_por_confianca_cruzada": plot_agreement_confidence_heatmap(
            agreement_pair, primary_label=primary_label, reference_label=reference_label
        ),
    }
    if not transitions.is_empty():
        figures["transicoes_divergencia"] = plot_divergence_transitions(
            transitions, primary_label=primary_label, reference_label=reference_label
        )
    saved: list[Path] = []
    for name, figure in figures.items():
        saved.extend(save_figure(figure, name, directory=directory))
    return saved


def _build_markdown_disclaimer(reference_name: str, primary: str, reference: str) -> list[str]:
    """Ressalva de interpretação: nenhuma das bases é rotulada por humanos."""
    if reference_name not in GOLD_SOURCE_NAMES:
        return [
            f"{primary} e {reference} são modelos do Hugging Face (não há rotulagem humana): estes",
            "números medem a concordância entre os dois modelos, não o acerto absoluto.",
        ]
    return [
        f"O gold set é rotulado pelo LLM {reference} (não por humanos): estes números medem a",
        "concordância com esse LLM, não o acerto absoluto.",
    ]


def _build_markdown_summary(
    summary: dict[str, Any], reference_name: str, primary: str, reference: str
) -> str:
    """Resumo legível dos números principais, para leitura rápida do relatório."""
    agreement = summary["agreement"]
    confidence = summary["confidence"]
    lines = [
        f"# Comparação {primary} × {reference}",
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
        *_build_markdown_disclaimer(reference_name, primary, reference),
    ]
    return "\n".join(lines) + "\n"


def _apply_summary_names(
    confidence_summary: dict[str, Any], primary: str, reference: str
) -> dict[str, Any]:
    """Troca os identificadores internos nas chaves do resumo de confiança."""
    internal = f"{MODEL_NAMES[0]}_minus_{MODEL_NAMES[1]}"
    return {
        key.replace(internal, f"{primary}_minus_{reference}"): value
        for key, value in confidence_summary.items()
    }


def run_reference_comparison(
    paths: ProjectPaths,
    reference_name: str,
    *,
    output_subdir: str,
    model_names: Mapping[str, str] | None = None,
    model_name: str = HUGGINGFACE_SOURCE,
    random_seed: int = 42,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    top_n_divergences: int = 50,
    high_confidence_threshold: float = 0.8,
    low_confidence_threshold: float = 0.5,
    n_length_bins: int = 5,
    examples_per_transition: int = 3,
) -> ComparativeEvaluationResult:
    """Executa toda a avaliação comparativa entre um modelo do Hugging Face e uma referência.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto.
    reference_name : str
        ``huggingface_secondary``, ``gold_openai`` ou ``gold_unb``.
    output_subdir : str
        Subpasta de ``reports/tables``, ``reports/figures`` e nome do JSON em
        ``reports/metrics``.
    model_names : Mapping[str, str] | None, optional
        Nome do modelo de cada fonte, exibido em tabelas e gráficos; ``None`` lê
        ``configs/labeling.yaml``, by default None. Os modelos adicionais são acrescentados.
    model_name : str, optional
        Fonte do modelo do Hugging Face comparado, by default ``huggingface`` (o primário).
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
    ...     paths, "gold_unb", output_subdir="comparativo_modelo-a_vs_modelo-b"
    ... )  # doctest: +SKIP
    """
    names = _merge_extra_names(
        model_names if model_names is not None else load_source_model_names(), paths
    )
    primary_label, reference_label = names[model_name], names[reference_name]
    frame = load_comparison_frame(paths, reference_name, model_name)
    logger.info("Comparando %d tweet(s): %s × %s.", frame.height, primary_label, reference_label)

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
    per_class_metrics = calculate_per_class_metrics(
        frame, n_bootstrap=n_bootstrap, confidence_level=confidence_level, random_seed=random_seed
    )
    agreement_by_confidence = analyze_agreement_by_confidence(
        frame, confidence_level=confidence_level
    )
    agreement_pair = analyze_agreement_by_confidence_pair(frame)
    tables = _write_tables(
        {
            "distribuicao_classes": distribution,
            "matriz_concordancia": _build_agreement_matrix_table(matrix, reference_label),
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
            "metricas_por_classe": per_class_metrics,
            "concordancia_por_confianca": agreement_by_confidence,
            "concordancia_por_confianca_cruzada": agreement_pair,
        },
        tables_dir,
        primary_label,
        reference_label,
    )
    figures = _save_figures(
        frame,
        distribution,
        matrix,
        length_analysis,
        transitions,
        per_class_metrics,
        agreement_by_confidence,
        agreement_pair,
        figures_dir,
        primary_label,
        reference_label,
    )

    summary: dict[str, Any] = {
        "models": {"primary": primary_label, "reference": reference_label},
        "agreement": agreement,
        "confidence": _apply_summary_names(confidence_summary, primary_label, reference_label),
        "confidence_by_agreement": dict(
            zip(
                (primary_label, reference_label),
                compare_confidence_by_agreement(frame).values(),
                strict=True,
            )
        ),
        "per_class_f1": {
            row["label"]: {k: row[k] for k in ("f1", "f1_ci_lower", "f1_ci_upper")}
            for row in per_class_metrics.to_dicts()
        },
        "thresholds": {
            "high_confidence": high_confidence_threshold,
            "low_confidence": low_confidence_threshold,
        },
        "ambiguity": ambiguity_summary.to_dicts(),
    }
    write_json(summary, metrics_path)
    (tables_dir / _SUMMARY_MARKDOWN_FILE_NAME).write_text(
        _build_markdown_summary(summary, reference_name, primary_label, reference_label),
        encoding="utf-8",
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
    output_subdir: str = "comparativo",
    model_names: Mapping[str, str] | None = None,
    **comparison_kwargs: Any,
) -> dict[str, ComparativeEvaluationResult]:
    """Compara cada modelo do Hugging Face com os gold sets e o primário com os demais modelos.

    Os modelos do Hugging Face são o primário, o secundário (``huggingface_secondary``) e os
    adicionais (``tweets_data_<modelo>.parquet``, ver :mod:`labeling.extra_sources`); os gold
    sets são ``gold_openai`` e ``gold_unb``. Os pares com base ausente são avisados no log e
    dispensados, como em ``label_consensus``. A comparação entre modelos do Hugging Face roda
    mesmo sem gold set.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto.
    output_subdir : str, optional
        Prefixo das subpastas de saída; cada referência grava em
        ``<output_subdir>_<modelo primário>_vs_<modelo de referência>`` (ex.:
        ``comparativo_twitter-xlm-roberta-base-sentiment-multilingual_vs_gpt-6-luna``),
        by default "comparativo".
    model_names : Mapping[str, str] | None, optional
        Nome do modelo de cada fonte; ``None`` lê ``configs/labeling.yaml``, by default None.
    **comparison_kwargs : Any
        Demais argumentos de :func:`run_reference_comparison` (semente, bootstrap, limiares).

    Returns
    -------
    dict[str, ComparativeEvaluationResult]
        Resultado por par comparado; a chave é a referência quando o modelo é o primário
        (ex.: ``gold_unb``) e ``<modelo>_vs_<referência>`` nos demais.

    Raises
    ------
    DataNotFoundError
        Se nenhuma base de referência foi gerada ou o Hugging Face ainda não foi rotulado.

    Examples
    --------
    >>> run_comparative_evaluation_stage(paths)  # doctest: +SKIP
    """
    names = _merge_extra_names(
        model_names if model_names is not None else load_source_model_names(), paths
    )
    file_names = _merge_extra_names(
        load_source_file_names() if model_names is None else names, paths
    )
    pairs = plan_comparison_pairs(paths)
    if not pairs:
        raise DataNotFoundError(
            "nenhuma base de referência encontrada; gere-as com `make label-hf-secondary`, "
            "`make label-gold-openai` e/ou `make label-gold-unb`"
        )
    available = set(list_available_reference_names(paths))
    for missing in sorted(set(REFERENCE_SOURCE_NAMES) - available):
        logger.warning("Base '%s' ainda não gerada: comparação dispensada.", missing)
    return {
        build_pair_key(model_name, reference_name): run_reference_comparison(
            paths,
            reference_name,
            output_subdir=build_comparison_subdir(
                output_subdir, file_names, reference_name, model_name
            ),
            model_names=names,
            model_name=model_name,
            **comparison_kwargs,
        )
        for model_name, reference_name in pairs
    }
