"""Estágio ``report``: figuras, tabelas, Model Cards e Datasheet.

Consome as saídas de ``evaluate`` (``reports/tables/avaliacao/``,
``reports/statistics/``, ``reports/ablation/``) e gera, sem recalcular nenhum
modelo:

* figuras (``reports/figures/avaliacao/``, PNG 300 dpi + SVG): comparação entre
  modelos com IC 95%, matriz de confusão por modelo, impacto da ablação e, quando
  ``evaluate`` calibrou os modelos, confiabilidade por modelo e curva risco-cobertura;
* tabelas (``reports/tables/relatorio/``): resultados principais e ablação em
  CSV, Markdown e LaTeX;
* um Model Card de resultados por categoria de modelo
  (``reports/model_cards/model_card_<categoria>_resultados.md``);
* um Datasheet com as estatísticas reais do corpus
  (``reports/datasheets/datasheet_corpus_tweets_estatisticas.md``).

Nenhum artefato contém texto bruto de tweets.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from config.paths import ProjectPaths
from constants.labels import SENTIMENT_CLASSES
from constants.metrics import PRIMARY_METRIC
from data.loader import load_training_example_dataset, read_dataset_file
from evaluation.predictions import CATEGORY_LABELS
from exceptions.data import DataNotFoundError, EmptyDatasetError
from io_utils.csv import read_csv, write_csv
from io_utils.json import read_json
from metrics.classification import calculate_confusion_matrix
from pipelines.calibration_abstention import RELIABILITY_FILE_NAME, RISK_COVERAGE_FILE_NAME
from pipelines.evaluate import (
    ABLATION_FILE_NAME,
    EVALUATION_SUBDIR,
    MCNEMAR_FILE_NAME,
    METRICS_TABLE_FILE_NAME,
    PER_CLASS_TABLE_FILE_NAME,
    PREDICTIONS_FILE_NAME,
    SLICE_TABLE_FILE_NAME,
    SUMMARY_FILE_NAME,
)
from pipelines.slice_error_analysis import SLICE_UNCERTAINTY_FILE_NAME
from reporting.documents import render_datasheet, render_model_card
from reporting.tables import build_results_table, dataframe_to_latex, dataframe_to_markdown
from visualization.confusion_matrix import plot_confusion_matrix_heatmap
from visualization.evaluation import (
    plot_ablation_impact,
    plot_model_comparison,
    plot_reliability_with_intervals,
    plot_risk_coverage_curves,
    plot_slice_differences,
)
from visualization.theme import apply_project_theme, save_figure

logger = logging.getLogger(__name__)

REPORT_SUBDIR = "relatorio"
_CARD_FILE_NAMES: dict[str, str] = {
    "classical": "model_card_ml_classico_resultados.md",
    "deep_learning": "model_card_deep_learning_resultados.md",
    "transformer": "model_card_transformer_resultados.md",
    "llm": "model_card_llm_local_resultados.md",
}
DATASHEET_FILE_NAME = "datasheet_corpus_tweets_estatisticas.md"


@dataclass(frozen=True)
class ReportResult:
    """Arquivos gerados pela etapa ``report``.

    Attributes
    ----------
    figures : list[Path]
        Figuras (PNG e SVG).
    tables : list[Path]
        Tabelas (CSV, Markdown e LaTeX).
    model_cards : list[Path]
        Model Cards de resultados, um por categoria avaliada.
    datasheet : Path
        Datasheet com as estatísticas do corpus.
    """

    figures: list[Path]
    tables: list[Path]
    model_cards: list[Path]
    datasheet: Path


def _read_numeric_csv(path: Path) -> pl.DataFrame:
    """Lê uma tabela de ``evaluate`` inferindo tipos (``read_csv`` lê tudo como texto)."""
    return read_csv(path, infer_schema_length=10_000)


def _require(path: Path) -> Path:
    """Garante que uma saída da etapa ``evaluate`` existe, com mensagem orientada."""
    if not path.is_file():
        raise DataNotFoundError(f"{path} — rode `make evaluate` antes de `make report`")
    return path


def _save_confusion_figures(predictions: pl.DataFrame, directory: Path) -> list[Path]:
    """Gera uma matriz de confusão normalizada por modelo (colunas ``pred_*``)."""
    saved: list[Path] = []
    for column in (name for name in predictions.columns if name.startswith("pred_")):
        evaluated = predictions.filter(pl.col(column).is_not_null())
        matrix = calculate_confusion_matrix(
            evaluated["y_true"].to_list(), evaluated[column].to_list(), labels=SENTIMENT_CLASSES
        )
        model_name = column.removeprefix("pred_")
        figure = plot_confusion_matrix_heatmap(
            matrix, normalize=True, title=f"Matriz de confusão (teste) — {model_name}"
        )
        saved.extend(save_figure(figure, f"matriz_confusao_{model_name}", directory=directory))
    return saved


def _save_calibration_figures(
    reliability: pl.DataFrame | None, risk_coverage: pl.DataFrame | None, directory: Path
) -> list[Path]:
    """Gera os diagramas de confiabilidade (por modelo) e as curvas risco-cobertura."""
    saved: list[Path] = []
    if reliability is not None and not reliability.is_empty():
        for model_name in reliability["model"].unique(maintain_order=True):
            figure = plot_reliability_with_intervals(reliability, model=model_name)
            saved.extend(save_figure(figure, f"confiabilidade_{model_name}", directory=directory))
    if risk_coverage is not None and not risk_coverage.is_empty():
        saved.extend(
            save_figure(
                plot_risk_coverage_curves(risk_coverage), "risco_cobertura", directory=directory
            )
        )
    return saved


def _save_slice_figures(
    slice_uncertainty: pl.DataFrame | None, metrics_table: pl.DataFrame, directory: Path
) -> list[Path]:
    """Gera a figura de diferença por fatia do melhor modelo de cada paradigma."""
    if slice_uncertainty is None or slice_uncertainty.is_empty():
        return []
    saved: list[Path] = []
    best = metrics_table.sort(PRIMARY_METRIC, descending=True).unique(
        subset="category", keep="first", maintain_order=True
    )
    for model_name in best["model"].to_list():
        try:
            figure = plot_slice_differences(slice_uncertainty, model=model_name)
        except EmptyDatasetError:
            logger.warning("Sem fatias com tamanho suficiente para %s; figura pulada.", model_name)
            continue
        saved.extend(save_figure(figure, f"fatias_{model_name}", directory=directory))
    return saved


def _save_figures(
    metrics_table: pl.DataFrame,
    predictions: pl.DataFrame,
    ablation: pl.DataFrame | None,
    directory: Path,
    *,
    reliability: pl.DataFrame | None = None,
    risk_coverage: pl.DataFrame | None = None,
    slice_uncertainty: pl.DataFrame | None = None,
) -> list[Path]:
    """Gera comparação entre modelos, matrizes de confusão, ablação, calibração e fatias."""
    apply_project_theme()
    saved: list[Path] = []
    saved.extend(
        save_figure(plot_model_comparison(metrics_table), "comparacao_modelos", directory=directory)
    )
    saved.extend(_save_confusion_figures(predictions, directory))
    if ablation is not None and not ablation.is_empty():
        saved.extend(
            save_figure(plot_ablation_impact(ablation), "ablacao_pipeline", directory=directory)
        )
    saved.extend(_save_calibration_figures(reliability, risk_coverage, directory))
    saved.extend(_save_slice_figures(slice_uncertainty, metrics_table, directory))
    return saved


def _write_table_files(
    table: pl.DataFrame, stem: str, directory: Path, *, caption: str, label: str
) -> list[Path]:
    """Grava uma tabela em CSV, Markdown e LaTeX."""
    csv_path = directory / f"{stem}.csv"
    markdown_path = directory / f"{stem}.md"
    latex_path = directory / f"{stem}.tex"
    write_csv(table, csv_path)
    markdown_path.write_text(dataframe_to_markdown(table) + "\n", encoding="utf-8")
    latex_path.write_text(dataframe_to_latex(table, caption=caption, label=label), encoding="utf-8")
    return [csv_path, markdown_path, latex_path]


def _build_datasheet_inputs(paths: ProjectPaths) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Estatísticas agregadas (sem texto) das partições treino/validação/teste."""
    corpora = {
        "treino": load_training_example_dataset(paths.training_corpus_file),
        "validacao": load_training_example_dataset(paths.validation_corpus_file),
        "teste": load_training_example_dataset(paths.test_corpus_file),
    }
    split_rows, distribution_rows, length_rows = [], [], []
    for split_name, corpus in corpora.items():
        split_rows.append({"split": split_name, "n_tweets": corpus.height})
        for label in SENTIMENT_CLASSES:
            count = corpus.filter(pl.col("sentiment_label") == label).height
            distribution_rows.append(
                {
                    "split": split_name,
                    "sentiment_label": label,
                    "count": count,
                    "proportion": count / corpus.height,
                }
            )
        words = np.array([len(text.split()) for text in corpus["text"].to_list()])
        length_rows.append(
            {
                "split": split_name,
                "mean_words": float(words.mean()),
                "median_words": float(np.median(words)),
                "p95_words": float(np.percentile(words, 95)),
            }
        )
    return pl.DataFrame(split_rows), pl.DataFrame(distribution_rows), pl.DataFrame(length_rows)


def run_report_stage(paths: ProjectPaths) -> ReportResult:
    """Gera figuras, tabelas, Model Cards e Datasheet a partir das saídas de ``evaluate``.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto.

    Returns
    -------
    ReportResult
        Arquivos gerados.

    Raises
    ------
    DataNotFoundError
        Se a etapa ``evaluate`` ainda não foi executada.

    Examples
    --------
    >>> run_report_stage(paths)  # doctest: +SKIP
    """
    evaluation_dir = paths.reports_tables_dir / EVALUATION_SUBDIR
    metrics_table = _read_numeric_csv(_require(evaluation_dir / METRICS_TABLE_FILE_NAME))
    per_class = _read_numeric_csv(_require(evaluation_dir / PER_CLASS_TABLE_FILE_NAME))
    slices = _read_numeric_csv(_require(evaluation_dir / SLICE_TABLE_FILE_NAME))
    predictions = read_dataset_file(_require(evaluation_dir / PREDICTIONS_FILE_NAME))
    summary = read_json(_require(paths.reports_metrics_dir / SUMMARY_FILE_NAME))

    mcnemar_path = paths.reports_statistics_dir / MCNEMAR_FILE_NAME
    mcnemar = _read_numeric_csv(mcnemar_path) if mcnemar_path.is_file() else None
    ablation_path = paths.reports_ablation_dir / ABLATION_FILE_NAME
    ablation = _read_numeric_csv(ablation_path) if ablation_path.is_file() else None

    reliability_path = evaluation_dir / RELIABILITY_FILE_NAME
    risk_coverage_path = evaluation_dir / RISK_COVERAGE_FILE_NAME
    slice_uncertainty_path = evaluation_dir / SLICE_UNCERTAINTY_FILE_NAME
    figures = _save_figures(
        metrics_table,
        predictions,
        ablation,
        paths.reports_figures_dir / EVALUATION_SUBDIR,
        reliability=_read_numeric_csv(reliability_path) if reliability_path.is_file() else None,
        risk_coverage=(
            _read_numeric_csv(risk_coverage_path) if risk_coverage_path.is_file() else None
        ),
        slice_uncertainty=(
            _read_numeric_csv(slice_uncertainty_path) if slice_uncertainty_path.is_file() else None
        ),
    )

    tables_dir = paths.reports_tables_dir / REPORT_SUBDIR
    tables = _write_table_files(
        build_results_table(metrics_table),
        "resultados_principais",
        tables_dir,
        caption="Desempenho no conjunto de teste (IC 95\\% bootstrap)",
        label="tab:resultados",
    )
    if ablation is not None:
        tables += _write_table_files(
            ablation.with_columns(pl.col(pl.Float64).round(4)),
            "ablacao",
            tables_dir,
            caption="Ablação do pipeline clássico no conjunto de validação",
            label="tab:ablacao",
        )

    model_cards: list[Path] = []
    for category in metrics_table["category"].unique(maintain_order=True):
        card = render_model_card(
            category,
            metrics_table.filter(pl.col("category") == category),
            per_class,
            slices,
            mcnemar=mcnemar,
            ablation=ablation,
            summary=summary,
        )
        card_path = paths.reports_model_cards_dir / _CARD_FILE_NAMES[category]
        card_path.parent.mkdir(parents=True, exist_ok=True)
        card_path.write_text(card, encoding="utf-8")
        model_cards.append(card_path)

    datasheet_path = paths.reports_datasheets_dir / DATASHEET_FILE_NAME
    datasheet_path.parent.mkdir(parents=True, exist_ok=True)
    datasheet_path.write_text(render_datasheet(*_build_datasheet_inputs(paths)), encoding="utf-8")

    logger.info(
        "Relatório concluído: %d figura(s), %d tabela(s), %d Model Card(s) (%s) e o Datasheet.",
        len(figures),
        len(tables),
        len(model_cards),
        ", ".join(CATEGORY_LABELS[card] for card in metrics_table["category"].unique()),
    )
    return ReportResult(
        figures=figures, tables=tables, model_cards=model_cards, datasheet=datasheet_path
    )
