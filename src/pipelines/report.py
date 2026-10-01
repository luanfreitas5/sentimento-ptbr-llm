"""Estágio ``report``: figuras, tabelas, Model Cards e Datasheet.

Consome as saídas de ``evaluate`` (``reports/tables/avaliacao/``,
``reports/statistics/``, ``reports/ablation/``) e gera, sem recalcular nenhum
modelo:

* figuras (``reports/figures/avaliacao/``, PNG 300 dpi + SVG): comparação entre
  modelos com IC 95%, matriz de confusão por modelo e impacto da ablação;
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
from data.loader import load_training_example_dataset, read_dataset_file
from evaluation.predictions import CATEGORY_LABELS
from exceptions.data import DataNotFoundError
from io_utils.csv import read_csv, write_csv
from io_utils.json import read_json
from metrics.classification import calculate_confusion_matrix
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
from reporting.documents import render_datasheet, render_model_card
from reporting.tables import build_results_table, dataframe_to_latex, dataframe_to_markdown
from visualization.confusion_matrix import plot_confusion_matrix_heatmap
from visualization.evaluation import plot_ablation_impact, plot_model_comparison
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


def _save_figures(
    metrics_table: pl.DataFrame,
    predictions: pl.DataFrame,
    ablation: pl.DataFrame | None,
    directory: Path,
) -> list[Path]:
    """Gera a comparação entre modelos, as matrizes de confusão e o gráfico da ablação."""
    apply_project_theme()
    saved: list[Path] = []
    saved.extend(
        save_figure(plot_model_comparison(metrics_table), "comparacao_modelos", directory=directory)
    )

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

    if ablation is not None and not ablation.is_empty():
        saved.extend(
            save_figure(plot_ablation_impact(ablation), "ablacao_pipeline", directory=directory)
        )
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

    figures = _save_figures(
        metrics_table, predictions, ablation, paths.reports_figures_dir / EVALUATION_SUBDIR
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
