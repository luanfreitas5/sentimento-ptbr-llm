"""Testes das visualizações (``src/visualization``)."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from exceptions.data import EmptyDatasetError
from visualization.confusion_matrix import plot_confusion_matrix_heatmap
from visualization.diagnostics import (
    plot_calibration_curve,
    plot_confidence_distribution_by_correctness,
)
from visualization.distributions import plot_class_distribution, plot_text_length_distribution
from visualization.embeddings import plot_embedding_scatter
from visualization.interpretability import plot_shap_summary, plot_top_feature_importances
from visualization.ngrams import plot_top_ngrams_bar
from visualization.roc_pr_curves import (
    plot_precision_recall_curves_one_vs_rest,
    plot_roc_curves_one_vs_rest,
)
from visualization.theme import SENTIMENT_COLOR_PALETTE, apply_project_theme, save_figure
from visualization.wordcloud import generate_sentiment_wordcloud


class TestApplyProjectTheme:
    """Testes de :func:`visualization.theme.apply_project_theme`."""

    def test_runs_without_raising(self) -> None:
        """A aplicação do tema não deve levantar exceções."""
        apply_project_theme()

    def test_palette_has_one_color_per_sentiment_class(self) -> None:
        """A paleta deve ter exatamente três cores, uma por classe de sentimento."""
        assert set(SENTIMENT_COLOR_PALETTE.keys()) == {"negativo", "neutro", "positivo"}


class TestSaveFigure:
    """Testes de :func:`visualization.theme.save_figure`."""

    def test_saves_png_and_svg_files(self, tmp_path: Path) -> None:
        """Deve salvar tanto o arquivo PNG quanto o SVG no diretório informado."""
        figure, axis = plt.subplots()
        axis.plot([0, 1], [0, 1])
        png_path, svg_path = save_figure(figure, "exemplo", directory=tmp_path)
        assert png_path.is_file()
        assert svg_path.is_file()
        assert png_path.suffix == ".png"
        assert svg_path.suffix == ".svg"

    def test_creates_missing_directories(self, tmp_path: Path) -> None:
        """Deve criar diretórios ausentes no caminho de destino."""
        figure, axis = plt.subplots()
        axis.plot([0, 1], [0, 1])
        output_directory = tmp_path / "figuras" / "subdiretorio"
        png_path, _ = save_figure(figure, "exemplo", directory=output_directory)
        assert png_path.exists()


class TestPlotConfusionMatrixHeatmap:
    """Testes de :func:`visualization.confusion_matrix.plot_confusion_matrix_heatmap`."""

    def test_returns_figure_with_expected_title(self) -> None:
        """A figura retornada deve ter o título esperado."""
        matrix = np.array([[5, 1], [2, 8]])
        figure = plot_confusion_matrix_heatmap(matrix, labels=["negativo", "positivo"])
        assert isinstance(figure, Figure)
        assert figure.axes[0].get_title() == "Matriz de Confusão"

    def test_raises_on_empty_matrix(self) -> None:
        """Deve levantar ``EmptyDatasetError`` quando a matriz está vazia."""
        with pytest.raises(EmptyDatasetError):
            plot_confusion_matrix_heatmap(np.empty((0, 0)))


class TestPlotRocCurvesOneVsRest:
    """Testes de :func:`visualization.roc_pr_curves.plot_roc_curves_one_vs_rest`."""

    def test_draws_one_line_per_class_plus_reference(self) -> None:
        """Deve desenhar uma linha por classe, além da diagonal de referência."""
        y_true = ["negativo", "positivo", "negativo", "positivo"]
        y_score = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]])
        figure = plot_roc_curves_one_vs_rest(y_true, y_score, labels=["negativo", "positivo"])
        assert len(figure.axes[0].lines) == 3

    def test_raises_on_empty_input(self) -> None:
        """Deve levantar ``EmptyDatasetError`` quando ``y_true`` está vazio."""
        with pytest.raises(EmptyDatasetError):
            plot_roc_curves_one_vs_rest([], np.empty((0, 2)))


class TestPlotPrecisionRecallCurvesOneVsRest:
    """Testes de :func:`visualization.roc_pr_curves.plot_precision_recall_curves_one_vs_rest`."""

    def test_draws_one_line_per_class(self) -> None:
        """Deve desenhar uma linha por classe."""
        y_true = ["negativo", "positivo", "negativo", "positivo"]
        y_score = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]])
        figure = plot_precision_recall_curves_one_vs_rest(
            y_true, y_score, labels=["negativo", "positivo"]
        )
        assert len(figure.axes[0].lines) == 2

    def test_raises_on_empty_input(self) -> None:
        """Deve levantar ``EmptyDatasetError`` quando ``y_true`` está vazio."""
        with pytest.raises(EmptyDatasetError):
            plot_precision_recall_curves_one_vs_rest([], np.empty((0, 2)))


class TestPlotCalibrationCurve:
    """Testes de :func:`visualization.diagnostics.plot_calibration_curve`."""

    def test_returns_figure_with_expected_title(self) -> None:
        """A figura retornada deve ter o título esperado."""
        curve = {
            "bin_confidence_means": np.array([0.6, 0.9]),
            "bin_accuracy": np.array([0.5, 0.85]),
            "bin_counts": np.array([2, 3]),
        }
        figure = plot_calibration_curve(curve)
        assert figure.axes[0].get_title() == "Curva de Confiabilidade"

    def test_raises_when_all_bins_are_empty(self) -> None:
        """Deve levantar ``EmptyDatasetError`` quando todas as faixas estão vazias."""
        curve = {
            "bin_confidence_means": np.array([np.nan, np.nan]),
            "bin_accuracy": np.array([np.nan, np.nan]),
            "bin_counts": np.array([0, 0]),
        }
        with pytest.raises(EmptyDatasetError):
            plot_calibration_curve(curve)


class TestPlotConfidenceDistributionByCorrectness:
    """Testes de :func:`visualization.diagnostics.plot_confidence_distribution_by_correctness`."""

    def test_returns_figure_with_expected_title(self) -> None:
        """A figura retornada deve ter o título esperado."""
        figure = plot_confidence_distribution_by_correctness(
            np.array([0.9, 0.6, 0.95, 0.55]), [True, False, True, False]
        )
        assert figure.axes[0].get_title() == "Distribuição de Confiança por Acerto"

    def test_raises_on_empty_input(self) -> None:
        """Deve levantar ``EmptyDatasetError`` quando ``confidences`` está vazio."""
        with pytest.raises(EmptyDatasetError):
            plot_confidence_distribution_by_correctness(np.empty(0), [])


class TestPlotClassDistribution:
    """Testes de :func:`visualization.distributions.plot_class_distribution`."""

    def test_returns_figure_with_expected_title(self) -> None:
        """A figura retornada deve ter o título esperado."""
        figure = plot_class_distribution(["positivo", "positivo", "negativo"])
        assert figure.axes[0].get_title() == "Distribuição das Classes de Sentimento"

    def test_raises_on_empty_input(self) -> None:
        """Deve levantar ``EmptyDatasetError`` quando ``labels`` está vazio."""
        with pytest.raises(EmptyDatasetError):
            plot_class_distribution([])


class TestPlotTextLengthDistribution:
    """Testes de :func:`visualization.distributions.plot_text_length_distribution`."""

    def test_returns_figure_with_expected_title(self) -> None:
        """A figura retornada deve ter o título esperado."""
        figure = plot_text_length_distribution([10, 20, 15, 30])
        assert figure.axes[0].get_title() == "Distribuição do Comprimento de Texto"

    def test_raises_on_empty_input(self) -> None:
        """Deve levantar ``EmptyDatasetError`` quando ``text_lengths`` está vazio."""
        with pytest.raises(EmptyDatasetError):
            plot_text_length_distribution([])


class TestPlotEmbeddingScatter:
    """Testes de :func:`visualization.embeddings.plot_embedding_scatter`."""

    def test_returns_figure_with_expected_title(self) -> None:
        """A figura retornada deve ter o título esperado."""
        coordinates = np.array([[0.1, 0.2], [0.9, 0.8]])
        figure = plot_embedding_scatter(coordinates, ["negativo", "positivo"])
        assert figure.axes[0].get_title() == "Projeção 2D dos Embeddings por Sentimento"

    def test_raises_on_empty_input(self) -> None:
        """Deve levantar ``EmptyDatasetError`` quando ``coordinates_2d`` está vazia."""
        with pytest.raises(EmptyDatasetError):
            plot_embedding_scatter(np.empty((0, 2)), [])

    def test_raises_when_not_two_dimensional(self) -> None:
        """Deve levantar ``ValueError`` quando as coordenadas não têm 2 colunas."""
        with pytest.raises(ValueError, match="2 colunas"):
            plot_embedding_scatter(np.array([[0.1, 0.2, 0.3]]), ["negativo"])

    def test_raises_on_length_mismatch(self) -> None:
        """Deve levantar ``ValueError`` quando o número de amostras difere de ``labels``."""
        with pytest.raises(ValueError, match="mesmo número de amostras"):
            plot_embedding_scatter(np.array([[0.1, 0.2], [0.3, 0.4]]), ["negativo"])


class TestPlotTopNgramsBar:
    """Testes de :func:`visualization.ngrams.plot_top_ngrams_bar`."""

    def test_returns_figure_with_expected_title(self) -> None:
        """A figura retornada deve ter o título esperado."""
        figure = plot_top_ngrams_bar({"bom_dia": 10, "não_gostei": 8, "ótimo": 5}, top_n=2)
        assert figure.axes[0].get_title() == "N-gramas Mais Frequentes"

    def test_raises_on_empty_input(self) -> None:
        """Deve levantar ``EmptyDatasetError`` quando ``ngram_frequencies`` está vazio."""
        with pytest.raises(EmptyDatasetError):
            plot_top_ngrams_bar({})

    def test_raises_on_invalid_top_n(self) -> None:
        """Deve levantar ``ValueError`` quando ``top_n`` é menor que 1."""
        with pytest.raises(ValueError, match="top_n"):
            plot_top_ngrams_bar({"a": 1}, top_n=0)


class TestPlotTopFeatureImportances:
    """Testes de :func:`visualization.interpretability.plot_top_feature_importances`."""

    def test_returns_figure_with_expected_title(self) -> None:
        """A figura retornada deve ter o título esperado."""
        figure = plot_top_feature_importances(
            np.array([0.5, -0.8, 0.1]), ["preco", "qualidade", "cor"], top_n=2
        )
        assert figure.axes[0].get_title() == "Importância das Features"

    def test_raises_on_empty_input(self) -> None:
        """Deve levantar ``EmptyDatasetError`` quando ``importances`` está vazio."""
        with pytest.raises(EmptyDatasetError):
            plot_top_feature_importances(np.empty(0), [])

    def test_raises_on_length_mismatch(self) -> None:
        """Deve levantar ``ValueError`` quando os tamanhos diferem."""
        with pytest.raises(ValueError, match="mesmo tamanho"):
            plot_top_feature_importances(np.array([0.5, 0.3]), ["a"])

    def test_raises_on_invalid_top_n(self) -> None:
        """Deve levantar ``ValueError`` quando ``top_n`` é menor que 1."""
        with pytest.raises(ValueError, match="top_n"):
            plot_top_feature_importances(np.array([0.5]), ["a"], top_n=0)


class TestPlotShapSummary:
    """Testes de :func:`visualization.interpretability.plot_shap_summary`."""

    def test_returns_figure(self) -> None:
        """Deve retornar uma figura do matplotlib para um conjunto pequeno de valores SHAP."""
        shap_values = np.array(
            [[0.10, -0.20, 0.05], [0.20, -0.10, 0.02], [-0.10, 0.30, -0.05], [0.05, -0.05, 0.10]]
        )
        feature_matrix = np.array(
            [[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [0.5, 1.5, 2.5], [2.0, 3.0, 4.0]]
        )
        figure = plot_shap_summary(shap_values, feature_matrix, ["f1", "f2", "f3"])
        assert isinstance(figure, Figure)

    def test_raises_on_empty_input(self) -> None:
        """Deve levantar ``EmptyDatasetError`` quando ``shap_values`` está vazio."""
        with pytest.raises(EmptyDatasetError):
            plot_shap_summary(np.empty((0, 3)), np.empty((0, 3)), ["f1", "f2", "f3"])


class TestGenerateSentimentWordcloud:
    """Testes de :func:`visualization.wordcloud.generate_sentiment_wordcloud`."""

    def test_raises_on_empty_input(self) -> None:
        """Deve levantar ``EmptyDatasetError`` quando ``word_frequencies`` está vazio."""
        with pytest.raises(EmptyDatasetError):
            generate_sentiment_wordcloud({})

    def test_raises_import_error_when_dependency_is_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deve levantar ``ImportError`` com mensagem clara quando ``wordcloud`` não está instalado."""
        monkeypatch.setitem(sys.modules, "wordcloud", None)
        with pytest.raises(ImportError, match="wordcloud"):
            generate_sentiment_wordcloud({"ótimo": 5})
