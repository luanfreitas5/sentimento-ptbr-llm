"""Visualizações para o projeto de análise de sentimento pt-BR.

Implementa a seção "Visualization" do CLAUDE.md: uma paleta de cores única
por classe de sentimento (compatível com daltonismo), gráficos de
distribuição, matriz de confusão, curvas ROC/Precisão-Revocação,
diagnósticos de calibração, embeddings, n-gramas, interpretabilidade
(SHAP) e nuvens de palavras — todos salvos de forma consistente (PNG
300dpi + SVG) em ``reports/figures/``.

Modules
-------
theme
    Paleta de cores por classe de sentimento e função de salvamento de
    figuras.
distributions
    Distribuição de classes e de comprimento de texto.
confusion_matrix
    Mapa de calor da matriz de confusão.
roc_pr_curves
    Curvas ROC e Precisão-Revocação (esquema *one-vs-rest*).
diagnostics
    Curva de confiabilidade e distribuição de confiança por acerto.
embeddings
    Dispersão de embeddings reduzidos a 2D, coloridos por classe.
ngrams
    Frequência dos n-gramas mais comuns.
interpretability
    Resumo de valores SHAP e importância de features.
wordcloud
    Nuvem de palavras por classe de sentimento (dependência opcional).
"""

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

__all__: list[str] = [
    "SENTIMENT_COLOR_PALETTE",
    "apply_project_theme",
    "generate_sentiment_wordcloud",
    "plot_calibration_curve",
    "plot_class_distribution",
    "plot_confidence_distribution_by_correctness",
    "plot_confusion_matrix_heatmap",
    "plot_embedding_scatter",
    "plot_precision_recall_curves_one_vs_rest",
    "plot_roc_curves_one_vs_rest",
    "plot_shap_summary",
    "plot_text_length_distribution",
    "plot_top_feature_importances",
    "plot_top_ngrams_bar",
    "save_figure",
]
