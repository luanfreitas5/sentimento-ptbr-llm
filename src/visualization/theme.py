"""Paleta de cores e utilitários de estilo compartilhados por ``src/visualization/``.

Centraliza a identidade visual do projeto (CLAUDE.md, "Visualization"):
uma paleta única por classe de sentimento, reaproveitada em todos os
gráficos, e uma função de salvamento consistente (PNG 300dpi + SVG) em
``reports/figures/``.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

from config.paths import load_project_paths
from constants.labels import NEGATIVE_LABEL, NEUTRAL_LABEL, POSITIVE_LABEL

logger = logging.getLogger(__name__)

# Paleta compatível com daltonismo (Okabe-Ito), reaproveitada em todo o projeto.
SENTIMENT_COLOR_PALETTE: dict[str, str] = {
    NEGATIVE_LABEL: "#D55E00",
    NEUTRAL_LABEL: "#999999",
    POSITIVE_LABEL: "#0072B2",
}

FIGURE_DPI = 300


def apply_project_theme() -> None:
    """Aplica o tema visual padrão do projeto (``seaborn``) aos gráficos subsequentes.

    Returns
    -------
    None

    Examples
    --------
    >>> apply_project_theme()
    """
    sns.set_theme(
        style="whitegrid", context="notebook", palette=list(SENTIMENT_COLOR_PALETTE.values())
    )


def save_figure(
    figure: Figure, filename: str, *, directory: Path | None = None
) -> tuple[Path, Path]:
    """Salva uma figura em PNG (300dpi) e SVG, em um diretório consistente.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figura a ser salva.
    filename : str
        Nome-base do arquivo, sem extensão.
    directory : Path | None, optional
        Diretório de destino; usa ``reports/figures`` (``configs/paths.yaml``
        -> ``config.paths.ProjectPaths.reports_figures_dir``) quando
        ``None``, by default None.

    Returns
    -------
    tuple[Path, Path]
        Caminhos dos arquivos ``.png`` e ``.svg`` salvos, respectivamente.

    Examples
    --------
    >>> save_figure(figura, "distribuicao_sentimento")  # doctest: +SKIP
    """
    output_directory = directory or load_project_paths().reports_figures_dir
    output_directory.mkdir(parents=True, exist_ok=True)

    png_path = output_directory / f"{filename}.png"
    svg_path = output_directory / f"{filename}.svg"
    figure.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")
    plt.close(figure)

    logger.info("Figura salva em '%s' e '%s'.", png_path, svg_path)
    return png_path, svg_path
