"""Dashboard Streamlit comparativo entre as abordagens do projeto.

Implementa a Fase 21 do plano de elaboração (ver ``PLANO-ELABORACAO.md``):
uma camada de visualização apenas leitura sobre os artefatos já produzidos
pelas etapas anteriores do pipeline (``reports/metrics``,
``reports/figures`` — ver ``notebooks/07_avaliacao_comparativa.ipynb``, que
gera as figuras lidas aqui), mais uma demonstração interativa de predição
sobre texto livre (mesma abordagem servida por ``app/api.py``).

Uso
---
    uv run streamlit run app/dashboard.py

Ver ``configs/deploy.yaml -> dashboard`` para porta e habilitação da visão
comparativa.
"""

from __future__ import annotations

import sys
from pathlib import Path

# `app/` não fica dentro de `src/` (raiz de importação do projeto — ver
# CLAUDE.md, "Import style"), então o diretório precisa ser inserido em
# `sys.path` manualmente antes de qualquer import próprio (ver
# `pyproject.toml -> [tool.ruff.lint.per-file-ignores] "app/*"`).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import logging
from typing import Any

import polars as pl
import streamlit as st

from config.constants import CONFIG_FILE_NAMES
from config.logging import configure_logging
from config.paths import CONFIGS_DIR, ProjectPaths, load_project_paths
from data.loader import read_dataset_file
from exceptions.base import ProjectError
from inference.online import OnlinePredictor
from inference.predictor import Predictor
from io_utils.yaml import read_yaml
from models.persistence import load_classifier

logger = logging.getLogger(__name__)

_COMPARATIVE_REPORT_FILE_NAME = "comparativo_modelos.csv"
_CONFUSION_MATRIX_FILE_PREFIX = "matriz_confusao_"
_CALIBRATION_CURVE_FILE_PREFIX = "curva_calibracao_"

# Mesma convenção adotada em `app/api.py` para resolver, por abordagem, qual
# checkpoint local carregar (o projeto ainda não associa, em
# `configs/deploy.yaml`, um nome de modelo específico a cada abordagem).
_DEFAULT_MODEL_NAME_BY_APPROACH: dict[str, str] = {
    "deep_learning": "lstm",
    "transformer": "bertimbau",
}
_CHECKPOINT_BACKEND_BY_APPROACH: dict[str, str] = {
    "deep_learning": "torch",
    "transformer": "torch",
}


@st.cache_resource(show_spinner="Carregando configuração e caminhos do projeto...")
def _load_project_context() -> tuple[dict[str, Any], ProjectPaths]:
    """Carrega ``configs/deploy.yaml`` e os caminhos resolvidos do projeto.

    Returns
    -------
    tuple[dict[str, Any], ProjectPaths]
        Conteúdo de ``configs/deploy.yaml`` e os caminhos resolvidos
        (``configs/paths.yaml``).
    """
    configure_logging()
    deploy_config = read_yaml(CONFIGS_DIR / CONFIG_FILE_NAMES["deploy"])
    paths = load_project_paths()
    return deploy_config, paths


@st.cache_resource(show_spinner="Carregando modelo para a demonstração interativa...")
def _load_demo_predictor(approach: str) -> OnlinePredictor | None:
    """Carrega, por abordagem, o preditor usado na demonstração interativa.

    Reaproveita apenas o checkpoint local (sem MLflow Model Registry, ao
    contrário de ``app/api.py``): o dashboard é uma ferramenta de inspeção
    local, não um serviço de produção.

    Parameters
    ----------
    approach : str
        Abordagem a carregar, uma das chaves de
        :data:`_DEFAULT_MODEL_NAME_BY_APPROACH`.

    Returns
    -------
    OnlinePredictor | None
        Preditor pronto para uso, ou ``None`` se o checkpoint ainda não
        existir em disco (nenhum treino executado para essa abordagem).
    """
    _, paths = _load_project_context()
    model_name = _DEFAULT_MODEL_NAME_BY_APPROACH[approach]
    backend = _CHECKPOINT_BACKEND_BY_APPROACH[approach]
    extension = "joblib" if backend == "joblib" else "pt"
    checkpoint_path = paths.models_checkpoints_dir / f"{model_name}.{extension}"
    if not checkpoint_path.exists():
        return None
    model = load_classifier(checkpoint_path, backend=backend)
    return OnlinePredictor(Predictor(model))


def _render_comparative_table(paths: ProjectPaths) -> None:
    """Renderiza a tabela comparativa entre abordagens, quando disponível.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto.
    """
    st.subheader("Comparação entre abordagens")
    report_path = paths.reports_metrics_dir / _COMPARATIVE_REPORT_FILE_NAME
    if not report_path.exists():
        st.info(
            "Relatório comparativo ainda não encontrado em "
            f"'{report_path}'. Execute a etapa 'comparative_evaluation' "
            "(ver `notebooks/07_avaliacao_comparativa.ipynb` ou "
            "`uv run python src/main.py --stage comparative_evaluation`)."
        )
        return
    comparative_report = read_dataset_file(report_path)
    st.dataframe(comparative_report.to_pandas(), use_container_width=True)


def _render_figure_gallery(paths: ProjectPaths, *, file_prefix: str, title: str) -> None:
    """Renderiza uma galeria de figuras já geradas, selecionável por modelo.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto.
    file_prefix : str
        Prefixo do nome de arquivo usado para localizar as figuras (ex.:
        ``"matriz_confusao_"``), seguido do nome do modelo e ``".png"``
        (convenção definida em
        ``notebooks/07_avaliacao_comparativa.ipynb``).
    title : str
        Título da seção exibida acima da galeria.
    """
    st.subheader(title)
    figure_paths = sorted(paths.reports_figures_dir.glob(f"{file_prefix}*.png"))
    if not figure_paths:
        st.info(
            f"Nenhuma figura encontrada em '{paths.reports_figures_dir}' com o "
            f"prefixo '{file_prefix}'. Gere-as em "
            "`notebooks/07_avaliacao_comparativa.ipynb`."
        )
        return
    model_names = [path.stem.removeprefix(file_prefix) for path in figure_paths]
    selected_model_name = st.selectbox("Modelo", model_names, key=f"selectbox_{file_prefix}")
    selected_path = figure_paths[model_names.index(selected_model_name)]
    st.image(str(selected_path), caption=selected_path.stem)


def _render_interactive_demo(deploy_config: dict[str, Any]) -> None:
    """Renderiza a demonstração interativa de predição sobre texto livre.

    Parameters
    ----------
    deploy_config : dict[str, Any]
        Conteúdo de ``configs/deploy.yaml``.
    """
    st.subheader("Teste você mesmo")
    approach = st.selectbox(
        "Abordagem",
        tuple(_DEFAULT_MODEL_NAME_BY_APPROACH),
        index=tuple(_DEFAULT_MODEL_NAME_BY_APPROACH).index(deploy_config["api"]["default_approach"])
        if deploy_config["api"]["default_approach"] in _DEFAULT_MODEL_NAME_BY_APPROACH
        else 0,
    )
    text = st.text_area(
        "Texto em português", placeholder="Digite um comentário para classificar..."
    )
    if not st.button("Classificar sentimento"):
        return

    predictor = _load_demo_predictor(approach)
    if predictor is None:
        st.warning(
            f"Nenhum checkpoint local encontrado para a abordagem '{approach}'. "
            "Treine o modelo correspondente antes de usar a demonstração "
            "(ver `uv run python src/main.py --stage training_deep_learning`)."
        )
        return

    try:
        record = predictor.predict(text)
    except ValueError as exception:
        st.error(f"Entrada inválida: {exception}")
        return
    except ProjectError as exception:
        st.error(f"Falha na inferência: {exception}")
        return

    st.metric(
        "Sentimento predito", record["sentiment_label"], f"{record['confidence']:.1%} de confiança"
    )
    st.bar_chart(pl.DataFrame([record["probabilities"]]).to_pandas().T, use_container_width=True)


def main() -> None:
    """Monta a página do dashboard, na ordem: comparação, figuras e demonstração interativa."""
    st.set_page_config(page_title="Análise de Sentimentos pt-BR", layout="wide")
    st.title("Dashboard Comparativo — Análise de Sentimentos pt-BR")

    deploy_config, paths = _load_project_context()

    if deploy_config["dashboard"]["comparative_view"]:
        _render_comparative_table(paths)
        _render_figure_gallery(
            paths, file_prefix=_CONFUSION_MATRIX_FILE_PREFIX, title="Matrizes de confusão"
        )
        _render_figure_gallery(
            paths, file_prefix=_CALIBRATION_CURVE_FILE_PREFIX, title="Curvas de calibração"
        )

    _render_interactive_demo(deploy_config)


main()
