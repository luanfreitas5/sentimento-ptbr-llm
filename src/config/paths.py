"""Centraliza todos os caminhos do projeto usando ``pathlib.Path``.

Os caminhos relativos vêm de ``configs/paths.yaml`` e são resolvidos aqui
para caminhos absolutos, a partir de :data:`PROJECT_ROOT`. Nenhum caminho
deve ser escrito diretamente (como string) em outro módulo do projeto.
"""

from dataclasses import dataclass
from pathlib import Path

from config.constants import CONFIG_FILE_NAMES
from io_utils.yaml import read_yaml

# src/config/paths.py -> parents[0]=src/config, [1]=src, [2]=raiz do repositório.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
DEFAULT_PATHS_CONFIG_FILE: Path = CONFIGS_DIR / CONFIG_FILE_NAMES["paths"]


def resolve_project_path(relative_path: str | Path) -> Path:
    """Resolve um caminho relativo à raiz do projeto em um caminho absoluto.

    Parameters
    ----------
    relative_path : str | Path
        Caminho relativo à raiz do repositório (ex.: ``"data/raw"``).

    Returns
    -------
    Path
        Caminho absoluto correspondente.

    Examples
    --------
    >>> resolve_project_path("configs").name
    'configs'
    """
    return PROJECT_ROOT / relative_path


@dataclass(frozen=True)
class ProjectPaths:
    """Caminhos absolutos de dados, modelos, relatórios, logs e documentação.

    Parameters
    ----------
    data_raw_dir, data_external_dir, data_interim_dir, data_processed_dir : Path
        Diretórios das etapas de dados (ver CLAUDE.md, "Project Structure").
    raw_tweets_file, tweetsentbr_file, repro_file : Path
        Arquivos de dados de entrada (coleta própria e gold sets externos).
    normalized_corpus_file, labeled_corpus_file : Path
        Arquivos intermediário normalizado e final rotulado.
    training_corpus_file, validation_corpus_file, test_corpus_file : Path
        Arquivos de particionamento treino/validação/teste.
    models_checkpoints_dir, models_artifacts_dir, models_registry_dir : Path
        Diretórios de artefatos de modelo.
    mlflow_tracking_dir : Path
        Diretório de tracking local do MLflow.
    logs_dir : Path
        Diretório de arquivos de log.
    reports_figures_dir, reports_tables_dir, reports_metrics_dir,
    reports_statistics_dir, reports_ablation_dir, reports_interpretability_dir,
    reports_model_cards_dir, reports_datasheets_dir : Path
        Subdiretórios de relatórios gerados.
    docs_root_dir, docs_guides_dir, docs_assets_dir : Path
        Diretórios de documentação (MkDocs Material).
    """

    data_raw_dir: Path
    data_external_dir: Path
    data_interim_dir: Path
    data_processed_dir: Path

    raw_tweets_file: Path
    tweetsentbr_file: Path
    repro_file: Path
    normalized_corpus_file: Path
    labeled_corpus_file: Path
    training_corpus_file: Path
    validation_corpus_file: Path
    test_corpus_file: Path

    models_checkpoints_dir: Path
    models_artifacts_dir: Path
    models_registry_dir: Path

    mlflow_tracking_dir: Path

    logs_dir: Path

    reports_figures_dir: Path
    reports_tables_dir: Path
    reports_metrics_dir: Path
    reports_statistics_dir: Path
    reports_ablation_dir: Path
    reports_interpretability_dir: Path
    reports_model_cards_dir: Path
    reports_datasheets_dir: Path

    docs_root_dir: Path
    docs_guides_dir: Path
    docs_assets_dir: Path


def load_project_paths(config_file_path: Path = DEFAULT_PATHS_CONFIG_FILE) -> ProjectPaths:
    """Carrega ``configs/paths.yaml`` e resolve todos os caminhos para absolutos.

    Parameters
    ----------
    config_file_path : Path, optional
        Caminho do arquivo YAML de caminhos, by default :data:`DEFAULT_PATHS_CONFIG_FILE`.

    Returns
    -------
    ProjectPaths
        Estrutura imutável com todos os caminhos absolutos do projeto.

    Raises
    ------
    DataNotFoundError
        Se o arquivo de configuração não existir.

    Examples
    --------
    >>> load_project_paths().data_raw_dir.name
    'raw'
    """
    dados = read_yaml(config_file_path)

    return ProjectPaths(
        data_raw_dir=resolve_project_path(dados["data"]["raw"]),
        data_external_dir=resolve_project_path(dados["data"]["external"]),
        data_interim_dir=resolve_project_path(dados["data"]["interim"]),
        data_processed_dir=resolve_project_path(dados["data"]["processed"]),
        raw_tweets_file=resolve_project_path(dados["data_files"]["raw_tweets"]),
        tweetsentbr_file=resolve_project_path(dados["data_files"]["tweetsentbr"]),
        repro_file=resolve_project_path(dados["data_files"]["repro"]),
        normalized_corpus_file=resolve_project_path(dados["data_files"]["corpus_normalizado"]),
        labeled_corpus_file=resolve_project_path(dados["data_files"]["corpus_rotulado"]),
        training_corpus_file=resolve_project_path(dados["data_files"]["corpus_treino"]),
        validation_corpus_file=resolve_project_path(dados["data_files"]["corpus_validacao"]),
        test_corpus_file=resolve_project_path(dados["data_files"]["corpus_teste"]),
        models_checkpoints_dir=resolve_project_path(dados["models"]["checkpoints"]),
        models_artifacts_dir=resolve_project_path(dados["models"]["artifacts"]),
        models_registry_dir=resolve_project_path(dados["models"]["registry"]),
        mlflow_tracking_dir=resolve_project_path(dados["mlflow"]["tracking_dir"]),
        logs_dir=resolve_project_path(dados["logs"]["dir"]),
        reports_figures_dir=resolve_project_path(dados["reports"]["figures"]),
        reports_tables_dir=resolve_project_path(dados["reports"]["tables"]),
        reports_metrics_dir=resolve_project_path(dados["reports"]["metrics"]),
        reports_statistics_dir=resolve_project_path(dados["reports"]["statistics"]),
        reports_ablation_dir=resolve_project_path(dados["reports"]["ablation"]),
        reports_interpretability_dir=resolve_project_path(dados["reports"]["interpretability"]),
        reports_model_cards_dir=resolve_project_path(dados["reports"]["model_cards"]),
        reports_datasheets_dir=resolve_project_path(dados["reports"]["datasheets"]),
        docs_root_dir=resolve_project_path(dados["docs"]["root"]),
        docs_guides_dir=resolve_project_path(dados["docs"]["guides"]),
        docs_assets_dir=resolve_project_path(dados["docs"]["assets"]),
    )
