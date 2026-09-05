"""Fixtures compartilhadas entre os testes do projeto."""

import matplotlib

# Backend não interativo: os testes de src/visualization/ geram figuras sem
# necessidade de um display, e o backend deve ser fixado antes de qualquer
# import de matplotlib.pyplot em todo o processo de teste.
matplotlib.use("Agg")

import logging
from collections.abc import Generator
from typing import Any

import polars as pl
import pytest


@pytest.fixture
def sample_labeled_corpus() -> pl.DataFrame:
    """DataFrame de exemplo de corpus rotulado, válido contra ``LabeledCorpusSchema``."""
    return pl.DataFrame(
        {
            "id": ["1", "2", "3"],
            "text": ["ótimo produto", "péssimo atendimento", "chegou no prazo"],
            "sentiment_label": ["positivo", "negativo", "neutro"],
        }
    )


@pytest.fixture
def minimal_general_config_dict() -> dict[str, Any]:
    """Dicionário mínimo válido contra ``GeneralConfig`` (espelha ``configs/config.yaml``)."""
    return {
        "project": {
            "name": "exemplo",
            "description": "desc",
            "version": "0.1.0",
            "language": "pt-BR",
        },
        "reproducibility": {
            "random_seed": 42,
            "pythonhashseed": 42,
            "deterministic_algorithms": True,
        },
        "experiment": {
            "name": "exemplo",
            "tracking_uri": "mlruns",
            "registry_stage_default": "Staging",
        },
        "labels": {"classes": ["negativo", "neutro", "positivo"], "target_column": "sentimento"},
        "data_split": {
            "test_size": 0.2,
            "validation_size": 0.1,
            "stratify": True,
            "random_state": 42,
        },
        "stages": ["ingestion"],
    }


@pytest.fixture
def minimal_paths_config_dict() -> dict[str, Any]:
    """Dicionário mínimo válido para :func:`config.paths.load_project_paths`."""
    return {
        "data": {
            "raw": "data/raw",
            "external": "data/external",
            "interim": "data/interim",
            "processed": "data/processed",
        },
        "data_files": {
            "raw_tweets": "data/raw/tweets.parquet",
            "tweetsentbr": "data/external/tweetsentbr.parquet",
            "repro": "data/external/repro.parquet",
            "corpus_normalizado": "data/interim/normalizado.parquet",
            "corpus_rotulado": "data/processed/rotulado.parquet",
            "corpus_treino": "data/processed/treino.parquet",
            "corpus_validacao": "data/processed/validacao.parquet",
            "corpus_teste": "data/processed/teste.parquet",
        },
        "models": {
            "checkpoints": "models/checkpoints",
            "artifacts": "models/artifacts",
            "registry": "models/registry",
        },
        "mlflow": {"tracking_dir": "mlruns"},
        "logs": {"dir": "logs"},
        "reports": {
            "figures": "reports/figures",
            "tables": "reports/tables",
            "metrics": "reports/metrics",
            "statistics": "reports/statistics",
            "ablation": "reports/ablation",
            "interpretability": "reports/interpretability",
            "model_cards": "reports/model_cards",
            "datasheets": "reports/datasheets",
        },
        "docs": {"root": "docs", "guides": "docs/guides", "assets": "docs/assets"},
    }


@pytest.fixture
def minimal_logging_config_dict() -> dict[str, Any]:
    """Dicionário mínimo válido para :func:`config.logging.configure_logging`."""
    return {
        "level": "INFO",
        "console": {"enabled": True, "rich_tracebacks": True, "show_path": True, "markup": True},
        "file": {
            "enabled": True,
            "dir": "logs",
            "filename_pattern": "log_{date}.log",
            "rotation": "daily",
            "backup_count": 3,
            "encoding": "utf-8",
        },
        "format": {
            "file": "%(asctime)s \t %(levelname)s \t %(name)s \t %(message)s",
            "console": "%(message)s",
            "date_format": "%Y-%m-%d %H:%M:%S",
        },
        "loggers": {
            "root_level": "WARNING",
            "project_level": "INFO",
            "third_party_level": "WARNING",
        },
    }


@pytest.fixture
def reset_root_logger() -> Generator[logging.Logger, Any, None]:
    """Restaura os handlers e o nível do logger raiz após o teste, evitando vazamento de estado."""
    root = logging.getLogger()
    handlers_originais = list(root.handlers)
    nivel_original = root.level
    yield root
    root.handlers.clear()
    for handler in handlers_originais:
        root.addHandler(handler)
    root.setLevel(nivel_original)
