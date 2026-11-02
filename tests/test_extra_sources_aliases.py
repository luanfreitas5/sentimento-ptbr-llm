"""Testes dos apelidos dos modelos adicionais do Hugging Face."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from config.paths import ProjectPaths
from io_utils.json import write_json
from labeling.extra_sources import load_extra_display_names, load_extra_model_names


def _write_extra_base(paths: ProjectPaths, name: str, model_id: str) -> None:
    """Cria uma base adicional vazia com o ``meta.json`` ao lado."""
    base = paths.data_gold_dir / f"tweets_data_{name}.parquet"
    base.parent.mkdir(parents=True, exist_ok=True)
    base.touch()
    write_json({"model": model_id}, base.with_suffix(".meta.json"))


@pytest.fixture
def pipeline_paths(tmp_path: Path) -> ProjectPaths:
    """Caminhos mínimos: só o diretório gold e as bases configuradas."""
    gold = tmp_path / "gold"
    fake = SimpleNamespace(
        data_gold_dir=gold,
        huggingface_labeled_file=gold / "tweets_data_xlmr.parquet",
        huggingface_secondary_file=gold / "tweets_data_bertweet.parquet",
    )
    return cast(ProjectPaths, fake)


def test_display_names_use_alias_and_fall_back_to_model_name(
    pipeline_paths: ProjectPaths, tmp_path: Path
) -> None:
    """Com alias o nome de exibição muda; sem alias vale o nome do modelo."""
    _write_extra_base(pipeline_paths, "com-alias", "org/com-alias")
    _write_extra_base(pipeline_paths, "sem-alias", "org/sem-alias")
    config = tmp_path / "labeling.yaml"
    config.write_text(
        'huggingface_extra:\n  models:\n    - model: "org/com-alias"\n      alias: "Curto"\n',
        encoding="utf-8",
    )

    display = load_extra_display_names(pipeline_paths, config)

    assert display["huggingface_extra:com-alias"] == "Curto"
    assert display["huggingface_extra:sem-alias"] == "sem-alias"
    assert load_extra_model_names(pipeline_paths)["huggingface_extra:com-alias"] == "com-alias"
