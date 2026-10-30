"""Testes da retomada por par na etapa ``hypotheses`` (modo ``disagreement``)."""

import json
from pathlib import Path

from pipelines.hypotheses import _load_completed_pair_summary


def _write(path: Path, statuses: dict[str, str]) -> None:
    path.write_text(
        json.dumps([{"target": t, "status": s} for t, s in statuses.items()]), encoding="utf-8"
    )


def test_load_completed_pair_summary_reuses_when_all_targets_done(tmp_path):
    """Todos os alvos concluídos: o resumo salvo é reaproveitado."""
    path = tmp_path / "hipoteses_par.json"
    _write(path, {"disagreement": "concluido", "uncertainty": "concluido"})
    assert _load_completed_pair_summary(path, ("disagreement", "uncertainty")) is not None


def test_load_completed_pair_summary_reprocesses_when_target_missing_or_failed(tmp_path):
    """Alvo ausente ou não concluído: o par é reprocessado."""
    path = tmp_path / "hipoteses_par.json"
    _write(path, {"disagreement": "concluido", "uncertainty": "falhou"})
    assert _load_completed_pair_summary(path, ("disagreement", "uncertainty")) is None
    _write(path, {"disagreement": "concluido"})
    assert _load_completed_pair_summary(path, ("disagreement", "uncertainty")) is None


def test_load_completed_pair_summary_handles_missing_and_corrupt_file(tmp_path):
    """Arquivo inexistente ou corrompido: retorna None sem levantar."""
    assert _load_completed_pair_summary(tmp_path / "nao_existe.json", ("disagreement",)) is None
    bad = tmp_path / "ruim.json"
    bad.write_text("{", encoding="utf-8")
    assert _load_completed_pair_summary(bad, ("disagreement",)) is None


def test_select_names_filters_by_full_name_or_suffix():
    """O filtro aceita o nome completo ou o sufixo após ':' e preserva a ordem original."""
    from pipelines.hypotheses import _select_names

    available = ["huggingface", "huggingface_extra:FinBERT-PT-BR", "huggingface_extra:bertweet"]
    assert _select_names(available, None, "modelo") == available
    assert _select_names(available, ["FinBERT-PT-BR", "huggingface"], "modelo") == [
        "huggingface",
        "huggingface_extra:FinBERT-PT-BR",
    ]


def test_select_names_rejects_unknown_name():
    """Nome pedido que não existe gera erro claro em vez de pular em silêncio."""
    import pytest

    from exceptions.data import DataValidationError
    from pipelines.hypotheses import _select_names

    with pytest.raises(DataValidationError):
        _select_names(["gold_openai"], ["gold_xyz"], "gold set")
