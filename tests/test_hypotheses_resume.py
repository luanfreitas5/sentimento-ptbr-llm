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
