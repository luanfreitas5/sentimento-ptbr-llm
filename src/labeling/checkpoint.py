"""Checkpoint incremental da rotulagem por LLM (retomada e deduplicação).

Cada rótulo válido é anexado a um arquivo JSON Lines assim que o lote é
processado. Numa nova execução, os ``id`` já presentes no checkpoint não são
reprocessados — evitando gastar chamadas de API/GPU duas vezes e permitindo
retomar após falhas (timeout, limite de taxa, queda de energia).

O nome do arquivo inclui uma impressão digital da configuração
(modelo + prompt + temperatura): mudar qualquer um deles inicia um checkpoint
novo, nunca misturando rótulos gerados por configurações diferentes.
"""

import json
import logging
from collections.abc import Mapping
from pathlib import Path

from utils.hashing import calculate_text_hash

logger = logging.getLogger(__name__)

_FINGERPRINT_LENGTH = 12


def build_checkpoint_path(
    directory: Path, source_name: str, *, model_name: str, prompt_template: str, temperature: float
) -> Path:
    """Resolve o caminho do checkpoint de uma fonte para uma configuração específica.

    Parameters
    ----------
    directory : Path
        Diretório dos checkpoints (``ProjectPaths.labeling_checkpoints_dir``).
    source_name : str
        Fonte de rotulagem (``huggingface`` ou ``openai``).
    model_name : str
        Modelo usado na rotulagem.
    prompt_template : str
        Conteúdo do template de prompt (não o nome: editar o arquivo invalida o checkpoint).
    temperature : float
        Temperatura de geração.

    Returns
    -------
    Path
        Caminho ``<directory>/<fonte>_<impressão digital>.jsonl``.

    Examples
    --------
    >>> path = build_checkpoint_path(
    ...     Path("ckpt"), "openai", model_name="m", prompt_template="p", temperature=0.0
    ... )
    >>> path.name.startswith("openai_") and path.suffix == ".jsonl"
    True
    """
    fingerprint = calculate_text_hash(f"{model_name}|{temperature}|{prompt_template}")
    return directory / f"{source_name}_{fingerprint[:_FINGERPRINT_LENGTH]}.jsonl"


def read_labeling_checkpoint(checkpoint_path: Path) -> dict[str, tuple[str, float]]:
    """Lê os rótulos já gravados no checkpoint.

    Linhas corrompidas (ex.: última linha truncada por uma interrupção) são
    ignoradas com aviso; o tweet correspondente será reprocessado.

    Parameters
    ----------
    checkpoint_path : Path
        Caminho do checkpoint; um arquivo inexistente equivale a checkpoint vazio.

    Returns
    -------
    dict[str, tuple[str, float]]
        Mapa ``id -> (rótulo, confiança)``; em duplicatas, vale a última ocorrência.

    Examples
    --------
    >>> read_labeling_checkpoint(Path("nao_existe.jsonl"))
    {}
    """
    if not checkpoint_path.is_file():
        return {}

    results: dict[str, tuple[str, float]] = {}
    n_invalid_lines = 0
    with checkpoint_path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                results[str(record["id"])] = (
                    str(record["sentiment_label"]),
                    float(record["confidence_score"]),
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                n_invalid_lines += 1
    if n_invalid_lines:
        logger.warning(
            "Checkpoint '%s': %d linha(s) inválida(s) ignorada(s).",
            checkpoint_path,
            n_invalid_lines,
        )
    return results


def append_labeling_checkpoint(
    checkpoint_path: Path, results: Mapping[str, tuple[str, float]]
) -> None:
    """Anexa rótulos ao checkpoint, criando o arquivo e os diretórios se necessário.

    Parameters
    ----------
    checkpoint_path : Path
        Caminho do checkpoint.
    results : Mapping[str, tuple[str, float]]
        Mapa ``id -> (rótulo, confiança)`` a gravar.

    Examples
    --------
    >>> append_labeling_checkpoint(Path("ckpt.jsonl"), {"1": ("positivo", 0.9)})  # doctest: +SKIP
    """
    if not results:
        return
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("a", encoding="utf-8") as file:
        for tweet_id, (label, confidence) in results.items():
            file.write(
                json.dumps(
                    {"id": tweet_id, "sentiment_label": label, "confidence_score": confidence},
                    ensure_ascii=False,
                )
                + "\n"
            )
