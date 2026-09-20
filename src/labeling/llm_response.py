"""Construção de prompts e interpretação das respostas de LLMs de rotulagem.

Compartilhado pelos rotuladores por LLM (``src/labeling/openai_labeler.py`` e
``src/labeling/huggingface.py``) e pela comparação de prompts da camada de
diagnóstico (``src/diagnostics/comparison.py``): todos usam os templates de
``prompts/`` (marcador ``{{TEXTO}}``) e o mesmo formato de resposta em JSON.
"""

import json
import re
from collections.abc import Sequence
from typing import Any

from constants.labels import SENTIMENT_CLASSES

TEXT_PLACEHOLDER = "{{TEXTO}}"
_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def build_labeling_prompt(prompt_template: str, text: str) -> str:
    """Insere o texto do tweet no marcador ``{{TEXTO}}`` do template.

    Parameters
    ----------
    prompt_template : str
        Template de prompt (ver :func:`hypothesaes.utils.load_prompt_template`).
    text : str
        Texto do tweet (já sanitizado: sem menções, URLs ou identificadores).

    Returns
    -------
    str
        Prompt pronto para envio ao LLM.

    Examples
    --------
    >>> build_labeling_prompt('Tweet: "{{TEXTO}}"', "adorei")
    'Tweet: "adorei"'
    """
    return prompt_template.replace(TEXT_PLACEHOLDER, text)


def parse_llm_label_response(
    raw_response: str, *, allowed_labels: Sequence[str] = SENTIMENT_CLASSES
) -> tuple[str, float] | None:
    """Interpreta a resposta em JSON do LLM (``{"label": ..., ...}``).

    Aceita tanto ``{"label": ..., "confidence": ...}`` quanto
    ``{"label": ..., "probs": {"positivo": ..., ...}}`` (formato do prompt
    padrão ``labeling_1_rubrica_few-shot_distribuicao_probabilidade``): a
    confiança é lida de ``probs[label]`` quando presente, senão de
    ``confidence``, senão assume 1.0.

    Parameters
    ----------
    raw_response : str
        Resposta bruta do LLM.
    allowed_labels : Sequence[str], optional
        Classes de sentimento aceitas, by default
        :data:`constants.labels.SENTIMENT_CLASSES`.

    Returns
    -------
    tuple[str, float] | None
        Par ``(rótulo, confiança)``, ou ``None`` se a resposta não puder ser
        interpretada como um rótulo válido.

    Examples
    --------
    >>> parse_llm_label_response('{"label": "positivo", "confidence": 0.9}')
    ('positivo', 0.9)
    >>> parse_llm_label_response(
    ...     '{"label":"negativo","probs":{"positivo":0.1,"negativo":0.8,"neutro":0.1}}'
    ... )
    ('negativo', 0.8)
    >>> parse_llm_label_response("resposta sem json") is None
    True
    """
    text = raw_response.strip()
    if "</think>" in text:
        text = text.split("</think>")[1].strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    match = _JSON_OBJECT_PATTERN.search(text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None

    label = str(parsed.get("label", "")).strip().lower()
    if label not in allowed_labels:
        return None

    return label, _extract_confidence(parsed, label)


def _extract_confidence(parsed: dict[str, Any], label: str) -> float:
    """Extrai a confiança de uma resposta já interpretada, de ``probs`` ou ``confidence``."""
    probs = parsed.get("probs")
    if isinstance(probs, dict) and label in probs:
        try:
            return min(1.0, max(0.0, float(probs[label])))
        except (TypeError, ValueError):
            return 1.0
    try:
        return min(1.0, max(0.0, float(parsed.get("confidence", 1.0))))
    except (TypeError, ValueError):
        return 1.0
