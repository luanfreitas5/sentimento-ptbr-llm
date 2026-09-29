"""Formatos de resposta rápidos para a rotulagem por LLM (resposta mínima e multi-tweet).

Complementa :mod:`labeling.llm_response` (um tweet por requisição, resposta em JSON)
com dois modos que reduzem o tempo de rotulagem:

* **Resposta mínima com logprobs** (:func:`build_label_only_prompt`,
  :func:`parse_first_token_label`): o modelo responde só o rótulo (1 a 3 tokens) e a
  confiança é a distribuição de probabilidade do primeiro token.
* **Multi-tweet** (:func:`build_multi_tweet_prompt`, :func:`parse_multi_tweet_response`):
  vários tweets numerados por requisição, resposta em uma lista JSON. Itens ausentes ou
  inválidos viram ``None`` para o chamador reclassificar individualmente.
"""

import json
import math
import re
from collections.abc import Mapping, Sequence

from constants.labels import SENTIMENT_CLASSES
from labeling.llm_response import TEXT_PLACEHOLDER, parse_llm_label_response

_JSON_ARRAY_PATTERN = re.compile(r"\[.*\]", re.DOTALL)
_LABEL_ONLY_SUFFIX = (
    "\n\nIGNORE o formato JSON descrito acima. Responda SOMENTE com uma palavra, "
    "sem pontuação nem explicação: {labels}."
)
_MULTI_TWEET_SUFFIX = (
    "\n\nClassifique CADA tweet numerado abaixo de forma independente. Responda APENAS com "
    "uma lista JSON válida, um objeto por tweet, no formato "
    '[{{"id": 1, "label": "{example}"}}, ...], com os rótulos permitidos: {labels}. '
    "Sem markdown, sem explicação.\n\n{tweets}"
)


def build_label_only_prompt(
    prompt_template: str, text: str, *, allowed_labels: Sequence[str] = SENTIMENT_CLASSES
) -> str:
    """Monta o prompt de resposta mínima: template padrão + instrução de uma palavra.

    Parameters
    ----------
    prompt_template : str
        Template do prompt (marcador ``{{TEXTO}}``).
    text : str
        Texto sanitizado do tweet.
    allowed_labels : Sequence[str], optional
        Classes aceitas, by default :data:`constants.labels.SENTIMENT_CLASSES`.

    Returns
    -------
    str
        Prompt que pede apenas o rótulo.

    Examples
    --------
    >>> "adorei" in build_label_only_prompt('Tweet: "{{TEXTO}}"', "adorei")
    True
    """
    prompt = prompt_template.replace(TEXT_PLACEHOLDER, text)
    return prompt + _LABEL_ONLY_SUFFIX.format(labels=", ".join(allowed_labels))


def parse_first_token_label(
    text: str,
    first_token_logprobs: Mapping[str, float],
    *,
    allowed_labels: Sequence[str] = SENTIMENT_CLASSES,
) -> tuple[str, float] | None:
    """Interpreta a resposta mínima; a confiança vem dos logprobs do primeiro token.

    Cada token alternativo é associado a uma classe quando é prefixo não vazio de
    exatamente uma delas (ex.: ``"pos"`` -> ``positivo``). As probabilidades das classes
    são normalizadas entre si. Sem ``logprobs`` (endpoint sem suporte), usa o texto
    gerado com confiança 1.0.

    Parameters
    ----------
    text : str
        Texto gerado pelo modelo.
    first_token_logprobs : Mapping[str, float]
        ``{token: logprob}`` do primeiro token gerado.
    allowed_labels : Sequence[str], optional
        Classes aceitas, by default :data:`constants.labels.SENTIMENT_CLASSES`.

    Returns
    -------
    tuple[str, float] | None
        ``(rótulo, confiança)`` ou ``None`` se nenhuma classe puder ser identificada.

    Examples
    --------
    >>> parse_first_token_label("positivo", {"pos": -0.1, "neg": -2.5, "neu": -3.5})[0]
    'positivo'
    >>> parse_first_token_label("???", {}) is None
    True
    """
    class_probs = dict.fromkeys(allowed_labels, 0.0)
    for token, logprob in first_token_logprobs.items():
        label = _match_label_prefix(token, allowed_labels)
        if label is not None:
            class_probs[label] += math.exp(logprob)

    total = sum(class_probs.values())
    if total > 0:
        best = max(class_probs, key=lambda label: class_probs[label])
        return best, class_probs[best] / total

    word = text.strip().lower().strip("\"'.,;:!")
    fallback = _match_label_prefix(word, allowed_labels) if word else None
    return (fallback, 1.0) if fallback is not None else None


def _match_label_prefix(token: str, allowed_labels: Sequence[str]) -> str | None:
    """Devolve a única classe da qual ``token`` é prefixo (ou ``None``)."""
    cleaned = token.strip().lower().lstrip("\"'")
    if not cleaned:
        return None
    matches = [label for label in allowed_labels if label.startswith(cleaned)]
    return matches[0] if len(matches) == 1 else None


def build_multi_tweet_prompt(
    prompt_template: str, texts: Sequence[str], *, allowed_labels: Sequence[str] = SENTIMENT_CLASSES
) -> str:
    """Monta um prompt com vários tweets numerados (ids ``1..N``).

    As instruções são as do template, sem a linha do marcador ``{{TEXTO}}``.

    Parameters
    ----------
    prompt_template : str
        Template do prompt (marcador ``{{TEXTO}}``).
    texts : Sequence[str]
        Textos sanitizados, um por tweet (quebras de linha viram espaços).
    allowed_labels : Sequence[str], optional
        Classes aceitas, by default :data:`constants.labels.SENTIMENT_CLASSES`.

    Returns
    -------
    str
        Prompt multi-tweet.

    Examples
    --------
    >>> prompt = build_multi_tweet_prompt('Instrução\\nTweet: "{{TEXTO}}"', ["a", "b"])
    >>> "1. a" in prompt and "2. b" in prompt and "{{TEXTO}}" not in prompt
    True
    """
    instructions = "\n".join(
        line for line in prompt_template.splitlines() if TEXT_PLACEHOLDER not in line
    ).rstrip()
    numbered = "\n".join(
        f"{index}. {' '.join(text.split())}" for index, text in enumerate(texts, 1)
    )
    return instructions + _MULTI_TWEET_SUFFIX.format(
        example=allowed_labels[0], labels=", ".join(allowed_labels), tweets=numbered
    )


def parse_multi_tweet_response(
    raw_response: str, n_items: int, *, allowed_labels: Sequence[str] = SENTIMENT_CLASSES
) -> list[tuple[str, float] | None]:
    """Interpreta a lista JSON de uma resposta multi-tweet.

    Parameters
    ----------
    raw_response : str
        Resposta bruta do LLM.
    n_items : int
        Quantidade de tweets enviados (ids esperados ``1..n_items``).
    allowed_labels : Sequence[str], optional
        Classes aceitas, by default :data:`constants.labels.SENTIMENT_CLASSES`.

    Returns
    -------
    list[tuple[str, float] | None]
        Uma posição por tweet, na ordem de envio; ``None`` para id ausente, duplicado
        ou com rótulo inválido. A confiança é lida como em
        :func:`labeling.llm_response.parse_llm_label_response` (1.0 se ausente).

    Examples
    --------
    >>> parse_multi_tweet_response('[{"id":1,"label":"positivo"},{"id":3,"label":"x"}]', 3)
    [('positivo', 1.0), None, None]
    """
    results: list[tuple[str, float] | None] = [None] * n_items
    seen: set[int] = set()
    for item in _extract_json_items(raw_response):
        item_id = _read_valid_item_id(item, n_items)
        if item_id is None:
            continue
        if item_id in seen:
            results[item_id - 1] = None  # id duplicado é ambíguo: reclassificar
            continue
        seen.add(item_id)
        results[item_id - 1] = parse_llm_label_response(
            json.dumps(item), allowed_labels=allowed_labels
        )
    return results


def _extract_json_items(raw_response: str) -> list[object]:
    """Extrai a lista JSON da resposta bruta (vazia se ausente ou inválida)."""
    text = raw_response.rsplit("</think>", maxsplit=1)[-1].strip()
    match = _JSON_ARRAY_PATTERN.search(text)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return items if isinstance(items, list) else []


def _read_valid_item_id(item: object, n_items: int) -> int | None:
    """Devolve o ``id`` do item se for inteiro em ``1..n_items``; senão ``None``."""
    if not isinstance(item, dict):
        return None
    item_id = item.get("id")
    if not isinstance(item_id, int) or isinstance(item_id, bool):
        return None
    return item_id if 1 <= item_id <= n_items else None
