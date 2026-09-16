"""Re-rotulagem via LLM de tweets com confiança de rótulo abaixo de um limiar.

Implementa a etapa opcional ``llm_relabeling`` de ``configs/labeling.yaml``:
para cada tweet cujo ``confidence_score`` (rótulo de consenso da cascata, ver
``labeling.consensus.aggregate_by_weighted_majority_vote``) esteja abaixo de
``score_threshold``, reenvia o texto a um LLM
(:func:`hypothesaes.llm_api.generate_completion`, provedor escolhido via
``provider`` — ``configs/llm.yaml -> active_provider``: ``"openai"``,
endpoint/credenciais em ``OPENAI_BASE_URL``/``OPENAI_KEY`` — ``.env``, modelo
padrão ``UnB-Llama-3.3-70B-Instruct``; ou ``"ollama"``, servidor local em
``ollama_base_url`` — ``configs/llm.yaml -> backends.ollama.base_url``) com
um prompt carregado de ``prompts/`` (:func:`hypothesaes.utils.load_prompt_template`,
``configs/labeling.yaml -> llm_relabeling.prompt_name``).

Falhas de chamada/parsing preservam o rótulo e a confiança originais da
cascata (fail-safe): uma re-rotulagem malsucedida nunca deve degradar
silenciosamente um rótulo já calculado.
"""

import concurrent.futures
import json
import logging
import re
import time
from collections.abc import Sequence
from typing import Any

import polars as pl
from tqdm.auto import tqdm

from constants.labels import SENTIMENT_CLASSES
from exceptions.data import DataValidationError
from hypothesaes.llm_api import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    LLMProvider,
    generate_completion,
)
from hypothesaes.utils import load_prompt_template
from utils.validation import validate_not_empty_collection

logger = logging.getLogger(__name__)

DEFAULT_RELABEL_MODEL = "UnB-Llama-3.3-70B-Instruct"
DEFAULT_RELABEL_MODEL_OLLAMA = DEFAULT_OLLAMA_MODEL
_TEXT_PLACEHOLDER = "{{TEXTO}}"
_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def parse_relabel_response(
    raw_response: str, *, allowed_labels: Sequence[str] = SENTIMENT_CLASSES
) -> tuple[str, float] | None:
    """Interpreta a resposta em JSON do LLM de re-rotulagem (``{"label": ..., ...}``).

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
    >>> parse_relabel_response('{"label": "positivo", "confidence": 0.9}')
    ('positivo', 0.9)
    >>> parse_relabel_response(
    ...     '{"label":"negativo","probs":{"positivo":0.1,"negativo":0.8,"neutro":0.1}}'
    ... )
    ('negativo', 0.8)
    >>> parse_relabel_response("resposta sem json") is None
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


def _relabel_single_text(
    text: str,
    prompt_template: str,
    *,
    model: str,
    temperature: float,
    max_retries: int,
    allowed_labels: Sequence[str],
    provider: LLMProvider = "openai",
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
    request_interval_seconds: float = 0.0,
) -> tuple[str, float] | None:
    """Reenvia um único texto ao LLM e tenta interpretar a resposta, com retentativas.

    Parameters
    ----------
    text : str
        Texto do tweet a re-rotular.
    prompt_template : str
        Template do prompt (ver :data:`_TEXT_PLACEHOLDER`).
    model : str
        Modelo a usar em :func:`hypothesaes.llm_api.generate_completion`.
    temperature : float
        Temperatura de amostragem repassada à chamada.
    max_retries : int
        Número máximo de tentativas até obter uma resposta interpretável.
    allowed_labels : Sequence[str]
        Classes de sentimento aceitas.
    provider : {"openai", "ollama"}, optional
        Provedor de LLM (``configs/llm.yaml -> active_provider``), by
        default "openai".
    ollama_base_url : str, optional
        URL do servidor Ollama local, usada apenas quando
        ``provider="ollama"``, by default
        :data:`hypothesaes.llm_api.DEFAULT_OLLAMA_BASE_URL`.
    request_interval_seconds : float, optional
        Pausa (``time.sleep``) antes de cada chamada/tentativa ao LLM, para
        reduzir a taxa de requisições e evitar bloqueios por limite de taxa
        (HTTP 429) da API OpenAI-compatível, by default 0.0 (sem pausa).

    Returns
    -------
    tuple[str, float] | None
        Par ``(rótulo, confiança)`` interpretado, ou ``None`` se todas as
        tentativas falharem (chamada ou parsing).
    """
    prompt = prompt_template.replace(_TEXT_PLACEHOLDER, text)
    for attempt in range(max_retries):
        if request_interval_seconds > 0:
            time.sleep(request_interval_seconds)
        try:
            raw_response = generate_completion(
                prompt=prompt,
                model=model,
                temperature=temperature,
                provider=provider,
                ollama_base_url=ollama_base_url,
            )
        except Exception:
            logger.exception(
                "Falha na chamada ao LLM de re-rotulagem (tentativa %d/%d).",
                attempt + 1,
                max_retries,
            )
            continue
        parsed = parse_relabel_response(raw_response, allowed_labels=allowed_labels)
        if parsed is not None:
            return parsed
    return None


def _validate_relabel_inputs(
    labeled_corpus: pl.DataFrame,
    *,
    id_column: str,
    text_column: str,
    label_column: str,
    confidence_column: str,
) -> None:
    """Valida que ``labeled_corpus`` não está vazio e contém as colunas exigidas."""
    validate_not_empty_collection(labeled_corpus, collection_name="labeled_corpus")
    required_columns = {id_column, text_column, label_column, confidence_column}
    missing_columns = required_columns - set(labeled_corpus.columns)
    if missing_columns:
        raise DataValidationError(
            schema_name="labeled_corpus",
            detail=f"coluna(s) ausente(s) para re-rotulagem via LLM: {sorted(missing_columns)}",
        )


def _run_relabel_workers(
    texts: list[str],
    prompt_template: str,
    *,
    model: str,
    temperature: float,
    max_retries: int,
    allowed_labels: Sequence[str],
    n_workers: int,
    show_progress: bool,
    provider: LLMProvider = "openai",
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
    request_interval_seconds: float = 0.0,
) -> list[tuple[str, float] | None]:
    """Dispara a re-rotulagem de ``texts`` em paralelo e coleta os resultados na ordem original."""
    results: list[tuple[str, float] | None] = [None] * len(texts)
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_index = {
            executor.submit(
                _relabel_single_text,
                text,
                prompt_template,
                model=model,
                temperature=temperature,
                max_retries=max_retries,
                allowed_labels=allowed_labels,
                provider=provider,
                ollama_base_url=ollama_base_url,
                request_interval_seconds=request_interval_seconds,
            ): index
            for index, text in enumerate(texts)
        }
        iterator = tqdm(
            concurrent.futures.as_completed(future_to_index),
            total=len(texts),
            desc="Re-rotulando via LLM (baixa confiança)",
            disable=not show_progress,
        )
        for future in iterator:
            results[future_to_index[future]] = future.result()
    return results


def _merge_relabel_results(
    labeled_corpus: pl.DataFrame,
    candidate_rows: pl.DataFrame,
    results: list[tuple[str, float] | None],
    *,
    id_column: str,
    label_column: str,
    confidence_column: str,
) -> pl.DataFrame:
    """Combina os resultados da re-rotulagem a `labeled_corpus`, preservando falhas (fail-safe).

    Além de atualizar as colunas de trabalho ``label_column``/
    ``confidence_column`` (usadas pelas etapas seguintes — validação humana e
    modelagem), grava o rótulo/confiança do LLM em colunas próprias
    (``{label_column}_llm_relabel``/``{confidence_column}_llm_relabel``),
    nulas para amostras não candidatas e para candidatas cuja re-rotulagem
    falhou — nunca sobrescrevendo ``sentiment_label_huggingface``/
    ``confidence_score_huggingface`` (ver
    ``src/labeling/consensus.py``'s ``merge_consensus_into_corpus``).
    """
    original_labels = candidate_rows[label_column].to_list()
    original_confidences = candidate_rows[confidence_column].to_list()

    relabel_labels = [result[0] if result is not None else None for result in results]
    relabel_confidences = [result[1] if result is not None else None for result in results]

    new_labels = [
        relabel_label if relabel_label is not None else original_label
        for relabel_label, original_label in zip(relabel_labels, original_labels, strict=True)
    ]
    new_confidences = [
        relabel_confidence if relabel_confidence is not None else original_confidence
        for relabel_confidence, original_confidence in zip(
            relabel_confidences, original_confidences, strict=True
        )
    ]

    llm_relabel_label_column = f"{label_column}_llm_relabel"
    llm_relabel_confidence_column = f"{confidence_column}_llm_relabel"

    relabel_updates = pl.DataFrame(
        {
            id_column: candidate_rows[id_column],
            f"__relabel_{label_column}": new_labels,
            f"__relabel_{confidence_column}": new_confidences,
            llm_relabel_label_column: relabel_labels,
            llm_relabel_confidence_column: relabel_confidences,
        }
    )

    return (
        labeled_corpus.join(relabel_updates, on=id_column, how="left")
        .with_columns(
            pl.coalesce([f"__relabel_{label_column}", label_column]).alias(label_column),
            pl.coalesce([f"__relabel_{confidence_column}", confidence_column]).alias(
                confidence_column
            ),
        )
        .drop([f"__relabel_{label_column}", f"__relabel_{confidence_column}"])
    )


def relabel_low_confidence_samples(
    labeled_corpus: pl.DataFrame,
    *,
    id_column: str = "id",
    text_column: str = "text_normalized",
    label_column: str = "sentiment_label",
    confidence_column: str = "confidence_score",
    score_threshold: float,
    prompt_name: str,
    model: str | None = None,
    temperature: float = 0.0,
    max_retries: int = 3,
    n_workers: int = 8,
    allowed_labels: Sequence[str] = SENTIMENT_CLASSES,
    show_progress: bool = True,
    provider: LLMProvider = "openai",
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
    request_interval_seconds: float = 0.0,
) -> pl.DataFrame:
    """Re-rotula, via LLM, as amostras de ``labeled_corpus`` com confiança abaixo do limiar.

    Amostras cuja re-rotulagem falhar (erro de chamada ou resposta não
    interpretável, mesmo após ``max_retries`` tentativas) preservam o
    rótulo/confiança originais da cascata — nunca são substituídas por um
    valor padrão arbitrário.

    Parameters
    ----------
    labeled_corpus : pl.DataFrame
        Corpus rotulado, contendo ao menos ``id_column``, ``text_column``,
        ``label_column`` e ``confidence_column`` (saída de
        ``labeling.consensus.merge_consensus_into_corpus``).
    id_column : str, optional
        Coluna identificadora de cada amostra, by default "id".
    text_column : str, optional
        Coluna de texto reenviada ao LLM, by default "text_normalized".
    label_column : str, optional
        Coluna de rótulo de sentimento a atualizar, by default
        "sentiment_label".
    confidence_column : str, optional
        Coluna de confiança usada para selecionar candidatos e atualizada
        com a confiança relatada pelo LLM, by default "confidence_score".
    score_threshold : float
        Amostras com ``confidence_column < score_threshold`` são
        candidatas à re-rotulagem (``configs/labeling.yaml ->
        llm_relabeling.score_threshold``).
    prompt_name : str
        Nome do template de prompt em ``prompts/`` (sem a extensão
        ``.txt``), carregado via
        :func:`hypothesaes.utils.load_prompt_template``
        (``configs/labeling.yaml -> llm_relabeling.prompt_name``).
    model : str | None, optional
        Modelo LLM usado na re-rotulagem, by default ``None`` — resolvido
        conforme ``provider`` para :data:`DEFAULT_RELABEL_MODEL`
        (``"UnB-Llama-3.3-70B-Instruct"``, ``provider="openai"``) ou
        :data:`DEFAULT_RELABEL_MODEL_OLLAMA` (``provider="ollama"``).
    temperature : float, optional
        Temperatura de amostragem, by default 0.0 (determinístico).
    max_retries : int, optional
        Novas tentativas por amostra até obter uma resposta interpretável,
        by default 3.
    n_workers : int, optional
        Threads paralelas para as chamadas ao LLM, by default 8.
    allowed_labels : Sequence[str], optional
        Classes de sentimento aceitas, by default
        :data:`constants.labels.SENTIMENT_CLASSES`.
    show_progress : bool, optional
        Se exibe uma barra de progresso no console, by default True.
    provider : {"openai", "ollama"}, optional
        Provedor de LLM (``configs/llm.yaml -> active_provider``), by
        default "openai".
    ollama_base_url : str, optional
        URL do servidor Ollama local, usada apenas quando
        ``provider="ollama"``, by default
        :data:`hypothesaes.llm_api.DEFAULT_OLLAMA_BASE_URL`
        (``configs/llm.yaml -> backends.ollama.base_url``).
    request_interval_seconds : float, optional
        Pausa (``time.sleep``) antes de cada chamada/tentativa ao LLM (por
        worker), para reduzir a taxa de requisições à API OpenAI-compatível
        e evitar bloqueios por limite de taxa (HTTP 429), by default 0.0
        (sem pausa) (``configs/labeling.yaml ->
        llm_relabeling.request_interval_seconds``).

    Returns
    -------
    pl.DataFrame
        ``labeled_corpus`` com ``label_column``/``confidence_column``
        atualizadas para as amostras re-rotuladas com sucesso, acrescido das
        colunas ``{label_column}_llm_relabel``/``{confidence_column}_llm_relabel``
        — o rótulo/confiança bruto do LLM, nulos para amostras não candidatas
        ou cuja re-rotulagem falhou.

    Raises
    ------
    EmptyDatasetError
        Se ``labeled_corpus`` estiver vazio.
    DataValidationError
        Se alguma das colunas exigidas estiver ausente.

    Examples
    --------
    >>> relabel_low_confidence_samples(
    ...     labeled_corpus, score_threshold=0.5, prompt_name="labeling_1_..."
    ... )  # doctest: +SKIP
    """
    _validate_relabel_inputs(
        labeled_corpus,
        id_column=id_column,
        text_column=text_column,
        label_column=label_column,
        confidence_column=confidence_column,
    )

    resolved_model = model or (
        DEFAULT_RELABEL_MODEL if provider == "openai" else DEFAULT_RELABEL_MODEL_OLLAMA
    )

    candidate_rows = labeled_corpus.filter(pl.col(confidence_column) < score_threshold)
    if candidate_rows.height == 0:
        logger.info(
            "Nenhuma amostra com confidence_score < %.2f; re-rotulagem via LLM ignorada.",
            score_threshold,
        )
        return labeled_corpus

    prompt_template = load_prompt_template(prompt_name)
    texts = candidate_rows[text_column].to_list()

    results = _run_relabel_workers(
        texts,
        prompt_template,
        model=resolved_model,
        temperature=temperature,
        max_retries=max_retries,
        allowed_labels=allowed_labels,
        n_workers=n_workers,
        show_progress=show_progress,
        provider=provider,
        ollama_base_url=ollama_base_url,
        request_interval_seconds=request_interval_seconds,
    )

    n_success = sum(result is not None for result in results)
    logger.info(
        "Re-rotulagem via LLM concluída: %d/%d amostra(s) reinterpretada(s) com sucesso "
        "(provider='%s', modelo='%s', limiar=%.2f).",
        n_success,
        len(texts),
        provider,
        resolved_model,
        score_threshold,
    )

    return _merge_relabel_results(
        labeled_corpus,
        candidate_rows,
        results,
        id_column=id_column,
        label_column=label_column,
        confidence_column=confidence_column,
    )
