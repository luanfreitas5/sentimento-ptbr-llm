"""Anotação de texto via LLM: verificação de presença/ausência de conceitos.

Dado um par (texto, conceito em linguagem natural), pergunta a um LLM se o
conceito está presente no texto ("Sim"/"Não"), com cache em disco por par
(conceito, texto) e execução paralela via ``ThreadPoolExecutor``. É a
primitiva usada tanto para pontuar a fidelidade de uma interpretação de
neurônio (``interpret_neurons.py``) quanto para avaliar hipóteses em um
conjunto de dados real (``evaluation.py``, ``quickstart.evaluate_hypotheses``).
"""

import concurrent.futures
import json
import logging
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from config.paths import PROJECT_ROOT
from hypothesaes.llm_api import generate_completion, normalize_llm_kwargs
from hypothesaes.utils import load_prompt_template, truncate_text

ANNOTATION_CACHE_DIR: Path = PROJECT_ROOT / "models" / "artifacts" / "hypothesaes_annotation_cache"
DEFAULT_N_WORKERS = 30

logger = logging.getLogger(__name__)


def load_annotation_cache(cache_path: Path | None) -> dict[str, int]:
    """Carrega anotações previamente cacheadas de um arquivo JSON.

    Parameters
    ----------
    cache_path : Path | None
        Caminho do arquivo de cache; se ``None`` ou inexistente, retorna
        vazio.

    Returns
    -------
    dict[str, int]
        Mapa de chave de cache (ver :func:`generate_cache_key`) para
        anotação (0 ou 1).
    """
    if cache_path is None or not cache_path.is_file():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Falha ao ler o cache '%s'; iniciando cache vazio.", cache_path)
        cache_path.unlink()
        return {}


def save_annotation_cache(cache_path: Path, cache: dict[str, int]) -> None:
    """Salva o dicionário de anotações em um arquivo JSON.

    Parameters
    ----------
    cache_path : Path
        Caminho do arquivo de destino.
    cache : dict[str, int]
        Anotações a serem persistidas.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache), encoding="utf-8")


def generate_cache_key(concept: str, text: str) -> str:
    """Gera a chave de cache para um par (conceito, texto).

    Usa apenas os 100 primeiros e 100 últimos caracteres do texto, o
    suficiente para diferenciar textos na prática sem inflar as chaves de
    cache com textos longos.

    Parameters
    ----------
    concept : str
        Hipótese/conceito em linguagem natural.
    text : str
        Texto anotado.

    Returns
    -------
    str
        Chave de cache determinística para o par.

    Examples
    --------
    >>> generate_cache_key("menciona atendimento", "ótimo atendimento")
    'menciona atendimento|||ótimo atendimento[...]ótimo atendimento'
    """
    return f"{concept}|||{text[:100]}[...]{text[-100:]}"


def _store_annotation(
    results: dict[str, dict[str, int]],
    concept: str,
    text: str,
    annotation: int,
    cache: dict[str, int] | None = None,
) -> None:
    """Insere uma anotação em ``results`` e, opcionalmente, no cache."""
    results.setdefault(concept, {})[text] = annotation
    if cache is not None:
        cache[generate_cache_key(concept, text)] = annotation


def parse_completion(completion: str) -> int | None:
    """Interpreta uma completion em texto livre como anotação binária.

    Parameters
    ----------
    completion : str
        Resposta do LLM (esperado: iniciar com "yes"/"no").

    Returns
    -------
    int | None
        1 se a resposta começa com "yes", 0 se começa com "no", ``None``
        caso contrário.

    Examples
    --------
    >>> parse_completion("yes. o texto menciona o atendimento.")
    1
    >>> parse_completion("no, não menciona.")
    0
    """
    if "</think>" in completion:
        completion = completion.split("</think>")[1].strip()
    if completion.startswith("yes"):
        return 1
    if completion.startswith("no"):
        return 0
    return None


_ANSWER_TO_ANNOTATION: dict[str, int] = {"yes": 1, "no": 0}


def parse_completion_json(completion: str) -> int | None:
    """Interpreta uma completion em formato JSON (``{"answer": "yes"|"no", ...}``).

    Parameters
    ----------
    completion : str
        Resposta do LLM, contendo um objeto JSON com a chave ``"answer"``.

    Returns
    -------
    int | None
        1 para "yes", 0 para "no", ``None`` se o JSON não puder ser
        interpretado ou a resposta for ambígua.
    """
    if "</think>" in completion:
        completion = completion.split("</think>")[1].strip()
    match = re.search(r"\{.*\}", completion, re.DOTALL)
    if not match:
        return None
    try:
        cleaned = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', "", match.group(0))
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    answer = parsed.get("answer", "").strip().lower()
    return _ANSWER_TO_ANNOTATION.get(answer)


def _build_annotation_request_kwargs(
    completion_kwargs: dict[str, Any],
    model: str,
    temperature: float | None,
    verbosity: str | None,
    reasoning_effort: str | None,
    timeout: float | None,
    max_output_tokens: int | None,
) -> dict[str, Any]:
    """Monta os kwargs de uma tentativa de anotação, aplicando padrões e overrides."""
    request_kwargs = normalize_llm_kwargs(
        completion_kwargs,
        default_verbosity=verbosity,
        default_reasoning_effort=reasoning_effort,
        default_timeout=timeout,
        default_max_output_tokens=max_output_tokens,
    )
    request_kwargs.setdefault("model", model)
    if temperature is not None:
        request_kwargs.setdefault("temperature", temperature)
    return request_kwargs


def _try_annotate_once(
    prompt: str,
    system_prompt: str | None,
    request_kwargs: dict[str, Any],
    parse_fn: Callable[[str], int | None],
) -> tuple[int | None, float]:
    """Executa uma única chamada ao LLM e tenta interpretar a resposta."""
    start_time = time.time()
    response_text = (
        generate_completion(prompt=prompt, system_prompt=system_prompt, **request_kwargs)
        .strip()
        .lower()
    )
    elapsed = time.time() - start_time
    return parse_fn(response_text), elapsed


def _run_annotation_attempts(
    prompt: str,
    system_prompt: str | None,
    parse_fn: Callable[[str], int | None],
    max_retries: int,
    build_request_kwargs: Callable[[], dict[str, Any]],
) -> tuple[int | None, float]:
    """Tenta anotar até ``max_retries`` vezes, retornando a primeira anotação válida."""
    total_api_time = 0.0
    for attempt in range(max_retries):
        try:
            annotation, elapsed = _try_annotate_once(
                prompt, system_prompt, build_request_kwargs(), parse_fn
            )
            total_api_time += elapsed
            if annotation is not None:
                return annotation, total_api_time
        except Exception:
            if attempt == max_retries - 1:
                logger.exception("Falha ao anotar texto após %d tentativas.", max_retries)

    return None, total_api_time


def annotate_single_text(
    text: str,
    concept: str,
    annotate_prompt_name: str = "annotate",
    model: str = "gpt-5-mini",
    parse_fn: Callable[[str], int | None] = parse_completion,
    system_prompt_name: str | None = None,
    max_words_per_example: int | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    max_retries: int = 3,
    timeout: float | None = None,
    reasoning_effort: str | None = None,
    verbosity: str | None = None,
    **completion_kwargs: Any,
) -> tuple[int | None, float]:
    """Anota um único texto com um conceito, via LLM.

    Parameters
    ----------
    text : str
        Texto a anotar.
    concept : str
        Conceito/hipótese em linguagem natural.
    annotate_prompt_name : str, optional
        Nome do template de prompt em ``prompts/``, by default "annotate".
    model : str, optional
        Modelo usado na anotação, by default "gpt-5-mini".
    parse_fn : Callable[[str], int | None], optional
        Função de interpretação da resposta, by default :func:`parse_completion`.
    system_prompt_name : str | None, optional
        Nome do template de mensagem de sistema, by default None.
    max_words_per_example : int | None, optional
        Trunca o texto para no máximo esta quantidade de palavras antes de
        anotar, by default None.
    temperature, max_output_tokens, timeout, reasoning_effort, verbosity :
        Repassados a :func:`hypothesaes.llm_api.generate_completion`.
    max_retries : int, optional
        Número máximo de tentativas em caso de falha, by default 3.
    **completion_kwargs : Any
        Argumentos adicionais repassados à API de completions.

    Returns
    -------
    tuple[int | None, float]
        Par (anotação, tempo total de chamadas à API em segundos). A
        anotação é 1 (presente), 0 (ausente) ou ``None`` (falha).
    """
    if max_words_per_example:
        text = truncate_text(text, max_words_per_example)

    annotate_prompt = load_prompt_template(annotate_prompt_name)
    system_prompt = (
        load_prompt_template(system_prompt_name) if system_prompt_name is not None else None
    )
    prompt = annotate_prompt.format(hypothesis=concept, text=text)

    def build_request_kwargs() -> dict[str, Any]:
        return _build_annotation_request_kwargs(
            completion_kwargs,
            model,
            temperature,
            verbosity,
            reasoning_effort,
            timeout,
            max_output_tokens,
        )

    return _run_annotation_attempts(
        prompt, system_prompt, parse_fn, max_retries, build_request_kwargs
    )


def _should_checkpoint(
    completed: int, checkpoint_every: int, cache_path: Path | None, cache: dict[str, int] | None
) -> bool:
    """Indica se o cache deve ser persistido neste ponto da execução paralela."""
    return completed % checkpoint_every == 0 and cache_path is not None and cache is not None


def _retry_failed_tasks(
    retry_tasks: list[tuple[str, str]],
    model: str,
    results: dict[str, dict[str, int]],
    cache: dict[str, int] | None,
    **annotation_kwargs: Any,
) -> None:
    """Reprocessa sequencialmente as tarefas que falharam na execução paralela."""
    logger.info("Reprocessando %d tarefa(s) que falharam...", len(retry_tasks))
    for text, concept in retry_tasks:
        try:
            annotation, _ = annotate_single_text(
                text=text, concept=concept, model=model, **annotation_kwargs
            )
            if annotation is not None:
                _store_annotation(results, concept, text, annotation, cache)
        except Exception:
            logger.exception("Falha ao reprocessar anotação para o conceito '%s'.", concept)


def _annotate_tasks_in_parallel(
    tasks: list[tuple[str, str]],
    model: str,
    n_workers: int,
    results: dict[str, dict[str, int]],
    cache: dict[str, int] | None = None,
    cache_path: Path | None = None,
    checkpoint_every: int = 1000,
    progress_desc: str = "Anotando",
    show_progress: bool = True,
    **annotation_kwargs: Any,
) -> None:
    """Anota uma lista de tarefas (texto, conceito) em paralelo, com checkpoint periódico."""
    retry_tasks: list[tuple[str, str]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_task = {
            executor.submit(
                annotate_single_text, text=text, concept=concept, model=model, **annotation_kwargs
            ): (text, concept)
            for text, concept in tasks
        }

        iterator = tqdm(
            concurrent.futures.as_completed(future_to_task),
            total=len(tasks),
            desc=progress_desc,
            disable=not show_progress,
        )

        completed = 0
        for future in iterator:
            text, concept = future_to_task[future]
            try:
                annotation, _ = future.result()
                if annotation is not None:
                    _store_annotation(results, concept, text, annotation, cache)
                completed += 1
                if _should_checkpoint(completed, checkpoint_every, cache_path, cache):
                    save_annotation_cache(cache_path, cache)
            except Exception:
                retry_tasks.append((text, concept))
                logger.exception("Falha ao anotar texto para o conceito '%s'.", concept)

    if retry_tasks:
        _retry_failed_tasks(retry_tasks, model, results, cache, **annotation_kwargs)


def _split_cached_and_uncached_tasks(
    tasks: list[tuple[str, str]],
    cache: dict[str, int],
    results: dict[str, dict[str, int]],
    use_cache_only: bool,
    uncached_value: int,
) -> list[tuple[str, str]]:
    """Separa tarefas já cacheadas das que precisam ser anotadas, populando ``results``."""
    uncached_tasks: list[tuple[str, str]] = []
    for text, concept in tasks:
        results.setdefault(concept, {})
        cache_key = generate_cache_key(concept, text)
        if cache_key in cache:
            results[concept][text] = cache[cache_key]
        elif use_cache_only:
            results[concept][text] = uncached_value
            uncached_tasks.append((text, concept))
        else:
            uncached_tasks.append((text, concept))
    return uncached_tasks


def annotate_tasks(
    tasks: list[tuple[str, str]],
    model: str = "gpt-5-mini",
    cache_path: Path | None = None,
    n_workers: int = DEFAULT_N_WORKERS,
    show_progress: bool = True,
    progress_desc: str = "Anotando",
    use_cache_only: bool = False,
    uncached_value: int = 0,
    **annotation_kwargs: Any,
) -> dict[str, dict[str, int]]:
    """Anota uma lista de tarefas (texto, conceito), reaproveitando um cache em disco.

    Parameters
    ----------
    tasks : list[tuple[str, str]]
        Lista de pares (texto, conceito) a anotar.
    model : str, optional
        Modelo usado na anotação, by default "gpt-5-mini".
    cache_path : Path | None, optional
        Caminho do arquivo de cache, by default None.
    n_workers : int, optional
        Número de threads paralelas, by default :data:`DEFAULT_N_WORKERS`.
    show_progress : bool, optional
        Se exibe barra de progresso, by default True.
    progress_desc : str, optional
        Descrição exibida na barra de progresso, by default "Anotando".
    use_cache_only : bool, optional
        Se ``True``, não faz novas chamadas ao LLM: itens fora do cache
        recebem ``uncached_value``, by default False.
    uncached_value : int, optional
        Valor atribuído a itens fora do cache quando ``use_cache_only=True``,
        by default 0.
    **annotation_kwargs : Any
        Argumentos adicionais repassados a :func:`annotate_single_text`.

    Returns
    -------
    dict[str, dict[str, int]]
        Mapa aninhado ``{conceito: {texto: anotação}}``.
    """
    cache = load_annotation_cache(cache_path) if cache_path else {}
    results: dict[str, dict[str, int]] = {}
    uncached_tasks = _split_cached_and_uncached_tasks(
        tasks, cache, results, use_cache_only, uncached_value
    )

    if use_cache_only:
        logger.info(
            "%d item(ns) recuperado(s) do cache; %d item(ns) sem cache mapeado(s) para %d.",
            len(tasks) - len(uncached_tasks),
            len(uncached_tasks),
            uncached_value,
        )
        return results

    logger.info(
        "%d item(ns) recuperado(s) do cache; anotando %d item(ns) sem cache.",
        len(tasks) - len(uncached_tasks),
        len(uncached_tasks),
    )

    if uncached_tasks:
        _annotate_tasks_in_parallel(
            tasks=uncached_tasks,
            model=model,
            n_workers=n_workers,
            cache=cache,
            cache_path=cache_path,
            results=results,
            show_progress=show_progress,
            progress_desc=progress_desc,
            **annotation_kwargs,
        )

    if cache_path:
        save_annotation_cache(cache_path, cache)

    return results


def annotate_texts_with_concepts(
    texts: list[str],
    concepts: list[str],
    model: str = "gpt-5-mini",
    cache_name: str | None = None,
    progress_desc: str = "Anotando",
    show_progress: bool = True,
    **annotation_kwargs: Any,
) -> dict[str, np.ndarray]:
    """Anota todos os textos com todos os conceitos (produto cartesiano).

    Parameters
    ----------
    texts : list[str]
        Textos a anotar.
    concepts : list[str]
        Conceitos/hipóteses a verificar em cada texto.
    model : str, optional
        Modelo usado na anotação, by default "gpt-5-mini".
    cache_name : str | None, optional
        Nome do cache em :data:`ANNOTATION_CACHE_DIR`, by default None.
    progress_desc : str, optional
        Descrição da barra de progresso, by default "Anotando".
    show_progress : bool, optional
        Se exibe barra de progresso, by default True.
    **annotation_kwargs : Any
        Argumentos adicionais repassados a :func:`annotate_tasks`.

    Returns
    -------
    dict[str, np.ndarray]
        Mapa de cada conceito para um vetor de anotações (0/1), na mesma
        ordem de ``texts``.
    """
    tasks = [(text, concept) for text in texts for concept in concepts]
    cache_path = (
        (ANNOTATION_CACHE_DIR / f"{cache_name}_hypothesis-eval.json") if cache_name else None
    )

    results = annotate_tasks(
        tasks=tasks,
        model=model,
        cache_path=cache_path,
        n_workers=annotation_kwargs.pop("n_workers", DEFAULT_N_WORKERS),
        show_progress=show_progress,
        progress_desc=progress_desc,
        **annotation_kwargs,
    )

    return _collect_concept_annotations(results, texts, concepts)


def _collect_concept_annotations(
    results: dict[str, dict[str, int]], texts: list[str], concepts: list[str]
) -> dict[str, np.ndarray]:
    """Converte os resultados aninhados de anotação em um vetor de anotações por conceito."""
    return {concept: np.array([results[concept][text] for text in texts]) for concept in concepts}
