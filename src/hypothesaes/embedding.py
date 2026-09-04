"""Cálculo e cache de embeddings de texto (OpenAI ou modelo local).

Ponto de entrada de ``hypothesaes`` para transformar tweets em vetores
densos antes do treinamento do Sparse Autoencoder (``sae.py``). Suporta a
API de embeddings da OpenAI (:func:`extract_openai_embeddings`) e modelos
locais ``sentence-transformers`` (:func:`extract_local_embeddings`, ex.:
BERTimbau/Sentence-BERT em português). Os embeddings calculados são
cacheados em disco por texto (chave = string do tweet), evitando
recomputar embeddings entre execuções.

``openai``, ``tiktoken`` e ``sentence-transformers`` não estão nas
dependências base do projeto: os imports ocorrem de forma tardia, para que
o módulo permaneça importável sem eles. Instale com
``uv add openai tiktoken sentence-transformers`` conforme o backend usado.
"""

import concurrent.futures
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from config.paths import PROJECT_ROOT
from exceptions.model import ModelError
from hypothesaes.utils import filter_invalid_texts

logger = logging.getLogger(__name__)

# Diretório de cache de embeddings; pode ser sobrescrito por EMB_CACHE_DIR.
EMBEDDING_CACHE_DIR: Path = Path(
    os.getenv("EMB_CACHE_DIR")
    or (PROJECT_ROOT / "models" / "artifacts" / "hypothesaes_embedding_cache")
)


def _truncate_batch_for_embedding(batch: list[str], encoding: Any, max_tokens: int) -> list[str]:
    """Trunca cada texto do lote para no máximo ``max_tokens`` tokens do encoding informado."""
    truncated_batch = []
    for text in batch:
        tokens = encoding.encode(text.strip())
        truncated_batch.append(
            encoding.decode(tokens[:max_tokens]) if len(tokens) > max_tokens else text
        )
    return truncated_batch


def _request_embeddings_once(
    client: Any, truncated_batch: list[str], model: str, timeout: float | None
) -> list[list[float]]:
    """Envia uma única requisição de embeddings à API, retornando os vetores na ordem do lote."""
    request_kwargs: dict[str, Any] = {"input": truncated_batch, "model": model}
    if timeout is not None:
        request_kwargs["timeout"] = timeout
    response = client.embeddings.create(**request_kwargs)
    return [item.embedding for item in response.data]


def _request_openai_embeddings_batch(
    batch: list[str],
    model: str,
    client: Any,
    max_tokens: int = 8192,
    max_retries: int = 3,
    backoff_factor: float = 3.0,
    timeout: float | None = None,
) -> list[list[float]]:
    """Solicita embeddings de um lote de textos à API da OpenAI, truncando por tokens."""
    import openai  # type: ignore[reportMissingImports]
    import tiktoken  # type: ignore[reportMissingImports]

    encoding = tiktoken.get_encoding("cl100k_base")
    truncated_batch = _truncate_batch_for_embedding(batch, encoding, max_tokens)

    for attempt in range(max_retries):
        try:
            return _request_embeddings_once(client, truncated_batch, model, timeout)

        except (openai.RateLimitError, openai.APITimeoutError) as exception:
            if attempt == max_retries - 1:
                raise

            base_wait = timeout if timeout is not None else 1.0
            wait_time = base_wait * (backoff_factor**attempt)
            if attempt > 0:
                logger.warning(
                    "Erro na API de embeddings (%s); nova tentativa em %.1fs... (%d/%d)",
                    exception,
                    wait_time,
                    attempt + 1,
                    max_retries,
                )
            time.sleep(wait_time)

    return []


def load_embedding_cache(cache_name: str | None) -> dict[str, np.ndarray]:
    """Carrega embeddings previamente cacheados a partir de arquivos fragmentados (chunks).

    Parameters
    ----------
    cache_name : str | None
        Nome do cache (subdiretório de :data:`EMBEDDING_CACHE_DIR`). Se
        ``None``/vazio, nenhum cache é usado.

    Returns
    -------
    dict[str, np.ndarray]
        Mapa de texto para embedding.
    """
    if not cache_name:
        return {}

    cache_dir = EMBEDDING_CACHE_DIR / cache_name
    if not cache_dir.is_dir():
        return {}

    text_to_embedding: dict[str, np.ndarray] = {}
    chunk_files = sorted(cache_dir.glob("chunk_*.npy"))

    start_time = time.time()
    for chunk_file in tqdm(chunk_files, desc="Carregando fragmentos de embeddings"):
        # allow_pickle=True: arquivos gerados apenas por _save_embedding_chunk()
        # deste mesmo módulo (cache local de embeddings, nunca dados externos).
        chunk_data = np.load(chunk_file, allow_pickle=True)
        text_to_embedding.update(dict(chunk_data))

    logger.info(
        "%d embedding(s) carregado(s) do cache '%s' em %.1fs.",
        len(text_to_embedding),
        cache_name,
        time.time() - start_time,
    )
    return text_to_embedding


def _find_next_chunk_index(cache_name: str | None) -> int:
    """Determina o próximo índice de fragmento disponível para um cache."""
    if not cache_name:
        return 0

    cache_dir = EMBEDDING_CACHE_DIR / cache_name
    if not cache_dir.is_dir():
        return 0

    chunk_files = list(cache_dir.glob("chunk_*.npy"))
    if not chunk_files:
        return 0

    indices = [int(chunk_file.stem.split("_")[1]) for chunk_file in chunk_files]
    return max(indices) + 1


def _save_embedding_chunk(
    cache_name: str | None, chunk_embeddings: dict[str, np.ndarray], chunk_index: int
) -> int:
    """Salva um fragmento de embeddings em disco e retorna o próximo índice livre."""
    if not cache_name or not chunk_embeddings:
        return chunk_index

    cache_dir = EMBEDDING_CACHE_DIR / cache_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    chunk_path = cache_dir / f"chunk_{chunk_index:03d}.npy"
    np.save(chunk_path, np.array(list(chunk_embeddings.items()), dtype=object))
    logger.info("%d embedding(s) salvo(s) em '%s'.", len(chunk_embeddings), chunk_path)

    return chunk_index + 1


def _prepare_texts_to_embed(
    texts: list[str], cache_name: str | None
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Filtra textos inválidos e separa os que ainda não estão no cache de embeddings."""
    texts = filter_invalid_texts(texts)
    text_to_embedding = load_embedding_cache(cache_name)
    texts_to_embed = [text for text in texts if text not in text_to_embedding]
    return text_to_embedding, texts_to_embed


def _compute_chunk_ranges(n_items: int, chunk_size: int) -> list[tuple[int, int]]:
    """Calcula os intervalos ``[início, fim)`` de cada fragmento de tamanho ``chunk_size``."""
    return [(start, min(start + chunk_size, n_items)) for start in range(0, n_items, chunk_size)]


def _embed_openai_batches(
    batches: list[list[str]],
    model: str,
    client: Any,
    n_workers: int,
    timeout: float | None,
    show_progress: bool,
    chunk_label: str,
) -> dict[str, np.ndarray]:
    """Executa em paralelo as requisições de um fragmento e retorna os embeddings calculados."""
    chunk_embeddings: dict[str, np.ndarray] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(_request_openai_embeddings_batch, batch, model, client, timeout=timeout)
            for batch in batches
        ]

        iterator = concurrent.futures.as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(batches), desc=chunk_label)

        for future in iterator:
            batch_result = future.result()
            batch = batches[futures.index(future)]
            for text, embedding in zip(batch, batch_result, strict=True):
                chunk_embeddings[text] = np.asarray(embedding)

    return chunk_embeddings


def extract_openai_embeddings(
    texts: list[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 256,
    n_workers: int = 5,
    cache_name: str | None = None,
    show_progress: bool = True,
    chunk_size: int = 50_000,
    timeout: float | None = None,
) -> dict[str, np.ndarray]:
    """Calcula embeddings via API da OpenAI, com paralelismo e cache fragmentado em disco.

    Parameters
    ----------
    texts : list[str]
        Textos a serem embedados; valores ``None``/vazios são descartados.
    model : str, optional
        Modelo de embedding da OpenAI, by default "text-embedding-3-small".
    batch_size : int, optional
        Tamanho do lote por requisição, by default 256.
    n_workers : int, optional
        Número de threads paralelas para requisições, by default 5.
    cache_name : str | None, optional
        Nome do cache em :data:`EMBEDDING_CACHE_DIR`; se ``None``, não
        cacheia, by default None.
    show_progress : bool, optional
        Se exibe barras de progresso, by default True.
    chunk_size : int, optional
        Número de textos por fragmento salvo em disco, by default 50000.
    timeout : float | None, optional
        Timeout por requisição, em segundos, by default None.

    Returns
    -------
    dict[str, np.ndarray]
        Mapa de texto para embedding (inclui itens recuperados do cache).

    Raises
    ------
    ModelError
        Se ``openai`` ou ``tiktoken`` não estiverem instalados.
    """
    text_to_embedding, texts_to_embed = _prepare_texts_to_embed(texts, cache_name)
    if not texts_to_embed:
        return text_to_embedding

    try:
        from hypothesaes.llm_api import create_client
    except ImportError as exception:  # pragma: no cover - guarda defensiva
        raise ModelError(
            "A biblioteca 'openai' não está instalada. Instale com `uv add openai tiktoken` "
            "para calcular embeddings via API da OpenAI."
        ) from exception
    client = create_client()

    next_chunk_index = _find_next_chunk_index(cache_name)
    chunk_ranges = _compute_chunk_ranges(len(texts_to_embed), chunk_size)
    chunk_iterator = (
        tqdm(chunk_ranges, desc="Processando fragmentos") if show_progress else chunk_ranges
    )

    for chunk_start, chunk_end in chunk_iterator:
        chunk_texts = texts_to_embed[chunk_start:chunk_end]
        batches = [chunk_texts[i : i + batch_size] for i in range(0, len(chunk_texts), batch_size)]

        chunk_embeddings = _embed_openai_batches(
            batches,
            model,
            client,
            n_workers,
            timeout,
            show_progress,
            f"Fragmento {next_chunk_index}",
        )
        text_to_embedding.update(chunk_embeddings)
        next_chunk_index = _save_embedding_chunk(cache_name, chunk_embeddings, next_chunk_index)

    return text_to_embedding


def _prefix_batch_for_model(model: str, batch: list[str]) -> Any:
    """Aplica o prefixo de instrução esperado por certas
    famílias de modelo (nomic-ai, instructor).
    """
    if "nomic-ai" in model:
        return ["clustering: " + text for text in batch]
    if "instructor" in model:
        return [["Represent the text for classification: ", text] for text in batch]
    return batch


def _embed_local_batches(
    chunk_texts: list[str],
    model: str,
    transformer_model: Any,
    batch_size: int,
    show_progress: bool,
    chunk_label: str,
) -> dict[str, np.ndarray]:
    """Codifica um fragmento de textos em lotes com o modelo local já carregado."""
    chunk_embeddings: dict[str, np.ndarray] = {}
    batch_iterator: Any = range(0, len(chunk_texts), batch_size)
    if show_progress:
        batch_iterator = tqdm(batch_iterator, desc=chunk_label)

    for i in batch_iterator:
        batch = chunk_texts[i : i + batch_size]
        prefixed_batch = _prefix_batch_for_model(model, batch)
        batch_embeddings = transformer_model.encode(prefixed_batch, batch_size=batch_size)
        chunk_embeddings.update(zip(batch, batch_embeddings, strict=True))

    return chunk_embeddings


def extract_local_embeddings(
    texts: list[str],
    model: str = "nomic-ai/modernbert-embed-base",
    batch_size: int = 128,
    show_progress: bool = True,
    cache_name: str | None = None,
    chunk_size: int = 50_000,
    device: str | None = None,
) -> dict[str, np.ndarray]:
    """Calcula embeddings com um modelo local ``sentence-transformers``, com cache fragmentado.

    Parameters
    ----------
    texts : list[str]
        Textos a serem embedados; valores ``None``/vazios são descartados.
    model : str, optional
        Identificador do modelo no Hugging Face Hub, by default
        "nomic-ai/modernbert-embed-base".
    batch_size : int, optional
        Tamanho do lote de codificação, by default 128.
    show_progress : bool, optional
        Se exibe barras de progresso, by default True.
    cache_name : str | None, optional
        Nome do cache em :data:`EMBEDDING_CACHE_DIR`; se ``None``, não
        cacheia, by default None.
    chunk_size : int, optional
        Número de textos por fragmento salvo em disco, by default 50000.
    device : str | None, optional
        Dispositivo PyTorch (``"cuda"``/``"cpu"``); se ``None``, detecta
        automaticamente, by default None.

    Returns
    -------
    dict[str, np.ndarray]
        Mapa de texto para embedding (inclui itens recuperados do cache).

    Raises
    ------
    ModelError
        Se ``sentence-transformers``/``torch`` não estiverem instalados.
    """
    text_to_embedding, texts_to_embed = _prepare_texts_to_embed(texts, cache_name)
    if not texts_to_embed:
        return text_to_embedding

    try:
        import torch  # type: ignore[reportMissingImports]
        from sentence_transformers import SentenceTransformer  # type: ignore[reportMissingImports]
    except ImportError as exception:
        raise ModelError(
            "As bibliotecas 'torch'/'sentence-transformers' não estão instaladas. Instale com "
            "`uv add torch sentence-transformers` para calcular embeddings localmente."
        ) from exception

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Habilita matmul float32 com TF32 em GPUs compatíveis, evitando avisos
    # repetidos e melhorando o throughput da codificação.
    torch.set_float32_matmul_precision("high")

    transformer_model = SentenceTransformer(model, device=resolved_device)
    logger.info("Modelo '%s' carregado em '%s'.", model, resolved_device)

    next_chunk_index = _find_next_chunk_index(cache_name)
    chunk_ranges = _compute_chunk_ranges(len(texts_to_embed), chunk_size)
    chunk_iterator = (
        tqdm(chunk_ranges, desc="Processando fragmentos") if show_progress else chunk_ranges
    )

    for chunk_start, chunk_end in chunk_iterator:
        chunk_texts = texts_to_embed[chunk_start:chunk_end]
        chunk_embeddings = _embed_local_batches(
            chunk_texts,
            model,
            transformer_model,
            batch_size,
            show_progress,
            f"Fragmento {next_chunk_index}",
        )
        text_to_embedding.update(chunk_embeddings)
        next_chunk_index = _save_embedding_chunk(cache_name, chunk_embeddings, next_chunk_index)

    del transformer_model
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return text_to_embedding
