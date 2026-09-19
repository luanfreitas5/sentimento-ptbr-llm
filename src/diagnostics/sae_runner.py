"""Partições disjuntas, embeddings e SAE compartilhado entre todos os alvos.

O SAE é treinado **uma vez** por conjunto de embeddings (checkpoint versionado
por modelo de embedding, M, K, tamanho do treino e semente) e reutilizado por
todos os alvos. As partições são criadas uma única vez sobre o corpus
deduplicado, de forma que descoberta (``treino``), validação do SAE
(``validacao``) e holdout (``teste``) nunca se sobrepõem.

Embeddings e SAE dependem de ``torch``/``sentence-transformers`` (extra
``hypothesaes``): os imports são tardios e encapsulados em funções de módulo,
que os testes substituem por dublês via ``monkeypatch``.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from config.paths import ProjectPaths
from data.splitter import create_stratified_split
from diagnostics.settings import DiagnosticsSettings
from exceptions.data import EmptyDatasetError
from schemas.diagnostics import list_model_label_columns
from utils.hashing import calculate_text_hash

logger = logging.getLogger(__name__)

PARTITION_COLUMN = "partition"
DISCOVERY_PARTITION = "treino"
SAE_VALIDATION_PARTITION = "validacao"
HOLDOUT_PARTITION = "teste"
_STRATUM_COLUMN = "_stratum"
_NO_LABEL_STRATUM = "sem_rotulo"
_TEXT_COLUMN = "text_normalized"


@dataclass(frozen=True)
class DiscoveryData:
    """Corpus particionado, embeddings alinhados às linhas e SAE treinado.

    Attributes
    ----------
    partitioned : pl.DataFrame
        Corpus deduplicado por texto, com a coluna ``partition``.
    embeddings : np.ndarray
        Matriz (n_linhas, dim), alinhada linha a linha a ``partitioned``.
    sae : Any
        SAE treinado (``hypothesaes.sae.SparseAutoencoder``).
    cache_name : str
        Prefixo do cache de embeddings/anotações (inclui o modelo de embedding).
    corpus_hash : str
        SHA-256 dos textos, para rastrear a versão dos dados no MLflow.
    """

    partitioned: pl.DataFrame
    embeddings: np.ndarray
    sae: Any
    cache_name: str
    corpus_hash: str


def extract_local_embeddings(texts: list[str], **kwargs: Any) -> dict[str, np.ndarray]:
    """Import tardio de :func:`hypothesaes.embedding.extract_local_embeddings`.

    Parameters
    ----------
    texts : list[str]
        Textos únicos a codificar.
    **kwargs : Any
        Repassados à função original (``model``, ``batch_size``, ``cache_name``).

    Returns
    -------
    dict[str, np.ndarray]
        Mapa texto -> embedding.
    """
    from hypothesaes.embedding import extract_local_embeddings as _extract

    return _extract(texts, **kwargs)


def train_sae(**kwargs: Any) -> Any:
    """Import tardio de :func:`hypothesaes.quickstart.train_sae` (treina ou carrega checkpoint)."""
    from hypothesaes.quickstart import train_sae as _train_sae

    return _train_sae(**kwargs)


def build_embedding_cache_name(model_name: str, n_texts: int) -> str:
    """Monta o prefixo de cache incluindo o modelo de embedding.

    Corrige o cache do estágio ``hypothesaes_analysis``, que não distinguia o
    modelo de embedding e podia reutilizar vetores incompatíveis.

    Parameters
    ----------
    model_name : str
        Nome do modelo (ex.: ``neuralmind/bert-base-portuguese-cased``).
    n_texts : int
        Nº de textos únicos.

    Returns
    -------
    str
        Nome seguro para arquivo, ex.: ``diagnostics_neuralmind-bert-...-5000texts``.

    Examples
    --------
    >>> build_embedding_cache_name("org/modelo-x", 10)
    'diagnostics_org-modelo-x_10texts'
    """
    slug = re.sub(r"[^A-Za-z0-9]+", "-", model_name).strip("-")
    return f"diagnostics_{slug}_{n_texts}texts"


def assign_partitions(
    corpus: pl.DataFrame, *, holdout_size: float, validation_size: float, random_seed: int
) -> pl.DataFrame:
    """Deduplica por texto e cria as partições disjuntas treino/validação/teste.

    A estratificação usa o primeiro rótulo de modelo disponível por tweet
    (``sem_rotulo`` se nenhum), mantendo a proporção de classes em cada partição.

    Parameters
    ----------
    corpus : pl.DataFrame
        Corpus no contrato de diagnóstico.
    holdout_size, validation_size : float
        Proporções de ``teste`` e ``validacao``.
    random_seed : int
        Semente da divisão.

    Returns
    -------
    pl.DataFrame
        Corpus deduplicado com a coluna ``partition``.

    Raises
    ------
    EmptyDatasetError
        Se o corpus estiver vazio.

    Examples
    --------
    >>> assign_partitions(corpus, holdout_size=0.2, validation_size=0.1, random_seed=42)
    ... # doctest: +SKIP
    """
    if corpus.is_empty():
        raise EmptyDatasetError("corpus de diagnóstico vazio")
    label_columns = list_model_label_columns(corpus)
    stratum = pl.coalesce([pl.col(name) for name in label_columns]).fill_null(_NO_LABEL_STRATUM)
    unique = corpus.unique(subset=[_TEXT_COLUMN], keep="first", maintain_order=True)
    split = create_stratified_split(
        unique.with_columns(stratum.alias(_STRATUM_COLUMN)),
        label_column=_STRATUM_COLUMN,
        split_column=PARTITION_COLUMN,
        test_size=holdout_size,
        validation_size=validation_size,
        random_seed=random_seed,
    )
    logger.info(
        "Partições: %s",
        split[PARTITION_COLUMN].value_counts().sort(PARTITION_COLUMN).to_dicts(),
    )
    return split.drop(_STRATUM_COLUMN)


def _stack_embeddings(texts: list[str], mapping: dict[str, np.ndarray]) -> np.ndarray:
    """Empilha os embeddings na ordem de ``texts``."""
    return np.vstack([np.asarray(mapping[text], dtype=np.float64) for text in texts])


def prepare_discovery_data(
    corpus: pl.DataFrame, settings: DiagnosticsSettings, paths: ProjectPaths
) -> DiscoveryData:
    """Particiona o corpus, calcula embeddings e treina (ou carrega) o SAE compartilhado.

    Parameters
    ----------
    corpus : pl.DataFrame
        Corpus no contrato de diagnóstico (texto já sanitizado).
    settings : DiagnosticsSettings
        Configuração validada.
    paths : ProjectPaths
        Caminhos do projeto (checkpoints em ``models/checkpoints``).

    Returns
    -------
    DiscoveryData
        Partições, embeddings alinhados, SAE, nome de cache e hash do corpus.

    Examples
    --------
    >>> prepare_discovery_data(corpus, settings, paths)  # doctest: +SKIP
    """
    partitioned = assign_partitions(
        corpus,
        holdout_size=settings.splits.holdout_size,
        validation_size=settings.splits.validation_size,
        random_seed=settings.random_seed,
    )
    texts = partitioned[_TEXT_COLUMN].to_list()
    cache_name = build_embedding_cache_name(settings.embedding.model_name, len(texts))
    logger.info("Calculando embeddings ('%s')...", settings.embedding.model_name)
    mapping = extract_local_embeddings(
        texts,
        model=settings.embedding.model_name,
        batch_size=settings.embedding.batch_size,
        cache_name=cache_name,
    )
    embeddings = _stack_embeddings(texts, mapping)

    is_train = (partitioned[PARTITION_COLUMN] == DISCOVERY_PARTITION).to_numpy()
    is_validation = (partitioned[PARTITION_COLUMN] == SAE_VALIDATION_PARTITION).to_numpy()
    sae_cfg = settings.sae
    checkpoint_dir = (
        paths.models_checkpoints_dir
        / sae_cfg.checkpoint_subdir
        / (
            f"{cache_name}_{sae_cfg.m_total_neurons}M_{sae_cfg.k_active_neurons}K_"
            f"{int(is_train.sum())}train_seed{settings.random_seed}"
        )
    )
    logger.info(
        "Treinando/carregando o SAE (M=%d, K=%d)...",
        sae_cfg.m_total_neurons,
        sae_cfg.k_active_neurons,
    )
    sae = train_sae(
        embeddings=embeddings[is_train],
        m_total_neurons=sae_cfg.m_total_neurons,
        k_active_neurons=sae_cfg.k_active_neurons,
        matryoshka_prefix_lengths=sae_cfg.matryoshka_prefix_lengths,
        val_embeddings=embeddings[is_validation] if is_validation.any() else None,
        checkpoint_dir=Path(checkpoint_dir),
    )
    return DiscoveryData(
        partitioned=partitioned,
        embeddings=embeddings,
        sae=sae,
        cache_name=cache_name,
        corpus_hash=calculate_text_hash("\x1f".join(texts)),
    )
