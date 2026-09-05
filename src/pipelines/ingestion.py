"""Coleta de tweets e datasets externos, com catalogação de rastreabilidade.

Implementa o estágio ``ingestion`` de ``configs/config.yaml -> stages``:
orquestra a coleta paralela de tweets (``src/data/downloader.py``) e o
download de gold sets externos (TweetSentBR/RePro), grava o corpus bruto
consolidado em ``paths.raw_tweets_file`` e monta o catálogo de
rastreabilidade (``src/data/catalog.py``) dos arquivos de dados de entrada
do projeto.
"""

import logging
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from config.paths import ProjectPaths
from data.catalog import build_dataset_catalog, write_dataset_catalog
from data.downloader import collect_tweets_by_query, download_external_dataset
from data.writer import write_dataset

logger = logging.getLogger(__name__)

_CATALOG_FILE_NAME = "catalog.json"


def run_ingestion_stage(
    paths: ProjectPaths,
    *,
    scrape_func: Callable[[str], list[dict[str, str]]],
    queries: Iterable[str],
    external_download_funcs: Mapping[str, Callable[[], bytes]] | None = None,
    max_workers: int | None = None,
) -> Path:
    """Executa a etapa de ingestão: coleta tweets, baixa gold sets e cataloga os dados brutos.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto (``configs/paths.yaml``).
    scrape_func : Callable[[str], list[dict[str, str]]]
        Função de coleta por consulta, repassada a
        :func:`data.downloader.collect_tweets_by_query`. Cada registro
        retornado deve conter as chaves exigidas por
        :class:`schemas.dataset.RawTweetSchema` (``id``, ``text``,
        ``data_source``, ``data_collected``).
    queries : Iterable[str]
        Consultas de coleta (termos de busca, nomes de usuário etc.).
    external_download_funcs : Mapping[str, Callable[[], bytes]] | None, optional
        Funções de download por gold set externo, nomeadas pela chave
        ``"tweetsentbr"``/``"repro"`` (correspondendo a
        ``paths.tweetsentbr_file``/``paths.repro_file``), by default None
        (nenhum gold set baixado nesta execução).
    max_workers : int | None, optional
        Repassado a :func:`data.downloader.collect_tweets_by_query`, by
        default None.

    Returns
    -------
    Path
        Caminho do corpus bruto escrito (``paths.raw_tweets_file``).

    Raises
    ------
    EmptyDatasetError
        Se a coleta não retornar nenhum tweet com sucesso.

    Examples
    --------
    >>> def coletar_exemplo(termo):
    ...     return [
    ...         {
    ...             "id": "1",
    ...             "text": termo,
    ...             "data_source": "scraping",
    ...             "data_collected": "2026-01-01",
    ...         }
    ...     ]
    >>> run_ingestion_stage(
    ...     paths, scrape_func=coletar_exemplo, queries=["python"]
    ... )  # doctest: +SKIP
    """
    collected_tweets = collect_tweets_by_query(scrape_func, queries, max_workers=max_workers)
    write_dataset(collected_tweets, paths.raw_tweets_file)

    external_destinations = {
        "tweetsentbr": paths.tweetsentbr_file,
        "repro": paths.repro_file,
    }
    for dataset_name, download_func in (external_download_funcs or {}).items():
        destination_path = external_destinations.get(dataset_name)
        if destination_path is None:
            logger.warning(
                "Gold set externo '%s' ignorado: nenhum caminho conhecido em ProjectPaths.",
                dataset_name,
            )
            continue
        download_external_dataset(download_func, destination_path)

    catalog_datasets = {
        "raw_tweets": paths.raw_tweets_file,
        "tweetsentbr": paths.tweetsentbr_file,
        "repro": paths.repro_file,
    }
    catalog_entries = build_dataset_catalog(catalog_datasets)
    write_dataset_catalog(catalog_entries, paths.data_raw_dir / _CATALOG_FILE_NAME)

    logger.info(
        "Etapa de ingestão concluída: %d tweet(s) coletado(s), %d dataset(s) catalogado(s).",
        collected_tweets.height,
        len(catalog_entries),
    )
    return paths.raw_tweets_file
