"""Coleta de tweets via scraping paralelo e download de gold sets externos.

A execução da coleta em si (chamadas de rede ao ``twscrape`` ou download
HTTP dos gold sets TweetSentBR/RePro) é responsabilidade do chamador,
injetada como função — este módulo apenas orquestra a execução paralela
(``src/parallel/scraping.py``) e a consolidação/persistência dos
resultados. Isso mantém o módulo testável sem depender de rede ou de
credenciais, e evita acoplar este módulo a uma dependência opcional
específica (ver CLAUDE.md, "Import style": dependências opcionais como
``twscrape`` devem ser importadas sob demanda pelo chamador).
"""

import logging
from collections.abc import Callable, Iterable
from itertools import chain
from pathlib import Path
from typing import TypeVar

import polars as pl

from exceptions.data import EmptyDatasetError
from parallel.scraping import run_parallel_scraping

logger = logging.getLogger(__name__)

QueryType = TypeVar("QueryType")


def collect_tweets_by_query(
    scrape_func: Callable[[QueryType], list[dict[str, str]]],
    queries: Iterable[QueryType],
    *,
    max_workers: int | None = None,
    show_progress: bool = True,
) -> pl.DataFrame:
    """Coleta tweets em paralelo a partir de uma função de scraping por consulta.

    Parameters
    ----------
    scrape_func : Callable[[QueryType], list[dict[str, str]]]
        Função que executa a coleta de uma única consulta (ex.: termo de
        busca ou nome de usuário), retornando uma lista de registros de
        tweet. Tipicamente um adaptador sobre o ``twscrape``.
    queries : Iterable[QueryType]
        Consultas a serem executadas.
    max_workers : int | None, optional
        Número máximo de threads usadas na coleta paralela, by default None
        (o executor escolhe automaticamente).
    show_progress : bool, optional
        Se ``True``, exibe uma barra de progresso no console, by default True.

    Returns
    -------
    pl.DataFrame
        Tweets coletados com sucesso, consolidados em um único DataFrame.
        Consultas que falharam são registradas em log e descartadas,
        sem interromper a coleta das demais.

    Raises
    ------
    EmptyDatasetError
        Se nenhuma consulta retornar resultados com sucesso.

    Examples
    --------
    >>> def coletar_exemplo(termo: str) -> list[dict[str, str]]:
    ...     return [{"id": "1", "text": termo}]
    >>> collect_tweets_by_query(coletar_exemplo, ["python"], show_progress=False).height
    1
    """
    query_results = run_parallel_scraping(
        scrape_func, queries, max_workers=max_workers, show_progress=show_progress
    )

    for query_failure in query_results.failures:
        logger.warning(
            "Falha ao coletar consulta '%s': %s", query_failure.item, query_failure.error
        )

    tweet_records = list(chain.from_iterable(query_results.successes))

    if not tweet_records:
        raise EmptyDatasetError("coleta de tweets via scraping")
    return pl.DataFrame(tweet_records)


def download_external_dataset(
    download_func: Callable[[], bytes],
    destination_path: Path,
) -> Path:
    """Baixa um dataset externo (gold set) e o salva em disco.

    Parameters
    ----------
    download_func : Callable[[], bytes]
        Função que executa o download e retorna o conteúdo bruto do
        arquivo (ex.: um wrapper sobre uma requisição HTTP a um repositório
        público do TweetSentBR/RePro).
    destination_path : Path
        Caminho de destino (tipicamente em ``data/external``) para o
        arquivo baixado.

    Returns
    -------
    Path
        O caminho onde o arquivo foi salvo (mesmo valor de
        ``destination_path``).

    Examples
    --------
    >>> download_external_dataset(
    ...     lambda: b"conteudo", Path("data/external/exemplo.bin")
    ... )  # doctest: +SKIP
    """
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    raw_data = download_func()
    destination_path.write_bytes(raw_data)
    logger.info("Dataset externo salvo em: %s (%d bytes)", destination_path, len(raw_data))
    return destination_path
