"""Ingestão, particionamento e catalogação de dados do projeto.

Implementa a Fase 5 do plano de elaboração (``PLANO-ELABORACAO.md``):
coleta (scraping paralelo e gold sets), carregamento com validação de
schema, particionamento estratificado, amostragem para validação humana,
escrita padronizada em ``data/interim``/``data/processed`` e catalogação
com hashes de rastreabilidade (ver CLAUDE.md, "Reprodutibilidade &
Determinismo").

Modules
-------
downloader
    Coleta paralela de tweets via scraping e download de gold sets externos.
loader
    Carregamento de datasets (CSV/Parquet) com validação de schema.
splitter
    Particionamento estratificado em treino/validação/teste, com seed fixa.
sampler
    Amostragem aleatória e estratificada para validação humana.
writer
    Escrita validada e padronizada em ``data/interim``/``data/processed``.
catalog
    Catálogo dos datasets disponíveis, com hash SHA-256 e tamanho.
"""

from data.catalog import (
    DatasetCatalogEntry,
    build_dataset_catalog,
    build_dataset_catalog_entry,
    write_dataset_catalog,
)
from data.downloader import collect_tweets_by_query, download_external_dataset
from data.loader import (
    load_labeled_corpus,
    load_raw_tweet_dataset,
    load_training_example_dataset,
    read_dataset_file,
)
from data.sampler import sample_random_subset, sample_stratified_subset
from data.splitter import create_stratified_split
from data.writer import write_dataset, write_labeled_corpus, write_training_example_dataset

__all__: list[str] = [
    "DatasetCatalogEntry",
    "build_dataset_catalog",
    "build_dataset_catalog_entry",
    "collect_tweets_by_query",
    "create_stratified_split",
    "download_external_dataset",
    "load_labeled_corpus",
    "load_raw_tweet_dataset",
    "load_training_example_dataset",
    "read_dataset_file",
    "sample_random_subset",
    "sample_stratified_subset",
    "write_dataset",
    "write_dataset_catalog",
    "write_labeled_corpus",
    "write_training_example_dataset",
]
