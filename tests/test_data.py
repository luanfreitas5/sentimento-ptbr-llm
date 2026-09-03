"""Testes dos módulos de ingestão e catalogação de dados (``src/data``)."""

import hashlib
from pathlib import Path

import polars as pl
import pytest

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
from exceptions.data import DataError, DataNotFoundError, DataValidationError, EmptyDatasetError
from io_utils.csv import write_csv
from io_utils.json import read_json
from io_utils.parquet import write_parquet
from schemas.training import DATA_SPLITS

# Funções auxiliares no nível de módulo, usadas pelos testes de coleta paralela.


def _scrape_ok(query: str) -> list[dict[str, str]]:
    """Simula uma coleta bem-sucedida para uma única consulta."""
    return [{"id": query, "text": f"tweet sobre {query}"}]


def _scrape_fail_on_specific_query(query: str) -> list[dict[str, str]]:
    """Simula uma coleta que falha apenas para a consulta 'falha'."""
    if query == "falha":
        raise ValueError("consulta inválida")
    return [{"id": query, "text": query}]


def _scrape_always_fails(query: str) -> list[dict[str, str]]:
    """Simula uma coleta que falha para qualquer consulta."""
    raise ValueError(f"falha ao coletar '{query}'")


class TestReadDatasetFile:
    """Testes do despacho de leitura por extensão (``read_dataset_file``)."""

    def test_dispatches_to_parquet_reader(self, tmp_path: Path) -> None:
        """Um arquivo ``.parquet`` deve ser lido via Parquet."""
        file_path = tmp_path / "exemplo.parquet"
        df = pl.DataFrame({"id": ["1", "2"], "valor": ["a", "b"]})
        write_parquet(df, file_path)
        assert read_dataset_file(file_path).to_dicts() == df.to_dicts()

    def test_dispatches_to_csv_reader(self, tmp_path: Path) -> None:
        """Um arquivo ``.csv`` deve ser lido via CSV."""
        file_path = tmp_path / "exemplo.csv"
        df = pl.DataFrame({"id": ["1", "2"], "valor": ["a", "b"]})
        write_csv(df, file_path)
        assert read_dataset_file(file_path).to_dicts() == df.to_dicts()

    def test_raises_for_unsupported_extension(self, tmp_path: Path) -> None:
        """Uma extensão não suportada deve levantar DataError."""
        file_path = tmp_path / "exemplo.txt"
        file_path.write_text("conteudo")
        with pytest.raises(DataError):
            read_dataset_file(file_path)

    def test_raises_for_missing_file(self, tmp_path: Path) -> None:
        """Deve levantar DataNotFoundError se o arquivo não existir."""
        with pytest.raises(DataNotFoundError):
            read_dataset_file(tmp_path / "inexistente.parquet")


class TestLoadRawTweetDataset:
    """Testes de carregamento e validação de tweets brutos."""

    def test_returns_validated_dataframe(self, tmp_path: Path) -> None:
        """Um dataset de tweets brutos válido deve ser carregado normalmente."""
        file_path = tmp_path / "tweets.parquet"
        df = pl.DataFrame(
            {
                "id": ["1", "2"],
                "text": ["ótimo produto", "não gostei"],
                "data_source": ["scraping", "scraping"],
                "data_collected": ["2026-01-01", "2026-01-02"],
            }
        )
        write_parquet(df, file_path)
        assert load_raw_tweet_dataset(file_path).height == 2

    def test_raises_for_invalid_schema(self, tmp_path: Path) -> None:
        """Colunas obrigatórias ausentes devem levantar DataValidationError."""
        file_path = tmp_path / "tweets_invalidos.parquet"
        write_parquet(pl.DataFrame({"id": ["1"], "text": ["ótimo"]}), file_path)
        with pytest.raises(DataValidationError):
            load_raw_tweet_dataset(file_path)


class TestLoadLabeledCorpus:
    """Testes de carregamento e validação do corpus rotulado."""

    def test_returns_validated_dataframe(
        self, tmp_path: Path, sample_labeled_corpus: pl.DataFrame
    ) -> None:
        """Um corpus rotulado válido deve ser carregado normalmente."""
        file_path = tmp_path / "corpus_rotulado.parquet"
        write_parquet(sample_labeled_corpus, file_path)
        assert load_labeled_corpus(file_path).height == 3

    def test_raises_for_invalid_label(self, tmp_path: Path) -> None:
        """Um rótulo fora das classes conhecidas deve levantar DataValidationError."""
        file_path = tmp_path / "corpus_invalido.parquet"
        df = pl.DataFrame({"id": ["1"], "text": ["ótimo"], "sentiment_label": ["desconhecido"]})
        write_parquet(df, file_path)
        with pytest.raises(DataValidationError):
            load_labeled_corpus(file_path)


class TestLoadTrainingExampleDataset:
    """Testes de carregamento e validação de conjuntos de treino/validação/teste."""

    def test_returns_validated_dataframe(self, tmp_path: Path) -> None:
        """Um conjunto particionado válido deve ser carregado normalmente."""
        file_path = tmp_path / "treino.parquet"
        df = pl.DataFrame(
            {"id": ["1"], "text": ["ótimo"], "sentiment_label": ["positivo"], "split": ["treino"]}
        )
        write_parquet(df, file_path)
        assert load_training_example_dataset(file_path).height == 1

    def test_raises_for_invalid_split(self, tmp_path: Path) -> None:
        """Um valor de split fora de treino/validacao/teste deve levantar DataValidationError."""
        file_path = tmp_path / "treino_invalido.parquet"
        df = pl.DataFrame(
            {"id": ["1"], "text": ["ótimo"], "sentiment_label": ["positivo"], "split": ["outro"]}
        )
        write_parquet(df, file_path)
        with pytest.raises(DataValidationError):
            load_training_example_dataset(file_path)


class TestWriteDataset:
    """Testes da escrita genérica de datasets (``write_dataset``)."""

    def test_creates_parquet_file(self, tmp_path: Path) -> None:
        """Deve criar o arquivo Parquet, inclusive diretórios pais ausentes."""
        file_path = tmp_path / "subdir" / "exemplo.parquet"
        write_dataset(pl.DataFrame({"id": ["1", "2"]}), file_path)
        assert file_path.is_file()

    def test_raises_for_empty_dataframe(self, tmp_path: Path) -> None:
        """Um DataFrame vazio deve levantar EmptyDatasetError, sem escrever arquivo."""
        file_path = tmp_path / "vazio.parquet"
        with pytest.raises(EmptyDatasetError):
            write_dataset(pl.DataFrame({"id": []}), file_path)
        assert not file_path.exists()


class TestWriteLabeledCorpus:
    """Testes da escrita validada do corpus rotulado."""

    def test_validates_before_writing(
        self, tmp_path: Path, sample_labeled_corpus: pl.DataFrame
    ) -> None:
        """Um corpus rotulado válido deve ser escrito normalmente."""
        file_path = tmp_path / "corpus_rotulado.parquet"
        write_labeled_corpus(sample_labeled_corpus, file_path)
        assert file_path.is_file()

    def test_raises_for_invalid_label_without_writing_file(self, tmp_path: Path) -> None:
        """Um rótulo inválido deve impedir a escrita do arquivo."""
        file_path = tmp_path / "corpus_invalido.parquet"
        df = pl.DataFrame({"id": ["1"], "text": ["ótimo"], "sentiment_label": ["desconhecido"]})
        with pytest.raises(DataValidationError):
            write_labeled_corpus(df, file_path)
        assert not file_path.exists()


class TestWriteTrainingExampleDataset:
    """Testes da escrita validada de conjuntos de treino/validação/teste."""

    def test_validates_before_writing(self, tmp_path: Path) -> None:
        """Um conjunto particionado válido deve ser escrito normalmente."""
        file_path = tmp_path / "treino.parquet"
        df = pl.DataFrame(
            {"id": ["1"], "text": ["ótimo"], "sentiment_label": ["positivo"], "split": ["treino"]}
        )
        write_training_example_dataset(df, file_path)
        assert file_path.is_file()

    def test_raises_for_invalid_split_without_writing_file(self, tmp_path: Path) -> None:
        """Um valor de split inválido deve impedir a escrita do arquivo."""
        file_path = tmp_path / "treino_invalido.parquet"
        df = pl.DataFrame(
            {"id": ["1"], "text": ["ótimo"], "sentiment_label": ["positivo"], "split": ["outro"]}
        )
        with pytest.raises(DataValidationError):
            write_training_example_dataset(df, file_path)
        assert not file_path.exists()


class TestCreateStratifiedSplit:
    """Testes do particionamento estratificado (``splitter``)."""

    def test_assigns_all_three_splits(self) -> None:
        """Um grupo com exemplos suficientes deve receber as três classes de split."""
        df = pl.DataFrame({"id": [str(i) for i in range(10)], "sentiment_label": ["positivo"] * 10})
        result = create_stratified_split(df, label_column="sentiment_label")
        assert sorted(result["split"].unique().to_list()) == sorted(DATA_SPLITS)

    def test_preserves_row_count(self) -> None:
        """O número de linhas do resultado deve ser igual ao da entrada."""
        df = pl.DataFrame({"id": [str(i) for i in range(15)], "sentiment_label": ["neutro"] * 15})
        result = create_stratified_split(df, label_column="sentiment_label")
        assert result.height == df.height

    def test_is_deterministic_given_same_seed(self) -> None:
        """A mesma semente deve produzir exatamente a mesma atribuição de split."""
        df = pl.DataFrame({"id": [str(i) for i in range(10)], "sentiment_label": ["positivo"] * 10})
        result_a = create_stratified_split(df, label_column="sentiment_label", random_seed=7)
        result_b = create_stratified_split(df, label_column="sentiment_label", random_seed=7)
        assert result_a["split"].to_list() == result_b["split"].to_list()

    def test_raises_for_invalid_proportions(self) -> None:
        """test_size + validation_size >= 1 deve levantar ValueError."""
        df = pl.DataFrame({"id": ["1"], "sentiment_label": ["positivo"]})
        with pytest.raises(ValueError):
            create_stratified_split(
                df, label_column="sentiment_label", test_size=0.6, validation_size=0.5
            )

    def test_preserves_per_class_split_counts(self) -> None:
        """Cada classe deve ser particionada nas mesmas proporções, de forma independente."""
        df = pl.DataFrame(
            {
                "id": [str(i) for i in range(20)],
                "sentiment_label": ["positivo"] * 10 + ["negativo"] * 10,
            }
        )
        resultado = create_stratified_split(df, label_column="sentiment_label")
        for classe in ("positivo", "negativo"):
            class_splits = resultado.filter(pl.col("sentiment_label") == classe)["split"].to_list()
            assert class_splits.count("teste") == 2
            assert class_splits.count("validacao") == 1
            assert class_splits.count("treino") == 7


class TestSampleRandomSubset:
    """Testes da amostragem aleatória para validação humana."""

    def test_returns_requested_size(self) -> None:
        """A amostra deve ter exatamente o tamanho solicitado, quando disponível."""
        df = pl.DataFrame({"id": [str(i) for i in range(10)]})
        assert sample_random_subset(df, sample_size=4).height == 4

    def test_clamps_to_dataframe_size(self) -> None:
        """Um tamanho maior que o DataFrame deve ser limitado ao total disponível."""
        df = pl.DataFrame({"id": ["1", "2", "3"]})
        assert sample_random_subset(df, sample_size=100).height == 3

    def test_raises_for_empty_dataframe(self) -> None:
        """Um DataFrame vazio deve levantar EmptyDatasetError."""
        with pytest.raises(EmptyDatasetError):
            sample_random_subset(pl.DataFrame({"id": []}), sample_size=1)

    def test_raises_for_non_positive_sample_size(self) -> None:
        """sample_size não positivo deve levantar ValueError."""
        with pytest.raises(ValueError):
            sample_random_subset(pl.DataFrame({"id": ["1"]}), sample_size=0)


class TestSampleStratifiedSubset:
    """Testes da amostragem estratificada para validação humana."""

    def test_includes_all_strata(self) -> None:
        """Todas as classes presentes no DataFrame original devem aparecer na amostra."""
        df = pl.DataFrame(
            {
                "id": [str(i) for i in range(20)],
                "sentiment_label": ["positivo"] * 10 + ["negativo"] * 10,
            }
        )
        result = sample_stratified_subset(df, stratify_column="sentiment_label", sample_size=10)
        assert sorted(result["sentiment_label"].unique().to_list()) == ["negativo", "positivo"]

    def test_raises_for_empty_dataframe(self) -> None:
        """Um DataFrame vazio deve levantar EmptyDatasetError."""
        with pytest.raises(EmptyDatasetError):
            sample_stratified_subset(
                pl.DataFrame({"id": [], "sentiment_label": []}),
                stratify_column="sentiment_label",
                sample_size=1,
            )

    def test_raises_for_non_positive_sample_size(self) -> None:
        """sample_size não positivo deve levantar ValueError."""
        df = pl.DataFrame({"id": ["1"], "sentiment_label": ["positivo"]})
        with pytest.raises(ValueError):
            sample_stratified_subset(df, stratify_column="sentiment_label", sample_size=0)


class TestDatasetCatalog:
    """Testes do catálogo de datasets e seus hashes de rastreabilidade."""

    def test_build_dataset_catalog_entry_computes_hash_and_size(self, tmp_path: Path) -> None:
        """A entrada de catálogo deve conter o hash SHA-256 e o tamanho corretos do arquivo."""
        file_path = tmp_path / "exemplo.txt"
        sample_data = b"conteudo de exemplo"
        file_path.write_bytes(sample_data)

        dataset_entry = build_dataset_catalog_entry("exemplo", file_path)

        assert dataset_entry.name == "exemplo"
        assert dataset_entry.file_path == str(file_path)
        assert dataset_entry.sha256_hash == hashlib.sha256(sample_data).hexdigest()
        assert dataset_entry.size_bytes == len(sample_data)

    def test_build_dataset_catalog_entry_raises_for_missing_file(self, tmp_path: Path) -> None:
        """Deve levantar DataNotFoundError se o arquivo não existir."""
        with pytest.raises(DataNotFoundError):
            build_dataset_catalog_entry("inexistente", tmp_path / "inexistente.txt")

    def test_build_dataset_catalog_skips_missing_datasets(self, tmp_path: Path) -> None:
        """Datasets ausentes devem ser ignorados, sem interromper a construção do catálogo."""
        available_dataset = tmp_path / "existente.txt"
        available_dataset.write_bytes(b"dados")

        catalogo = build_dataset_catalog(
            {"existente": available_dataset, "ausente": tmp_path / "ausente.txt"}
        )

        assert [entrada.name for entrada in catalogo] == ["existente"]

    def test_write_dataset_catalog_writes_json(self, tmp_path: Path) -> None:
        """O catálogo deve ser serializado em JSON, preservando todos os campos."""
        dataset_entries = [
            DatasetCatalogEntry(
                name="a", file_path="a.parquet", sha256_hash="hash_a", size_bytes=10
            )
        ]
        catalog_file_path = tmp_path / "catalogo.json"

        write_dataset_catalog(dataset_entries, catalog_file_path)

        assert read_json(catalog_file_path) == [
            {"name": "a", "file_path": "a.parquet", "sha256_hash": "hash_a", "size_bytes": 10}
        ]


class TestCollectTweetsByQuery:
    """Testes da coleta paralela de tweets por consulta."""

    def test_consolidates_successful_results(self) -> None:
        """Os registros de todas as consultas bem-sucedidas devem ser consolidados."""
        result = collect_tweets_by_query(_scrape_ok, ["python", "nlp"], show_progress=False)
        assert result.height == 2
        assert sorted(result["id"].to_list()) == ["nlp", "python"]

    def test_skips_failed_queries(self) -> None:
        """Consultas que falham devem ser descartadas, sem interromper as demais."""
        result = collect_tweets_by_query(
            _scrape_fail_on_specific_query, ["ok", "falha"], show_progress=False
        )
        assert result.height == 1
        assert result["id"].to_list() == ["ok"]

    def test_raises_when_all_queries_fail(self) -> None:
        """Se nenhuma consulta retornar resultado, deve levantar EmptyDatasetError."""
        with pytest.raises(EmptyDatasetError):
            collect_tweets_by_query(_scrape_always_fails, ["a", "b"], show_progress=False)


class TestDownloadExternalDataset:
    """Testes do download de datasets externos (gold sets)."""

    def test_writes_file_and_creates_parent_directories(self, tmp_path: Path) -> None:
        """O conteúdo baixado deve ser salvo no destino, criando diretórios pais ausentes."""
        output_file_path = tmp_path / "subdir" / "gold_set.bin"
        result = download_external_dataset(lambda: b"conteudo binario", output_file_path)
        assert result == output_file_path
        assert output_file_path.read_bytes() == b"conteudo binario"
