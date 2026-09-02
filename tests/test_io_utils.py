"""Testes dos utilitários de entrada/saída de arquivos (``io_utils``)."""

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from exceptions.data import DataNotFoundError
from exceptions.model import ModelPersistenceError
from io_utils.csv import read_csv, write_csv
from io_utils.json import read_json, write_json
from io_utils.model import load_model, save_model
from io_utils.parquet import read_parquet, write_parquet
from io_utils.yaml import read_yaml, write_yaml


class TestYamlIO:
    """Testes de leitura e escrita de arquivos YAML."""

    def test_write_then_read_roundtrip(self, tmp_path: Path) -> None:
        """Escrever e reler um YAML deve preservar os dados originais."""
        caminho = tmp_path / "exemplo.yaml"
        dados = {"chave": "valor", "lista": [1, 2, 3]}
        write_yaml(dados, caminho)
        assert read_yaml(caminho) == dados

    def test_read_yaml_raises_for_missing_file(self, tmp_path: Path) -> None:
        """Deve levantar DataNotFoundError se o arquivo não existir."""
        with pytest.raises(DataNotFoundError):
            read_yaml(tmp_path / "inexistente.yaml")

    def test_read_yaml_returns_empty_dict_for_empty_file(self, tmp_path: Path) -> None:
        """Um arquivo YAML vazio deve retornar um dicionário vazio, não None."""
        caminho = tmp_path / "vazio.yaml"
        caminho.write_text("")
        assert read_yaml(caminho) == {}

    def test_write_yaml_creates_parent_directories(self, tmp_path: Path) -> None:
        """Deve criar diretórios pais ausentes automaticamente."""
        caminho = tmp_path / "subdir" / "exemplo.yaml"
        write_yaml({"a": 1}, caminho)
        assert caminho.is_file()


class TestJsonIO:
    """Testes de leitura e escrita de arquivos JSON."""

    def test_write_then_read_roundtrip(self, tmp_path: Path) -> None:
        """Escrever e reler um JSON deve preservar os dados originais."""
        caminho = tmp_path / "exemplo.json"
        dados: dict[str, Any] = {"f1_macro": 0.82, "classes": ["negativo", "neutro", "positivo"]}
        write_json(dados, caminho)
        assert read_json(caminho) == dados

    def test_write_json_preserves_non_ascii_characters(self, tmp_path: Path) -> None:
        """Caracteres acentuados em pt-BR devem ser preservados (ensure_ascii=False)."""
        caminho = tmp_path / "exemplo.json"
        write_json({"texto": "análise de sentimentos"}, caminho)
        assert "análise" in caminho.read_text(encoding="utf-8")

    def test_read_json_raises_for_missing_file(self, tmp_path: Path) -> None:
        """Deve levantar DataNotFoundError se o arquivo não existir."""
        with pytest.raises(DataNotFoundError):
            read_json(tmp_path / "inexistente.json")


class TestCsvIO:
    """Testes de leitura e escrita de arquivos CSV via Polars."""

    def test_write_then_read_roundtrip(self, tmp_path: Path) -> None:
        """Escrever e reler um CSV deve preservar os dados originais."""
        caminho = tmp_path / "exemplo.csv"
        dataframe = pl.DataFrame({"id": ["1", "2"], "sentimento": ["positivo", "negativo"]})
        write_csv(dataframe, caminho)
        resultado = read_csv(caminho)
        assert resultado.to_dicts() == dataframe.to_dicts()

    def test_read_csv_raises_for_missing_file(self, tmp_path: Path) -> None:
        """Deve levantar DataNotFoundError se o arquivo não existir."""
        with pytest.raises(DataNotFoundError):
            read_csv(tmp_path / "inexistente.csv")


class TestParquetIO:
    """Testes de leitura e escrita de arquivos Parquet via Polars."""

    def test_write_then_read_roundtrip(self, tmp_path: Path) -> None:
        """Escrever e reler um Parquet deve preservar os dados originais."""
        caminho = tmp_path / "exemplo.parquet"
        dataframe = pl.DataFrame({"id": ["1", "2"], "confianca": [0.9, 0.8]})
        write_parquet(dataframe, caminho)
        resultado = read_parquet(caminho)
        assert resultado.to_dicts() == dataframe.to_dicts()

    def test_read_parquet_raises_for_missing_file(self, tmp_path: Path) -> None:
        """Deve levantar DataNotFoundError se o arquivo não existir."""
        with pytest.raises(DataNotFoundError):
            read_parquet(tmp_path / "inexistente.parquet")


class TestModelIO:
    """Testes de persistência de modelos via joblib."""

    def test_save_then_load_roundtrip(self, tmp_path: Path) -> None:
        """Salvar e carregar um objeto simples deve preservar seu conteúdo."""
        caminho = tmp_path / "modelo.joblib"
        modelo_original = {"coeficientes": [0.1, 0.2, 0.3], "intercepto": 0.5}
        save_model(modelo_original, caminho)
        assert load_model(caminho) == modelo_original

    def test_save_model_creates_parent_directories(self, tmp_path: Path) -> None:
        """Deve criar diretórios pais ausentes automaticamente."""
        caminho = tmp_path / "subdir" / "modelo.joblib"
        save_model([1, 2, 3], caminho)
        assert caminho.is_file()

    def test_load_model_raises_for_missing_file(self, tmp_path: Path) -> None:
        """Deve levantar DataNotFoundError se o arquivo não existir."""
        with pytest.raises(DataNotFoundError):
            load_model(tmp_path / "inexistente.joblib")

    def test_load_model_raises_model_persistence_error_for_corrupted_file(
        self, tmp_path: Path
    ) -> None:
        """Um arquivo corrompido deve levantar ModelPersistenceError, não uma exceção genérica.

        Usa bytes que não correspondem a nenhum opcode válido de pickle
        (0xFF não é um opcode definido em nenhum protocolo), garantindo uma
        falha de desserialização determinística.
        """
        caminho = tmp_path / "corrompido.joblib"
        caminho.write_bytes(b"\xff\xff\xff\xff")
        with pytest.raises(ModelPersistenceError):
            load_model(caminho)
