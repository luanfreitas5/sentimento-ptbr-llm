"""Testes das funções utilitárias de propósito geral do projeto."""

import hashlib
import logging
import random
from pathlib import Path

import numpy as np
import pytest

from exceptions.data import DataNotFoundError, EmptyDatasetError
from utils.decorators import log_execution_time, retry_on_exception
from utils.hashing import calculate_file_hash, calculate_text_hash
from utils.seed import seed_everything
from utils.text import normalize_whitespace, remove_accents, truncate_text
from utils.timing import ExecutionTiming, format_duration, measure_execution_time
from utils.validation import (
    validate_directory_exists,
    validate_file_exists,
    validate_not_empty_collection,
    validate_value_in_choices,
)


class TestLogExecutionTime:
    """Testes do decorador de log de tempo de execução."""

    def test_preserves_return_value(self) -> None:
        """A função decorada deve retornar o mesmo valor da função original."""

        @log_execution_time
        def somar(a: int, b: int) -> int:
            return a + b

        assert somar(2, 3) == 5

    def test_logs_execution_message(self, caplog: pytest.LogCaptureFixture) -> None:
        """Deve registrar uma mensagem de log informando o tempo de execução."""

        @log_execution_time
        def funcao_exemplo() -> None:
            return None

        with caplog.at_level(logging.INFO, logger="utils.decorators"):
            funcao_exemplo()
        assert any("funcao_exemplo" in registro.message for registro in caplog.records)


class TestRetryOnException:
    """Testes do decorador de retentativa em caso de falha."""

    def test_succeeds_after_transient_failures(self) -> None:
        """Deve reexecutar a função até obter sucesso, dentro do limite de tentativas."""
        contador = {"tentativas": 0}

        @retry_on_exception(exceptions=(ValueError,), max_attempts=3, delay_seconds=0)
        def funcao_instavel() -> str:
            contador["tentativas"] += 1
            if contador["tentativas"] < 2:
                raise ValueError("falha transitória")
            return "sucesso"

        assert funcao_instavel() == "sucesso"
        assert contador["tentativas"] == 2

    def test_raises_after_exhausting_attempts(self) -> None:
        """Deve levantar a última exceção após esgotar as tentativas."""

        @retry_on_exception(exceptions=(ValueError,), max_attempts=2, delay_seconds=0)
        def sempre_falha() -> None:
            raise ValueError("falha proposital")

        with pytest.raises(ValueError, match="falha proposital"):
            sempre_falha()

    def test_rejects_invalid_max_attempts(self) -> None:
        """max_attempts menor que 1 deve levantar ValueError na configuração do decorador."""
        with pytest.raises(ValueError):
            retry_on_exception(max_attempts=0)


class TestHashing:
    """Testes das funções de hashing de arquivos e texto."""

    def test_calculate_file_hash_matches_hashlib(self, tmp_path: Path) -> None:
        """O hash calculado deve coincidir com o cálculo direto via hashlib."""
        arquivo = tmp_path / "exemplo.txt"
        arquivo.write_bytes(b"conteudo de exemplo")
        esperado = hashlib.sha256(b"conteudo de exemplo").hexdigest()
        assert calculate_file_hash(arquivo) == esperado

    def test_calculate_file_hash_raises_for_missing_file(self, tmp_path: Path) -> None:
        """Deve levantar DataNotFoundError se o arquivo não existir."""
        with pytest.raises(DataNotFoundError):
            calculate_file_hash(tmp_path / "inexistente.txt")

    def test_calculate_text_hash_is_deterministic(self) -> None:
        """O hash do mesmo texto deve ser idêntico em chamadas repetidas."""
        assert calculate_text_hash("exemplo") == calculate_text_hash("exemplo")

    def test_calculate_text_hash_differs_for_different_text(self) -> None:
        """Textos diferentes devem produzir hashes diferentes."""
        assert calculate_text_hash("exemplo_a") != calculate_text_hash("exemplo_b")


class TestSeedEverything:
    """Testes da fixação de sementes aleatórias."""

    def test_same_seed_produces_same_random_sequence(self) -> None:
        """A mesma semente deve produzir a mesma sequência em random e NumPy."""
        seed_everything(123)
        sequencia_random_1 = [random.random() for _ in range(5)]
        sequencia_numpy_1 = np.random.rand(5).tolist()

        seed_everything(123)
        sequencia_random_2 = [random.random() for _ in range(5)]
        sequencia_numpy_2 = np.random.rand(5).tolist()

        assert sequencia_random_1 == sequencia_random_2
        assert sequencia_numpy_1 == sequencia_numpy_2


class TestText:
    """Testes das funções utilitárias genéricas de texto."""

    def test_normalize_whitespace_collapses_and_strips(self) -> None:
        """Deve colapsar espaços múltiplos e remover espaços nas bordas."""
        assert normalize_whitespace("  ola   mundo\n") == "ola mundo"

    def test_remove_accents_strips_diacritics(self) -> None:
        """Deve remover acentos preservando os caracteres-base."""
        assert remove_accents("análise de sentimentos") == "analise de sentimentos"

    def test_truncate_text_returns_original_when_short_enough(self) -> None:
        """Um texto já dentro do limite não deve ser alterado."""
        assert truncate_text("curto", 10) == "curto"

    def test_truncate_text_truncates_and_appends_suffix(self) -> None:
        """Um texto acima do limite deve ser truncado com o sufixo anexado."""
        assert truncate_text("um texto muito longo", 10) == "um text..."

    def test_truncate_text_rejects_max_length_smaller_than_suffix(self) -> None:
        """max_length menor que o sufixo deve levantar ValueError."""
        with pytest.raises(ValueError):
            truncate_text("texto", 2, suffix="...")


class TestTiming:
    """Testes de medição e formatação de duração."""

    def test_measure_execution_time_returns_non_negative_duration(self) -> None:
        """A duração medida deve ser não negativa."""
        with measure_execution_time() as tempo:
            _ = sum(range(1000))
        assert isinstance(tempo, ExecutionTiming)
        assert tempo.elapsed_seconds >= 0

    def test_format_duration_seconds_only(self) -> None:
        """Durações abaixo de um minuto devem ser formatadas apenas em segundos."""
        assert format_duration(3.4) == "3.40s"

    def test_format_duration_with_hours(self) -> None:
        """Durações acima de uma hora devem incluir horas, minutos e segundos."""
        assert format_duration(3723.4) == "1h 02min 03.40s"

    def test_format_duration_rejects_negative(self) -> None:
        """Duração negativa deve levantar ValueError."""
        with pytest.raises(ValueError):
            format_duration(-1.0)


class TestValidation:
    """Testes das validações genéricas de propósito geral."""

    def test_validate_file_exists_returns_path_when_valid(self, tmp_path: Path) -> None:
        """Deve retornar o mesmo caminho quando o arquivo existe."""
        arquivo = tmp_path / "exemplo.txt"
        arquivo.write_text("conteudo")
        assert validate_file_exists(arquivo) == arquivo

    def test_validate_file_exists_raises_for_missing_file(self, tmp_path: Path) -> None:
        """Deve levantar DataNotFoundError quando o arquivo não existe."""
        with pytest.raises(DataNotFoundError):
            validate_file_exists(tmp_path / "inexistente.txt")

    def test_validate_directory_exists_creates_when_requested(self, tmp_path: Path) -> None:
        """Deve criar o diretório quando create_if_missing=True."""
        diretorio = tmp_path / "novo_diretorio"
        resultado = validate_directory_exists(diretorio, create_if_missing=True)
        assert resultado.is_dir()

    def test_validate_directory_exists_raises_when_missing_and_not_created(self, tmp_path: Path) -> None:
        """Deve levantar DataNotFoundError quando o diretório não existe e create_if_missing=False."""
        with pytest.raises(DataNotFoundError):
            validate_directory_exists(tmp_path / "inexistente", create_if_missing=False)

    def test_validate_value_in_choices_accepts_valid_value(self) -> None:
        """Deve retornar o próprio valor quando ele pertence às escolhas."""
        assert validate_value_in_choices("f1_macro", ["f1_macro", "accuracy"]) == "f1_macro"

    def test_validate_value_in_choices_rejects_invalid_value(self) -> None:
        """Deve levantar ValueError quando o valor não pertence às escolhas."""
        with pytest.raises(ValueError):
            validate_value_in_choices("rmse", ["f1_macro", "accuracy"])

    def test_validate_not_empty_collection_accepts_non_empty(self) -> None:
        """Deve retornar a própria coleção quando ela não está vazia."""
        assert validate_not_empty_collection([1, 2, 3], collection_name="exemplo") == [1, 2, 3]

    def test_validate_not_empty_collection_rejects_empty(self) -> None:
        """Deve levantar EmptyDatasetError quando a coleção está vazia."""
        with pytest.raises(EmptyDatasetError):
            validate_not_empty_collection([], collection_name="exemplo")
