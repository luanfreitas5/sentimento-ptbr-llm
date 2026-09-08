"""Testes dos utilitários de execução paralela do projeto."""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import polars as pl
import pytest

from io_utils.parquet import write_parquet
from parallel.core import ParallelExecutionResult, execute_parallel_tasks
from parallel.data_loading import run_parallel_parquet_loading
from parallel.experiments import run_parallel_experiments
from parallel.inference import run_parallel_predictions
from parallel.labeling import run_parallel_sentiment_labeling
from parallel.preprocessing import run_parallel_text_cleaning
from parallel.scraping import run_parallel_scraping

# Funções auxiliares no nível de módulo: exigido pelo ProcessPoolExecutor, que
# serializa (pickle) a função a ser executada nos processos filhos — funções
# locais/lambdas não são serializáveis.


def _double(value: int) -> int:
    """Dobra um número inteiro."""
    return value * 2


def _raise_if_negative(value: int) -> int:
    """Retorna o valor recebido, ou levanta ValueError se for negativo."""
    if value < 0:
        raise ValueError(f"valor negativo: {value}")
    return value


def _fail_on_specific_text(text: str) -> str:
    """Retorna o texto em maiúsculas, ou levanta ValueError para o texto 'erro'."""
    if text == "erro":
        raise ValueError("texto inválido")
    return text.upper()


def _fail_on_empty_string(text: str) -> int:
    """Retorna o tamanho do texto, ou levanta ValueError se ele estiver vazio."""
    if not text:
        raise ValueError("texto vazio")
    return len(text)


def _run_experiment(config: dict[str, float]) -> dict[str, float]:
    """Simula a execução de um experimento, retornando uma métrica derivada."""
    return {"score": config["lr"] * 10}


def _fail_experiment_on_negative_lr(config: dict[str, float]) -> dict[str, float]:
    """Simula um experimento que falha para taxas de aprendizado negativas."""
    if config["lr"] < 0:
        raise ValueError("lr inválido")
    return {"score": config["lr"]}


def _fail_on_specific_query(query: str) -> str:
    """Retorna a consulta em maiúsculas, ou levanta ValueError para 'falha'."""
    if query == "falha":
        raise ValueError("consulta inválida")
    return query.upper()


def _label_or_fail(text: str) -> str:
    """Retorna o texto em maiúsculas, ou levanta ValueError para o texto 'erro'."""
    if text == "erro":
        raise ValueError("texto inválido")
    return text.upper()


def _fail_on_multiples_of_five(value: int) -> int:
    """Retorna o valor recebido, ou levanta ValueError se for múltiplo de 5."""
    if value % 5 == 0:
        raise ValueError(f"valor inválido (múltiplo de 5): {value}")
    return value * 2


class TestExecuteParallelTasks:
    """Testes do motor genérico de execução paralela (``parallel.core``)."""

    def test_returns_all_successes_when_no_failures(self) -> None:
        """Todos os itens processados com sucesso devem aparecer em `successes`."""
        result = execute_parallel_tasks(_double, [1, 2, 3], show_progress=False, max_workers=2)
        assert sorted(result.successes) == [2, 4, 6]
        assert result.failures == []
        assert result.total_items == 3
        assert result.success_rate == 1.0

    def test_isolates_failure_per_item(self) -> None:
        """A falha de um item não deve interromper o processamento dos demais."""
        result = execute_parallel_tasks(
            _raise_if_negative, [1, -1, 2], show_progress=False, max_workers=2
        )
        assert sorted(result.successes) == [1, 2]
        assert len(result.failures) == 1
        assert result.failures[0].item == -1
        assert isinstance(result.failures[0].error, ValueError)
        assert result.total_items == 3
        assert result.success_rate == pytest.approx(2 / 3)

    def test_returns_empty_result_for_no_items(self) -> None:
        """Uma coleção vazia de itens deve retornar um resultado vazio, sem erro."""
        result: ParallelExecutionResult[int, int] = execute_parallel_tasks(
            _double, [], show_progress=False
        )
        assert result.successes == []
        assert result.failures == []
        assert result.total_items == 0
        assert result.success_rate == 0.0
        assert result.elapsed_seconds == 0.0

    def test_rejects_invalid_max_workers(self) -> None:
        """max_workers menor que 1 deve levantar ValueError antes de iniciar a execução."""
        with pytest.raises(ValueError):
            execute_parallel_tasks(_double, [1], max_workers=0, show_progress=False)

    def test_elapsed_seconds_is_non_negative(self) -> None:
        """O tempo total de execução medido deve ser não negativo."""
        result = execute_parallel_tasks(_double, [1, 2], show_progress=False)
        assert result.elapsed_seconds >= 0

    def test_works_with_process_pool_executor(self) -> None:
        """Deve funcionar corretamente com ProcessPoolExecutor, exigindo função serializável."""
        result = execute_parallel_tasks(
            _double,
            [1, 2, 3],
            executor_class=ProcessPoolExecutor,
            max_workers=2,
            show_progress=False,
        )
        assert sorted(result.successes) == [2, 4, 6]
        assert result.failures == []

    def test_works_with_thread_pool_executor(self) -> None:
        """Deve funcionar corretamente com ThreadPoolExecutor (executor padrão)."""
        result = execute_parallel_tasks(
            _double,
            [1, 2, 3],
            executor_class=ThreadPoolExecutor,
            max_workers=2,
            show_progress=False,
        )
        assert sorted(result.successes) == [2, 4, 6]


class TestRunParallelTextCleaning:
    """Testes da paralelização de limpeza de texto (``parallel.preprocessing``)."""

    def test_cleans_all_texts_successfully(self) -> None:
        """Todos os textos devem ser limpos com sucesso quando não há erro."""
        result = run_parallel_text_cleaning(
            str.strip, ["  a  ", " b "], show_progress=False, max_workers=1
        )
        assert sorted(result.successes) == ["a", "b"]
        assert result.failures == []

    def test_isolates_failure_per_text(self) -> None:
        """A falha de limpeza de um texto não deve interromper os demais."""
        result = run_parallel_text_cleaning(
            _fail_on_specific_text, ["ok", "erro"], show_progress=False, max_workers=1
        )
        assert result.successes == ["OK"]
        assert len(result.failures) == 1
        assert result.failures[0].item == "erro"


class TestRunParallelPredictions:
    """Testes da paralelização de inferência (``parallel.inference``)."""

    def test_predicts_all_items_successfully(self) -> None:
        """Todos os itens devem ser preditos com sucesso quando não há erro."""
        result = run_parallel_predictions(len, ["ab", "cde"], show_progress=False, max_workers=2)
        assert sorted(result.successes) == [2, 3]
        assert result.failures == []

    def test_isolates_failure_per_item(self) -> None:
        """A falha de predição de um item não deve interromper os demais."""
        result = run_parallel_predictions(
            _fail_on_empty_string, ["ab", "", "c"], show_progress=False, max_workers=2
        )
        assert sorted(result.successes) == [1, 2]
        assert len(result.failures) == 1
        assert result.failures[0].item == ""


class TestRunParallelExperiments:
    """Testes da paralelização de experimentos (``parallel.experiments``)."""

    def test_runs_all_experiments_successfully(self) -> None:
        """Todos os experimentos devem ser executados com sucesso quando não há erro."""
        result = run_parallel_experiments(
            _run_experiment,
            [{"lr": 0.1}, {"lr": 0.2}],
            show_progress=False,
            max_workers=1,
        )
        scores = sorted(metrics["score"] for metrics in result.successes)
        assert scores == pytest.approx([1.0, 2.0])
        assert result.failures == []

    def test_isolates_failed_experiment(self) -> None:
        """A falha de um experimento não deve interromper os demais."""
        failing_config = {"lr": -1.0}
        result = run_parallel_experiments(
            _fail_experiment_on_negative_lr,
            [{"lr": 0.1}, failing_config],
            show_progress=False,
            max_workers=1,
        )
        assert len(result.successes) == 1
        assert len(result.failures) == 1
        assert result.failures[0].item == failing_config


class TestRunParallelScraping:
    """Testes da paralelização de coleta de dados (``parallel.scraping``)."""

    def test_scrapes_all_queries_successfully(self) -> None:
        """Todas as consultas devem ser coletadas com sucesso quando não há erro."""
        result = run_parallel_scraping(
            str.upper, ["python", "nlp"], show_progress=False, max_workers=2
        )
        assert sorted(result.successes) == ["NLP", "PYTHON"]
        assert result.failures == []

    def test_isolates_failure_per_query(self) -> None:
        """A falha de coleta de uma consulta não deve interromper as demais."""
        result = run_parallel_scraping(
            _fail_on_specific_query, ["ok", "falha"], show_progress=False, max_workers=2
        )
        assert result.successes == ["OK"]
        assert len(result.failures) == 1
        assert result.failures[0].item == "falha"


class TestRunParallelParquetLoading:
    """Testes da leitura paralela de lote Parquet (``parallel.data_loading``)."""

    def test_reads_all_files_successfully(self, tmp_path: Path) -> None:
        """Todos os arquivos devem ser lidos com sucesso quando não há erro."""
        file_a = tmp_path / "a.parquet"
        file_b = tmp_path / "b.parquet"
        write_parquet(pl.DataFrame({"valor": [1]}), file_a)
        write_parquet(pl.DataFrame({"valor": [2]}), file_b)

        result = run_parallel_parquet_loading([file_a, file_b], show_progress=False, max_workers=2)

        assert result.failures == []
        valores = sorted(df["valor"].to_list()[0] for df in result.successes)
        assert valores == [1, 2]

    def test_isolates_failure_for_missing_file(self, tmp_path: Path) -> None:
        """A falha de leitura de um arquivo não deve interromper a leitura dos demais."""
        file_ok = tmp_path / "ok.parquet"
        write_parquet(pl.DataFrame({"valor": [1]}), file_ok)
        missing_file = tmp_path / "inexistente.parquet"

        result = run_parallel_parquet_loading(
            [file_ok, missing_file], show_progress=False, max_workers=2
        )

        assert len(result.successes) == 1
        assert len(result.failures) == 1
        assert result.failures[0].item == missing_file


class TestRunParallelSentimentLabeling:
    """Testes da paralelização de rotulagem de sentimento (``parallel.labeling``)."""

    def test_labels_all_items_successfully(self) -> None:
        """Todos os itens devem ser rotulados com sucesso quando não há erro."""
        result = run_parallel_sentiment_labeling(
            str.upper, ["a", "b"], show_progress=False, max_workers=1
        )
        assert sorted(result.successes) == ["A", "B"]
        assert result.failures == []

    def test_isolates_failure_per_item(self) -> None:
        """A falha de rotulagem de um item não deve interromper os demais."""
        result = run_parallel_sentiment_labeling(
            _label_or_fail, ["ok", "erro"], show_progress=False, max_workers=1
        )
        assert result.successes == ["OK"]
        assert len(result.failures) == 1
        assert result.failures[0].item == "erro"


class TestExecuteParallelTasksChunkSize:
    """Testes do agrupamento em lotes (``chunk_size``) de ``execute_parallel_tasks``.

    Cobre a correção do Achado Crítico 1 da revisão final: submeter uma
    ``Future`` por item é mais lento que a execução serial para itens
    baratos, e piora com a escala (custo fixo de IPC dominando o trabalho
    real) — ``chunk_size`` agrupa vários itens por ``Future``, mantendo o
    isolamento de falha por item.
    """

    def test_chunked_execution_matches_unchunked_successes(self) -> None:
        """A execução em lotes deve produzir exatamente os mesmos sucessos que a não agrupada."""
        items = list(range(1, 51))
        unchunked = execute_parallel_tasks(_double, items, show_progress=False, max_workers=2)
        chunked = execute_parallel_tasks(
            _double, items, show_progress=False, max_workers=2, chunk_size=7
        )
        assert sorted(chunked.successes) == sorted(unchunked.successes)
        assert chunked.failures == []

    def test_chunked_execution_with_process_pool_matches_unchunked(self) -> None:
        """O agrupamento em lotes deve funcionar com ``ProcessPoolExecutor`` (uso real)."""
        items = list(range(1, 31))
        chunked = execute_parallel_tasks(
            _double,
            items,
            executor_class=ProcessPoolExecutor,
            show_progress=False,
            max_workers=2,
            chunk_size=4,
        )
        assert sorted(chunked.successes) == sorted(item * 2 for item in items)
        assert chunked.failures == []

    def test_isolates_failure_within_a_chunk_without_failing_the_chunk(self) -> None:
        """Um item que falha não deve derrubar os demais itens do mesmo lote nem de outros lotes."""
        items = list(range(1, 21))  # múltiplos de 5: 5, 10, 15, 20 (4 falhas esperadas)
        result = execute_parallel_tasks(
            _fail_on_multiples_of_five,
            items,
            show_progress=False,
            max_workers=2,
            chunk_size=5,
        )
        expected_failures = {5, 10, 15, 20}
        expected_successes = sorted(value * 2 for value in items if value not in expected_failures)
        assert sorted(result.successes) == expected_successes
        assert {failure.item for failure in result.failures} == expected_failures
        assert all(isinstance(failure.error, ValueError) for failure in result.failures)
        assert result.total_items == len(items)

    def test_rejects_invalid_chunk_size(self) -> None:
        """``chunk_size`` menor que 1 deve levantar ``ValueError`` antes de iniciar a execução."""
        with pytest.raises(ValueError):
            execute_parallel_tasks(_double, [1, 2], chunk_size=0, show_progress=False)

    def test_chunk_size_none_preserves_default_behavior(self) -> None:
        """``chunk_size=None`` (padrão) deve manter o comportamento original, sem agrupamento."""
        result = execute_parallel_tasks(_double, [1, 2, 3], show_progress=False, max_workers=2)
        assert sorted(result.successes) == [2, 4, 6]
        assert result.failures == []
