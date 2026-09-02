"""Testes dos utilitários de logging do projeto (``logging_utils``)."""

import logging
from datetime import date
from pathlib import Path

import pytest
from rich.logging import RichHandler

from logging_utils.formatter import (
    DEFAULT_FILE_LOG_FORMAT,
    build_console_log_formatter,
    build_file_log_formatter,
)
from logging_utils.handlers import (
    build_daily_log_file_path,
    create_console_handler,
    create_file_handler,
    remove_old_log_files,
)
from logging_utils.logger import (
    configure_logger_handlers,
    get_logger,
    set_third_party_loggers_level,
)
from logging_utils.timer import time_block


class TestFormatter:
    """Testes dos formatadores de log."""

    def test_build_file_log_formatter_uses_expected_format(self) -> None:
        """O formatador de arquivo deve usar o formato tabular padrão do projeto."""
        file_log_formatter = build_file_log_formatter()
        assert (
            file_log_formatter._fmt == DEFAULT_FILE_LOG_FORMAT
        )  # atributo interno do logging.Formatter

    def test_build_console_log_formatter_returns_formatter_instance(self) -> None:
        """Deve retornar uma instância válida de logging.Formatter."""
        assert isinstance(build_console_log_formatter(), logging.Formatter)


class TestHandlers:
    """Testes da construção de handlers de console e arquivo."""

    def test_create_console_handler_returns_rich_handler(self) -> None:
        """Deve retornar uma instância de RichHandler configurada com o nível informado."""
        handler = create_console_handler(level=logging.WARNING)
        assert isinstance(handler, RichHandler)
        assert handler.level == logging.WARNING

    def test_build_daily_log_file_path_formats_date_correctly(self) -> None:
        """O nome do arquivo deve conter a data no formato YYYY-MM-DD."""
        log_file_path = build_daily_log_file_path(Path("logs"), reference_date=date(2026, 1, 5))
        assert log_file_path.name == "log_2026-01-05.log"

    def test_create_file_handler_creates_directory_and_file(self, tmp_path: Path) -> None:
        """Deve criar o diretório de logs e o arquivo de log do dia."""
        log_directory = tmp_path / "logs"
        handler = create_file_handler(log_directory)
        try:
            assert log_directory.is_dir()
            assert Path(handler.baseFilename).is_file()
        finally:
            handler.close()

    def test_remove_old_log_files_keeps_only_recent_files(self, tmp_path: Path) -> None:
        """Deve manter apenas os arquivos mais recentes, respeitando backup_count."""
        for index in range(5):
            log_file = tmp_path / f"log_2026-01-0{index}.log"
            log_file.write_text("linha de log")

        removed_file_count = remove_old_log_files(tmp_path, backup_count=2)

        assert removed_file_count == 3
        assert len(list(tmp_path.glob("log_*.log"))) == 2

    def test_remove_old_log_files_returns_zero_for_missing_directory(self, tmp_path: Path) -> None:
        """Deve retornar 0 sem levantar exceção quando o diretório não existe."""
        assert remove_old_log_files(tmp_path / "inexistente", backup_count=5) == 0


class TestLogger:
    """Testes da fábrica de loggers e aplicação de handlers/níveis."""

    def test_get_logger_returns_logger_with_given_name(self) -> None:
        """Deve retornar um logger com o nome solicitado."""
        assert get_logger("sentimento_ptbr_llm.exemplo").name == "sentimento_ptbr_llm.exemplo"

    def test_configure_logger_handlers_replaces_existing_handlers(self) -> None:
        """Deve remover handlers antigos e aplicar apenas os novos, além do nível e propagação."""
        logger = logging.getLogger("sentimento_ptbr_llm.teste_configure")
        logger.addHandler(logging.NullHandler())

        new_handler = logging.NullHandler()
        configure_logger_handlers(logger, [new_handler], level=logging.DEBUG, propagate=True)

        assert logger.handlers == [new_handler]
        assert logger.level == logging.DEBUG
        assert logger.propagate is True

    def test_set_third_party_loggers_level_applies_to_all_names(self) -> None:
        """Deve aplicar o nível informado a todos os loggers de terceiros listados."""
        set_third_party_loggers_level(
            logging.ERROR, ["biblioteca_exemplo_a", "biblioteca_exemplo_b"]
        )
        assert logging.getLogger("biblioteca_exemplo_a").level == logging.ERROR
        assert logging.getLogger("biblioteca_exemplo_b").level == logging.ERROR


class TestTimeBlock:
    """Testes do gerenciador de contexto que loga a duração de um bloco."""

    def test_time_block_logs_start_and_completion_messages(
        self, log_capture_fixture: pytest.LogCaptureFixture
    ) -> None:
        """Deve registrar uma mensagem de início e uma de conclusão com a duração."""
        logger = logging.getLogger("sentimento_ptbr_llm.teste_time_block")
        with (
            log_capture_fixture.at_level(logging.INFO, logger=logger.name),
            time_block(logger, "bloco de teste"),
        ):
            _ = sum(range(100))

        log_messages = [log_record.message for log_record in log_capture_fixture.records]
        assert any("bloco de teste: iniciado" in log_message for log_message in log_messages)
        assert any("bloco de teste: concluído em" in log_message for log_message in log_messages)
