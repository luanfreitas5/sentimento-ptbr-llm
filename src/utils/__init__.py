"""Funções utilitárias de propósito geral do projeto ``sentimento-ptbr-llm``.

Reúne helpers reutilizáveis, sem conhecimento do domínio de análise de
sentimentos, usados por múltiplos módulos do projeto.

Modules
-------
decorators
    Decoradores genéricos (log de tempo de execução, retentativa).
hashing
    Cálculo de hash de arquivos e strings para rastreabilidade.
seed
    Fixação de sementes aleatórias para reprodutibilidade.
text
    Manipulação genérica de texto (espaçamento, acentos, truncamento).
timing
    Medição e formatação de duração de execução.
validation
    Validações genéricas usadas nos limites do sistema.
"""

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

__all__: list[str] = [
    "log_execution_time",
    "retry_on_exception",
    "calculate_file_hash",
    "calculate_text_hash",
    "seed_everything",
    "normalize_whitespace",
    "remove_accents",
    "truncate_text",
    "ExecutionTiming",
    "format_duration",
    "measure_execution_time",
    "validate_directory_exists",
    "validate_file_exists",
    "validate_not_empty_collection",
    "validate_value_in_choices",
]
