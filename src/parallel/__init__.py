"""Execução paralela e concorrente de etapas do pipeline.

Camada de utilitários genéricos de paralelismo (``concurrent.futures``),
usada pelos módulos de pré-processamento, inferência, experimentos e coleta
de dados para distribuir trabalho entre processos (tarefas ligadas a CPU) ou
threads (tarefas ligadas a I/O), isolando a falha de um item sem interromper
o restante do lote.

Modules
-------
core
    Motor genérico de execução paralela (``execute_parallel_tasks``) e os
    tipos de resultado (``ParallelExecutionResult``, ``ParallelTaskFailure``)
    compartilhados pelos demais módulos.
experiments
    Execução paralela de múltiplos experimentos/configurações de treino.
inference
    Execução paralela de inferência/predição de modelos.
preprocessing
    Execução paralela de limpeza e normalização de texto.
scraping
    Execução paralela de coleta de dados (scraping).
"""

from parallel.core import ParallelExecutionResult, ParallelTaskFailure, execute_parallel_tasks
from parallel.experiments import run_parallel_experiments
from parallel.inference import run_parallel_predictions
from parallel.preprocessing import run_parallel_text_cleaning
from parallel.scraping import run_parallel_scraping

__all__: list[str] = [
    "ParallelExecutionResult",
    "ParallelTaskFailure",
    "execute_parallel_tasks",
    "run_parallel_experiments",
    "run_parallel_predictions",
    "run_parallel_scraping",
    "run_parallel_text_cleaning",
]
