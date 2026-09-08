"""Execução paralela e concorrente de etapas do pipeline.

Camada de utilitários genéricos de paralelismo (``concurrent.futures``),
usada pelos módulos de pré-processamento, rotulagem, inferência,
experimentos, coleta de dados e carregamento em lote para distribuir
trabalho entre processos (tarefas ligadas a CPU) ou threads (tarefas
ligadas a I/O), isolando a falha de um item sem interromper o restante do
lote.

Modules
-------
core
    Motor genérico de execução paralela (``execute_parallel_tasks``) e os
    tipos de resultado (``ParallelExecutionResult``, ``ParallelTaskFailure``)
    compartilhados pelos demais módulos.
data_loading
    Leitura paralela de lotes de arquivos Parquet.
experiments
    Execução paralela de múltiplos experimentos/configurações de treino.
inference
    Execução paralela de inferência/predição de modelos.
labeling
    Execução paralela da rotulagem automática de sentimento.
preprocessing
    Execução paralela de limpeza e normalização de texto.
scraping
    Execução paralela de coleta de dados (scraping).
"""

from parallel.core import ParallelExecutionResult, ParallelTaskFailure, execute_parallel_tasks
from parallel.data_loading import run_parallel_parquet_loading
from parallel.experiments import run_parallel_experiments
from parallel.inference import run_parallel_predictions
from parallel.labeling import run_parallel_sentiment_labeling
from parallel.preprocessing import run_parallel_text_cleaning
from parallel.scraping import run_parallel_scraping

__all__: list[str] = [
    "ParallelExecutionResult",
    "ParallelTaskFailure",
    "execute_parallel_tasks",
    "run_parallel_experiments",
    "run_parallel_parquet_loading",
    "run_parallel_predictions",
    "run_parallel_scraping",
    "run_parallel_sentiment_labeling",
    "run_parallel_text_cleaning",
]
