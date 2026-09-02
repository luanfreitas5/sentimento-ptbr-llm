"""Paralelização da execução de múltiplos experimentos de treino/avaliação.

Usa múltiplos processos (``ProcessPoolExecutor``), adequado para rodar
experimentos com diferentes configurações de modelo (ver
``src/experiment/`` e ``src/training/``) de forma isolada e ligada a CPU.
"""

from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from typing import TypeVar

from parallel.core import ParallelExecutionResult, execute_parallel_tasks

ExperimentConfigType = TypeVar("ExperimentConfigType")
ExperimentResultType = TypeVar("ExperimentResultType")


def run_parallel_experiments(
    run_experiment_func: Callable[[ExperimentConfigType], ExperimentResultType],
    experiment_configs: Iterable[ExperimentConfigType],
    *,
    max_workers: int | None = None,
    show_progress: bool = True,
) -> ParallelExecutionResult[ExperimentConfigType, ExperimentResultType]:
    """Executa múltiplos experimentos (configurações de treino/avaliação) em paralelo.

    Cada configuração de experimento (ex.: combinação de hiperparâmetros ou
    de modelo) é executada em um processo separado, isolando a falha de um
    experimento específico sem interromper os demais.

    Parameters
    ----------
    run_experiment_func : Callable[[ExperimentConfigType], ExperimentResultType]
        Função que executa um único experimento a partir de sua
        configuração (ex.: treina e avalia um modelo, retornando métricas).
    experiment_configs : Iterable[ExperimentConfigType]
        Configurações dos experimentos a serem executados.
    max_workers : int | None, optional
        Número máximo de processos usados, by default None (o executor
        escolhe automaticamente com base nos núcleos disponíveis).
    show_progress : bool, optional
        Se ``True``, exibe uma barra de progresso no console, by default
        True.

    Returns
    -------
    ParallelExecutionResult[ExperimentConfigType, ExperimentResultType]
        Resultados dos experimentos bem-sucedidos e falhas isoladas por
        configuração, cada uma preservando a configuração que causou o
        erro.

    Examples
    --------
    >>> resultado = run_parallel_experiments(
    ...     lambda cfg: cfg["lr"] * 2, [{"lr": 0.1}, {"lr": 0.2}]
    ... )  # doctest: +SKIP
    """
    return execute_parallel_tasks(
        run_experiment_func,
        experiment_configs,
        executor_class=ProcessPoolExecutor,
        max_workers=max_workers,
        task_description="Execução paralela de experimentos",
        show_progress=show_progress,
    )
