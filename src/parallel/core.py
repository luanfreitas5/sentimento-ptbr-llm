"""Motor genérico de execução paralela usado pelos demais módulos de ``parallel``.

Implementa o padrão comum a todas as etapas paralelizáveis do projeto
(pré-processamento, inferência, experimentos, scraping): distribuir a
aplicação de uma função sobre uma coleção de itens entre múltiplos
processos ou threads, isolando a falha de um item sem interromper o
restante do lote (consistente com a ressalva de ``PERF203`` documentada em
``pyproject.toml``).
"""

import logging
from collections.abc import Callable, Iterable
from concurrent.futures import Executor, ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from utils.timing import format_duration, measure_execution_time

logger = logging.getLogger(__name__)

ItemType = TypeVar("ItemType")
ResultType = TypeVar("ResultType")


@dataclass
class ParallelTaskFailure(Generic[ItemType]):
    """Falha isolada de um único item durante a execução paralela.

    Parameters
    ----------
    item : ItemType
        Item de entrada cujo processamento falhou.
    error : Exception
        Exceção levantada durante o processamento do item.
    """

    item: ItemType
    error: Exception


@dataclass
class ParallelExecutionResult(Generic[ItemType, ResultType]):
    """Resultado agregado de uma execução paralela sobre múltiplos itens.

    Parameters
    ----------
    successes : list[ResultType]
        Resultados dos itens processados com sucesso.
    failures : list[ParallelTaskFailure[ItemType]]
        Itens cujo processamento falhou, com a exceção correspondente.
    elapsed_seconds : float
        Tempo total de execução, em segundos.
    """

    successes: list[ResultType] = field(default_factory=list)
    failures: list[ParallelTaskFailure[ItemType]] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def total_items(self) -> int:
        """Número total de itens processados (sucesso + falha).

        Returns
        -------
        int
            Soma do número de sucessos e falhas.
        """
        return len(self.successes) + len(self.failures)

    @property
    def success_rate(self) -> float:
        """Proporção de itens processados com sucesso.

        Returns
        -------
        float
            Valor entre 0.0 e 1.0. Retorna 0.0 quando não há itens.
        """
        if self.total_items == 0:
            return 0.0
        return len(self.successes) / self.total_items


def _build_progress_bar() -> Progress:
    """Monta a barra de progresso padrão do projeto para tarefas paralelas.

    Returns
    -------
    Progress
        Instância de ``rich.progress.Progress`` configurada com as colunas
        padrão definidas em ``CLAUDE.md`` (spinner, descrição, barra,
        contagem, percentual e tempos decorrido/restante).
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )


def execute_parallel_tasks(
    func: Callable[[ItemType], ResultType],
    items: Iterable[ItemType],
    *,
    executor_class: type[Executor] = ThreadPoolExecutor,
    max_workers: int | None = None,
    task_description: str = "Processando itens em paralelo",
    show_progress: bool = True,
) -> ParallelExecutionResult[ItemType, ResultType]:
    """Aplica uma função a múltiplos itens em paralelo, isolando falhas por item.

    Cada item é processado independentemente: uma exceção levantada para um
    item específico é registrada em log e armazenada em
    :attr:`ParallelExecutionResult.failures`, sem interromper o
    processamento dos demais itens do lote.

    Parameters
    ----------
    func : Callable[[ItemType], ResultType]
        Função aplicada a cada item. Ao usar ``executor_class=ProcessPoolExecutor``,
        deve ser importável no nível de módulo (não pode ser uma função
        local ou lambda), pois é serializada para os processos filhos.
    items : Iterable[ItemType]
        Itens a serem processados. É consumido integralmente (materializado
        em lista) antes do início da execução, para permitir o cálculo do
        total de itens exibido na barra de progresso.
    executor_class : type[Executor], optional
        Classe do executor usada para paralelizar o trabalho —
        ``ThreadPoolExecutor`` para tarefas ligadas a I/O ou
        ``ProcessPoolExecutor`` para tarefas ligadas a CPU, by default
        ``ThreadPoolExecutor``.
    max_workers : int | None, optional
        Número máximo de workers (processos ou threads) usados, by default
        None (o executor escolhe automaticamente).
    task_description : str, optional
        Descrição exibida na barra de progresso e nas mensagens de log, by
        default "Processando itens em paralelo".
    show_progress : bool, optional
        Se ``True``, exibe uma barra de progresso no console, by default
        True.

    Returns
    -------
    ParallelExecutionResult[ItemType, ResultType]
        Resultado agregando sucessos, falhas isoladas por item e o tempo
        total de execução.

    Raises
    ------
    ValueError
        Se ``max_workers`` for informado e for menor que 1.

    Examples
    --------
    >>> resultado = execute_parallel_tasks(
    ...     str.upper, ["a", "b"], show_progress=False
    ... )  # doctest: +SKIP
    >>> sorted(resultado.successes)  # doctest: +SKIP
    ['A', 'B']
    """
    if max_workers is not None and max_workers < 1:
        raise ValueError(f"max_workers deve ser >= 1, recebido: {max_workers}")

    items_list = list(items)
    result: ParallelExecutionResult[ItemType, ResultType] = ParallelExecutionResult()

    if not items_list:
        logger.warning("Nenhum item recebido para '%s'; nada a processar", task_description)
        return result

    progress = _build_progress_bar() if show_progress else None
    progress_context = progress if progress is not None else nullcontext()

    with measure_execution_time() as tempo, executor_class(max_workers=max_workers) as executor:
        futures = {executor.submit(func, item): item for item in items_list}
        with progress_context:
            task_id = (
                progress.add_task(task_description, total=len(items_list)) if progress else None
            )
            for future in as_completed(futures):
                item = futures[future]
                try:
                    result.successes.append(future.result())
                except (
                    Exception
                ) as exception:  # captura ampla e proposital: isola a falha de um único item
                    logger.exception(
                        "Falha ao processar item em '%s'", task_description
                    )
                    result.failures.append(ParallelTaskFailure(item=item, error=exception))
                if progress:
                    progress.update(task_id, advance=1)

    result.elapsed_seconds = tempo.elapsed_seconds
    logger.info(
        "'%s' concluída: %d sucesso(s), %d falha(s) em %s",
        task_description,
        len(result.successes),
        len(result.failures),
        format_duration(result.elapsed_seconds),
    )
    return result
