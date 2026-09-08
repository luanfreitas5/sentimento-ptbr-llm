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
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
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


def _collect_results(
    futures: dict[Future[ResultType], ItemType],
    result: ParallelExecutionResult[ItemType, ResultType],
    progress: Progress | None,
    task_id: TaskID | None,
    task_description: str,
) -> None:
    """Aguarda a conclusão das tarefas e agrega sucessos e falhas por item.

    Parameters
    ----------
    futures : dict[Future[ResultType], ItemType]
        Mapeamento de cada ``Future`` submetida ao item de origem correspondente.
    result : ParallelExecutionResult[ItemType, ResultType]
        Objeto de resultado, atualizado in-place com cada sucesso ou falha.
    progress : Progress | None
        Barra de progresso a ser atualizada a cada item concluído, ou ``None``
        se a exibição de progresso estiver desabilitada.
    task_id : TaskID | None
        Identificador da tarefa na barra de progresso, ou ``None`` se a
        exibição de progresso estiver desabilitada.
    task_description : str
        Descrição da tarefa, usada nas mensagens de log de falha.
    """
    for future in as_completed(futures):
        item = futures[future]
        try:
            result.successes.append(future.result())
        except Exception as exception:  # captura ampla e proposital: isola a falha de um único item
            logger.exception("Falha ao processar item em '%s'", task_description)
            result.failures.append(ParallelTaskFailure(item=item, error=exception))
        if progress is not None and task_id is not None:
            progress.update(task_id, advance=1)


def _split_into_chunks(items_list: list[ItemType], chunk_size: int) -> list[list[ItemType]]:
    """Divide uma lista em sublistas (lotes/chunks) de tamanho no máximo ``chunk_size``.

    Parameters
    ----------
    items_list : list[ItemType]
        Lista completa de itens a dividir.
    chunk_size : int
        Tamanho máximo de cada lote.

    Returns
    -------
    list[list[ItemType]]
        Lista de lotes, na mesma ordem dos itens originais.
    """
    return [
        items_list[start : start + chunk_size] for start in range(0, len(items_list), chunk_size)
    ]


def _run_chunk_of_items(
    func: Callable[[ItemType], ResultType], chunk_items: list[ItemType]
) -> list[tuple[ItemType, ResultType | None, Exception | None]]:
    """Aplica ``func`` a cada item de um lote (chunk), isolando a falha de cada item.

    Função de nível de módulo (não uma closure/lambda), exigida para ser
    serializável e submetida como uma única tarefa por ``Future`` ao
    executor (ver ``execute_parallel_tasks`` com ``chunk_size`` informado):
    amortiza o custo fixo de IPC por item (dominante para tarefas baratas —
    ex.: contagem léxica/regex em um único tweet), mantendo o isolamento de
    falha por item dentro do próprio lote.

    Parameters
    ----------
    func : Callable[[ItemType], ResultType]
        Função aplicada a cada item do lote.
    chunk_items : list[ItemType]
        Itens do lote a processar.

    Returns
    -------
    list[tuple[ItemType, ResultType | None, Exception | None]]
        Uma tripla por item de entrada: ``(item, resultado, None)`` em caso
        de sucesso, ou ``(item, None, exceção)`` em caso de falha.
    """
    outcomes: list[tuple[ItemType, ResultType | None, Exception | None]] = []
    for item in chunk_items:
        try:
            outcomes.append((item, func(item), None))
        except Exception as exception:  # captura ampla e proposital: isola a falha de um único item
            outcomes.append((item, None, exception))
    return outcomes


def _collect_chunked_results(
    futures: dict[
        Future[list[tuple[ItemType, ResultType | None, Exception | None]]], list[ItemType]
    ],
    result: ParallelExecutionResult[ItemType, ResultType],
    progress: Progress | None,
    task_id: TaskID | None,
    task_description: str,
) -> None:
    """Aguarda a conclusão dos lotes (chunks) e agrega sucessos/falhas por item.

    A barra de progresso avança pelo tamanho do lote de uma só vez, quando
    o ``Future`` do lote inteiro é concluído — os itens de um mesmo lote
    terminam atomicamente do ponto de vista do executor, não um a um.

    Parameters
    ----------
    futures : dict[Future[...], list[ItemType]]
        Mapeamento de cada ``Future`` de lote (retornando a lista de triplas
        produzida por :func:`_run_chunk_of_items`) à lista de itens
        originais daquele lote.
    result : ParallelExecutionResult[ItemType, ResultType]
        Objeto de resultado, atualizado in-place com cada sucesso ou falha.
    progress : Progress | None
        Barra de progresso a ser atualizada a cada lote concluído, ou
        ``None`` se a exibição de progresso estiver desabilitada.
    task_id : TaskID | None
        Identificador da tarefa na barra de progresso, ou ``None`` se a
        exibição de progresso estiver desabilitada.
    task_description : str
        Descrição da tarefa, usada nas mensagens de log de falha.
    """
    for future in as_completed(futures):
        chunk_items = futures[future]
        try:
            chunk_outcomes = future.result()
        except Exception as exception:  # captura ampla e proposital: isola a falha do lote inteiro
            logger.exception("Falha ao processar lote (chunk) em '%s'", task_description)
            for item in chunk_items:
                result.failures.append(ParallelTaskFailure(item=item, error=exception))
        else:
            for item, item_result, error in chunk_outcomes:
                if error is not None:
                    result.failures.append(ParallelTaskFailure(item=item, error=error))
                else:
                    result.successes.append(item_result)  # type: ignore[arg-type]
        if progress is not None and task_id is not None:
            progress.update(task_id, advance=len(chunk_items))


def execute_parallel_tasks(
    func: Callable[[ItemType], ResultType],
    items: Iterable[ItemType],
    *,
    executor_class: type[ThreadPoolExecutor] | type[ProcessPoolExecutor] = ThreadPoolExecutor,
    max_workers: int | None = None,
    task_description: str = "Processando itens em paralelo",
    show_progress: bool = True,
    chunk_size: int | None = None,
) -> ParallelExecutionResult[ItemType, ResultType]:
    """Aplica uma função a múltiplos itens em paralelo, isolando falhas por item.

    Cada item é processado independentemente: uma exceção levantada para um
    item específico é registrada em log e armazenada em
    :attr:`ParallelExecutionResult.failures`, sem interromper o
    processamento dos demais itens do lote.

    Por padrão (``chunk_size=None``), submete uma ``Future`` por item —
    adequado quando o trabalho por item já é caro o bastante para amortizar
    o custo fixo de round-trip do executor (serialização/IPC). Para itens
    muito baratos (ex.: contagem léxica/regex em um único tweet), esse
    custo fixo passa a dominar e a paralelização por item fica mais lenta
    que a execução serial, piorando com a escala — nesse caso, informe
    ``chunk_size`` para agrupar vários itens por ``Future`` (ver
    :func:`_run_chunk_of_items`), amortizando o custo fixo por lote em vez
    de por item, mantendo o isolamento de falha por item dentro do lote.

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
    executor_class : type[ThreadPoolExecutor] | type[ProcessPoolExecutor], optional
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
    chunk_size : int | None, optional
        Se informado (e maior que 1), agrupa os itens em lotes desse
        tamanho e submete uma ``Future`` por lote em vez de uma por item,
        reduzindo o overhead de IPC/serialização por item — recomendado
        para itens de processamento muito barato em grande volume. Por
        padrão ``None``: comportamento idêntico ao existente antes deste
        parâmetro (uma ``Future`` por item).

    Returns
    -------
    ParallelExecutionResult[ItemType, ResultType]
        Resultado agregando sucessos, falhas isoladas por item e o tempo
        total de execução.

    Raises
    ------
    ValueError
        Se ``max_workers`` for informado e for menor que 1, ou se
        ``chunk_size`` for informado e for menor que 1.

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
    if chunk_size is not None and chunk_size < 1:
        raise ValueError(f"chunk_size deve ser >= 1, recebido: {chunk_size}")

    items_list = list(items)
    result: ParallelExecutionResult[ItemType, ResultType] = ParallelExecutionResult()

    if not items_list:
        logger.warning("Nenhum item recebido para '%s'; nada a processar", task_description)
        return result

    progress = _build_progress_bar() if show_progress else None
    progress_context = progress if progress is not None else nullcontext()

    with measure_execution_time() as tempo, executor_class(max_workers=max_workers) as executor:
        if chunk_size is None:
            futures = {executor.submit(func, item): item for item in items_list}
            with progress_context:
                task_id = (
                    progress.add_task(task_description, total=len(items_list)) if progress else None
                )
                _collect_results(futures, result, progress, task_id, task_description)
        else:
            chunks = _split_into_chunks(items_list, chunk_size)
            chunk_futures = {
                executor.submit(_run_chunk_of_items, func, chunk): chunk for chunk in chunks
            }
            with progress_context:
                task_id = (
                    progress.add_task(task_description, total=len(items_list)) if progress else None
                )
                _collect_chunked_results(chunk_futures, result, progress, task_id, task_description)

    result.elapsed_seconds = tempo.elapsed_seconds
    logger.info(
        "'%s' concluída: %d sucesso(s), %d falha(s) em %s",
        task_description,
        len(result.successes),
        len(result.failures),
        format_duration(result.elapsed_seconds),
    )
    return result
