"""Barras de progresso ``rich`` compartilhadas pelos módulos de ``hypothesaes``.

Substitui o ``tqdm`` do código original do HypotheSAEs pelo padrão do projeto
(``rich.progress`` com as colunas definidas em ``CLAUDE.md``, "Progress Bars").
O ``rich`` permite apenas uma barra ativa por vez (``LiveError`` ao aninhar
duas instâncias de ``Progress``); por isso :func:`open_progress` reaproveita a
barra já aberta, o que permite laços aninhados (ex.: fragmentos > lotes de
embeddings) exibidos como tarefas irmãs na mesma tela.

Functions
---------
open_progress
    Context manager que abre (ou reaproveita) a barra de progresso do projeto.
iterate_with_progress
    Envolve um iterável exibindo o progresso, como substituto de ``tqdm(...)``.
"""

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import TypeVar

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

T = TypeVar("T")

_active_progress: Progress | None = None


def _build_progress_bar(*, disable: bool) -> Progress:
    """Monta a barra de progresso padrão do projeto.

    Parameters
    ----------
    disable : bool
        Se ``True``, a barra não renderiza nada (mantém a mesma API).

    Returns
    -------
    Progress
        Instância de ``rich.progress.Progress`` com spinner, descrição, barra,
        contagem, percentual e tempos decorrido/restante.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        disable=disable,
    )


@contextmanager
def open_progress(*, disable: bool = False) -> Iterator[Progress]:
    """Abre a barra de progresso do projeto, reaproveitando uma já ativa.

    Parameters
    ----------
    disable : bool, optional
        Se ``True``, não exibe progresso, by default False. Ignorado quando já
        existe uma barra ativa (a barra externa decide a exibição).

    Yields
    ------
    Progress
        Barra ativa, na qual novas tarefas podem ser registradas com
        ``add_task``.

    Examples
    --------
    >>> with open_progress() as progress:
    ...     task = progress.add_task("Treinando", total=10)
    ...     progress.advance(task)
    """
    global _active_progress  # noqa: PLW0603 - estado necessário p/ aninhamento
    if _active_progress is not None:
        yield _active_progress
        return

    progress = _build_progress_bar(disable=disable)
    _active_progress = progress
    try:
        with progress:
            yield progress
    finally:
        _active_progress = None


def iterate_with_progress(
    iterable: Iterable[T],
    description: str,
    *,
    total: int | None = None,
    disable: bool = False,
) -> Iterator[T]:
    """Itera sobre ``iterable`` exibindo o progresso com ``rich``.

    Substituto direto de ``tqdm(iterable, desc=..., total=..., disable=...)``.

    Parameters
    ----------
    iterable : Iterable[T]
        Itens a percorrer.
    description : str
        Descrição exibida ao lado da barra.
    total : int | None, optional
        Número total de itens; se ``None``, usa ``len(iterable)`` quando
        disponível, by default None.
    disable : bool, optional
        Se ``True``, apenas repassa os itens sem exibir progresso, by default
        False.

    Yields
    ------
    T
        Cada item de ``iterable``, na ordem original.

    Examples
    --------
    >>> for chunk in iterate_with_progress(chunks, "Processando fragmentos"):
    ...     process(chunk)
    """
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = None

    is_nested = _active_progress is not None
    with open_progress(disable=disable) as progress:
        task_id = progress.add_task(description, total=total)
        try:
            for item in iterable:
                yield item
                progress.advance(task_id)
        finally:
            if is_nested:
                # Evita acumular barras concluídas em laços aninhados.
                progress.remove_task(task_id)
