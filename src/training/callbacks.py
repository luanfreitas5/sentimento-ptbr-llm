"""Callbacks de treino: logging, checkpoint e parada antecipada sob uma interface comum.

Define a interface :class:`Callback`, observada a cada passo de treino por
:class:`training.trainer.Trainer` — um "passo" é uma época para um laço de
treino iterativo ou, no caso mais comum deste projeto (modelos com ``fit``
atômico), uma dobra de validação cruzada
(:func:`training.cross_validation.run_stratified_cross_validation`) ou a
própria chamada única de treino. Os adaptadores
:class:`EarlyStoppingCallback` e :class:`ModelCheckpointCallback` permitem
reaproveitar :class:`training.early_stopping.EarlyStopping` e
:class:`training.checkpoint.ModelCheckpoint` — que não conhecem a interface
``Callback`` — dentro de uma mesma lista de callbacks.
"""

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from training.checkpoint import CheckpointMode, ModelCheckpoint
from training.early_stopping import EarlyStopping, EarlyStoppingMode

logger = logging.getLogger(__name__)


@runtime_checkable
class Callback(Protocol):
    """Interface comum a todo callback observável por :class:`training.trainer.Trainer`."""

    def on_train_begin(self) -> None:
        """Notifica o início do treino, antes do primeiro passo."""
        ...

    def on_step_end(self, step_index: int, model: Any, metrics: Mapping[str, float]) -> bool:
        """Notifica o fim de um passo de treino (época ou dobra).

        Parameters
        ----------
        step_index : int
            Índice do passo concluído (indexado a partir de 0).
        model : Any
            Instância do modelo no estado atual (pode ser ``None`` quando
            indisponível no ponto de chamada).
        metrics : Mapping[str, float]
            Métricas calculadas ao final do passo.

        Returns
        -------
        bool
            ``True`` se este callback solicita a interrupção do treino.
        """
        ...

    def on_train_end(self) -> None:
        """Notifica o fim do treino, após o último passo (ou uma interrupção antecipada)."""
        ...


class LoggingCallback:
    """Callback que registra em log o progresso do treino a cada passo.

    Parameters
    ----------
    stage_name : str, optional
        Nome descritivo da etapa de treino, usado nas mensagens de log, by
        default "treino".
    """

    def __init__(self, stage_name: str = "treino") -> None:
        self.stage_name = stage_name

    def on_train_begin(self) -> None:
        """Registra em log o início do treino."""
        logger.info("Início do treino: '%s'.", self.stage_name)

    def on_step_end(self, step_index: int, model: Any, metrics: Mapping[str, float]) -> bool:  # noqa: ARG002
        """Registra em log as métricas do passo concluído; nunca solicita parada."""
        metrics_text = ", ".join(f"{name}={value:.4f}" for name, value in metrics.items())
        logger.info("'%s' - passo %d concluído (%s).", self.stage_name, step_index, metrics_text)
        return False

    def on_train_end(self) -> None:
        """Registra em log o fim do treino."""
        logger.info("Fim do treino: '%s'.", self.stage_name)


class EarlyStoppingCallback:
    """Adapta :class:`training.early_stopping.EarlyStopping` à interface :class:`Callback`.

    Parameters
    ----------
    monitor : str
        Nome da métrica, em :attr:`Callback.on_step_end` ``metrics``, a ser
        monitorada.
    patience : int, optional
        Repassado a :class:`training.early_stopping.EarlyStopping`, by
        default 3.
    mode : {"min", "max"}, optional
        Repassado a :class:`training.early_stopping.EarlyStopping`, by
        default "max".
    min_delta : float, optional
        Repassado a :class:`training.early_stopping.EarlyStopping`, by
        default 0.0.
    """

    def __init__(
        self,
        monitor: str,
        *,
        patience: int = 3,
        mode: EarlyStoppingMode = "max",
        min_delta: float = 0.0,
    ) -> None:
        self.monitor = monitor
        self.early_stopping = EarlyStopping(patience=patience, mode=mode, min_delta=min_delta)

    def on_train_begin(self) -> None:
        """Reinicia o monitor de parada antecipada no início de um novo treino."""
        self.early_stopping.reset()

    def on_step_end(self, step_index: int, model: Any, metrics: Mapping[str, float]) -> bool:  # noqa: ARG002
        """Repassa a métrica monitorada ao monitor de parada antecipada."""
        if self.monitor not in metrics:
            logger.warning(
                "Métrica '%s' ausente no passo %d; parada antecipada ignorada.",
                self.monitor,
                step_index,
            )
            return False
        return self.early_stopping.step(metrics[self.monitor])

    def on_train_end(self) -> None:
        """Sem efeito: nada a fazer ao final do treino."""


class ModelCheckpointCallback:
    """Adapta :class:`training.checkpoint.ModelCheckpoint` à interface :class:`Callback`.

    Parameters
    ----------
    checkpoint_dir : Path
        Repassado a :class:`training.checkpoint.ModelCheckpoint`.
    monitor : str
        Nome da métrica, em :attr:`Callback.on_step_end` ``metrics``, usada
        para decidir quando salvar.
    mode : {"min", "max"}, optional
        Repassado a :class:`training.checkpoint.ModelCheckpoint`, by
        default "max".
    save_best_only : bool, optional
        Repassado a :class:`training.checkpoint.ModelCheckpoint`, by
        default True.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        monitor: str,
        *,
        mode: CheckpointMode = "max",
        save_best_only: bool = True,
    ) -> None:
        self.monitor = monitor
        self.checkpoint = ModelCheckpoint(
            checkpoint_dir, monitor=monitor, mode=mode, save_best_only=save_best_only
        )

    def on_train_begin(self) -> None:
        """Sem efeito: nada a fazer no início do treino."""

    def on_step_end(self, step_index: int, model: Any, metrics: Mapping[str, float]) -> bool:
        """Salva o modelo se a métrica monitorada melhorou; nunca solicita parada."""
        if model is None or self.monitor not in metrics:
            return False
        self.checkpoint.step(model, metrics[self.monitor], step_index=step_index)
        return False

    def on_train_end(self) -> None:
        """Sem efeito: nada a fazer ao final do treino."""


class CallbackList:
    """Agrega múltiplos callbacks e os notifica em sequência, na ordem informada.

    Parameters
    ----------
    callbacks : Sequence[Callback] | None, optional
        Callbacks a agregar, by default None (lista vazia).
    """

    def __init__(self, callbacks: Sequence[Callback] | None = None) -> None:
        self.callbacks: list[Callback] = list(callbacks) if callbacks is not None else []

    def on_train_begin(self) -> None:
        """Notifica todos os callbacks do início do treino."""
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_step_end(self, step_index: int, model: Any, metrics: Mapping[str, float]) -> bool:
        """Notifica todos os callbacks do fim de um passo, agregando o pedido de parada.

        Parameters
        ----------
        step_index : int
            Índice do passo concluído.
        model : Any
            Instância do modelo no estado atual.
        metrics : Mapping[str, float]
            Métricas calculadas ao final do passo.

        Returns
        -------
        bool
            ``True`` se qualquer callback da lista solicitou a interrupção
            do treino (todos são notificados, mesmo após um pedido de
            parada).
        """
        should_stop = False
        for callback in self.callbacks:
            should_stop = callback.on_step_end(step_index, model, metrics) or should_stop
        return should_stop

    def on_train_end(self) -> None:
        """Notifica todos os callbacks do fim do treino."""
        for callback in self.callbacks:
            callback.on_train_end()
