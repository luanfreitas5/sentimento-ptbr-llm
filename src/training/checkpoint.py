"""Checkpointing de modelos durante o treino, a partir de uma métrica monitorada.

Usa ``src/models/persistence.py`` como mecanismo de serialização (``joblib``
para modelos clássicos, ``torch`` para DL/Transformer), decidindo apenas
*quando* salvar: a cada melhora da métrica monitorada, opcionalmente
mantendo somente o melhor checkpoint em disco.
"""

import logging
from pathlib import Path
from typing import Any, Literal

from models.persistence import PersistenceBackend, save_classifier

logger = logging.getLogger(__name__)

CheckpointMode = Literal["min", "max"]


class ModelCheckpoint:
    """Salva um modelo em disco sempre que a métrica monitorada melhora.

    Parameters
    ----------
    checkpoint_dir : Path
        Diretório de destino dos checkpoints (ver
        ``configs/paths.yaml -> models.checkpoints``). Criado se ausente.
    monitor : str, optional
        Nome descritivo da métrica monitorada, usado apenas no nome do
        arquivo e nas mensagens de log, by default "score".
    mode : {"min", "max"}, optional
        Se ``"max"``, valores maiores são melhores (ex.: F1-macro); se
        ``"min"``, valores menores são melhores (ex.: perda de validação),
        by default "max".
    save_best_only : bool, optional
        Se ``True``, cada novo checkpoint sobrescreve o anterior (apenas o
        melhor modelo permanece em disco); se ``False``, um arquivo
        numerado é mantido por melhora, by default True.
    backend : {"joblib", "torch"}, optional
        Mecanismo de serialização repassado a
        :func:`models.persistence.save_classifier`, by default "joblib".

    Examples
    --------
    >>> from sklearn.naive_bayes import MultinomialNB
    >>> checkpoint = ModelCheckpoint(Path("models/checkpoints/exemplo"), mode="max")
    >>> checkpoint.step(MultinomialNB(), 0.80, step_index=0) is not None  # doctest: +SKIP
    True
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        *,
        monitor: str = "score",
        mode: CheckpointMode = "max",
        save_best_only: bool = True,
        backend: PersistenceBackend = "joblib",
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.backend = backend

        self.best_value: float | None = None
        self.best_path: Path | None = None

    def _is_improvement(self, value: float) -> bool:
        """Indica se ``value`` melhora :attr:`best_value` segundo :attr:`mode`.

        Parameters
        ----------
        value : float
            Novo valor da métrica monitorada.

        Returns
        -------
        bool
            ``True`` se ``value`` for uma melhora (ou se ainda não houver
            valor registrado).
        """
        if self.best_value is None:
            return True
        if self.mode == "max":
            return value > self.best_value
        return value < self.best_value

    def _build_checkpoint_path(self, step_index: int) -> Path:
        """Monta o caminho de destino do checkpoint para um passo específico.

        Parameters
        ----------
        step_index : int
            Índice do passo/época/dobra corrente.

        Returns
        -------
        Path
            Caminho do arquivo de checkpoint, único quando
            ``save_best_only=False`` e fixo (``best.<ext>``) caso contrário.
        """
        extension = "joblib" if self.backend == "joblib" else "pt"
        filename = "best" if self.save_best_only else f"{self.monitor}_step{step_index:04d}"
        return self.checkpoint_dir / f"{filename}.{extension}"

    def step(self, model: Any, value: float, *, step_index: int = 0) -> Path | None:
        """Avalia a métrica do passo corrente e salva o modelo se houver melhora.

        Parameters
        ----------
        model : Any
            Instância de modelo a ser potencialmente salva.
        value : float
            Valor da métrica monitorada (:attr:`monitor`) no passo corrente.
        step_index : int, optional
            Índice do passo/época/dobra corrente, usado no nome do arquivo
            quando ``save_best_only=False``, by default 0.

        Returns
        -------
        Path | None
            Caminho onde o modelo foi salvo, ou ``None`` se não houve
            melhora (nenhum arquivo escrito).
        """
        if not self._is_improvement(value):
            logger.debug(
                "Sem melhora em '%s' no passo %d (valor=%.4f, melhor=%.4f); checkpoint ignorado.",
                self.monitor,
                step_index,
                value,
                self.best_value,
            )
            return None

        checkpoint_path = self._build_checkpoint_path(step_index)
        save_classifier(model, checkpoint_path, backend=self.backend)
        self.best_value = value
        self.best_path = checkpoint_path
        logger.info(
            "Checkpoint salvo em '%s' ('%s'=%.4f no passo %d).",
            checkpoint_path,
            self.monitor,
            value,
            step_index,
        )
        return checkpoint_path
