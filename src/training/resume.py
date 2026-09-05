"""Retomada de treino a partir de um checkpoint salvo em disco.

Complementa ``src/training/checkpoint.py``: enquanto :class:`ModelCheckpoint`
decide *quando* salvar um modelo durante o treino, este módulo trata da
persistência/recuperação do estado completo de um treino interrompido — o
modelo em si mais os metadados necessários para retomá-lo (passos já
concluídos e histórico de métricas) — permitindo, por exemplo, continuar uma
validação cruzada interrompida sem repetir as dobras já executadas.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from io_utils.json import read_json, write_json
from models.persistence import PersistenceBackend, load_classifier, save_classifier

logger = logging.getLogger(__name__)

_STATE_FILE_NAME = "state.json"


@dataclass
class TrainingCheckpointState:
    """Estado completo de um treino em um ponto de interrupção/retomada.

    Parameters
    ----------
    model : Any
        Instância do modelo no ponto salvo.
    completed_steps : int
        Número de passos (épocas ou dobras, conforme o contexto de uso) já
        concluídos com sucesso.
    metrics_history : list[dict[str, float]]
        Métricas registradas a cada passo concluído, na ordem de execução.
    """

    model: Any
    completed_steps: int = 0
    metrics_history: list[dict[str, float]] = field(default_factory=list)


def _model_file_path(checkpoint_dir: Path, backend: PersistenceBackend) -> Path:
    """Monta o caminho do arquivo de modelo dentro do diretório de checkpoint.

    Parameters
    ----------
    checkpoint_dir : Path
        Diretório do checkpoint.
    backend : PersistenceBackend
        Mecanismo de serialização (define a extensão do arquivo).

    Returns
    -------
    Path
        Caminho ``checkpoint_dir/model.<extensão>``.
    """
    extension = "joblib" if backend == "joblib" else "pt"
    return checkpoint_dir / f"model.{extension}"


def save_training_state(
    state: TrainingCheckpointState,
    checkpoint_dir: Path,
    *,
    backend: PersistenceBackend = "joblib",
) -> Path:
    """Salva o estado completo de um treino (modelo + metadados) em disco.

    Parameters
    ----------
    state : TrainingCheckpointState
        Estado a ser persistido.
    checkpoint_dir : Path
        Diretório de destino, criado se ausente.
    backend : {"joblib", "torch"}, optional
        Mecanismo de serialização do modelo, repassado a
        :func:`models.persistence.save_classifier`, by default "joblib".

    Returns
    -------
    Path
        O próprio ``checkpoint_dir``, após a escrita bem-sucedida do modelo
        e dos metadados.

    Examples
    --------
    >>> from sklearn.naive_bayes import MultinomialNB
    >>> estado = TrainingCheckpointState(model=MultinomialNB(), completed_steps=2)
    >>> save_training_state(estado, Path("models/checkpoints/exemplo"))  # doctest: +SKIP
    """
    save_classifier(state.model, _model_file_path(checkpoint_dir, backend), backend=backend)
    write_json(
        {"completed_steps": state.completed_steps, "metrics_history": state.metrics_history},
        checkpoint_dir / _STATE_FILE_NAME,
    )
    logger.info(
        "Estado de treino salvo em '%s' (%d passo(s) concluído(s)).",
        checkpoint_dir,
        state.completed_steps,
    )
    return checkpoint_dir


def resume_training_state(
    checkpoint_dir: Path, *, backend: PersistenceBackend = "joblib"
) -> TrainingCheckpointState:
    """Carrega o estado completo de um treino previamente salvo por :func:`save_training_state`.

    Parameters
    ----------
    checkpoint_dir : Path
        Diretório do checkpoint a ser restaurado.
    backend : {"joblib", "torch"}, optional
        Mecanismo de desserialização do modelo, deve corresponder ao usado
        em :func:`save_training_state`, by default "joblib".

    Returns
    -------
    TrainingCheckpointState
        Estado restaurado, pronto para continuar o treino a partir de
        :attr:`TrainingCheckpointState.completed_steps`.

    Raises
    ------
    DataNotFoundError
        Se o diretório de checkpoint ou algum de seus arquivos não existir.
    ModelPersistenceError
        Se a desserialização do modelo falhar.

    Examples
    --------
    >>> resume_training_state(Path("models/checkpoints/exemplo"))  # doctest: +SKIP
    """
    model = load_classifier(_model_file_path(checkpoint_dir, backend), backend=backend)
    metadata = read_json(checkpoint_dir / _STATE_FILE_NAME)
    logger.info(
        "Estado de treino retomado de '%s' (%d passo(s) já concluído(s)).",
        checkpoint_dir,
        metadata["completed_steps"],
    )
    return TrainingCheckpointState(
        model=model,
        completed_steps=metadata["completed_steps"],
        metrics_history=metadata["metrics_history"],
    )
