"""Persistência de modelos treinados (joblib/PyTorch) e registro no MLflow Model Registry.

Implementa a Fase 9 do plano de elaboração: unifica o salvamento/
carregamento dos diferentes paradigmas de modelo do projeto (ML clássico via
``joblib``/scikit-learn/XGBoost, DL/Transformer via
``torch.save``/``torch.load``) e o registro de qualquer um deles no MLflow
Model Registry, evitando que cada módulo de modelo reimplemente sua própria
lógica de I/O.
"""

import logging
from pathlib import Path
from typing import Any, Literal, NoReturn

import joblib

from exceptions.model import ModelPersistenceError
from utils.validation import validate_file_exists

logger = logging.getLogger(__name__)

PersistenceBackend = Literal["joblib", "torch"]


def _raise_unsupported_backend(backend: str) -> NoReturn:
    """Levanta ``ValueError`` para um backend de persistência desconhecido.

    Isolada em função própria para que o ``raise`` não fique diretamente
    dentro do bloco ``try`` das funções de (des)serialização, mantendo a
    intenção de "erro de programação, não falha de I/O" explícita mesmo
    sendo capturado e reembalado como :class:`ModelPersistenceError`.

    Parameters
    ----------
    backend : str
        Nome do backend informado pelo chamador.

    Raises
    ------
    ValueError
        Sempre, com mensagem indicando o backend não suportado.
    """
    raise ValueError(f"Backend de persistência '{backend}' não suportado.")


def save_classifier(model: Any, file_path: Path, *, backend: PersistenceBackend = "joblib") -> Path:
    """Salva um classificador treinado em disco.

    Parameters
    ----------
    model : Any
        Instância de modelo treinado (estimator scikit-learn/XGBoost para
        ``backend="joblib"``; instância de ``src/models/`` baseada em
        PyTorch, ex. :class:`models.lstm.LSTMSentimentClassifier`, para
        ``backend="torch"``).
    file_path : Path
        Caminho de destino (ver ``configs/paths.yaml -> models``). O
        diretório pai é criado quando ausente.
    backend : {"joblib", "torch"}, optional
        Mecanismo de serialização, by default "joblib".

    Returns
    -------
    Path
        O mesmo ``file_path``, após a escrita bem-sucedida.

    Raises
    ------
    ModelPersistenceError
        Se a serialização falhar (ex.: ``backend`` desconhecido, objeto não
        serializável, ``torch`` ausente).

    Examples
    --------
    >>> from sklearn.naive_bayes import MultinomialNB
    >>> save_classifier(MultinomialNB(), Path("modelo.joblib"))  # doctest: +SKIP
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if backend == "joblib":
            joblib.dump(model, file_path)
        elif backend == "torch":
            import torch  # type: ignore[reportMissingImports]

            torch.save(model, file_path)
        else:
            _raise_unsupported_backend(backend)
    except (OSError, ValueError, ImportError, TypeError) as exception:
        raise ModelPersistenceError(str(file_path), str(exception)) from exception

    logger.info("Modelo salvo em '%s' (backend=%s).", file_path, backend)
    return file_path


def load_classifier(file_path: Path, *, backend: PersistenceBackend = "joblib") -> Any:
    """Carrega um classificador previamente salvo por :func:`save_classifier`.

    Parameters
    ----------
    file_path : Path
        Caminho do arquivo de modelo.
    backend : {"joblib", "torch"}, optional
        Mecanismo de desserialização, deve corresponder ao usado em
        :func:`save_classifier`, by default "joblib".

    Returns
    -------
    Any
        Instância do modelo carregado.

    Raises
    ------
    DataNotFoundError
        Se ``file_path`` não existir.
    ModelPersistenceError
        Se a desserialização falhar (ex.: ``backend`` desconhecido, arquivo
        corrompido, ``torch`` ausente).

    Examples
    --------
    >>> load_classifier(Path("modelo.joblib"))  # doctest: +SKIP
    """
    validate_file_exists(file_path)
    try:
        if backend == "joblib":
            # joblib.load desserializa via pickle: seguro aqui porque o arquivo é sempre
            # um artefato produzido pelo próprio projeto (save_classifier), nunca uma
            # entrada de usuário/rede não confiável (ver CLAUDE.md, "Data Privacy & LGPD").
            model = joblib.load(file_path)
        elif backend == "torch":
            import torch  # type: ignore[reportMissingImports]

            # weights_only=False é necessário pois os artefatos deste projeto são
            # objetos Python completos (ex.: LSTMSentimentClassifier), não apenas
            # tensores; mesma justificativa de confiança do ramo "joblib" acima.
            model = torch.load(file_path, weights_only=False)  # nosec B614
        else:
            _raise_unsupported_backend(backend)
    except (OSError, ValueError, ImportError, TypeError, EOFError) as exception:
        raise ModelPersistenceError(str(file_path), str(exception)) from exception

    logger.info("Modelo carregado de '%s' (backend=%s).", file_path, backend)
    return model


def log_classifier_to_mlflow(
    model: Any,
    artifact_path: str,
    *,
    backend: PersistenceBackend = "joblib",
    registered_model_name: str | None = None,
) -> str:
    """Registra um classificador treinado como artefato no MLflow, dentro de um run ativo.

    Requer uma execução (``run``) do MLflow ativa no momento da chamada
    (ver ``configs/config.yaml -> experiment``); esta função não abre nem
    encerra runs.

    Parameters
    ----------
    model : Any
        Instância de modelo treinado.
    artifact_path : str
        Caminho relativo do artefato dentro do run do MLflow.
    backend : {"joblib", "torch"}, optional
        Flavor do MLflow usado no registro (``mlflow.sklearn`` para
        ``"joblib"``, ``mlflow.pytorch`` para ``"torch"``), by default
        "joblib".
    registered_model_name : str | None, optional
        Se informado, também registra/promove o modelo no MLflow Model
        Registry sob este nome, by default None.

    Returns
    -------
    str
        URI do modelo registrado (``model_uri``).

    Raises
    ------
    ModelPersistenceError
        Se o registro falhar (ex.: ``backend`` desconhecido, nenhum run
        ativo, ``torch`` ausente).

    Examples
    --------
    >>> log_classifier_to_mlflow(model, "modelo")  # doctest: +SKIP
    """
    try:
        if backend == "joblib":
            import mlflow.sklearn as mlflow_sklearn  # type: ignore[reportPrivateImportUsage]

            model_info = mlflow_sklearn.log_model(
                model, artifact_path, registered_model_name=registered_model_name
            )
        elif backend == "torch":
            import mlflow.pytorch as mlflow_pytorch  # type: ignore[reportPrivateImportUsage]

            model_info = mlflow_pytorch.log_model(
                model, artifact_path, registered_model_name=registered_model_name
            )
        else:
            _raise_unsupported_backend(backend)
    except (OSError, ValueError, ImportError, TypeError) as exception:
        raise ModelPersistenceError(artifact_path, str(exception)) from exception

    logger.info("Modelo registrado no MLflow em '%s'.", model_info.model_uri)
    return str(model_info.model_uri)
