"""Registro e promoção de modelos no MLflow Model Registry.

Implementa CLAUDE.md, "Model & Data Versioning": promove modelos entre
estágios (``Staging`` -> ``Production``, ver ``configs/config.yaml`` ->
``experiment.registry_stage_default``) via MLflow Model Registry, em vez
de depender de arquivos ``joblib`` soltos sem rastreabilidade.
"""

import logging

from exceptions.model import ModelError

logger = logging.getLogger(__name__)

MODEL_REGISTRY_STAGES: tuple[str, ...] = ("None", "Staging", "Production", "Archived")


def _validate_stage(stage: str) -> str:
    """Valida se um nome de estágio é reconhecido pelo MLflow Model Registry.

    Parameters
    ----------
    stage : str
        Nome do estágio a validar.

    Returns
    -------
    str
        O próprio estágio, quando válido.

    Raises
    ------
    ValueError
        Se ``stage`` não pertencer a :data:`MODEL_REGISTRY_STAGES`.
    """
    if stage not in MODEL_REGISTRY_STAGES:
        raise ValueError(f"stage '{stage}' inválido. Estágios disponíveis: {MODEL_REGISTRY_STAGES}")
    return stage


def register_model_version(model_uri: str, model_name: str) -> str:
    """Registra uma nova versão de um modelo no MLflow Model Registry.

    Parameters
    ----------
    model_uri : str
        URI do modelo dentro de uma execução MLflow (ex.:
        ``"runs:/<run_id>/modelo"``, ver
        ``models.persistence.log_classifier_to_mlflow``).
    model_name : str
        Nome do modelo registrado (ver ``configs/config.yaml`` ->
        ``experiment.name``).

    Returns
    -------
    str
        Número da versão recém-registrada.

    Examples
    --------
    >>> register_model_version("runs:/abc123/modelo", "sentimento-ptbr-llm")  # doctest: +SKIP
    """
    import mlflow

    model_version = mlflow.register_model(model_uri, model_name)
    logger.info("Modelo '%s' registrado como versão %s.", model_name, model_version.version)
    return model_version.version


def transition_model_stage(model_name: str, model_version: str, stage: str) -> None:
    """Promove (ou rebaixa) uma versão de modelo para um novo estágio.

    Parameters
    ----------
    model_name : str
        Nome do modelo registrado.
    model_version : str
        Número da versão a promover.
    stage : str
        Estágio de destino, um de :data:`MODEL_REGISTRY_STAGES`.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        Se ``stage`` não for um estágio reconhecido.

    Examples
    --------
    >>> transition_model_stage("sentimento-ptbr-llm", "1", "Production")  # doctest: +SKIP
    """
    _validate_stage(stage)
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    client.transition_model_version_stage(name=model_name, version=model_version, stage=stage)
    logger.info("Modelo '%s' versão %s promovido para '%s'.", model_name, model_version, stage)


def get_latest_model_version(model_name: str, *, stage: str = "Production") -> str:
    """Obtém o número da versão mais recente de um modelo em um dado estágio.

    Parameters
    ----------
    model_name : str
        Nome do modelo registrado.
    stage : str, optional
        Estágio a consultar, by default "Production".

    Returns
    -------
    str
        Número da versão mais recente no estágio informado.

    Raises
    ------
    ValueError
        Se ``stage`` não for um estágio reconhecido.
    ModelError
        Se não houver nenhuma versão registrada no estágio informado.

    Examples
    --------
    >>> get_latest_model_version("sentimento-ptbr-llm", stage="Production")  # doctest: +SKIP
    """
    _validate_stage(stage)
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    versions = client.get_latest_versions(model_name, stages=[stage])
    if not versions:
        raise ModelError(
            f"Nenhuma versão do modelo '{model_name}' encontrada no estágio '{stage}'."
        )
    return versions[0].version
