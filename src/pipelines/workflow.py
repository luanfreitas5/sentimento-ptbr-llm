"""Orquestração das etapas do pipeline por nome.

Implementa a orquestração final de ``configs/config.yaml -> stages``:
registra cada estágio implementado nos demais módulos de
``src/pipelines/`` sob um nome único (:data:`STAGE_REGISTRY`), despacha a
execução de um estágio individual ou da sequência completa configurada, e
traduz falhas inesperadas em
:class:`exceptions.pipeline.PipelineStageError`.
"""

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from exceptions.base import ProjectError
from exceptions.pipeline import PipelineStageError, UnknownPipelineStageError
from pipelines.comparative_evaluation import run_comparative_evaluation_stage
from pipelines.features import run_features_stage
from pipelines.ingestion import run_ingestion_stage
from pipelines.labeling import run_labeling_stage
from pipelines.llm_evaluation import run_llm_evaluation_stage
from pipelines.preprocessing import run_preprocessing_stage
from pipelines.training_classical import run_training_classical_stage
from pipelines.training_deep_learning import run_training_deep_learning_stage

logger = logging.getLogger(__name__)

STAGE_REGISTRY: dict[str, Callable[..., Any]] = {
    "ingestion": run_ingestion_stage,
    "preprocessing": run_preprocessing_stage,
    "labeling": run_labeling_stage,
    "features": run_features_stage,
    "training_classical": run_training_classical_stage,
    "training_deep_learning": run_training_deep_learning_stage,
    "llm_evaluation": run_llm_evaluation_stage,
    "comparative_evaluation": run_comparative_evaluation_stage,
}


def run_pipeline_stage(stage_name: str, **stage_kwargs: Any) -> Any:
    """Executa um único estágio do pipeline pelo nome, registrado em :data:`STAGE_REGISTRY`.

    Parameters
    ----------
    stage_name : str
        Nome do estágio a executar, uma das chaves de :data:`STAGE_REGISTRY`
        (espelhando ``configs/config.yaml -> stages``).
    **stage_kwargs : Any
        Argumentos repassados à função do estágio correspondente.

    Returns
    -------
    Any
        Resultado da função do estágio executado.

    Raises
    ------
    UnknownPipelineStageError
        Se ``stage_name`` não estiver registrado em :data:`STAGE_REGISTRY`.
    PipelineStageError
        Se a execução do estágio falhar de forma inesperada. Exceções já
        tipadas do projeto (:class:`exceptions.base.ProjectError`, ex.:
        ``EmptyDatasetError``, ``DataValidationError``) são repropagadas
        sem modificação, preservando o tipo original do erro.

    Examples
    --------
    >>> run_pipeline_stage("etapa_inexistente")  # doctest: +SKIP
    """
    stage_function = STAGE_REGISTRY.get(stage_name)
    if stage_function is None:
        raise UnknownPipelineStageError(stage_name, list(STAGE_REGISTRY))

    logger.info("Iniciando etapa do pipeline: '%s'.", stage_name)
    try:
        result = stage_function(**stage_kwargs)
    except ProjectError:
        raise
    except Exception as exception:
        logger.exception("Falha inesperada na etapa do pipeline '%s'.", stage_name)
        raise PipelineStageError(stage_name, str(exception)) from exception

    logger.info("Etapa do pipeline '%s' concluída com sucesso.", stage_name)
    return result


def run_full_workflow(
    stages: Sequence[str],
    *,
    stage_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Executa a sequência completa de estágios configurados, na ordem informada.

    Interrompe a execução no primeiro estágio que falhar (falha rápida),
    propagando a exceção original de :func:`run_pipeline_stage`.

    Parameters
    ----------
    stages : Sequence[str]
        Nomes dos estágios a executar, na ordem de execução (tipicamente
        ``general_config.stages``, de ``configs/config.yaml``).
    stage_kwargs : Mapping[str, Mapping[str, Any]] | None, optional
        Argumentos por estágio, indexados pelo nome do estágio, by default
        None (nenhum argumento extra para nenhum estágio).

    Returns
    -------
    dict[str, Any]
        Resultado de cada estágio executado com sucesso, indexado pelo nome
        do estágio, na ordem de ``stages``.

    Raises
    ------
    UnknownPipelineStageError
        Se algum nome em ``stages`` não estiver registrado.
    PipelineStageError
        Se a execução de algum estágio falhar inesperadamente.

    Examples
    --------
    >>> run_full_workflow(["ingestion"], stage_kwargs={"ingestion": {}})  # doctest: +SKIP
    """
    resolved_stage_kwargs = stage_kwargs or {}
    results: dict[str, Any] = {}
    for stage_name in stages:
        results[stage_name] = run_pipeline_stage(
            stage_name, **resolved_stage_kwargs.get(stage_name, {})
        )
    logger.info("Workflow completo concluído: %d etapa(s) executada(s).", len(results))
    return results
