"""Exceções relacionadas à execução de pipelines.

Cobrem falhas de execução de etapas (``src/pipelines/``) e solicitação de
etapas desconhecidas via orquestração (``src/main.py``).
"""

from exceptions.base import ProjectError


class PipelineError(ProjectError):
    """Erro genérico relacionado à execução de um pipeline."""


class PipelineStageError(PipelineError):
    """Levantada quando uma etapa específica do pipeline falha durante a execução.

    Parameters
    ----------
    stage_name : str
        Nome da etapa do pipeline que falhou.
    detail : str
        Descrição da causa da falha.
    """

    def __init__(self, stage_name: str, detail: str) -> None:
        super().__init__(
            f"Falha na etapa do pipeline '{stage_name}': {detail}",
            context={"stage_name": stage_name},
        )


class UnknownPipelineStageError(PipelineError):
    """Levantada quando uma etapa de pipeline solicitada não existe.

    Parameters
    ----------
    stage_name : str
        Nome da etapa solicitada.
    available_stages : list[str]
        Lista de etapas conhecidas/registradas, para orientar a correção.
    """

    def __init__(self, stage_name: str, available_stages: list[str]) -> None:
        super().__init__(
            f"Nome da etapa '{stage_name}' desconhecida. Etapas disponíveis: {available_stages}",
            context={"stage_name": stage_name},
        )
