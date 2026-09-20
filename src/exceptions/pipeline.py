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


class SanityGateFailedError(PipelineError):
    """Levantada quando o gate de sanidade indica que o alvo não é previsível pelos embeddings.

    Parameters
    ----------
    target_name : str
        Nome do alvo de diagnóstico avaliado.
    detail : str
        Resumo do resultado do gate (métrica, IC e p-valor).
    """

    def __init__(self, target_name: str, detail: str) -> None:
        super().__init__(
            f"Gate de sanidade reprovou o alvo '{target_name}': {detail}",
            context={"target_name": target_name},
        )


class IncompleteLabelingError(PipelineError):
    """Levantada quando a rotulagem termina com tweets sem rótulo válido.

    O ponto de retomada (checkpoint) preserva os tweets já rotulados: uma nova
    execução reprocessa apenas os ``n_failed`` tweets pendentes.

    Parameters
    ----------
    source_name : str
        Fonte de rotulagem (``huggingface`` ou ``openai``).
    n_failed : int
        Quantidade de tweets sem rótulo válido após todas as tentativas.
    checkpoint_path : str
        Caminho do checkpoint que permite retomar a execução.
    """

    def __init__(self, source_name: str, n_failed: int, checkpoint_path: str) -> None:
        super().__init__(
            f"Rotulagem '{source_name}' incompleta: {n_failed} tweet(s) sem rótulo válido; "
            "execute a etapa novamente para reprocessar apenas os pendentes.",
            context={"source_name": source_name, "checkpoint_path": checkpoint_path},
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
