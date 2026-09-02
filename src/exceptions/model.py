"""Exceções relacionadas a modelos de Machine Learning.

Cobrem falhas de treinamento, uso de modelo não treinado e persistência de
artefatos de modelo (``models/``).
"""

from exceptions.base import ProjectError


class ModelError(ProjectError):
    """Erro genérico relacionado a modelos."""


class ModelNotFittedError(ModelError):
    """Levantada ao tentar usar um modelo antes de treiná-lo.

    Parameters
    ----------
    model_name : str
        Nome ou identificador do modelo não treinado.
    """

    def __init__(self, model_name: str) -> None:
        super().__init__(
            f"Modelo '{model_name}' ainda não foi treinado (fit não executado)",
            context={"model_name": model_name},
        )


class ModelPersistenceError(ModelError):
    """Levantada quando ocorre falha ao salvar ou carregar um modelo.

    Parameters
    ----------
    file_path : str
        Caminho do arquivo de modelo envolvido na falha.
    detail : str
        Descrição da causa da falha.
    """

    def __init__(self, file_path: str, detail: str) -> None:
        super().__init__(
            f"Falha ao persistir/carregar modelo em '{file_path}': {detail}",
            context={"file_path": file_path},
        )


class UnsupportedModelError(ModelError):
    """Levantada quando um nome de modelo não reconhecido é solicitado à factory.

    Parameters
    ----------
    model_name : str
        Nome do modelo não suportado.
    available_models : list[str]
        Lista de nomes de modelos suportados, para orientar a correção.
    """

    def __init__(self, model_name: str, available_models: list[str]) -> None:
        super().__init__(
            f"Modelo '{model_name}' não suportado. Modelos disponíveis: {available_models}",
            context={"model_name": model_name},
        )
