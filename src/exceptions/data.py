"""Exceções relacionadas a dados.

Cobrem falhas de leitura, ausência e violação de contratos de dados
(schemas ``pandera`` definidos em ``src/schemas/``).
"""

from exceptions.base import ProjectError


class DataError(ProjectError):
    """Erro genérico relacionado a dados."""


class DataNotFoundError(DataError):
    """Levantada quando um arquivo ou fonte de dados esperada não é encontrada.

    Parameters
    ----------
    source : str
        Caminho ou identificador da fonte de dados não encontrada.
    """

    def __init__(self, source: str) -> None:
        super().__init__(
            f"Fonte de dados não encontrada: {source}",
            context={"source": source},
        )


class EmptyDatasetError(DataError):
    """Levantada quando um dataset esperado está vazio.

    Parameters
    ----------
    source : str
        Caminho ou identificador do dataset vazio.
    """

    def __init__(self, source: str) -> None:
        super().__init__(
            f"Dataset vazio: {source}",
            context={"source": source},
        )


class DataValidationError(DataError):
    """Levantada quando um DataFrame viola um contrato de dados (schema).

    Parameters
    ----------
    schema_name : str
        Nome do schema de validação que falhou.
    detail : str
        Descrição do erro de validação retornado pelo ``pandera``.
    """

    def __init__(self, schema_name: str, detail: str) -> None:
        super().__init__(
            f"Falha na validação do schema '{schema_name}': {detail}",
            context={"schema_name": schema_name},
        )
