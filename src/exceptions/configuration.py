"""Exceções relacionadas à configuração do projeto.

Cobrem falhas de carregamento e validação de arquivos YAML em ``configs/``,
variáveis de ambiente ausentes e valores de configuração inválidos.
"""

from exceptions.base import ProjectError


class ConfigurationError(ProjectError):
    """Erro genérico de configuração do projeto."""


class ConfigurationFileNotFoundError(ConfigurationError):
    """Levantada quando um arquivo de configuração esperado não é encontrado.

    Parameters
    ----------
    file_path : str
        Caminho do arquivo de configuração que não foi localizado.
    """

    def __init__(self, file_path: str) -> None:
        super().__init__(
            f"Arquivo de configuração não encontrado: {file_path}",
            context={"file_path": file_path},
        )


class InvalidConfigurationError(ConfigurationError):
    """Levantada quando o conteúdo de uma configuração é inválido ou incompleto.

    Parameters
    ----------
    detail : str
        Descrição do problema de validação encontrado.
    """

    def __init__(self, detail: str) -> None:
        super().__init__(f"Configuração inválida: {detail}")


class MissingEnvironmentVariableError(ConfigurationError):
    """Levantada quando uma variável de ambiente obrigatória não está definida.

    Parameters
    ----------
    variable_name : str
        Nome da variável de ambiente ausente.
    """

    def __init__(self, variable_name: str) -> None:
        super().__init__(
            f"Variável de ambiente obrigatória não definida: {variable_name}",
            context={"variable_name": variable_name},
        )
