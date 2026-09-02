"""Exceção-base do projeto.

Todas as exceções customizadas do projeto devem herdar de :class:`ProjectError`,
permitindo capturar qualquer falha originada internamente com um único
``except ProjectError`` quando apropriado, sem mascarar exceções de terceiros.
"""

from typing import Any


class ProjectError(Exception):
    """Exceção-base para todos os erros customizados do projeto.

    Parameters
    ----------
    message : str
        Mensagem de erro em pt-BR, descrevendo a falha de forma clara.
    context : dict[str, Any] | None, optional
        Informações adicionais de contexto (ex.: caminho de arquivo, nome de
        coluna, etapa do pipeline) úteis para diagnóstico, by default None.

    Examples
    --------
    >>> raise ProjectError("Falha genérica no projeto")
    Traceback (most recent call last):
        ...
    exceptions.base.ProjectError: Falha genérica no projeto
    """

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        self.message = message
        self.context = context or {}
        super().__init__(self._build_full_message())

    def _build_full_message(self) -> str:
        """Monta a mensagem final incluindo o contexto, quando houver.

        Returns
        -------
        str
            Mensagem de erro formatada com o contexto anexado.
        """
        if not self.context:
            return self.message
        contexto_formatado = ", ".join(
            f"{chave}={valor!r}" for chave, valor in self.context.items()
        )
        return f"{self.message} (contexto: {contexto_formatado})"
