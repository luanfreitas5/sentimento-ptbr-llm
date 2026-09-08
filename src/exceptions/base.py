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
        context_string = ", ".join(f"{key}={value!r}" for key, value in self.context.items())
        return f"{self.message} (contexto: {context_string})"

    def __reduce__(self) -> tuple[Any, ...]:
        """Permite que subclasses com ``__init__`` próprio sejam serializadas (``pickle``).

        Subclasses de :class:`ProjectError` costumam expor um ``__init__``
        com parâmetros específicos (ex.: ``PipelineStageError(stage_name,
        detail)``), incompatível com a reconstrução padrão do ``pickle``
        (que chamaria ``type(self)(*self.args)`` usando a mensagem já
        formatada de ``Exception.__init__``). Necessário para que uma
        exceção levantada dentro de um worker de ``ProcessPoolExecutor``
        (ver ``src/parallel/core.py``) chegue intacta ao processo pai, em
        vez de quebrar o pool inteiro (``BrokenProcessPool``) durante a
        desserialização.

        Returns
        -------
        tuple[Any, ...]
            Par ``(função_reconstrutora, argumentos)`` usado pelo ``pickle``.
        """
        return (_reconstruct_project_error, (self.__class__, self.message, self.context))


def _reconstruct_project_error(
    error_class: type[ProjectError], message: str, context: dict[str, Any]
) -> ProjectError:
    """Reconstrói uma instância de :class:`ProjectError` (ou subclasse) sem chamar seu ``__init__``.

    Parameters
    ----------
    error_class : type[ProjectError]
        Classe concreta a instanciar (``PipelineStageError``, ``DataError`` etc.).
    message : str
        Mensagem original (sem o contexto formatado), igual a ``self.message``.
    context : dict[str, Any]
        Contexto original, igual a ``self.context``.

    Returns
    -------
    ProjectError
        Instância reconstruída, equivalente à original.
    """
    instance = error_class.__new__(error_class)
    ProjectError.__init__(instance, message, context=context)
    return instance
