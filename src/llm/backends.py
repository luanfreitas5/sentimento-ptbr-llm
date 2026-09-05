"""Interface única sobre os backends Ollama e Hugging Face, via LangChain.

Implementa a Fase 11 (``configs/llm.yaml -> backends``): diferente de
``src/models/llm.py`` (que fala diretamente com o SDK ``ollama``), este
módulo passa pelos integradores oficiais LangChain
(``langchain-ollama``/``langchain-huggingface``, extra ``llm`` de
``pyproject.toml``), permitindo compor os backends com
``src/llm/chains.py`` (LCEL). São dependências pesadas e opcionais: o
import ocorre de forma tardia, dentro das funções ``load_*_backend``, para
que o restante do módulo permaneça importável sem elas.
"""

import logging
from typing import Literal, Protocol

from exceptions.model import ModelError, UnsupportedModelError

logger = logging.getLogger(__name__)

LLMBackendName = Literal["ollama", "huggingface"]
LLM_BACKEND_NAMES: tuple[LLMBackendName, ...] = ("huggingface", "ollama")


class LLMBackend(Protocol):
    """Interface mínima de um backend de geração de texto por LLM.

    Satisfeita por :class:`OllamaLLMBackend`, :class:`HuggingFaceLLMBackend`
    e por qualquer dublê de teste (ex.: ``_FakeLLMBackend`` de
    ``tests/test_llm.py``), permitindo que
    :class:`llm.classifier.LangChainSentimentClassifier` opere sobre
    qualquer um deles sem conhecer a implementação concreta.
    """

    def generate(self, prompt: str) -> str:
        """Gera a resposta do modelo para um prompt.

        Parameters
        ----------
        prompt : str
            Prompt de entrada, via ``src/llm/prompts.py``.

        Returns
        -------
        str
            Texto bruto gerado pelo modelo.
        """
        ...


class OllamaLLMBackend:
    """Backend LLM via servidor Ollama local, através do integrador LangChain.

    Parameters
    ----------
    llm_runnable : Any
        Instância de ``langchain_ollama.OllamaLLM`` (ou qualquer objeto com
        método ``invoke(prompt: str) -> str`` compatível), já configurada.
    """

    def __init__(self, llm_runnable: object) -> None:
        self._llm_runnable = llm_runnable

    def generate(self, prompt: str) -> str:
        """Gera a resposta do servidor Ollama para um prompt, via LangChain.

        Parameters
        ----------
        prompt : str
            Prompt de entrada.

        Returns
        -------
        str
            Texto bruto gerado pelo modelo.
        """
        return str(self._llm_runnable.invoke(prompt))


class HuggingFaceLLMBackend:
    """Backend LLM via um modelo Hugging Face local, através do integrador LangChain.

    Parameters
    ----------
    llm_runnable : Any
        Instância de ``langchain_huggingface.HuggingFacePipeline`` (ou
        qualquer objeto com método ``invoke(prompt: str) -> str``
        compatível), já configurada.
    """

    def __init__(self, llm_runnable: object) -> None:
        self._llm_runnable = llm_runnable

    def generate(self, prompt: str) -> str:
        """Gera a resposta do modelo Hugging Face local para um prompt, via LangChain.

        Parameters
        ----------
        prompt : str
            Prompt de entrada.

        Returns
        -------
        str
            Texto bruto gerado pelo modelo.
        """
        return str(self._llm_runnable.invoke(prompt))


def load_ollama_llm_backend(
    model_name: str = "llama3.1:8b",
    *,
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_new_tokens: int = 300,
    seed: int = 42,
) -> LLMBackend:
    """Carrega um backend LLM via servidor Ollama local, através do integrador LangChain.

    Parameters
    ----------
    model_name : str, optional
        Nome do modelo Ollama (previamente baixado via ``ollama pull``), by
        default "llama3.1:8b" (``configs/llm.yaml ->
        backends.ollama.models``).
    base_url : str, optional
        URL do servidor Ollama local, by default "http://localhost:11434"
        (``configs/llm.yaml -> backends.ollama.base_url``).
    temperature : float, optional
        Temperatura de amostragem, by default 0.0 (determinístico).
    top_p : float, optional
        Amostragem por núcleo (nucleus sampling), by default 1.0.
    max_new_tokens : int, optional
        Número máximo de tokens gerados por resposta, by default 300.
    seed : int, optional
        Semente de geração, by default 42.

    Returns
    -------
    LLMBackend
        Backend pronto para uso em
        :class:`llm.classifier.LangChainSentimentClassifier`.

    Raises
    ------
    ModelError
        Se a biblioteca ``langchain-ollama`` não estiver instalada.

    Examples
    --------
    >>> load_ollama_llm_backend()  # doctest: +SKIP
    """
    try:
        from langchain_ollama import OllamaLLM
    except ImportError as exception:
        raise ModelError(
            "A biblioteca 'langchain-ollama' não está instalada. Instale com "
            "`uv add langchain-ollama` (ou `uv sync --extra llm`) para usar o backend "
            f"Ollama '{model_name}' via LangChain."
        ) from exception

    llm_runnable = OllamaLLM(
        model=model_name,
        base_url=base_url,
        temperature=temperature,
        top_p=top_p,
        num_predict=max_new_tokens,
        seed=seed,
    )
    logger.info(
        "Backend Ollama '%s' configurado via LangChain (base_url=%s).", model_name, base_url
    )
    return OllamaLLMBackend(llm_runnable)


def load_huggingface_llm_backend(
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    *,
    device: str = "auto",
    temperature: float = 0.0,
    max_new_tokens: int = 300,
) -> LLMBackend:
    """Carrega um backend LLM via um modelo Hugging Face local, através do integrador LangChain.

    Parameters
    ----------
    model_name : str, optional
        Identificador do modelo no Hugging Face Hub (previamente baixado
        ou disponível localmente), by default
        "meta-llama/Meta-Llama-3.1-8B-Instruct" (``configs/llm.yaml ->
        backends.huggingface.models``).
    device : str, optional
        Dispositivo de execução (``"auto"``, ``"cpu"``, ``"cuda"``), by
        default "auto" (``configs/llm.yaml -> backends.huggingface.device``).
    temperature : float, optional
        Temperatura de amostragem; ``0.0`` desativa a amostragem
        (geração gulosa/determinística), by default 0.0.
    max_new_tokens : int, optional
        Número máximo de tokens gerados por resposta, by default 300.

    Returns
    -------
    LLMBackend
        Backend pronto para uso em
        :class:`llm.classifier.LangChainSentimentClassifier`.

    Raises
    ------
    ModelError
        Se as bibliotecas ``langchain-huggingface``/``transformers`` não
        estiverem instaladas.

    Examples
    --------
    >>> load_huggingface_llm_backend()  # doctest: +SKIP
    """
    try:
        from langchain_huggingface import HuggingFacePipeline
    except ImportError as exception:
        raise ModelError(
            "As bibliotecas 'langchain-huggingface'/'transformers' não estão instaladas. "
            "Instale com `uv add langchain-huggingface transformers` (ou `uv sync --extra "
            f"llm`) para usar o backend Hugging Face '{model_name}' via LangChain."
        ) from exception

    llm_runnable = HuggingFacePipeline.from_model_id(
        model_id=model_name,
        task="text-generation",
        device_map=device,
        pipeline_kwargs={
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature if temperature > 0 else None,
        },
    )
    logger.info(
        "Backend Hugging Face '%s' configurado via LangChain (device=%s).", model_name, device
    )
    return HuggingFaceLLMBackend(llm_runnable)


_BACKEND_LOADERS = {
    "ollama": load_ollama_llm_backend,
    "huggingface": load_huggingface_llm_backend,
}


def create_llm_backend(backend_name: LLMBackendName, **overrides: object) -> LLMBackend:
    """Constrói um backend LLM a partir do nome do backend (fábrica única).

    Parameters
    ----------
    backend_name : {"ollama", "huggingface"}
        Nome do backend, uma das chaves de :data:`LLM_BACKEND_NAMES`.
    **overrides : object
        Hiperparâmetros repassados ao carregador correspondente
        (:func:`load_ollama_llm_backend` ou
        :func:`load_huggingface_llm_backend`).

    Returns
    -------
    LLMBackend
        Backend pronto para uso.

    Raises
    ------
    UnsupportedModelError
        Se ``backend_name`` não for um dos backends suportados.

    Examples
    --------
    >>> "ollama" in LLM_BACKEND_NAMES
    True
    """
    loader = _BACKEND_LOADERS.get(backend_name)
    if loader is None:
        raise UnsupportedModelError(backend_name, list(LLM_BACKEND_NAMES))
    logger.info("Criando backend LLM '%s' (overrides=%s).", backend_name, overrides)
    return loader(**overrides)
