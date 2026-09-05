"""Orquestração LangChain de LLMs locais (Ollama/Hugging Face) para classificação de sentimento.

Implementa a Fase 11 do plano de elaboração (``PLANO-ELABORACAO.md``) e a
Seção 4.6 do documento mestre: a contraparte mais rica de
``src/models/llm.py`` (autocontida, apenas Ollama), com backend unificado
Ollama/Hugging Face, templates de prompt versionados
(zero-shot/few-shot/cadeia de pensamento), parsing estruturado com
retentativas e um encadeamento LangChain (LCEL) opcional.

Modules
-------
prompts
    Templates de prompt versionados (zero-shot/few-shot/cadeia de
    pensamento). Sem dependências opcionais.
parsers
    Output parser estruturado (Pydantic) com retentativas. Sem
    dependências opcionais.
backends
    Interface única sobre os backends Ollama e Hugging Face, via
    ``langchain-ollama``/``langchain-huggingface`` (dependências pesadas e
    opcionais, importadas sob demanda).
chains
    Encadeamento LangChain (LCEL) prompt -> geração -> parsing, via
    ``langchain-core`` (dependência pesada e opcional, importada sob
    demanda).
classifier
    Classificador LLM completo (backend + prompt + parser), satisfazendo
    :class:`models.base.SentimentClassifier`.
"""

from llm.backends import (
    LLM_BACKEND_NAMES,
    HuggingFaceLLMBackend,
    LLMBackend,
    LLMBackendName,
    OllamaLLMBackend,
    create_llm_backend,
    load_huggingface_llm_backend,
    load_ollama_llm_backend,
)
from llm.chains import build_sentiment_classification_chain, run_chain_with_retry
from llm.classifier import LangChainSentimentClassifier
from llm.parsers import (
    SentimentLLMOutput,
    extract_json_object,
    generate_and_parse_with_retry,
    parse_structured_llm_output,
)
from llm.prompts import (
    DEFAULT_PROMPT_TEMPLATE_VERSION,
    PROMPT_STRATEGIES,
    PromptStrategy,
    build_sentiment_prompt,
)

__all__: list[str] = [
    "DEFAULT_PROMPT_TEMPLATE_VERSION",
    "LLM_BACKEND_NAMES",
    "PROMPT_STRATEGIES",
    "HuggingFaceLLMBackend",
    "LLMBackend",
    "LLMBackendName",
    "LangChainSentimentClassifier",
    "OllamaLLMBackend",
    "PromptStrategy",
    "SentimentLLMOutput",
    "build_sentiment_classification_chain",
    "build_sentiment_prompt",
    "create_llm_backend",
    "extract_json_object",
    "generate_and_parse_with_retry",
    "load_huggingface_llm_backend",
    "load_ollama_llm_backend",
    "parse_structured_llm_output",
    "run_chain_with_retry",
]
