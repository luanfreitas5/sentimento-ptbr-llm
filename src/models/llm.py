"""Classificador de sentimento baseado em LLM (zero-shot/few-shot), via Ollama.

Implementa a Fase 9 (adaptador do classificador LLM à interface de
``base.py``) e a Seção 4.6 do documento mestre. ``src/llm/`` (Fase 11,
orquestração LangChain completa) ainda não está implementado - ver
``src/labeling/automatic.py``; este módulo é autocontido, dependendo apenas
de um backend de geração de texto (:func:`load_ollama_backend`, sobre
``configs/llm.yaml -> backends.ollama``) para produzir um classificador
utilizável desde já por ``src/models/factory.py``.

O LLM realiza apenas a classificação; toda agregação/estatística é feita em
código Python determinístico (ver ``configs/llm.yaml``, seção
``parsing``). ``ollama`` é uma dependência opcional pesada (extra ``llm`` do
projeto): o import ocorre de forma tardia, em :func:`load_ollama_backend`,
para que o restante do módulo permaneça importável sem ela.
"""

import json
import logging
import re
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from constants.labels import SENTIMENT_CLASSES
from exceptions.model import ModelError
from utils.validation import validate_not_empty_collection

logger = logging.getLogger(__name__)

_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


class LLMBackend(Protocol):
    """Interface mínima de um backend de geração de texto por LLM.

    Permite injetar um backend real (:func:`load_ollama_backend`) ou um
    dublê de teste em :class:`LLMSentimentClassifier`, sem acoplamento à
    implementação concreta (Ollama, Hugging Face).
    """

    def generate(self, prompt: str) -> str:
        """Gera a resposta do modelo para um prompt.

        Parameters
        ----------
        prompt : str
            Prompt de entrada, via :func:`build_sentiment_prompt`.

        Returns
        -------
        str
            Texto bruto gerado pelo modelo.
        """
        ...


class _OllamaBackend:
    """Backend LLM via servidor Ollama local (``configs/llm.yaml -> backends.ollama``)."""

    def __init__(
        self,
        client: Any,
        model_name: str,
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seed: int,
    ) -> None:
        self._client = client
        self._model_name = model_name
        self._temperature = temperature
        self._top_p = top_p
        self._max_new_tokens = max_new_tokens
        self._seed = seed

    def generate(self, prompt: str) -> str:
        """Gera a resposta do servidor Ollama para um prompt.

        Parameters
        ----------
        prompt : str
            Prompt de entrada.

        Returns
        -------
        str
            Texto bruto gerado pelo modelo.
        """
        response = self._client.generate(
            model=self._model_name,
            prompt=prompt,
            options={
                "temperature": self._temperature,
                "top_p": self._top_p,
                "num_predict": self._max_new_tokens,
                "seed": self._seed,
            },
        )
        return response["response"]


def load_ollama_backend(
    model_name: str = "llama3.1:8b",
    *,
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_new_tokens: int = 300,
    seed: int = 42,
) -> LLMBackend:
    """Carrega um backend LLM via servidor Ollama local.

    Parameters
    ----------
    model_name : str, optional
        Nome do modelo Ollama (previamente baixado via ``ollama pull``), by
        default "llama3.1:8b" (``configs/llm.yaml ->
        backends.ollama.models``).
    base_url : str, optional
        URL do servidor Ollama local, by default "http://localhost:11434".
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
        Backend pronto para uso em :class:`LLMSentimentClassifier`.

    Raises
    ------
    ModelError
        Se a biblioteca ``ollama`` não estiver instalada.

    Examples
    --------
    >>> load_ollama_backend()  # doctest: +SKIP
    """
    try:
        import ollama  # type: ignore[reportMissingImports]
    except ImportError as exception:
        raise ModelError(
            "A biblioteca 'ollama' não está instalada. Instale com `uv add ollama` "
            "(ou `uv sync --extra llm`) para usar o classificador LLM."
        ) from exception

    client = ollama.Client(host=base_url)
    logger.info("Backend Ollama '%s' configurado (base_url=%s).", model_name, base_url)
    return _OllamaBackend(
        client,
        model_name,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )


def build_sentiment_prompt(
    text: str,
    *,
    few_shot_examples: Sequence[tuple[str, str]] = (),
    allowed_labels: Sequence[str] = SENTIMENT_CLASSES,
) -> str:
    """Monta o prompt de classificação de sentimento (zero-shot ou few-shot).

    Segue ``configs/llm.yaml -> prompting``: instrui o LLM a responder
    apenas com um objeto JSON contendo ``sentimento``, ``confianca`` e
    ``justificativa``, opcionalmente precedido de exemplos balanceados
    (estratégia ``few_shot``).

    Parameters
    ----------
    text : str
        Texto a ser classificado.
    few_shot_examples : Sequence[tuple[str, str]], optional
        Pares ``(texto, rótulo)`` de exemplo, via
        :func:`select_balanced_few_shot_examples`, by default ().
    allowed_labels : Sequence[str], optional
        Classes de sentimento aceitas, by default
        :data:`constants.labels.SENTIMENT_CLASSES`.

    Returns
    -------
    str
        Prompt completo, pronto para :meth:`LLMBackend.generate`.

    Examples
    --------
    >>> prompt = build_sentiment_prompt("ótimo produto", allowed_labels=("positivo", "negativo"))
    >>> prompt.endswith("Resposta:") and "\n" in prompt
    True
    """
    labels_text = ", ".join(f'"{label}"' for label in allowed_labels)
    instructions = (
        "Classifique o sentimento do texto em português brasileiro abaixo em "
        f"uma das classes {labels_text}. Responda apenas com um objeto JSON "
        'contendo as chaves "sentimento", "confianca" (entre 0.0 e 1.0) e '
        '"justificativa".'
    )
    example_blocks = [
        f'Texto: "{example_text}"\nResposta: '
        f'{{"sentimento": "{example_label}", "confianca": 1.0, "justificativa": "exemplo"}}'
        for example_text, example_label in few_shot_examples
    ]
    prompt_sections = [instructions, *example_blocks, f'Texto: "{text}"\nResposta:']
    return "\n\n".join(prompt_sections)


def parse_llm_sentiment_output(
    raw_output: str, *, allowed_labels: Sequence[str] = SENTIMENT_CLASSES
) -> tuple[str | None, float]:
    """Extrai rótulo e confiança da resposta JSON (ou quase-JSON) de um LLM.

    Parameters
    ----------
    raw_output : str
        Texto bruto gerado pelo LLM, esperado no formato instruído por
        :func:`build_sentiment_prompt`.
    allowed_labels : Sequence[str], optional
        Classes de sentimento aceitas, by default
        :data:`constants.labels.SENTIMENT_CLASSES`.

    Returns
    -------
    tuple[str | None, float]
        Par ``(rótulo, confiança)`` quando a resposta é interpretável, ou
        ``(None, 0.0)`` quando não contém um objeto JSON válido, falta a
        chave ``"sentimento"`` ou o rótulo extraído não pertence a
        ``allowed_labels``.

    Examples
    --------
    >>> parse_llm_sentiment_output('{"sentimento": "positivo", "confianca": 0.9}')
    ('positivo', 0.9)
    >>> parse_llm_sentiment_output("resposta sem json")
    (None, 0.0)
    """
    match = _JSON_OBJECT_PATTERN.search(raw_output)
    if match is None:
        logger.warning("Resposta do LLM sem objeto JSON reconhecível: %r", raw_output)
        return None, 0.0
    try:
        parsed = json.loads(match.group(0))
        label = str(parsed["sentimento"]).strip().lower()
        confidence = float(parsed.get("confianca", 0.0))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exception:
        logger.warning("Falha ao decodificar resposta do LLM: %s", exception)
        return None, 0.0

    if label not in allowed_labels:
        logger.warning("Rótulo '%s' fora das classes conhecidas %s.", label, allowed_labels)
        return None, 0.0
    return label, min(max(confidence, 0.0), 1.0)


def select_balanced_few_shot_examples(
    texts: Sequence[str],
    labels: Sequence[str],
    *,
    n_examples_per_class: int = 2,
    random_state: int = 42,
) -> list[tuple[str, str]]:
    """Seleciona exemplos balanceados por classe para compor o prompt few-shot.

    Parameters
    ----------
    texts : Sequence[str]
        Textos de treino disponíveis para seleção. Não vazio.
    labels : Sequence[str]
        Rótulos de sentimento de treino, mesmo tamanho de ``texts``.
    n_examples_per_class : int, optional
        Número de exemplos sorteados por classe, by default 2
        (``configs/llm.yaml -> prompting.few_shot.n_examples_per_class``).
    random_state : int, optional
        Semente aleatória da seleção, by default 42.

    Returns
    -------
    list[tuple[str, str]]
        Pares ``(texto, rótulo)`` selecionados, agrupados por classe na
        ordem de :data:`constants.labels.SENTIMENT_CLASSES`.

    Raises
    ------
    EmptyDatasetError
        Se ``texts`` estiver vazio.

    Examples
    --------
    >>> exemplos = select_balanced_few_shot_examples(
    ...     ["bom", "ruim", "ok"], ["positivo", "negativo", "neutro"], n_examples_per_class=1
    ... )
    >>> len(exemplos)
    3
    """
    validate_not_empty_collection(texts, collection_name="texts")
    rng = np.random.default_rng(random_state)
    examples: list[tuple[str, str]] = []
    for label in SENTIMENT_CLASSES:
        candidate_indices = [
            index for index, candidate_label in enumerate(labels) if candidate_label == label
        ]
        if not candidate_indices:
            continue
        chosen_indices = rng.choice(
            candidate_indices, size=min(n_examples_per_class, len(candidate_indices)), replace=False
        )
        examples.extend((texts[index], label) for index in chosen_indices)
    return examples


class LLMSentimentClassifier:
    """Classificador de sentimento zero-shot/few-shot baseado em LLM.

    Diferente dos demais modelos do projeto, não há otimização de
    parâmetros via gradiente: :meth:`fit` apenas seleciona exemplos
    balanceados para compor prompts few-shot
    (``configs/llm.yaml -> prompting.few_shot``), quando ``few_shot=True``.

    Parameters
    ----------
    backend : LLMBackend
        Backend de geração de texto, via :func:`load_ollama_backend` (ou um
        dublê de teste que implemente :class:`LLMBackend`).
    few_shot : bool, optional
        Se ``True``, usa os exemplos balanceados selecionados em
        :meth:`fit` na composição do prompt, by default True.
    n_examples_per_class : int, optional
        Exemplos por classe no prompt few-shot, by default 2.
    allowed_labels : Sequence[str], optional
        Classes de sentimento aceitas, by default
        :data:`constants.labels.SENTIMENT_CLASSES`.
    fallback_label : str, optional
        Rótulo usado quando, mesmo após ``max_retries`` tentativas, a
        resposta do LLM não pode ser interpretada, by default "neutro"
        (``configs/llm.yaml -> parsing.fallback_label``).
    max_retries : int, optional
        Novas tentativas de geração ao receber uma resposta não
        interpretável, by default 3.
    random_state : int, optional
        Semente da seleção de exemplos few-shot, by default 42.
    """

    def __init__(
        self,
        backend: LLMBackend,
        *,
        few_shot: bool = True,
        n_examples_per_class: int = 2,
        allowed_labels: Sequence[str] = SENTIMENT_CLASSES,
        fallback_label: str = "neutro",
        max_retries: int = 3,
        random_state: int = 42,
    ) -> None:
        self.backend = backend
        self.few_shot = few_shot
        self.n_examples_per_class = n_examples_per_class
        self.allowed_labels = tuple(allowed_labels)
        self.fallback_label = fallback_label
        self.max_retries = max_retries
        self.random_state = random_state
        self.classes_ = np.array(self.allowed_labels)
        self._few_shot_examples: list[tuple[str, str]] = []

    def fit(self, X: Sequence[str], y: Sequence[str]) -> "LLMSentimentClassifier":
        """Seleciona exemplos few-shot balanceados a partir do conjunto de treino.

        Quando ``few_shot=False``, é um no-op que apenas retorna ``self``,
        mantendo a paridade com a API scikit-learn (``fit`` antes de
        ``predict``).

        Parameters
        ----------
        X : Sequence[str]
            Textos de treino.
        y : Sequence[str]
            Rótulos de sentimento de treino, mesmo tamanho de ``X``.

        Returns
        -------
        LLMSentimentClassifier
            A própria instância.
        """
        if self.few_shot:
            validate_not_empty_collection(X, collection_name="X")
            self._few_shot_examples = select_balanced_few_shot_examples(
                list(X),
                list(y),
                n_examples_per_class=self.n_examples_per_class,
                random_state=self.random_state,
            )
            logger.info("Selecionados %d exemplo(s) few-shot.", len(self._few_shot_examples))
        return self

    def _classify_one(self, text: str) -> tuple[str, float]:
        """Classifica um único texto, com retentativas em resposta não interpretável.

        Parameters
        ----------
        text : str
            Texto a classificar.

        Returns
        -------
        tuple[str, float]
            Par ``(rótulo_de_sentimento, confiança)``. Retorna
            ``(self.fallback_label, 0.0)`` se todas as ``max_retries``
            tentativas produzirem respostas não interpretáveis.
        """
        prompt = build_sentiment_prompt(
            text, few_shot_examples=self._few_shot_examples, allowed_labels=self.allowed_labels
        )
        for attempt in range(self.max_retries):
            raw_output = self.backend.generate(prompt)
            label, confidence = parse_llm_sentiment_output(
                raw_output, allowed_labels=self.allowed_labels
            )
            if label is not None:
                return label, confidence
            logger.warning(
                "Tentativa %d/%d sem resposta interpretável; retentando.",
                attempt + 1,
                self.max_retries,
            )
        logger.warning(
            "Todas as %d tentativa(s) falharam; usando rótulo de fallback '%s'.",
            self.max_retries,
            self.fallback_label,
        )
        return self.fallback_label, 0.0

    def predict(self, X: Sequence[str]) -> np.ndarray:
        """Classifica um lote de textos.

        Parameters
        ----------
        X : Sequence[str]
            Textos a classificar.

        Returns
        -------
        np.ndarray
            Vetor de rótulos de sentimento preditos.
        """
        return np.array([self._classify_one(text)[0] for text in X])

    def predict_proba(self, X: Sequence[str]) -> np.ndarray:
        """Estima uma distribuição de probabilidade por classe a partir da confiança do LLM.

        Não é uma probabilidade calibrada: atribui a confiança relatada pelo
        LLM à classe predita e distribui o restante uniformemente entre as
        demais classes - aproximação simples, a ser usada com cautela em
        decisões que exijam calibração (ver CLAUDE.md, "Rigorous
        evaluation").

        Parameters
        ----------
        X : Sequence[str]
            Textos a classificar.

        Returns
        -------
        np.ndarray
            Matriz ``(len(X), n_classes)`` de probabilidades.
        """
        n_classes = len(self.allowed_labels)
        probabilities = np.zeros((len(X), n_classes), dtype=np.float64)
        for row_index, text in enumerate(X):
            label, confidence = self._classify_one(text)
            predicted_index = self.allowed_labels.index(label)
            remaining_probability = (1.0 - confidence) / max(n_classes - 1, 1)
            probabilities[row_index] = remaining_probability
            probabilities[row_index, predicted_index] = confidence
        return probabilities
