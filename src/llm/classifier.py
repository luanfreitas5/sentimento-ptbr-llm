"""Classificador LLM completo (backend + prompt + parser), via LangChain.

Implementa a Fase 11: a contraparte mais rica de
:class:`models.llm.LLMSentimentClassifier` (que fala diretamente com o SDK
``ollama`` e é autocontida), compondo os blocos de ``src/llm/backends.py``
(Ollama **ou** Hugging Face, via LangChain), ``src/llm/prompts.py``
(zero-shot/few-shot/cadeia de pensamento) e ``src/llm/parsers.py`` (parsing
estruturado com retentativas). Satisfaz
:class:`models.base.SentimentClassifier` por duck typing, podendo ser usado
por ``src/training`` e ``src/inference`` como qualquer outro modelo do
projeto.

Compõe ``backend`` + ``prompt`` + ``parser`` diretamente (não via
``src/llm/chains.py``), para permanecer utilizável mesmo quando apenas
``pydantic`` está disponível (o backend concreto passado ao construtor é
que eventualmente depende de ``langchain-ollama``/``langchain-huggingface``
- ver ``src/llm/backends.py``). ``src/llm/chains.py`` oferece, à parte, uma
composição LCEL equivalente para quem precisa de um ``Runnable``
LangChain nativo (ex.: integração com ferramentas de observabilidade
LangChain).
"""

import logging
from collections.abc import Sequence

import numpy as np

from constants.labels import NEUTRAL_LABEL, SENTIMENT_CLASSES
from llm.backends import LLMBackend
from llm.parsers import SentimentLLMOutput, generate_and_parse_with_retry
from llm.prompts import DEFAULT_PROMPT_TEMPLATE_VERSION, PromptStrategy, build_sentiment_prompt
from models.llm import select_balanced_few_shot_examples
from utils.validation import validate_not_empty_collection

logger = logging.getLogger(__name__)


class LangChainSentimentClassifier:
    """Classificador de sentimento zero-shot/few-shot/CoT, orquestrado via LangChain.

    Assim como :class:`models.llm.LLMSentimentClassifier`, não há
    otimização de parâmetros via gradiente: :meth:`fit` apenas seleciona
    exemplos balanceados quando ``strategy`` exige exemplos
    (``"few_shot"``/``"chain_of_thought"``).

    Parameters
    ----------
    backend : LLMBackend
        Backend de geração de texto, via
        :func:`llm.backends.create_llm_backend` (ou um dublê de teste que
        implemente :class:`llm.backends.LLMBackend`).
    strategy : {"zero_shot", "few_shot", "chain_of_thought"}, optional
        Estratégia de engenharia de prompt, by default "few_shot"
        (``configs/llm.yaml -> prompting.default_strategy``).
    n_examples_per_class : int, optional
        Exemplos por classe no prompt, quando ``strategy`` usa exemplos, by
        default 2 (``configs/llm.yaml ->
        prompting.few_shot.n_examples_per_class``).
    allowed_labels : Sequence[str], optional
        Classes de sentimento aceitas, by default
        :data:`constants.labels.SENTIMENT_CLASSES`.
    fallback_label : str, optional
        Rótulo usado quando, mesmo após ``max_retries`` tentativas, a
        resposta do LLM não pode ser interpretada, by default
        :data:`constants.labels.NEUTRAL_LABEL` (``configs/llm.yaml ->
        parsing.fallback_label``).
    max_retries : int, optional
        Novas tentativas de geração ao receber uma resposta não
        interpretável, by default 3 (``configs/llm.yaml ->
        parsing.max_retries``).
    prompt_version : str, optional
        Versão do template de prompt, by default
        :data:`llm.prompts.DEFAULT_PROMPT_TEMPLATE_VERSION`.
    random_state : int, optional
        Semente da seleção de exemplos, by default 42.
    """

    def __init__(
        self,
        backend: LLMBackend,
        *,
        strategy: PromptStrategy = "few_shot",
        n_examples_per_class: int = 2,
        allowed_labels: Sequence[str] = SENTIMENT_CLASSES,
        fallback_label: str = NEUTRAL_LABEL,
        max_retries: int = 3,
        prompt_version: str = DEFAULT_PROMPT_TEMPLATE_VERSION,
        random_state: int = 42,
    ) -> None:
        self.backend = backend
        self.strategy = strategy
        self.n_examples_per_class = n_examples_per_class
        self.allowed_labels = tuple(allowed_labels)
        self.fallback_label = fallback_label
        self.max_retries = max_retries
        self.prompt_version = prompt_version
        self.random_state = random_state
        self.classes_ = np.array(self.allowed_labels)
        self._few_shot_examples: list[tuple[str, str]] = []

    def fit(self, X: Sequence[str], y: Sequence[str]) -> "LangChainSentimentClassifier":
        """Seleciona exemplos balanceados a partir do treino, quando a estratégia exige exemplos.

        Não tem efeito quando ``strategy="zero_shot"``, mantendo a paridade
        com a API scikit-learn (``fit`` antes de ``predict``).

        Parameters
        ----------
        X : Sequence[str]
            Textos de treino.
        y : Sequence[str]
            Rótulos de sentimento de treino, mesmo tamanho de ``X``.

        Returns
        -------
        LangChainSentimentClassifier
            A própria instância.
        """
        if self.strategy != "zero_shot":
            validate_not_empty_collection(X, collection_name="X")
            self._few_shot_examples = select_balanced_few_shot_examples(
                list(X),
                list(y),
                n_examples_per_class=self.n_examples_per_class,
                random_state=self.random_state,
            )
            logger.info("Selecionados %d exemplo(s) para o prompt.", len(self._few_shot_examples))
        return self

    def _classify_one(self, text: str) -> SentimentLLMOutput:
        """Classifica um único texto, retornando a saída estruturada completa.

        Parameters
        ----------
        text : str
            Texto a classificar.

        Returns
        -------
        SentimentLLMOutput
            Rótulo, confiança e justificativa preditos.
        """
        prompt = build_sentiment_prompt(
            text,
            strategy=self.strategy,
            few_shot_examples=self._few_shot_examples,
            allowed_labels=self.allowed_labels,
            version=self.prompt_version,
        )
        return generate_and_parse_with_retry(
            self.backend,
            prompt,
            allowed_labels=self.allowed_labels,
            max_retries=self.max_retries,
            fallback_label=self.fallback_label,
        )

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
        return np.array([self._classify_one(text).sentimento for text in X])

    def predict_proba(self, X: Sequence[str]) -> np.ndarray:
        """Estima uma distribuição de probabilidade por classe a partir da confiança do LLM.

        Não é uma probabilidade calibrada: atribui a confiança relatada
        pelo LLM à classe predita e distribui o restante uniformemente
        entre as demais classes (ver CLAUDE.md, "Rigorous evaluation").

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
            output = self._classify_one(text)
            predicted_index = self.allowed_labels.index(output.sentimento)
            remaining_probability = (1.0 - output.confianca) / max(n_classes - 1, 1)
            probabilities[row_index] = remaining_probability
            probabilities[row_index, predicted_index] = output.confianca
        return probabilities

    def predict_with_justification(self, X: Sequence[str]) -> list[SentimentLLMOutput]:
        """Classifica um lote de textos, preservando a justificativa textual de cada predição.

        Parameters
        ----------
        X : Sequence[str]
            Textos a classificar.

        Returns
        -------
        list[SentimentLLMOutput]
            Uma saída estruturada completa (rótulo, confiança,
            justificativa) por texto de entrada.

        Examples
        --------
        >>> class _Backend:
        ...     def generate(self, prompt: str) -> str:
        ...         return '{"sentimento": "positivo", "confianca": 0.9, "justificativa": "ok"}'
        >>> classificador = LangChainSentimentClassifier(_Backend(), strategy="zero_shot")
        >>> classificador.predict_with_justification(["ótimo"])[0].justificativa
        'ok'
        """
        return [self._classify_one(text) for text in X]
