"""Anotação assíncrona de tweets com conceitos (hipóteses) via LLM.

``annotate_texts_with_concepts`` do HypotheSAEs faz N tweets × C conceitos
chamadas de LLM: por isso este módulo **exige subamostra** (trava
``validation.max_annotation_calls``), roda de forma assíncrona com limite de
concorrência, usa cache em disco e estima o custo em ``--dry-run``.

O prompt é o ``annotate`` do porte (``prompts/annotate.txt``). A resposta é
interpretada aceitando ``Yes``/``No`` e também ``Sim``/``Não`` (modelos locais
costumam responder em português). Respostas não parseáveis são contadas e, se
excederem ``llm.max_annotation_failure_rate``, o fluxo aborta: um modelo que
não segue o formato produz anotações sem valor (smoke test: ``llama3.2:1b``
teve 0/6 respostas parseáveis).
"""

import asyncio
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from config.paths import resolve_project_path
from diagnostics.llm_client import AsyncLLMClient, CallStats, DiskCompletionCache
from diagnostics.settings import DiagnosticsSettings
from exceptions.pipeline import PipelineStageError

logger = logging.getLogger(__name__)

ANNOTATION_TEMPLATE_NAME = "annotate"
ANNOTATION_MAX_TOKENS = 120
_YES_WORDS = frozenset({"yes", "sim"})
_NO_WORDS = frozenset({"no", "não", "nao"})
_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
_FIRST_WORD_PATTERN = re.compile(r"[\W\d_]*([^\W\d_]+)")


@dataclass(frozen=True)
class AnnotationResult:
    """Anotações 0/1 por conceito e contadores de qualidade.

    Attributes
    ----------
    annotations : dict[str, np.ndarray]
        Conceito -> vetor 0/1 na ordem dos textos (falhas contam como 0).
    n_total : int
        Total de anotações solicitadas.
    n_failed : int
        Respostas ausentes ou não parseáveis.
    """

    annotations: dict[str, np.ndarray]
    n_total: int
    n_failed: int

    @property
    def failure_rate(self) -> float:
        """Fração de respostas não parseáveis (0 se nada foi solicitado)."""
        return self.n_failed / self.n_total if self.n_total else 0.0


def parse_yes_no(completion: str | None) -> int | None:
    """Interpreta a primeira palavra da resposta como sim (1) ou não (0).

    Parameters
    ----------
    completion : str | None
        Resposta bruta do LLM.

    Returns
    -------
    int | None
        1 para ``yes``/``sim``, 0 para ``no``/``não``, ``None`` se ilegível.

    Examples
    --------
    >>> parse_yes_no('Yes. "sun sets" is a natural scene.')
    1
    >>> parse_yes_no("Não, o texto é neutro.")
    0
    >>> parse_yes_no("PROPERTY: ...") is None
    True
    """
    if not completion:
        return None
    cleaned = _THINK_PATTERN.sub("", completion).strip().lower()
    match = _FIRST_WORD_PATTERN.match(cleaned)
    if match is None:
        return None
    word = match.group(1)
    if word in _YES_WORDS:
        return 1
    if word in _NO_WORDS:
        return 0
    return None


def truncate_words(text: str, max_words: int) -> str:
    """Trunca um texto em ``max_words`` palavras.

    Parameters
    ----------
    text : str
        Texto de entrada.
    max_words : int
        Máximo de palavras.

    Returns
    -------
    str
        Texto truncado.

    Examples
    --------
    >>> truncate_words("a b c d", 2)
    'a b'
    """
    return " ".join(text.split()[:max_words])


def build_annotation_prompt(template: str, concept: str, text: str, *, max_words: int) -> str:
    """Preenche o template de anotação com o conceito e o texto truncado.

    Parameters
    ----------
    template : str
        Template com ``{hypothesis}`` e ``{text}`` (``prompts/annotate.txt``).
    concept : str
        Hipótese/conceito a verificar.
    text : str
        Texto sanitizado do tweet.
    max_words : int
        Máximo de palavras do texto.

    Returns
    -------
    str
        Prompt pronto.
    """
    return template.format(hypothesis=concept, text=truncate_words(text, max_words))


def enforce_annotation_budget(n_tweets: int, n_concepts: int, *, max_calls: int) -> int:
    """Garante que N × C não excede o orçamento de chamadas (sempre subamostrar).

    Parameters
    ----------
    n_tweets : int
        Tweets a anotar.
    n_concepts : int
        Conceitos a verificar.
    max_calls : int
        Orçamento (``validation.max_annotation_calls``).

    Returns
    -------
    int
        Nº de chamadas previstas.

    Raises
    ------
    PipelineStageError
        Se ``n_tweets * n_concepts`` exceder ``max_calls``.

    Examples
    --------
    >>> enforce_annotation_budget(10, 5, max_calls=100)
    50
    """
    n_calls = n_tweets * n_concepts
    if n_calls > max_calls:
        raise PipelineStageError(
            "annotation",
            f"{n_tweets} tweets x {n_concepts} conceitos = {n_calls} chamadas excede o "
            f"orçamento de {max_calls}; subamostre os tweets ou reduza os conceitos.",
        )
    return n_calls


def assert_failure_rate_acceptable(result: AnnotationResult, *, max_rate: float) -> None:
    """Aborta se muitas respostas não puderam ser interpretadas.

    Parameters
    ----------
    result : AnnotationResult
        Resultado da anotação.
    max_rate : float
        Taxa máxima tolerada (``llm.max_annotation_failure_rate``).

    Raises
    ------
    PipelineStageError
        Se ``result.failure_rate`` exceder ``max_rate``.

    Examples
    --------
    >>> assert_failure_rate_acceptable(AnnotationResult({}, 10, 1), max_rate=0.2)
    """
    if result.failure_rate > max_rate:
        raise PipelineStageError(
            "annotation",
            f"{result.failure_rate:.0%} das respostas não seguiram o formato Yes/No "
            f"(limite {max_rate:.0%}); use um modelo anotador mais capaz.",
        )


async def annotate_concepts_async(
    client: AsyncLLMClient,
    texts: Sequence[str],
    concepts: Sequence[str],
    *,
    model: str,
    template: str,
    max_words: int,
    namespace: str = ANNOTATION_TEMPLATE_NAME,
) -> AnnotationResult:
    """Anota todos os textos com todos os conceitos (produto cartesiano) de forma assíncrona.

    Parameters
    ----------
    client : AsyncLLMClient
        Cliente (com semáforo, cache e/ou dry-run).
    texts : Sequence[str]
        Textos sanitizados.
    concepts : Sequence[str]
        Hipóteses a verificar.
    model : str
        Modelo anotador.
    template : str
        Template de anotação.
    max_words : int
        Máximo de palavras por texto.
    namespace : str, optional
        Separador de cache (versão do prompt), by default "annotate".

    Returns
    -------
    AnnotationResult
        Anotações 0/1 por conceito e contadores.
    """
    prompts = [
        build_annotation_prompt(template, concept, text, max_words=max_words)
        for concept in concepts
        for text in texts
    ]
    completions = await client.complete_many(
        prompts, model=model, max_tokens=ANNOTATION_MAX_TOKENS, namespace=namespace
    )
    parsed = [parse_yes_no(completion) for completion in completions]
    n_failed = sum(value is None for value in parsed)
    values = np.array([0 if value is None else value for value in parsed], dtype=int)
    n_texts = len(texts)
    annotations = {
        concept: values[index * n_texts : (index + 1) * n_texts]
        for index, concept in enumerate(concepts)
    }
    return AnnotationResult(annotations, len(prompts), n_failed)


def load_annotation_template(name: str = ANNOTATION_TEMPLATE_NAME) -> str:
    """Carrega o template de anotação do porte (import tardio de ``hypothesaes.utils``).

    Parameters
    ----------
    name : str, optional
        Nome do template em ``prompts/`` (sem extensão), by default "annotate".

    Returns
    -------
    str
        Conteúdo do template.
    """
    from hypothesaes.utils import load_prompt_template

    return load_prompt_template(name)


def annotate_concepts(
    settings: DiagnosticsSettings,
    texts: Sequence[str],
    concepts: Sequence[str],
    *,
    dry_run: bool = False,
    client: AsyncLLMClient | None = None,
    template: str | None = None,
) -> tuple[AnnotationResult, CallStats]:
    """Versão síncrona: valida o orçamento, anota (ou estima, em dry-run) e verifica a qualidade.

    Parameters
    ----------
    settings : DiagnosticsSettings
        Configuração validada.
    texts : Sequence[str]
        Textos sanitizados (subamostrados).
    concepts : Sequence[str]
        Hipóteses a verificar.
    dry_run : bool, optional
        Se verdadeiro, não faz rede e devolve só as estatísticas previstas, by default False.
    client : AsyncLLMClient | None, optional
        Cliente pronto (testes); por padrão cria um com cache em disco, by default None.
    template : str | None, optional
        Template de anotação; por padrão o do porte, by default None.

    Returns
    -------
    tuple[AnnotationResult, CallStats]
        Resultado e contadores de chamadas/cache/tokens.

    Raises
    ------
    PipelineStageError
        Se o orçamento for excedido ou a taxa de falha de parsing for alta (fora do dry-run).
    """
    enforce_annotation_budget(
        len(texts), len(concepts), max_calls=settings.validation.max_annotation_calls
    )
    resolved_template = template if template is not None else load_annotation_template()

    async def _run() -> tuple[AnnotationResult, CallStats]:
        active = client or AsyncLLMClient(
            settings.llm,
            cache=DiskCompletionCache(resolve_project_path(settings.llm.cache_dir)),
            dry_run=dry_run,
        )
        result = await annotate_concepts_async(
            active,
            texts,
            concepts,
            model=settings.llm.annotator_model,
            template=resolved_template,
            max_words=settings.hypotheses.max_words_per_example,
        )
        return result, active.stats

    result, stats = asyncio.run(_run())
    if not dry_run:
        assert_failure_rate_acceptable(result, max_rate=settings.llm.max_annotation_failure_rate)
    logger.info(
        "Anotação: %d requisições, %d do cache, %d falhas de parsing.",
        stats.n_requests,
        stats.n_cache_hits,
        result.n_failed,
    )
    return result, stats
