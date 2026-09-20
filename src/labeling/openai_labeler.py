"""Rotulagem de sentimento via API OpenAI-compatível (base ``tweets_data_openai``).

Implementa a seção ``openai`` de ``configs/labeling.yaml``: cada tweet é
classificado por um LLM acessado pelo cliente único do projeto
(:func:`hypothesaes.llm_api.generate_completion`, autenticado via
``OPENAI_BASE_URL``/``OPENAI_KEY`` do ``.env``), com o prompt versionado em
``prompts/``.

Robustez frente à API:

* pausa (``time.sleep``) de ``request_interval_seconds`` antes de cada
  chamada, para respeitar o limite de taxa e evitar HTTP 429;
* *timeout* por requisição e nova tentativa com backoff exponencial em caso
  de erro de rede, limite de taxa ou resposta fora do formato;
* falhas de configuração (chave ausente, biblioteca não instalada) não são
  retentadas: interrompem a execução, e o checkpoint preserva o progresso;
* um tweet que esgota as tentativas devolve ``None`` (ver
  :mod:`labeling.incremental`) sem derrubar os demais do lote.
"""

import concurrent.futures
import logging
import time
from collections.abc import Sequence

from constants.labels import SENTIMENT_CLASSES
from exceptions.base import ProjectError
from hypothesaes.llm_api import generate_completion
from labeling.incremental import BatchClassifier, LabelPrediction
from labeling.llm_response import build_labeling_prompt, parse_llm_label_response

logger = logging.getLogger(__name__)

_MINIMUM_RETRY_WAIT_SECONDS = 1.0


def _classify_single_text(
    text: str,
    prompt_template: str,
    *,
    model: str,
    temperature: float,
    max_retries: int,
    request_interval_seconds: float,
    request_timeout_seconds: float,
    allowed_labels: Sequence[str],
) -> LabelPrediction | None:
    """Classifica um tweet, com pausa entre chamadas e novas tentativas com backoff.

    Parameters
    ----------
    text : str
        Texto sanitizado do tweet.
    prompt_template : str
        Template do prompt (marcador ``{{TEXTO}}``).
    model : str
        Modelo da API OpenAI-compatível.
    temperature : float
        Temperatura de amostragem (0.0 = determinístico).
    max_retries : int
        Máximo de tentativas por tweet (chamada + interpretação da resposta).
    request_interval_seconds : float
        Pausa antes de cada chamada; também é a base do backoff entre tentativas.
    request_timeout_seconds : float
        Timeout de cada requisição, em segundos.
    allowed_labels : Sequence[str]
        Classes de sentimento aceitas.

    Returns
    -------
    tuple[str, float] | None
        ``(rótulo, confiança)`` ou ``None`` se todas as tentativas falharem.

    Raises
    ------
    ProjectError
        Falhas de configuração (ex.: ``OPENAI_KEY`` ausente) são propagadas sem retentativa.
    """
    prompt = build_labeling_prompt(prompt_template, text)
    for attempt in range(max_retries):
        if request_interval_seconds > 0:
            time.sleep(request_interval_seconds)
        try:
            # max_retries=1: a retentativa (com backoff próprio) é feita neste laço.
            raw_response = generate_completion(
                prompt=prompt,
                model=model,
                temperature=temperature,
                provider="openai",
                timeout=request_timeout_seconds,
                max_retries=1,
            )
        except ProjectError:
            raise
        except Exception as exception:
            logger.warning(
                "Falha na chamada à API (tentativa %d/%d): %s", attempt + 1, max_retries, exception
            )
            time.sleep(max(_MINIMUM_RETRY_WAIT_SECONDS, request_interval_seconds) * (2**attempt))
            continue
        parsed = parse_llm_label_response(raw_response, allowed_labels=allowed_labels)
        if parsed is not None:
            return parsed
        logger.warning("Resposta fora do formato (tentativa %d/%d).", attempt + 1, max_retries)
    return None


def create_openai_batch_classifier(
    prompt_template: str,
    *,
    model: str,
    temperature: float = 0.0,
    max_retries: int = 3,
    n_workers: int = 4,
    request_interval_seconds: float = 1.0,
    request_timeout_seconds: float = 60.0,
    allowed_labels: Sequence[str] = SENTIMENT_CLASSES,
) -> BatchClassifier:
    """Cria o classificador de lotes da base ``tweets_data_openai``.

    Parameters
    ----------
    prompt_template : str
        Template do prompt (marcador ``{{TEXTO}}``).
    model : str
        Modelo da API OpenAI-compatível (``configs/labeling.yaml -> openai.model``).
    temperature : float, optional
        Temperatura de amostragem, by default 0.0.
    max_retries : int, optional
        Máximo de tentativas por tweet, by default 3.
    n_workers : int, optional
        Chamadas simultâneas dentro de um lote; a taxa efetiva é
        ``n_workers / request_interval_seconds`` requisições por segundo, by default 4.
    request_interval_seconds : float, optional
        Pausa (``time.sleep``) antes de cada chamada, para evitar HTTP 429, by default 1.0.
    request_timeout_seconds : float, optional
        Timeout por requisição, by default 60.0.
    allowed_labels : Sequence[str], optional
        Classes aceitas, by default :data:`constants.labels.SENTIMENT_CLASSES`.

    Returns
    -------
    BatchClassifier
        Função ``textos -> [(rótulo, confiança) | None]`` para
        :func:`labeling.incremental.run_incremental_labeling`.

    Examples
    --------
    >>> classifier = create_openai_batch_classifier(
    ...     'Tweet: "{{TEXTO}}"', model="gpt-5-mini"
    ... )  # doctest: +SKIP
    """

    def classify_batch(texts: Sequence[str]) -> list[LabelPrediction | None]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    _classify_single_text,
                    text,
                    prompt_template,
                    model=model,
                    temperature=temperature,
                    max_retries=max_retries,
                    request_interval_seconds=request_interval_seconds,
                    request_timeout_seconds=request_timeout_seconds,
                    allowed_labels=allowed_labels,
                )
                for text in texts
            ]
            return [future.result() for future in futures]

    return classify_batch
