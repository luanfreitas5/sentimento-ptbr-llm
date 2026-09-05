"""Classificação e avaliação via LLM local (Ollama/Hugging Face).

Implementa o estágio ``llm_evaluation`` de ``configs/config.yaml ->
stages``: classifica um conjunto de teste com um classificador LLM
(``src/llm/classifier.py``), em lote concorrente
(``src/inference/llm_batch.py``), e avalia o resultado com
:func:`evaluation.evaluator.evaluate_classifier`.
"""

import logging
from collections.abc import Mapping
from typing import Any

import polars as pl

from evaluation.evaluator import EvaluationResult, evaluate_classifier
from inference.llm_batch import run_llm_batch_inference
from inference.predictor import Predictor
from llm.backends import LLMBackendName, create_llm_backend
from llm.classifier import LangChainSentimentClassifier
from llm.prompts import DEFAULT_PROMPT_TEMPLATE_VERSION, PromptStrategy
from models.base import SentimentClassifier

logger = logging.getLogger(__name__)


def run_llm_evaluation_stage(
    test_dataframe: pl.DataFrame,
    *,
    text_column: str = "text",
    label_column: str = "sentiment_label",
    id_column: str = "id",
    classifier: SentimentClassifier | None = None,
    backend_name: LLMBackendName = "ollama",
    backend_overrides: Mapping[str, Any] | None = None,
    strategy: PromptStrategy = "few_shot",
    prompt_version: str = DEFAULT_PROMPT_TEMPLATE_VERSION,
    max_workers: int | None = None,
) -> tuple[pl.DataFrame, EvaluationResult]:
    """Classifica e avalia um conjunto de teste com um classificador LLM.

    Parameters
    ----------
    test_dataframe : pl.DataFrame
        Conjunto de teste, contendo ao menos ``id_column``, ``text_column``
        e ``label_column``.
    text_column : str, optional
        Coluna de texto a classificar, by default "text".
    label_column : str, optional
        Coluna de rótulo verdadeiro, by default "sentiment_label".
    id_column : str, optional
        Coluna identificadora de cada amostra, by default "id".
    classifier : SentimentClassifier | None, optional
        Classificador LLM já construído (ex.: um dublê de teste,
        satisfazendo :class:`models.base.SentimentClassifier`). Quando
        informado, ``backend_name``/``backend_overrides``/``strategy``/
        ``prompt_version`` são ignorados, by default None (constrói um
        :class:`llm.classifier.LangChainSentimentClassifier` a partir dos
        demais parâmetros).
    backend_name : {"ollama", "huggingface"}, optional
        Backend usado para construir o classificador, quando ``classifier``
        não é informado, by default "ollama".
    backend_overrides : Mapping[str, Any] | None, optional
        Hiperparâmetros do backend LLM (ver
        :func:`llm.backends.create_llm_backend`), by default None.
    strategy : {"zero_shot", "few_shot", "chain_of_thought"}, optional
        Estratégia de prompt do classificador construído, by default
        "few_shot".
    prompt_version : str, optional
        Versão do template de prompt, by default
        :data:`llm.prompts.DEFAULT_PROMPT_TEMPLATE_VERSION`.
    max_workers : int | None, optional
        Repassado a :func:`inference.llm_batch.run_llm_batch_inference`, by
        default None.

    Returns
    -------
    tuple[pl.DataFrame, EvaluationResult]
        Predições em lote (apenas as amostras classificadas com sucesso) e
        o resultado consolidado da avaliação sobre essas mesmas amostras.

    Raises
    ------
    EmptyDatasetError
        Se ``test_dataframe`` estiver vazio ou nenhuma predição for bem-sucedida.

    Examples
    --------
    >>> run_llm_evaluation_stage(test_dataframe, classifier=fake_classifier)  # doctest: +SKIP
    """
    resolved_classifier = classifier
    if resolved_classifier is None:
        backend = create_llm_backend(backend_name, **(backend_overrides or {}))
        resolved_classifier = LangChainSentimentClassifier(
            backend, strategy=strategy, prompt_version=prompt_version
        )

    predictor = Predictor(resolved_classifier)
    predictions = run_llm_batch_inference(
        predictor,
        test_dataframe[text_column].to_list(),
        ids=test_dataframe[id_column].to_list(),
        max_workers=max_workers,
    )

    ground_truth = test_dataframe.select([id_column, label_column]).rename(
        {label_column: "true_sentiment_label"}
    )
    joined = predictions.join(ground_truth, on=id_column, how="inner")
    evaluation_result = evaluate_classifier(
        joined["true_sentiment_label"].to_list(), joined["sentiment_label"].to_list()
    )

    logger.info(
        "Etapa de avaliação LLM concluída: %d/%d predição(ões) bem-sucedida(s).",
        predictions.height,
        test_dataframe.height,
    )
    return predictions, evaluation_result
