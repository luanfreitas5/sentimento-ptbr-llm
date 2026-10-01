"""Análise de *ablation study*: impacto de cada componente no desempenho do modelo.

Implementa ``configs/evaluation.yaml`` -> ``ablation``: compara a métrica
principal de uma configuração completa (baseline) contra versões com um
componente removido de cada vez (ex.: ``sem_embeddings_contextuais``),
quantificando a contribuição de cada peça do pipeline.
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import f1_score

from constants.defaults import (
    DEFAULT_BOOTSTRAP_ITERATIONS,
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_RANDOM_SEED,
)
from constants.metrics import PRIMARY_METRIC
from exceptions.data import EmptyDatasetError
from features.lexical import LexicalTfidfVectorizer
from metrics.classification import calculate_classification_metrics
from models.factory import create_classifier

logger = logging.getLogger(__name__)


def calculate_ablation_impact(
    baseline_metrics: dict[str, float],
    ablated_metrics: dict[str, dict[str, float]],
    *,
    metric_name: str = PRIMARY_METRIC,
) -> pl.DataFrame:
    """Calcula o impacto da remoção de cada componente sobre a métrica principal.

    Parameters
    ----------
    baseline_metrics : dict[str, float]
        Métricas da configuração completa (todos os componentes ativos),
        no formato de :func:`metrics.classification.calculate_classification_metrics`.
    ablated_metrics : dict[str, dict[str, float]]
        Métricas de cada configuração com um componente removido, indexadas
        pelo nome do componente (ver ``configs/evaluation.yaml`` ->
        ``ablation.components``).
    metric_name : str, optional
        Métrica usada para comparação, by default
        :data:`constants.metrics.PRIMARY_METRIC`.

    Returns
    -------
    pl.DataFrame
        Uma linha por componente removido, com ``component``,
        ``baseline_value``, ``ablated_value`` e ``impact`` (queda de
        desempenho ao remover o componente; quanto maior, mais importante o
        componente), ordenada do componente mais para o menos importante.

    Raises
    ------
    EmptyDatasetError
        Se ``ablated_metrics`` estiver vazio.
    ValueError
        Se ``metric_name`` não existir em ``baseline_metrics`` ou em algum
        item de ``ablated_metrics``.

    Examples
    --------
    >>> baseline = {"f1_macro": 0.80}
    >>> ablado = {
    ...     "sem_embeddings_contextuais": {"f1_macro": 0.65},
    ...     "sem_autoencoder": {"f1_macro": 0.78},
    ... }
    >>> resultado = calculate_ablation_impact(baseline, ablado)
    >>> resultado["component"].to_list()
    ['sem_embeddings_contextuais', 'sem_autoencoder']
    """
    if not ablated_metrics:
        raise EmptyDatasetError("ablated_metrics")
    if metric_name not in baseline_metrics:
        raise ValueError(f"metric_name '{metric_name}' não encontrado em baseline_metrics")

    baseline_value = baseline_metrics[metric_name]
    rows: list[dict[str, str | float]] = []
    for component_name, component_metrics in ablated_metrics.items():
        if metric_name not in component_metrics:
            raise ValueError(
                f"metric_name '{metric_name}' não encontrado nas métricas do componente "
                f"'{component_name}'"
            )
        ablated_value = component_metrics[metric_name]
        rows.append(
            {
                "component": component_name,
                "baseline_value": baseline_value,
                "ablated_value": ablated_value,
                "impact": baseline_value - ablated_value,
            }
        )

    result = pl.DataFrame(rows).sort("impact", descending=True)
    logger.info("Impacto de ablation calculado para %d componente(s).", result.height)
    return result


def identify_most_impactful_component(ablation_impact: pl.DataFrame) -> dict[str, str | float]:
    """Identifica o componente cuja remoção mais degrada a métrica principal.

    Parameters
    ----------
    ablation_impact : pl.DataFrame
        Saída de :func:`calculate_ablation_impact`, já ordenada por
        ``impact`` decrescente.

    Returns
    -------
    dict[str, str | float]
        Primeira linha de ``ablation_impact`` (maior impacto), como
        dicionário.

    Raises
    ------
    EmptyDatasetError
        Se ``ablation_impact`` estiver vazio.

    Examples
    --------
    >>> import polars as pl
    >>> impacto = pl.DataFrame(
    ...     {
    ...         "component": ["a", "b"],
    ...         "baseline_value": [0.8, 0.8],
    ...         "ablated_value": [0.5, 0.7],
    ...         "impact": [0.3, 0.1],
    ...     }
    ... )
    >>> identify_most_impactful_component(impacto)["component"]
    'a'
    """
    if ablation_impact.height == 0:
        raise EmptyDatasetError("ablation_impact")
    return ablation_impact.row(0, named=True)


def calculate_paired_bootstrap_difference(
    y_true: Sequence[str],
    y_pred_full: Sequence[str],
    y_pred_ablated: Sequence[str],
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> tuple[float, float]:
    """Intervalo bootstrap pareado da queda de F1-macro ao remover um componente.

    As duas configurações são avaliadas sobre as mesmas reamostragens, de modo que o intervalo
    reflete a incerteza da *diferença* (``F1 completo - F1 ablado``) e não de cada métrica isolada.

    Parameters
    ----------
    y_true : Sequence[str]
        Rótulos verdadeiros.
    y_pred_full : Sequence[str]
        Predições da configuração completa.
    y_pred_ablated : Sequence[str]
        Predições da configuração sem o componente, mesmo tamanho de ``y_true``.
    n_bootstrap : int, optional
        Número de reamostragens, by default
        :data:`constants.defaults.DEFAULT_BOOTSTRAP_ITERATIONS`.
    confidence_level : float, optional
        Nível de confiança, by default :data:`constants.defaults.DEFAULT_CONFIDENCE_LEVEL`.
    random_state : int, optional
        Semente, by default :data:`constants.defaults.DEFAULT_RANDOM_SEED`.

    Returns
    -------
    tuple[float, float]
        Limites inferior e superior do intervalo da diferença.

    Raises
    ------
    EmptyDatasetError
        Se ``y_true`` estiver vazio.

    Examples
    --------
    >>> low, high = calculate_paired_bootstrap_difference(
    ...     ["positivo", "negativo"] * 10,
    ...     ["positivo", "negativo"] * 10,
    ...     ["positivo", "positivo"] * 10,
    ...     n_bootstrap=50,
    ... )
    >>> low <= high
    True
    """
    if len(y_true) == 0:
        raise EmptyDatasetError("y_true")

    generator = np.random.default_rng(random_state)
    truth = np.asarray(y_true)
    full = np.asarray(y_pred_full)
    ablated = np.asarray(y_pred_ablated)

    differences = np.empty(n_bootstrap)
    for iteration in range(n_bootstrap):
        sample = generator.integers(0, len(truth), size=len(truth))
        f1_full = f1_score(
            truth[sample],
            full[sample],
            average="macro",
            zero_division=0,  # type: ignore[reportArgumentType]
        )
        f1_ablated = f1_score(
            truth[sample],
            ablated[sample],
            average="macro",
            zero_division=0,  # type: ignore[reportArgumentType]
        )
        differences[iteration] = f1_full - f1_ablated

    alpha = 1 - confidence_level
    return (
        float(np.percentile(differences, 100 * alpha / 2)),
        float(np.percentile(differences, 100 * (1 - alpha / 2))),
    )


def run_pipeline_ablation(
    train_texts: Sequence[str],
    train_labels: Sequence[str],
    eval_texts: Sequence[str],
    eval_labels: Sequence[str],
    *,
    components: Mapping[str, Mapping[str, Any]],
    model_name: str = "logistic_regression",
    model_params: Mapping[str, Any] | None = None,
    tfidf_params: Mapping[str, Any] | None = None,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> pl.DataFrame:
    """Ablação do pipeline clássico: retreina removendo um componente por vez.

    A configuração completa (``model_params`` + ``tfidf_params``) é comparada a variantes em que
    cada componente é desligado por sobrescritas declaradas em ``components`` (ex.:
    ``{"sem_bigramas": {"tfidf": {"ngram_range": [1, 1]}}}``). Cada variante é treinada no
    treino e avaliada no conjunto ``eval_*`` (use a validação, nunca o teste, para não orientar
    decisões de projeto com o conjunto final).

    Parameters
    ----------
    train_texts : Sequence[str]
        Textos de treino normalizados.
    train_labels : Sequence[str]
        Rótulos de treino.
    eval_texts : Sequence[str]
        Textos do conjunto de avaliação da ablação.
    eval_labels : Sequence[str]
        Rótulos do conjunto de avaliação.
    components : Mapping[str, Mapping[str, Any]]
        Para cada componente, sobrescritas ``{"tfidf": {...}, "model": {...}}`` (ambas opcionais).
    model_name : str, optional
        Modelo da fábrica usado na ablação, by default "logistic_regression".
    model_params : Mapping[str, Any] | None, optional
        Hiperparâmetros da configuração completa, by default None.
    tfidf_params : Mapping[str, Any] | None, optional
        Parâmetros do :class:`features.lexical.LexicalTfidfVectorizer` da configuração completa,
        by default None.
    n_bootstrap : int, optional
        Reamostragens do IC pareado, by default
        :data:`constants.defaults.DEFAULT_BOOTSTRAP_ITERATIONS`.
    confidence_level : float, optional
        Nível de confiança do IC, by default :data:`constants.defaults.DEFAULT_CONFIDENCE_LEVEL`.
    random_state : int, optional
        Semente do bootstrap, by default :data:`constants.defaults.DEFAULT_RANDOM_SEED`.

    Returns
    -------
    pl.DataFrame
        Saída de :func:`calculate_ablation_impact` acrescida de ``impact_ci_low`` e
        ``impact_ci_high`` (IC pareado da queda de F1-macro).

    Raises
    ------
    EmptyDatasetError
        Se ``components`` estiver vazio.

    Examples
    --------
    >>> run_pipeline_ablation(
    ...     textos, rotulos, textos_val, rotulos_val, components={"sem_bigramas": {}}
    ... )  # doctest: +SKIP
    """
    if not components:
        raise EmptyDatasetError("components")

    def _fit_predict(
        tfidf_overrides: Mapping[str, Any], model_overrides: Mapping[str, Any]
    ) -> list[str]:
        vectorizer = LexicalTfidfVectorizer(**dict(tfidf_overrides))
        features_train = vectorizer.fit(train_texts).transform(train_texts)
        model = create_classifier(model_name, **dict(model_overrides))
        model.fit(features_train, list(train_labels))
        return [str(label) for label in model.predict(vectorizer.transform(eval_texts))]

    base_tfidf = dict(tfidf_params or {})
    base_model = dict(model_params or {})
    full_predictions = _fit_predict(base_tfidf, base_model)
    baseline_metrics = calculate_classification_metrics(list(eval_labels), full_predictions)

    ablated_metrics: dict[str, dict[str, float]] = {}
    intervals: dict[str, tuple[float, float]] = {}
    for component_name, overrides in components.items():
        ablated_predictions = _fit_predict(
            {**base_tfidf, **overrides.get("tfidf", {})},
            {**base_model, **overrides.get("model", {})},
        )
        ablated_metrics[component_name] = calculate_classification_metrics(
            list(eval_labels), ablated_predictions
        )
        intervals[component_name] = calculate_paired_bootstrap_difference(
            eval_labels,
            full_predictions,
            ablated_predictions,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_state=random_state,
        )

    impact = calculate_ablation_impact(baseline_metrics, ablated_metrics)
    return impact.with_columns(
        pl.col("component")
        .map_elements(lambda name: intervals[name][0], return_dtype=pl.Float64)
        .alias("impact_ci_low"),
        pl.col("component")
        .map_elements(lambda name: intervals[name][1], return_dtype=pl.Float64)
        .alias("impact_ci_high"),
    )
