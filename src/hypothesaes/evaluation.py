"""Avaliação estatística de hipóteses em um conjunto de dados real.

Uma vez que uma hipótese (interpretação de neurônio) é anotada em cada
texto de um corpus (0/1, ver ``annotate.py``), este módulo mede o quanto
ela realmente prediz a variável-alvo: score de separação (diferença de
médias), significância estatística (teste t, regressão com correção de
Bonferroni) e, opcionalmente, similaridade de superfície entre pares de
hipóteses via LLM (útil para deduplicar hipóteses redundantes).
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, ttest_ind
from sklearn.metrics import average_precision_score, roc_auc_score

from hypothesaes.llm_api import generate_completion
from hypothesaes.utils import load_prompt_template

logger = logging.getLogger(__name__)


def compute_pairwise_correlation_matrix(
    reference_hypotheses: dict[str, np.ndarray], predicted_hypotheses: dict[str, np.ndarray]
) -> dict[tuple[str, str], float]:
    """Calcula a correlação de Pearson entre cada par (hipótese de referência, hipótese predita).

    Parameters
    ----------
    reference_hypotheses : dict[str, np.ndarray]
        Mapa de texto da hipótese para vetor binário de anotações
        (referência/gold).
    predicted_hypotheses : dict[str, np.ndarray]
        Mapa de texto da hipótese para vetor binário de anotações
        (geradas pelo pipeline).

    Returns
    -------
    dict[tuple[str, str], float]
        Correlação de Pearson para cada par (hipótese de referência,
        hipótese predita).
    """
    return {
        (reference_hypothesis, predicted_hypothesis): pearsonr(
            reference_hypotheses[reference_hypothesis], predicted_hypotheses[predicted_hypothesis]
        )[0]
        for reference_hypothesis in reference_hypotheses
        for predicted_hypothesis in predicted_hypotheses
    }


def match_hypothesis_pairs(
    hypothesis_list_1: list[str],
    hypothesis_list_2: list[str],
    similarity_scores: dict[tuple[str, str], float],
) -> list[tuple[str, str, float]]:
    """Encontra o pareamento ótimo entre dois conjuntos de hipóteses (algoritmo húngaro).

    Parameters
    ----------
    hypothesis_list_1 : list[str]
        Primeiro conjunto de hipóteses.
    hypothesis_list_2 : list[str]
        Segundo conjunto de hipóteses (mesmo tamanho do primeiro).
    similarity_scores : dict[tuple[str, str], float]
        Score de similaridade para cada par ``(hyp1, hyp2)``.

    Returns
    -------
    list[tuple[str, str, float]]
        Lista de triplas ``(hyp1, hyp2, score_de_similaridade)`` que
        maximizam a similaridade total do pareamento.
    """
    n_hypotheses = len(hypothesis_list_1)
    if n_hypotheses != len(hypothesis_list_2):
        raise ValueError("hypothesis_list_1 e hypothesis_list_2 devem ter o mesmo tamanho")

    similarity_matrix = np.zeros((n_hypotheses, n_hypotheses))
    for i, hypothesis_i in enumerate(hypothesis_list_1):
        for j, hypothesis_j in enumerate(hypothesis_list_2):
            similarity_matrix[i, j] = similarity_scores[(hypothesis_i, hypothesis_j)]

    row_indices, col_indices = linear_sum_assignment(-similarity_matrix)  # negativo p/ maximizar

    return [
        (hypothesis_list_1[i], hypothesis_list_2[j], float(similarity_matrix[i, j]))
        for i, j in zip(row_indices, col_indices, strict=True)
    ]


def evaluate_predicate_surface_similarity(
    predicate1: str, predicate2: str, n_samples: int = 5, **kwargs: Any
) -> float:
    """Avalia a similaridade de superfície entre dois predicados/hipóteses, via LLM.

    Parameters
    ----------
    predicate1 : str
        Primeiro predicado/hipótese.
    predicate2 : str
        Segundo predicado/hipótese.
    n_samples : int, optional
        Número de amostragens do LLM (o score final é a média), by default 5.
    **kwargs : Any
        Argumentos extras repassados a :func:`~hypothesaes.llm_api.generate_completion`
        (ex.: ``model``, ``temperature``).

    Returns
    -------
    float
        Score de similaridade entre 0.0 (diferentes) e 1.0 (equivalentes),
        com 0.5 indicando relação parcial.
    """
    prompt = load_prompt_template("surface-similarity")
    scores = []

    for _ in range(n_samples):
        response = generate_completion(
            prompt=prompt.format(text_a=predicate1, text_b=predicate2), **kwargs
        )
        response = response.strip().lower()
        if response.startswith("yes"):
            scores.append(1.0)
        elif response.startswith("related"):
            scores.append(0.5)
        elif response.startswith("no"):
            scores.append(0.0)

    return sum(scores) / len(scores)


def compute_hypothesis_separation_scores(
    hypothesis_annotations: dict[str, np.ndarray], y_true: np.ndarray
) -> dict[str, tuple[float, float]]:
    """Calcula o score de separação e o p-valor (teste t) para cada hipótese.

    O score de separação é a diferença de média do alvo entre os itens que
    têm e não têm o conceito da hipótese.

    Parameters
    ----------
    hypothesis_annotations : dict[str, np.ndarray]
        Mapa de hipótese para vetor de anotações (0/1, ou -1 para dados
        pareados).
    y_true : np.ndarray
        Variável-alvo observada.

    Returns
    -------
    dict[str, tuple[float, float]]
        Mapa de hipótese para ``(tamanho_do_efeito, p_valor)``.
    """
    results = {}
    for hypothesis, annotations in hypothesis_annotations.items():
        if -1 in annotations:
            # Para dados pareados: E[Y | A == 1] + E[1 - Y | A == -1], em média.
            positive_mean = 0.5 * (
                np.mean(y_true[annotations == 1]) + np.mean(1 - y_true[annotations == -1])
            )
        else:
            positive_mean = np.mean(y_true[annotations == 1])

        negative_mean = np.mean(y_true[annotations == 0])
        effect_size = positive_mean - negative_mean

        positive_values = np.concatenate([y_true[annotations == 1], -1 * y_true[annotations == -1]])
        negative_values = y_true[annotations == 0]
        _, p_value = ttest_ind(positive_values, negative_values)

        results[hypothesis] = (effect_size, p_value)

    return results


def _compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, results: Any, classification: bool
) -> dict[str, float]:
    """Calcula as métricas de qualidade de ajuste (AUROC/AUPRC ou R²) do modelo de regressão."""
    if not classification:
        correlation, _ = pearsonr(y_true, y_pred)
        return {"r2": correlation**2}

    metrics: dict[str, float] = {
        "auroc": roc_auc_score(y_true, y_pred),
        "auprc": average_precision_score(y_true, y_pred),
    }
    if hasattr(results, "prsquared"):
        metrics["r2"] = results.prsquared
    return metrics


def compute_ols_metrics(
    hypothesis_annotations: dict[str, np.ndarray],
    y_true: np.ndarray,
    classification: bool = False,
    print_summary: bool = False,
) -> tuple[dict[str, float | tuple[int, int, float]], dict[str, tuple[float, float]]]:
    """Ajusta uma regressão (OLS ou logística) do alvo sobre as anotações de todas as hipóteses.

    Parameters
    ----------
    hypothesis_annotations : dict[str, np.ndarray]
        Mapa de hipótese para vetor de anotações (0/1).
    y_true : np.ndarray
        Variável-alvo observada.
    classification : bool, optional
        Se ``True``, ajusta regressão logística; caso contrário, OLS, by
        default False.
    print_summary : bool, optional
        Se registra o resumo estatístico completo do modelo, by default
        False.

    Returns
    -------
    tuple[dict[str, float | tuple[int, int, float]], dict[str, tuple[float, float]]]
        Par (métricas agregadas do modelo, mapa de hipótese para
        ``(coeficiente, p_valor)``).
    """
    hypotheses = list(hypothesis_annotations.keys())
    x = np.array([hypothesis_annotations[hypothesis] for hypothesis in hypotheses]).T
    x = sm.add_constant(x)

    model = sm.Logit(y_true, x) if classification else sm.OLS(y_true, x)

    try:
        results = model.fit()
    except Exception:
        logger.warning("Falha ao ajustar o modelo solicitado; tentando OLS como alternativa.")
        results = sm.OLS(y_true, x).fit()

    if print_summary:
        logger.info("%s", results.summary())

    y_pred = results.predict(x)
    metrics = _compute_regression_metrics(y_true, y_pred, results, classification)

    coefficients_and_pvalues = {
        hypothesis: (coefficient, p_value)
        for hypothesis, coefficient, p_value in zip(
            hypotheses, results.params[1:], results.pvalues[1:], strict=True
        )
    }

    return metrics, coefficients_and_pvalues


def _build_hypothesis_dataframe(
    coefficients_and_pvalues: dict[str, tuple[float, float]],
    hypothesis_annotations: dict[str, np.ndarray],
    y_true: np.ndarray,
) -> pd.DataFrame:
    """Monta o DataFrame de avaliação por hipótese
    (coeficientes, p-valores, scores de separação).
    """
    hypotheses = list(coefficients_and_pvalues.keys())
    separation_scores = compute_hypothesis_separation_scores(
        hypothesis_annotations=hypothesis_annotations, y_true=y_true
    )

    hypothesis_df = pd.DataFrame(
        {
            "hypothesis": hypotheses,
            "regression_coef": [coefficients_and_pvalues[h][0] for h in hypotheses],
            "regression_pval": [coefficients_and_pvalues[h][1] for h in hypotheses],
            "feature_prevalence": [np.mean(hypothesis_annotations[h] != 0) for h in hypotheses],
            "separation_score": [separation_scores[h][0] for h in hypotheses],
            "separation_pval": [separation_scores[h][1] for h in hypotheses],
        }
    )

    columns = [
        "hypothesis",
        "separation_score",
        "separation_pval",
        "regression_coef",
        "regression_pval",
        "feature_prevalence",
    ]
    return hypothesis_df[columns].sort_values("separation_score", ascending=False)


def score_hypotheses(
    hypothesis_annotations: dict[str, np.ndarray],
    y_true: np.ndarray,
    classification: bool = False,
    corrected_pval_threshold: float = 0.1,
    print_summary: bool = False,
) -> tuple[dict[str, float | tuple[int, int, float]], pd.DataFrame]:
    """Avalia um conjunto de hipóteses em um dataset real (rotulado).

    Parameters
    ----------
    hypothesis_annotations : dict[str, np.ndarray]
        Mapa de hipótese para vetor de anotações (0/1).
    y_true : np.ndarray
        Variável-alvo observada.
    classification : bool, optional
        Se esta é uma tarefa de classificação binária, by default False.
    corrected_pval_threshold : float, optional
        Limiar de significância antes da correção de Bonferroni, by
        default 0.1.
    print_summary : bool, optional
        Se registra o resumo estatístico completo do modelo de regressão,
        by default False.

    Returns
    -------
    tuple[dict[str, float | tuple[int, int, float]], pd.DataFrame]
        Par (métricas agregadas, DataFrame com uma linha por hipótese:
        ``hypothesis``, ``separation_score``, ``separation_pval``,
        ``regression_coef``, ``regression_pval``, ``feature_prevalence``,
        ordenado por ``separation_score`` decrescente).
    """
    metrics, coefficients_and_pvalues = compute_ols_metrics(
        hypothesis_annotations=hypothesis_annotations,
        y_true=y_true,
        classification=classification,
        print_summary=print_summary,
    )

    corrected_pvalue_threshold = corrected_pval_threshold / len(hypothesis_annotations)
    significant_hypotheses = [
        hypothesis
        for hypothesis, (_, p_value) in coefficients_and_pvalues.items()
        if p_value < corrected_pvalue_threshold
    ]
    metrics["Significant"] = (
        len(significant_hypotheses),
        len(hypothesis_annotations),
        corrected_pvalue_threshold,
    )

    hypothesis_df = _build_hypothesis_dataframe(
        coefficients_and_pvalues, hypothesis_annotations, y_true
    )

    return metrics, hypothesis_df
