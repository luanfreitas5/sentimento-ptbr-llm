"""Testes estatísticos de significância para comparação entre classificadores.

Implementa a Seção 4.9 do documento mestre (Análises Estatísticas) e
``configs/evaluation.yaml`` -> ``significance_tests``: nunca afirmar que um
modelo é "melhor" que outro a partir de uma única diferença de métrica sem
testar se ela é estatisticamente significativa (ver CLAUDE.md, "Rigorous
evaluation").
"""

import logging
from collections.abc import Sequence

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, studentized_range, wilcoxon
from statsmodels.stats.contingency_tables import mcnemar

from constants.defaults import DEFAULT_SIGNIFICANCE_ALPHA
from exceptions.data import EmptyDatasetError

logger = logging.getLogger(__name__)


def run_mcnemar_test(
    y_true: Sequence[str],
    y_pred_a: Sequence[str],
    y_pred_b: Sequence[str],
    *,
    exact: bool = True,
) -> dict[str, float]:
    """Compara dois classificadores no mesmo conjunto de teste com o teste de McNemar.

    Apropriado para a comparação par a par definida em
    ``configs/evaluation.yaml`` -> ``significance_tests.pairwise``: avalia
    se a diferença de acertos entre os dois modelos é estatisticamente
    significativa, considerando apenas as amostras em que os modelos
    discordam.

    Parameters
    ----------
    y_true : Sequence[str]
        Rótulos de sentimento verdadeiros.
    y_pred_a : Sequence[str]
        Rótulos preditos pelo modelo A, mesmo tamanho de ``y_true``.
    y_pred_b : Sequence[str]
        Rótulos preditos pelo modelo B, mesmo tamanho de ``y_true``.
    exact : bool, optional
        Se ``True``, usa a distribuição binomial exata (recomendado para
        poucas discordâncias); caso contrário usa a aproximação
        qui-quadrado, by default True.

    Returns
    -------
    dict[str, float]
        Dicionário com as chaves ``statistic`` e ``p_value``.

    Raises
    ------
    EmptyDatasetError
        Se ``y_true`` estiver vazio.

    Examples
    --------
    >>> y_true = ["positivo", "negativo", "positivo", "negativo"]
    >>> y_pred_a = ["positivo", "negativo", "positivo", "negativo"]
    >>> y_pred_b = ["negativo", "negativo", "positivo", "negativo"]
    >>> resultado = run_mcnemar_test(y_true, y_pred_a, y_pred_b)
    >>> resultado["p_value"] >= 0.0
    True
    """
    if len(y_true) == 0:
        raise EmptyDatasetError("y_true")

    correct_a = np.array(
        [true == predicted for true, predicted in zip(y_true, y_pred_a)], dtype=bool
    )
    correct_b = np.array(
        [true == predicted for true, predicted in zip(y_true, y_pred_b)], dtype=bool
    )

    contingency_table = [
        [int(np.sum(correct_a & correct_b)), int(np.sum(correct_a & ~correct_b))],
        [int(np.sum(~correct_a & correct_b)), int(np.sum(~correct_a & ~correct_b))],
    ]
    result = mcnemar(contingency_table, exact=exact)
    logger.info(
        "Teste de McNemar: estatística=%.4f, p-valor=%.4f.", result.statistic, result.pvalue
    )
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}


def run_wilcoxon_signed_rank_test(
    scores_a: Sequence[float], scores_b: Sequence[float]
) -> dict[str, float]:
    """Compara dois classificadores ao longo de múltiplas dobras com o teste de Wilcoxon.

    Apropriado para a comparação pareada definida em
    ``configs/evaluation.yaml`` -> ``significance_tests.paired_multi_fold``:
    não assume normalidade das diferenças de métrica entre dobras.

    Parameters
    ----------
    scores_a : Sequence[float]
        Métrica do modelo A em cada dobra de validação cruzada.
    scores_b : Sequence[float]
        Métrica do modelo B nas mesmas dobras, mesmo tamanho de ``scores_a``.

    Returns
    -------
    dict[str, float]
        Dicionário com as chaves ``statistic`` e ``p_value``.

    Raises
    ------
    EmptyDatasetError
        Se ``scores_a`` estiver vazio.

    Examples
    --------
    >>> resultado = run_wilcoxon_signed_rank_test([0.80, 0.82, 0.79], [0.75, 0.78, 0.74])
    >>> resultado["p_value"] >= 0.0
    True
    """
    if len(scores_a) == 0:
        raise EmptyDatasetError("scores_a")
    statistic, p_value = wilcoxon(scores_a, scores_b)
    return {"statistic": float(statistic), "p_value": float(p_value)}


def run_friedman_test(*model_scores: Sequence[float]) -> dict[str, float]:
    """Compara três ou mais classificadores simultaneamente com o teste de Friedman.

    Apropriado para a comparação multi-modelo definida em
    ``configs/evaluation.yaml`` -> ``significance_tests.multi_model``,
    tipicamente seguido pelo post-hoc de Nemenyi
    (:func:`run_nemenyi_post_hoc_test`) quando significativo.

    Parameters
    ----------
    *model_scores : Sequence[float]
        Uma sequência de métricas por modelo (mesma dobra/dataset na mesma
        posição), ao menos três modelos.

    Returns
    -------
    dict[str, float]
        Dicionário com as chaves ``statistic`` e ``p_value``.

    Raises
    ------
    EmptyDatasetError
        Se algum dos vetores de ``model_scores`` estiver vazio.
    ValueError
        Se menos de três modelos forem informados.

    Examples
    --------
    >>> resultado = run_friedman_test([0.80, 0.82, 0.79], [0.75, 0.78, 0.74], [0.70, 0.71, 0.69])
    >>> resultado["p_value"] >= 0.0
    True
    """
    if len(model_scores) < 3:
        raise ValueError(
            f"run_friedman_test requer ao menos 3 modelos, recebido: {len(model_scores)}"
        )
    for scores in model_scores:
        if len(scores) == 0:
            raise EmptyDatasetError("model_scores")
    statistic, p_value = friedmanchisquare(*model_scores)
    return {"statistic": float(statistic), "p_value": float(p_value)}


def run_nemenyi_post_hoc_test(
    scores_matrix: np.ndarray, *, alpha: float = DEFAULT_SIGNIFICANCE_ALPHA
) -> dict[str, object]:
    """Calcula os ranks médios e a diferença crítica do post-hoc de Nemenyi.

    Executado após um teste de Friedman significativo
    (``configs/evaluation.yaml`` -> ``significance_tests.post_hoc``): dois
    modelos diferem significativamente quando a diferença entre seus ranks
    médios excede a diferença crítica (CD).

    Parameters
    ----------
    scores_matrix : np.ndarray
        Matriz ``(n_dobras, n_modelos)``: uma linha por dobra/dataset, uma
        coluna por modelo, com a métrica principal de cada um. Valores
        maiores devem indicar melhor desempenho.
    alpha : float, optional
        Nível de significância, by default
        :data:`constants.defaults.DEFAULT_SIGNIFICANCE_ALPHA`.

    Returns
    -------
    dict[str, object]
        Dicionário com ``average_ranks`` (``np.ndarray``, um rank médio por
        modelo/coluna; menor é melhor) e ``critical_difference`` (``float``).

    Raises
    ------
    EmptyDatasetError
        Se ``scores_matrix`` estiver vazia.
    ValueError
        Se ``scores_matrix`` tiver menos de duas colunas (modelos).

    Examples
    --------
    >>> import numpy as np
    >>> matriz = np.array([[0.80, 0.75, 0.70], [0.82, 0.78, 0.71], [0.79, 0.74, 0.69]])
    >>> resultado = run_nemenyi_post_hoc_test(matriz)
    >>> resultado["average_ranks"].tolist()
    [1.0, 2.0, 3.0]
    """
    if scores_matrix.size == 0:
        raise EmptyDatasetError("scores_matrix")
    n_folds, n_models = scores_matrix.shape
    if n_models < 2:
        raise ValueError(
            f"scores_matrix deve ter ao menos 2 modelos (colunas), recebido: {n_models}"
        )

    ranks_per_fold = np.apply_along_axis(lambda row: rankdata(-row), 1, scores_matrix)
    average_ranks = ranks_per_fold.mean(axis=0)

    studentized_range_quantile = studentized_range.ppf(1 - alpha, n_models, np.inf)
    critical_difference = (studentized_range_quantile / np.sqrt(2)) * np.sqrt(
        n_models * (n_models + 1) / (6 * n_folds)
    )
    logger.info(
        "Post-hoc de Nemenyi: ranks médios=%s, diferença crítica=%.4f.",
        average_ranks.tolist(),
        critical_difference,
    )
    return {"average_ranks": average_ranks, "critical_difference": float(critical_difference)}
