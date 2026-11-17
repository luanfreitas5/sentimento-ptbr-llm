"""Testes estatísticos de significância para comparação entre classificadores.

Implementa a Seção 4.9 do documento mestre (Análises Estatísticas) e
``configs/evaluation.yaml`` -> ``significance_tests``: nunca afirmar que um
modelo é "melhor" que outro a partir de uma única diferença de métrica sem
testar se ela é estatisticamente significativa (ver CLAUDE.md, "Rigorous
evaluation").
"""

import logging
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, studentized_range, wilcoxon
from scipy.stats import t as student_t
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

from constants.defaults import (
    DEFAULT_BOOTSTRAP_ITERATIONS,
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SIGNIFICANCE_ALPHA,
)
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
    if not y_true:
        raise EmptyDatasetError("y_true")

    correct_a = np.array(
        [true == predicted for true, predicted in zip(y_true, y_pred_a, strict=True)], dtype=bool
    )
    correct_b = np.array(
        [true == predicted for true, predicted in zip(y_true, y_pred_b, strict=True)], dtype=bool
    )

    contingency_table = [
        [int(np.sum(correct_a & correct_b)), int(np.sum(correct_a & ~correct_b))],
        [int(np.sum(~correct_a & correct_b)), int(np.sum(~correct_a & ~correct_b))],
    ]
    # statsmodels não distribui stubs de tipo para `mcnemar`: o retorno em tempo de
    # execução expõe `.statistic`/`.pvalue`, mas o checker só o vê como `_Bunch` opaco.
    result = cast(Any, mcnemar(contingency_table, exact=exact))
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
    if not scores_a:
        raise EmptyDatasetError("scores_a")
    # scipy tipa o retorno de `wilcoxon` como uma tupla genérica: o desempacotamento
    # perde o tipo concreto (`np.float64`), daí o cast explícito antes de `float()`.
    statistic, p_value = cast(tuple[float, float], wilcoxon(scores_a, scores_b))
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
        if not scores:
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

    average_ranks = np.apply_along_axis(lambda row: rankdata(-row), 1, scores_matrix).mean(axis=0)

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


_MULTIPLE_TESTING_METHODS = {"holm": "holm", "bonferroni": "bonferroni", "bh": "fdr_bh"}


def adjust_p_values(p_values: Sequence[float], *, method: str = "holm") -> list[float]:
    """Corrige uma família de p-valores para comparações múltiplas.

    Com ``k`` comparações a 5%, a chance de ao menos um falso positivo cresce com ``k``.
    Holm controla a taxa de erro por família (FWER) e é uniformemente mais potente que
    Bonferroni; Benjamini-Hochberg controla a taxa de falsas descobertas (FDR) e é mais
    permissivo, adequado a análises exploratórias com muitas comparações.

    Parameters
    ----------
    p_values : Sequence[float]
        P-valores brutos da família de testes.
    method : str, optional
        ``"holm"``, ``"bonferroni"`` ou ``"bh"`` (Benjamini-Hochberg), by default "holm".

    Returns
    -------
    list[float]
        P-valores ajustados, na ordem original.

    Raises
    ------
    ValueError
        Se ``method`` não for um dos aceitos.

    Examples
    --------
    >>> [round(p, 2) for p in adjust_p_values([0.01, 0.04, 0.03], method="holm")]
    [0.03, 0.06, 0.06]
    >>> [round(p, 2) for p in adjust_p_values([0.01, 0.04, 0.03], method="bh")]
    [0.03, 0.04, 0.04]
    """
    if method not in _MULTIPLE_TESTING_METHODS:
        raise ValueError(
            f"Método de correção desconhecido: {method!r}; use {sorted(_MULTIPLE_TESTING_METHODS)}"
        )
    if not p_values:
        return []
    # statsmodels não distribui stubs: o segundo item do retorno é o vetor de p-valores ajustados.
    adjusted = np.asarray(
        cast(Any, multipletests(list(p_values), method=_MULTIPLE_TESTING_METHODS[method]))[1]
    )
    return [float(value) for value in adjusted]


def calculate_mcnemar_effect(
    y_true: Sequence[str], y_pred_a: Sequence[str], y_pred_b: Sequence[str]
) -> dict[str, float]:
    """Resume o tamanho do efeito de uma comparação McNemar (não só o p-valor).

    Em amostras grandes qualquer diferença minúscula é "significativa"; o tamanho do
    efeito diz se ela importa. Devolve os acertos exclusivos de cada modelo, a razão de
    chances das discordâncias (com correção de 0,5 para contagens nulas) e a diferença
    de acurácia.

    Parameters
    ----------
    y_true : Sequence[str]
        Rótulos verdadeiros.
    y_pred_a : Sequence[str]
        Predições do modelo A.
    y_pred_b : Sequence[str]
        Predições do modelo B.

    Returns
    -------
    dict[str, float]
        ``a_only_correct``, ``b_only_correct``, ``odds_ratio`` (A sobre B; acima de 1
        favorece A) e ``accuracy_difference`` (A menos B).

    Raises
    ------
    EmptyDatasetError
        Se ``y_true`` estiver vazio.

    Examples
    --------
    >>> efeito = calculate_mcnemar_effect(["a", "a", "b"], ["a", "a", "a"], ["b", "b", "b"])
    >>> efeito["a_only_correct"], efeito["b_only_correct"]
    (2.0, 1.0)
    """
    if not y_true:
        raise EmptyDatasetError("y_true")
    truth = np.asarray(y_true)
    correct_a = np.asarray(y_pred_a) == truth
    correct_b = np.asarray(y_pred_b) == truth
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    return {
        "a_only_correct": float(a_only),
        "b_only_correct": float(b_only),
        "odds_ratio": (a_only + 0.5) / (b_only + 0.5),
        "accuracy_difference": float(np.mean(correct_a) - np.mean(correct_b)),
    }


def calculate_macro_f1_from_codes(
    true_codes: np.ndarray, pred_codes: np.ndarray, n_classes: int
) -> float:
    """Macro-F1 pela matriz de confusão (``bincount``), rápido o bastante para o bootstrap."""
    confusion = np.bincount(true_codes * n_classes + pred_codes, minlength=n_classes**2).reshape(
        n_classes, n_classes
    )
    true_positive = np.diag(confusion).astype(float)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1_per_class = np.divide(
        2 * true_positive, denominator, out=np.zeros(n_classes), where=denominator > 0
    )
    return float(f1_per_class.mean())


def calculate_paired_bootstrap_f1_difference(
    y_true: Sequence[str],
    y_pred_a: Sequence[str],
    y_pred_b: Sequence[str],
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> dict[str, float]:
    """Estima a diferença de macro-F1 entre dois modelos com IC por bootstrap pareado.

    Cada reamostragem usa os **mesmos** tweets para os dois modelos, preservando a
    correlação entre eles. Se o IC da diferença não contém zero, a vantagem é
    consistente; o McNemar olha acertos, este olha a métrica principal do projeto.

    Parameters
    ----------
    y_true : Sequence[str]
        Rótulos verdadeiros.
    y_pred_a : Sequence[str]
        Predições do modelo A.
    y_pred_b : Sequence[str]
        Predições do modelo B.
    n_bootstrap : int, optional
        Reamostragens, by default :data:`constants.defaults.DEFAULT_BOOTSTRAP_ITERATIONS`.
    confidence_level : float, optional
        Nível do intervalo, by default :data:`constants.defaults.DEFAULT_CONFIDENCE_LEVEL`.
    random_state : int, optional
        Semente, by default :data:`constants.defaults.DEFAULT_RANDOM_SEED`.

    Returns
    -------
    dict[str, float]
        ``f1_macro_difference`` (A menos B), ``ci_low`` e ``ci_high``.

    Raises
    ------
    EmptyDatasetError
        Se ``y_true`` estiver vazio.

    Examples
    --------
    >>> y = ["a", "b"] * 50
    >>> r = calculate_paired_bootstrap_f1_difference(y, y, ["a"] * 100, n_bootstrap=50)
    >>> r["f1_macro_difference"] > 0
    True
    """
    if not y_true:
        raise EmptyDatasetError("y_true")
    classes = sorted(set(y_true) | set(y_pred_a) | set(y_pred_b))
    code_by_class = {label: index for index, label in enumerate(classes)}
    true_codes = np.array([code_by_class[label] for label in y_true])
    codes_a = np.array([code_by_class[label] for label in y_pred_a])
    codes_b = np.array([code_by_class[label] for label in y_pred_b])
    n_classes = len(classes)
    n_samples = len(true_codes)

    point = calculate_macro_f1_from_codes(
        true_codes, codes_a, n_classes
    ) - calculate_macro_f1_from_codes(true_codes, codes_b, n_classes)
    generator = np.random.default_rng(random_state)
    differences = np.empty(n_bootstrap)
    for iteration in range(n_bootstrap):
        sample = generator.integers(0, n_samples, size=n_samples)
        differences[iteration] = calculate_macro_f1_from_codes(
            true_codes[sample], codes_a[sample], n_classes
        ) - calculate_macro_f1_from_codes(true_codes[sample], codes_b[sample], n_classes)
    tail = 100 * (1 - confidence_level) / 2
    return {
        "f1_macro_difference": float(point),
        "ci_low": float(np.percentile(differences, tail)),
        "ci_high": float(np.percentile(differences, 100 - tail)),
    }


def run_corrected_resampled_ttest(
    scores_a: Sequence[float], scores_b: Sequence[float], *, n_train: int, n_test: int
) -> dict[str, float]:
    """Teste t reamostrado corrigido (Nadeau e Bengio) para dobras de validação cruzada.

    As dobras compartilham dados de treino, então as diferenças entre elas são
    correlacionadas e o teste t comum subestima a variância (p-valores otimistas). A
    correção multiplica a variância por ``1/k + n_test/n_train``.

    Parameters
    ----------
    scores_a : Sequence[float]
        Métrica do modelo A em cada dobra.
    scores_b : Sequence[float]
        Métrica do modelo B nas mesmas dobras.
    n_train : int
        Tamanho do conjunto de treino de cada dobra.
    n_test : int
        Tamanho do conjunto de teste de cada dobra.

    Returns
    -------
    dict[str, float]
        ``statistic``, ``p_value`` (bilateral) e ``mean_difference`` (A menos B).

    Raises
    ------
    EmptyDatasetError
        Se ``scores_a`` estiver vazio.
    ValueError
        Se houver menos de duas dobras ou se os vetores tiverem tamanhos diferentes.

    Examples
    --------
    >>> r = run_corrected_resampled_ttest(
    ...     [0.80, 0.82, 0.81, 0.79, 0.83], [0.70, 0.72, 0.71, 0.69, 0.73], n_train=800, n_test=200
    ... )
    >>> r["mean_difference"] > 0
    True
    """
    if not scores_a:
        raise EmptyDatasetError("scores_a")
    if len(scores_a) != len(scores_b) or len(scores_a) < 2:
        raise ValueError("São necessárias ao menos 2 dobras, com o mesmo número para A e B.")
    differences = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    n_folds = len(differences)
    variance = differences.var(ddof=1)
    mean_difference = float(differences.mean())
    if variance == 0:
        return {
            "statistic": 0.0 if mean_difference == 0 else float("inf"),
            "p_value": 1.0 if mean_difference == 0 else 0.0,
            "mean_difference": mean_difference,
        }
    statistic = mean_difference / np.sqrt((1 / n_folds + n_test / n_train) * variance)
    p_value = 2 * student_t.sf(abs(statistic), df=n_folds - 1)
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "mean_difference": mean_difference,
    }
