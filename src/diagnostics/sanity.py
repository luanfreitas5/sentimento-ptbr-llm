"""Gate de sanidade: os embeddings preveem o alvo acima do acaso?

Antes de gastar chamadas de LLM gerando hipóteses, ajusta uma regressão Ridge
nos embeddings de treino e mede o desempenho no holdout (ROC-AUC para alvo
binário, correlação de Spearman para alvo contínuo). Se o desempenho for indistinguível do acaso,
não há sinal nos embeddings para o SAE explicar e o fluxo deve abortar (e
registrar o motivo no MLflow).

A pergunta do gate é "há sinal?", não "a previsão é bem calibrada?". Por isso o
alvo contínuo usa Spearman (invariante à escala e a transformações monótonas), e
não R², que penaliza previsões com escala errada mesmo quando ordenam bem o alvo.

O critério é conjunto: o limite inferior do IC (bootstrap do holdout) precisa
superar o nível de acaso e o teste de permutação precisa rejeitar a hipótese
nula de associação nula entre predição e alvo.
"""

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from constants.defaults import DEFAULT_BOOTSTRAP_ITERATIONS, DEFAULT_RANDOM_SEED
from exceptions.data import DataValidationError
from exceptions.pipeline import SanityGateFailedError

logger = logging.getLogger(__name__)

MetricName = Literal["roc_auc", "spearman"]
_CHANCE_LEVEL: dict[str, float] = {"roc_auc": 0.5, "spearman": 0.0}
DEFAULT_RIDGE_ALPHAS: tuple[float, ...] = (1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SanityGateResult:
    """Resultado do gate de sanidade para um alvo.

    Attributes
    ----------
    target_name : str
        Nome do alvo avaliado.
    metric_name : {"roc_auc", "spearman"}
        Métrica usada (AUC para binário, Spearman para contínuo).
    holdout_score : float
        Métrica observada no holdout (``nan`` se o alvo for degenerado).
    ci_lower, ci_upper : float
        Limites do IC por bootstrap do holdout.
    permutation_pvalue : float
        p-valor unilateral do teste de permutação.
    n_train, n_holdout : int
        Tamanhos das partições.
    go : bool
        ``True`` se o alvo passa no gate.
    reason : str
        Explicação legível da decisão.
    null_median : float
        Mediana da métrica sob permutação do alvo (nível nulo empírico). Para
        Spearman é ~0 (alvo embaralhado não se associa à previsão).
    """

    target_name: str
    metric_name: MetricName
    holdout_score: float
    ci_lower: float
    ci_upper: float
    permutation_pvalue: float
    n_train: int
    n_holdout: int
    go: bool
    reason: str
    null_median: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        """Converte o resultado em dicionário (ex.: para registrar no MLflow)."""
        return asdict(self)


def _score(metric_name: MetricName, y_true: FloatArray, y_pred: FloatArray) -> float:
    """Calcula a métrica escolhida (AUC ou Spearman; ``nan`` se um vetor for constante)."""
    if metric_name == "roc_auc":
        return float(roc_auc_score(y_true, y_pred))
    if 0 in (np.std(y_true), np.std(y_pred)):
        return float("nan")
    result = spearmanr(y_true, y_pred)
    return float(result.statistic)  # pyright: ignore[reportAttributeAccessIssue]


def _has_two_classes(values: FloatArray) -> bool:
    """Indica se um vetor binário contém as duas classes."""
    return np.unique(values).size > 1


def _bootstrap_interval(
    metric_name: MetricName,
    y_true: FloatArray,
    y_pred: FloatArray,
    *,
    n_bootstrap: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """IC percentil por reamostragem do holdout (descarta reamostras degeneradas)."""
    n_rows = y_true.size
    scores: list[float] = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n_rows, n_rows)
        if metric_name == "roc_auc" and not _has_two_classes(y_true[indices]):
            continue
        score = _score(metric_name, y_true[indices], y_pred[indices])
        if not np.isnan(score):
            scores.append(score)
    if not scores:
        return float("nan"), float("nan")
    tail = (1.0 - confidence_level) / 2.0 * 100.0
    lower, upper = np.percentile(scores, [tail, 100.0 - tail])
    return float(lower), float(upper)


def _permutation_test(
    metric_name: MetricName,
    y_true: FloatArray,
    y_pred: FloatArray,
    *,
    observed: float,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Teste de permutação unilateral: retorna (p-valor, mediana da métrica nula).

    O p-valor é a fração de permutações do alvo com métrica >= observada. Para
    Spearman o nulo se concentra em 0.
    """
    null_scores = np.array(
        [_score(metric_name, rng.permutation(y_true), y_pred) for _ in range(n_permutations)]
    )
    exceed = int(np.sum(null_scores >= observed))
    return (1 + exceed) / (1 + n_permutations), float(np.median(null_scores))


def _permutation_pvalue(
    metric_name: MetricName,
    y_true: FloatArray,
    y_pred: FloatArray,
    *,
    observed: float,
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    """p-valor unilateral: fração de permutações do alvo com métrica >= observada."""
    return _permutation_test(
        metric_name, y_true, y_pred, observed=observed, n_permutations=n_permutations, rng=rng
    )[0]


def _holdout_correlation(y_true: FloatArray, y_pred: FloatArray) -> float:
    """Correlação de Pearson previsão x alvo (``nan`` se algum vetor for constante)."""
    if 0 in (np.std(y_true), np.std(y_pred)):
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _validate_inputs(
    x_train: FloatArray, y_train: FloatArray, x_holdout: FloatArray, y_holdout: FloatArray
) -> None:
    """Verifica consistência de formas entre embeddings e alvos."""
    if x_train.ndim != 2 or x_holdout.ndim != 2:
        raise DataValidationError(
            schema_name="SanityGateInputs", detail="embeddings devem ser matrizes 2D"
        )
    if x_train.shape[0] != y_train.shape[0] or x_holdout.shape[0] != y_holdout.shape[0]:
        raise DataValidationError(
            schema_name="SanityGateInputs",
            detail="número de linhas dos embeddings difere do número de alvos",
        )
    if x_train.shape[1] != x_holdout.shape[1]:
        raise DataValidationError(
            schema_name="SanityGateInputs", detail="dimensão dos embeddings difere entre partições"
        )
    if 0 in (x_train.shape[0], x_holdout.shape[0]):
        raise DataValidationError(schema_name="SanityGateInputs", detail="partição vazia")


def _degenerate_result(
    target_name: str, metric_name: MetricName, n_train: int, n_holdout: int
) -> SanityGateResult:
    """Resultado no-go para alvo constante (uma única classe), sem ajustar o modelo."""
    nan = float("nan")
    return SanityGateResult(
        target_name=target_name,
        metric_name=metric_name,
        holdout_score=nan,
        ci_lower=nan,
        ci_upper=nan,
        permutation_pvalue=1.0,
        n_train=n_train,
        n_holdout=n_holdout,
        go=False,
        reason="alvo constante em treino ou holdout: não há o que explicar",
    )


def _decide(
    *,
    metric_name: MetricName,
    ci_lower: float,
    pvalue: float,
    significance_alpha: float,
    min_effect: float,
) -> tuple[bool, str]:
    """Aplica o critério conjunto (IC acima do acaso + permutação significativa)."""
    threshold = _CHANCE_LEVEL[metric_name] + min_effect
    ci_ok = bool(ci_lower > threshold)
    p_ok = bool(pvalue < significance_alpha)
    if ci_ok and p_ok:
        return True, "IC inferior acima do acaso e permutação significativa"
    reasons = []
    if not ci_ok:
        reasons.append(f"IC inferior ({ci_lower:.3f}) não supera {threshold:.3f}")
    if not p_ok:
        reasons.append(f"p de permutação ({pvalue:.4f}) >= {significance_alpha}")
    return False, "; ".join(reasons)


def evaluate_sanity_gate(
    x_train: FloatArray,
    y_train: FloatArray,
    x_holdout: FloatArray,
    y_holdout: FloatArray,
    *,
    target_name: str,
    classification: bool,
    ridge_alphas: Sequence[float] = DEFAULT_RIDGE_ALPHAS,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    n_permutations: int = 1000,
    confidence_level: float = 0.95,
    significance_alpha: float = 0.05,
    min_effect: float = 0.0,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> SanityGateResult:
    """Avalia se os embeddings preveem o alvo acima do acaso (decisão go/no-go).

    Parameters
    ----------
    x_train, y_train : np.ndarray
        Embeddings (n, d) e alvo de treino.
    x_holdout, y_holdout : np.ndarray
        Embeddings e alvo do holdout, disjuntos do treino.
    target_name : str
        Nome do alvo (para log e relatório).
    classification : bool
        ``True`` para alvo binário (usa ROC-AUC); ``False`` para contínuo (Spearman).
    ridge_alphas : Sequence[float], optional
        Grade de regularização do ``RidgeCV`` (validação cruzada leave-one-out
        eficiente no treino), by default :data:`DEFAULT_RIDGE_ALPHAS`. Com
        embeddings de alta dimensão e poucos exemplos, ``alpha`` fixo pequeno
        sobreajusta e degrada a previsão no holdout.
    n_bootstrap : int, optional
        Reamostragens do holdout para o IC, by default
        :data:`constants.defaults.DEFAULT_BOOTSTRAP_ITERATIONS`.
    n_permutations : int, optional
        Permutações do alvo para o p-valor, by default 1000.
    confidence_level : float, optional
        Nível do IC, by default 0.95.
    significance_alpha : float, optional
        Nível do teste de permutação, by default 0.05.
    min_effect : float, optional
        Margem mínima acima do acaso exigida do limite inferior do IC, by
        default 0.0.
    random_seed : int, optional
        Semente do bootstrap e das permutações, by default
        :data:`constants.defaults.DEFAULT_RANDOM_SEED`.

    Returns
    -------
    SanityGateResult
        Métrica no holdout, IC, p-valor e decisão.

    Raises
    ------
    DataValidationError
        Se formas/tamanhos das entradas forem inconsistentes.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=(200, 4))
    >>> y = (x[:, 0] > 0).astype(float)
    >>> evaluate_sanity_gate(
    ...     x[:150], y[:150], x[150:], y[150:], target_name="demo", classification=True
    ... ).go
    True
    """
    arrays = [np.asarray(a, dtype=np.float64) for a in (x_train, y_train, x_holdout, y_holdout)]
    x_tr, y_tr, x_ho, y_ho = arrays
    _validate_inputs(x_tr, y_tr, x_ho, y_ho)
    metric_name: MetricName = "roc_auc" if classification else "spearman"
    has_variation = _has_two_classes(y_tr) and _has_two_classes(y_ho)
    if not has_variation:
        result = _degenerate_result(target_name, metric_name, x_tr.shape[0], x_ho.shape[0])
        logger.warning("Gate de sanidade '%s': %s.", target_name, result.reason)
        return result

    ridge = RidgeCV(alphas=np.asarray(ridge_alphas, dtype=float))
    predictions = make_pipeline(StandardScaler(), ridge).fit(x_tr, y_tr).predict(x_ho)
    logger.info(
        "Gate de sanidade '%s': alpha do RidgeCV selecionado = %.4g.",
        target_name,
        cast("float", ridge.alpha_),
    )
    rng = np.random.default_rng(random_seed)
    observed = _score(metric_name, y_ho, predictions)
    ci_lower, ci_upper = _bootstrap_interval(
        metric_name,
        y_ho,
        predictions,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        rng=rng,
    )
    pvalue, null_median = _permutation_test(
        metric_name, y_ho, predictions, observed=observed, n_permutations=n_permutations, rng=rng
    )
    go, reason = _decide(
        metric_name=metric_name,
        ci_lower=ci_lower,
        pvalue=pvalue,
        significance_alpha=significance_alpha,
        min_effect=min_effect,
    )
    logger.info(
        "Gate de sanidade '%s': %s=%.4f IC[%.4f, %.4f] p=%.4f (nulo mediano=%.4f, "
        "correlação=%.3f) -> %s.",
        target_name,
        metric_name,
        observed,
        ci_lower,
        ci_upper,
        pvalue,
        null_median,
        _holdout_correlation(y_ho, predictions),
        "GO" if go else "NO-GO",
    )
    return SanityGateResult(
        target_name=target_name,
        metric_name=metric_name,
        holdout_score=observed,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        permutation_pvalue=pvalue,
        n_train=x_tr.shape[0],
        n_holdout=x_ho.shape[0],
        go=go,
        reason=reason,
        null_median=null_median,
    )


def assert_sanity_gate_passed(result: SanityGateResult) -> None:
    """Interrompe o fluxo se o gate de sanidade reprovou o alvo.

    Parameters
    ----------
    result : SanityGateResult
        Resultado de :func:`evaluate_sanity_gate`.

    Raises
    ------
    SanityGateFailedError
        Se ``result.go`` for falso; a mensagem traz métrica, IC, p-valor e motivo.

    Examples
    --------
    >>> ok = SanityGateResult("t", "roc_auc", 0.7, 0.6, 0.8, 0.01, 10, 10, True, "ok")
    >>> assert_sanity_gate_passed(ok) is None
    True
    """
    if result.go:
        return
    detail = (
        f"{result.metric_name}={result.holdout_score:.4f} "
        f"IC95%[{result.ci_lower:.4f}, {result.ci_upper:.4f}] "
        f"p={result.permutation_pvalue:.4f}; {result.reason}"
    )
    logger.error("Abortando: %s", detail)
    raise SanityGateFailedError(result.target_name, detail)
