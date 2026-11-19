"""Avaliação de calibração das probabilidades preditas.

Implementa ``configs/evaluation.yaml`` -> ``calibration``: verifica se a
confiança do modelo reflete sua real probabilidade de acerto — essencial
quando as probabilidades são usadas para decisões (ex.: classificação
seletiva em :func:`metrics.confidence.calculate_selective_prediction_accuracy`).
"""

import logging
from collections.abc import Sequence

import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from scipy.stats import beta

from constants.labels import SENTIMENT_CLASSES
from exceptions.data import EmptyDatasetError
from metrics.confidence import calculate_multiclass_brier_score

logger = logging.getLogger(__name__)

DEFAULT_N_CALIBRATION_BINS = 10
MIN_TEMPERATURE = 0.05
MAX_TEMPERATURE = 20.0
_PROBABILITY_FLOOR = 1e-12


def calculate_reliability_curve(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    confidences: np.ndarray,
    *,
    n_bins: int = DEFAULT_N_CALIBRATION_BINS,
) -> dict[str, np.ndarray]:
    """Calcula a curva de confiabilidade (confiança predita vs. acurácia observada).

    Agrupa as predições em ``n_bins`` faixas de confiança e, em cada faixa,
    compara a confiança média com a fração de acertos observada. Uma
    calibração perfeita produz pontos sobre a diagonal.

    Parameters
    ----------
    y_true : Sequence[str]
        Rótulos de sentimento verdadeiros.
    y_pred : Sequence[str]
        Rótulos de sentimento preditos, mesmo tamanho de ``y_true``.
    confidences : np.ndarray
        Confiança (probabilidade da classe predita) por amostra, mesmo
        tamanho de ``y_true``, com valores em ``[0, 1]``.
    n_bins : int, optional
        Número de faixas de confiança de largura igual, by default
        :data:`DEFAULT_N_CALIBRATION_BINS`.

    Returns
    -------
    dict[str, np.ndarray]
        Dicionário com ``bin_confidence_means``, ``bin_accuracy`` (``nan``
        em faixas vazias) e ``bin_counts``, um valor por faixa.

    Raises
    ------
    EmptyDatasetError
        Se ``y_true`` estiver vazio.
    ValueError
        Se ``n_bins`` for menor que 1.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = ["positivo", "positivo", "negativo", "negativo"]
    >>> y_pred = ["positivo", "positivo", "negativo", "positivo"]
    >>> confidences = np.array([0.9, 0.95, 0.85, 0.3])
    >>> curva = calculate_reliability_curve(y_true, y_pred, confidences, n_bins=2)
    >>> curva["bin_counts"].tolist()
    [1, 3]
    """
    if not y_true:
        raise EmptyDatasetError("y_true")
    if n_bins < 1:
        raise ValueError(f"n_bins deve ser >= 1, recebido: {n_bins}")

    correctness = np.array(
        [
            true_label == predicted_label
            for true_label, predicted_label in zip(y_true, y_pred, strict=True)
        ],
        dtype=float,
    )
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(confidences, bin_edges[1:-1], right=True), 0, n_bins - 1)

    bin_confidence_means = np.full(n_bins, np.nan)
    bin_accuracy = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=int)

    for bin_index in range(n_bins):
        mask = bin_indices == bin_index
        count = int(np.sum(mask))
        bin_counts[bin_index] = count
        if count > 0:
            bin_confidence_means[bin_index] = float(np.mean(confidences[mask]))
            bin_accuracy[bin_index] = float(np.mean(correctness[mask]))

    return {
        "bin_confidence_means": bin_confidence_means,
        "bin_accuracy": bin_accuracy,
        "bin_counts": bin_counts,
    }


def calculate_expected_calibration_error(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    confidences: np.ndarray,
    *,
    n_bins: int = DEFAULT_N_CALIBRATION_BINS,
) -> float:
    """Calcula o Erro de Calibração Esperado (ECE).

    Média ponderada, por faixa de confiança, da diferença absoluta entre
    confiança e acurácia observada (ver :func:`calculate_reliability_curve`).

    Parameters
    ----------
    y_true : Sequence[str]
        Rótulos de sentimento verdadeiros.
    y_pred : Sequence[str]
        Rótulos de sentimento preditos, mesmo tamanho de ``y_true``.
    confidences : np.ndarray
        Confiança (probabilidade da classe predita) por amostra, mesmo
        tamanho de ``y_true``.
    n_bins : int, optional
        Número de faixas de confiança, by default
        :data:`DEFAULT_N_CALIBRATION_BINS`.

    Returns
    -------
    float
        ECE, entre 0.0 (calibração perfeita) e 1.0.

    Raises
    ------
    EmptyDatasetError
        Se ``y_true`` estiver vazio.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = ["positivo", "positivo", "negativo", "negativo"]
    >>> y_pred = ["positivo", "positivo", "negativo", "positivo"]
    >>> confidences = np.array([0.9, 0.9, 0.9, 0.9])
    >>> round(calculate_expected_calibration_error(y_true, y_pred, confidences, n_bins=1), 4)
    0.15
    """
    curve = calculate_reliability_curve(y_true, y_pred, confidences, n_bins=n_bins)
    total_samples = len(y_true)
    expected_calibration_error = 0.0
    for count, confidence_mean, accuracy in zip(
        curve["bin_counts"], curve["bin_confidence_means"], curve["bin_accuracy"], strict=True
    ):
        if count > 0:
            expected_calibration_error += (count / total_samples) * abs(accuracy - confidence_mean)
    return float(expected_calibration_error)


def calculate_calibration_metrics(
    y_true: Sequence[str],
    y_score: np.ndarray,
    *,
    labels: Sequence[str] = SENTIMENT_CLASSES,
    n_bins: int = DEFAULT_N_CALIBRATION_BINS,
) -> dict[str, float]:
    """Calcula o conjunto de métricas de calibração usado no projeto.

    Combina o Brier score multiclasse (:func:`metrics.confidence.calculate_multiclass_brier_score`)
    com o ECE calculado sobre a confiança da classe predita (ver
    ``configs/evaluation.yaml`` -> ``calibration``).

    Parameters
    ----------
    y_true : Sequence[str]
        Rótulos de sentimento verdadeiros.
    y_score : np.ndarray
        Matriz ``(n_amostras, n_classes)`` de probabilidades preditas, na
        ordem de ``labels``.
    labels : Sequence[str], optional
        Ordem das classes nas colunas de ``y_score``, by default
        :data:`constants.labels.SENTIMENT_CLASSES`.
    n_bins : int, optional
        Número de faixas de confiança usadas no ECE, by default
        :data:`DEFAULT_N_CALIBRATION_BINS`.

    Returns
    -------
    dict[str, float]
        Dicionário com as chaves ``brier_score`` e ``expected_calibration_error``.

    Raises
    ------
    EmptyDatasetError
        Se ``y_true`` estiver vazio.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = ["positivo", "negativo"]
    >>> y_score = np.array([[0.1, 0.9], [0.8, 0.2]])
    >>> resultado = calculate_calibration_metrics(y_true, y_score, labels=["negativo", "positivo"])
    >>> sorted(resultado.keys())
    ['brier_score', 'expected_calibration_error']
    """
    labels_list = list(labels)
    y_pred = [labels_list[index] for index in np.argmax(y_score, axis=1)]
    confidences = np.max(y_score, axis=1)

    metrics = {
        "brier_score": calculate_multiclass_brier_score(y_true, y_score, labels=labels_list),
        "expected_calibration_error": calculate_expected_calibration_error(
            y_true, y_pred, confidences, n_bins=n_bins
        ),
    }
    logger.info(
        "Métricas de calibração: Brier score=%.4f, ECE=%.4f.",
        metrics["brier_score"],
        metrics["expected_calibration_error"],
    )
    return metrics


def apply_temperature_scaling(y_score: np.ndarray, temperature: float) -> np.ndarray:
    """Aplica *temperature scaling* a probabilidades já normalizadas.

    Como os modelos do projeto expõem probabilidades (e não *logits*), usa-se
    ``softmax(log(p) / T)``, equivalente ao escalonamento dos *logits* quando a
    saída original é um *softmax*. ``T > 1`` suaviza (reduz a confiança) e
    ``T < 1`` afia a distribuição. A operação preserva o ``argmax`` de cada linha:
    acurácia e F1 não mudam, apenas a confiança.

    Parameters
    ----------
    y_score : np.ndarray
        Matriz ``(n_amostras, n_classes)`` de probabilidades.
    temperature : float
        Temperatura positiva.

    Returns
    -------
    np.ndarray
        Matriz de probabilidades recalibradas, com cada linha somando 1.

    Raises
    ------
    ValueError
        Se ``temperature`` não for positiva.

    Examples
    --------
    >>> import numpy as np
    >>> p = apply_temperature_scaling(np.array([[0.8, 0.1, 0.1]]), 2.0)
    >>> bool(p[0, 0] < 0.8) and bool(np.isclose(p.sum(), 1.0))
    True
    """
    if temperature <= 0:
        raise ValueError(f"temperature deve ser > 0, recebido: {temperature}")
    log_probabilities = np.log(np.clip(y_score, _PROBABILITY_FLOOR, 1.0))
    return softmax(log_probabilities / temperature, axis=1)


def calculate_negative_log_likelihood(
    y_true: Sequence[str], y_score: np.ndarray, *, labels: Sequence[str] = SENTIMENT_CLASSES
) -> np.ndarray:
    """Calcula a log-verossimilhança negativa por amostra (perda de log).

    Parameters
    ----------
    y_true : Sequence[str]
        Rótulos verdadeiros.
    y_score : np.ndarray
        Matriz ``(n_amostras, n_classes)`` de probabilidades, na ordem de ``labels``.
    labels : Sequence[str], optional
        Ordem das classes nas colunas, by default
        :data:`constants.labels.SENTIMENT_CLASSES`.

    Returns
    -------
    np.ndarray
        Vetor com ``-log(p_classe_verdadeira)`` por amostra.

    Raises
    ------
    EmptyDatasetError
        Se ``y_true`` estiver vazio.
    ValueError
        Se algum rótulo não estiver em ``labels``.

    Examples
    --------
    >>> import numpy as np
    >>> nll = calculate_negative_log_likelihood(["a"], np.array([[0.5, 0.5]]), labels=["a", "b"])
    >>> round(float(nll[0]), 4)
    0.6931
    """
    if not y_true:
        raise EmptyDatasetError("y_true")
    label_to_column = {label: column for column, label in enumerate(labels)}
    unknown = set(y_true) - set(label_to_column)
    if unknown:
        raise ValueError(f"Rótulos fora de `labels`: {sorted(unknown)}")
    columns = np.array([label_to_column[label] for label in y_true])
    true_probabilities = np.asarray(y_score)[np.arange(len(columns)), columns]
    return -np.log(np.clip(true_probabilities, _PROBABILITY_FLOOR, 1.0))


def fit_temperature_scaling(
    y_true: Sequence[str], y_score: np.ndarray, *, labels: Sequence[str] = SENTIMENT_CLASSES
) -> float:
    """Ajusta a temperatura que minimiza a log-verossimilhança negativa.

    Deve ser ajustada em dados **disjuntos do teste** (aqui, a validação): um único
    parâmetro limita o sobreajuste, mas ajustar e avaliar nas mesmas linhas
    esconderia a descalibração real.

    Parameters
    ----------
    y_true : Sequence[str]
        Rótulos verdadeiros do conjunto de ajuste.
    y_score : np.ndarray
        Probabilidades ``(n_amostras, n_classes)`` do conjunto de ajuste.
    labels : Sequence[str], optional
        Ordem das classes nas colunas, by default
        :data:`constants.labels.SENTIMENT_CLASSES`.

    Returns
    -------
    float
        Temperatura ótima em ``[MIN_TEMPERATURE, MAX_TEMPERATURE]``.

    Raises
    ------
    EmptyDatasetError
        Se ``y_true`` estiver vazio.

    Examples
    --------
    >>> import numpy as np
    >>> y = ["a", "a", "b", "b"]
    >>> p = np.array([[0.99, 0.01], [0.01, 0.99], [0.99, 0.01], [0.01, 0.99]])
    >>> fit_temperature_scaling(y, p, labels=["a", "b"]) > 1.0
    True
    """

    def _objective(temperature: float) -> float:
        scaled = apply_temperature_scaling(y_score, temperature)
        return float(np.mean(calculate_negative_log_likelihood(y_true, scaled, labels=labels)))

    result = minimize_scalar(
        _objective, bounds=(MIN_TEMPERATURE, MAX_TEMPERATURE), method="bounded"
    )
    return float(result.x)


def calculate_reliability_table(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    confidences: np.ndarray,
    *,
    n_bins: int = DEFAULT_N_CALIBRATION_BINS,
    confidence_level: float = 0.95,
) -> pl.DataFrame:
    """Tabela de confiabilidade com ``n`` e IC de Clopper-Pearson por faixa.

    Estende :func:`calculate_reliability_curve`: uma curva sem o tamanho de cada faixa
    leva a ler como calibrado (ou descalibrado) o que é ruído de poucas amostras.

    Parameters
    ----------
    y_true : Sequence[str]
        Rótulos verdadeiros.
    y_pred : Sequence[str]
        Rótulos preditos.
    confidences : np.ndarray
        Confiança da classe predita por amostra.
    n_bins : int, optional
        Número de faixas de largura igual, by default
        :data:`DEFAULT_N_CALIBRATION_BINS`.
    confidence_level : float, optional
        Nível do IC, by default 0.95.

    Returns
    -------
    pl.DataFrame
        Uma linha por faixa não vazia: ``bin``, ``n``, ``mean_confidence``,
        ``accuracy``, ``ci_low``, ``ci_high``.

    Raises
    ------
    EmptyDatasetError
        Se ``y_true`` estiver vazio.

    Examples
    --------
    >>> import numpy as np
    >>> tabela = calculate_reliability_table(
    ...     ["a", "a", "b"], ["a", "b", "b"], np.array([0.9, 0.9, 0.6]), n_bins=2
    ... )
    >>> tabela["n"].to_list()
    [1, 2]
    """
    curve = calculate_reliability_curve(y_true, y_pred, confidences, n_bins=n_bins)
    alpha = 1 - confidence_level
    rows = []
    for bin_index, count in enumerate(curve["bin_counts"]):
        if count == 0:
            continue
        accuracy = float(curve["bin_accuracy"][bin_index])
        hits = round(accuracy * count)
        rows.append(
            {
                "bin": bin_index,
                "n": int(count),
                "mean_confidence": float(curve["bin_confidence_means"][bin_index]),
                "accuracy": accuracy,
                "ci_low": float(beta.ppf(alpha / 2, hits, count - hits + 1)) if hits > 0 else 0.0,
                "ci_high": float(beta.ppf(1 - alpha / 2, hits + 1, count - hits))
                if hits < count
                else 1.0,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "bin": pl.Int64,
            "n": pl.Int64,
            "mean_confidence": pl.Float64,
            "accuracy": pl.Float64,
            "ci_low": pl.Float64,
            "ci_high": pl.Float64,
        },
    )
