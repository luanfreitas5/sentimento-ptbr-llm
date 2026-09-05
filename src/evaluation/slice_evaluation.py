"""Avaliação por fatia (*slice*) dos dados.

Implementa ``configs/evaluation.yaml`` -> ``slice_evaluation``: métricas
agregadas sobre todo o conjunto de teste escondem falhas em subgrupos
específicos (ex.: por ``fonte_dados``, ``classe`` ou faixa de comprimento
de texto). Não há atributos sensíveis definidos no domínio deste projeto
(ver ``projeto-mestrado-analise-sentimentos-ptbr.md``), portanto a análise
aqui é de qualidade por segmento operacional, não de auditoria de
fairness.
"""

import logging
from collections.abc import Sequence

import polars as pl

from constants.defaults import DEFAULT_F1_MACRO_MINIMUM
from exceptions.data import EmptyDatasetError
from metrics.classification import calculate_classification_metrics

logger = logging.getLogger(__name__)


def evaluate_metrics_by_slice(
    y_true: Sequence[str], y_pred: Sequence[str], slice_labels: Sequence[str]
) -> pl.DataFrame:
    """Calcula as métricas de classificação separadamente para cada valor de fatia.

    Parameters
    ----------
    y_true : Sequence[str]
        Rótulos de sentimento verdadeiros.
    y_pred : Sequence[str]
        Rótulos de sentimento preditos, mesmo tamanho de ``y_true``.
    slice_labels : Sequence[str]
        Valor de fatia de cada amostra (ex.: fonte de dados, faixa de
        comprimento de texto), mesmo tamanho de ``y_true``.

    Returns
    -------
    pl.DataFrame
        Uma linha por valor único de fatia, com a coluna ``slice``,
        ``n_samples`` e as métricas de
        :func:`metrics.classification.calculate_classification_metrics`.

    Raises
    ------
    EmptyDatasetError
        Se ``y_true`` estiver vazio.
    ValueError
        Se ``y_true``, ``y_pred`` e ``slice_labels`` não tiverem o mesmo
        tamanho.

    Examples
    --------
    >>> y_true = ["positivo", "positivo", "negativo", "negativo"]
    >>> y_pred = ["positivo", "negativo", "negativo", "negativo"]
    >>> fatias = ["twitter", "twitter", "reddit", "reddit"]
    >>> resultado = evaluate_metrics_by_slice(y_true, y_pred, fatias)
    >>> sorted(resultado["slice"].to_list())
    ['reddit', 'twitter']
    """
    if len(y_true) == 0:
        raise EmptyDatasetError("y_true")
    if not (len(y_true) == len(y_pred) == len(slice_labels)):
        raise ValueError(
            "y_true, y_pred e slice_labels devem ter o mesmo tamanho, recebido "
            f"{len(y_true)}, {len(y_pred)} e {len(slice_labels)}"
        )

    rows: list[dict[str, float | int | str]] = []
    for slice_value in sorted(set(slice_labels)):
        sliced_true = [
            true_label
            for true_label, current_slice in zip(y_true, slice_labels)
            if current_slice == slice_value
        ]
        sliced_pred = [
            predicted_label
            for predicted_label, current_slice in zip(y_pred, slice_labels)
            if current_slice == slice_value
        ]
        metrics = calculate_classification_metrics(sliced_true, sliced_pred)
        rows.append({"slice": slice_value, "n_samples": len(sliced_true), **metrics})

    result = pl.DataFrame(rows)
    logger.info("Avaliação por fatia concluída para %d valor(es) de fatia.", result.height)
    return result


def identify_underperforming_slices(
    slice_metrics: pl.DataFrame,
    *,
    metric_column: str = "f1_macro",
    threshold: float = DEFAULT_F1_MACRO_MINIMUM,
) -> pl.DataFrame:
    """Filtra as fatias cuja métrica principal fica abaixo de um limiar aceitável.

    Parameters
    ----------
    slice_metrics : pl.DataFrame
        Saída de :func:`evaluate_metrics_by_slice`.
    metric_column : str, optional
        Nome da coluna de métrica a comparar contra ``threshold``, by
        default "f1_macro".
    threshold : float, optional
        Limiar mínimo aceitável, by default
        :data:`constants.defaults.DEFAULT_F1_MACRO_MINIMUM`.

    Returns
    -------
    pl.DataFrame
        Subconjunto de ``slice_metrics`` com ``metric_column`` abaixo de
        ``threshold``, ordenado da pior para a melhor fatia.

    Raises
    ------
    EmptyDatasetError
        Se ``slice_metrics`` estiver vazio.

    Examples
    --------
    >>> import polars as pl
    >>> metricas = pl.DataFrame({"slice": ["a", "b"], "f1_macro": [0.9, 0.4]})
    >>> identify_underperforming_slices(metricas, threshold=0.65)["slice"].to_list()
    ['b']
    """
    if slice_metrics.height == 0:
        raise EmptyDatasetError("slice_metrics")
    underperforming = slice_metrics.filter(pl.col(metric_column) < threshold).sort(metric_column)
    if underperforming.height > 0:
        logger.warning(
            "%d fatia(s) abaixo do limiar de %.2f em '%s'.",
            underperforming.height,
            threshold,
            metric_column,
        )
    return underperforming
