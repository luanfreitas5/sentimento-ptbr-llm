"""Validação dos rótulos de consenso contra gold sets de referência (TweetSentBR/RePro).

Implementa a seção ``validation`` de ``configs/labeling.yaml``: mede a
concordância entre os rótulos produzidos pelo pipeline e os rótulos de
referência via Kappa de Cohen (dois avaliadores — consenso vs. gold set) e
Alpha de Krippendorff (múltiplos avaliadores — ex.: os próprios
rotuladores da cascata entre si). As duas métricas são implementadas com
``numpy`` a partir das fórmulas padrão da literatura, sem dependência de
``scikit-learn``/``krippendorff`` (ainda não instalados — ver CLAUDE.md,
"Install only what the project needs").
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from exceptions.data import EmptyDatasetError
from utils.validation import validate_not_empty_collection

logger = logging.getLogger(__name__)


def calculate_cohen_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    """Calcula o coeficiente Kappa de Cohen entre dois conjuntos de rótulos pareados.

    Mede a concordância entre dois avaliadores (ex.: rótulo de consenso vs.
    rótulo do gold set) descontando a concordância esperada ao acaso:
    ``kappa = (p_observado - p_esperado) / (1 - p_esperado)``.

    Parameters
    ----------
    labels_a : Sequence[str]
        Rótulos do primeiro avaliador. Não vazio.
    labels_b : Sequence[str]
        Rótulos do segundo avaliador, na mesma ordem/tamanho de
        ``labels_a``.

    Returns
    -------
    float
        Coeficiente Kappa de Cohen. 1.0 indica concordância perfeita, 0.0
        indica concordância igual ao acaso; valores negativos indicam
        concordância pior que o acaso.

    Raises
    ------
    EmptyDatasetError
        Se ``labels_a`` estiver vazio.
    ValueError
        Se ``labels_a`` e ``labels_b`` tiverem tamanhos diferentes.

    Examples
    --------
    >>> round(
    ...     calculate_cohen_kappa(
    ...         ["positivo", "negativo", "neutro"], ["positivo", "negativo", "positivo"]
    ...     ),
    ...     4,
    ... )
    0.5
    """
    validate_not_empty_collection(labels_a, collection_name="labels_a")
    if len(labels_a) != len(labels_b):
        raise ValueError(
            f"labels_a e labels_b devem ter o mesmo tamanho: {len(labels_a)} != {len(labels_b)}"
        )

    categories = sorted(set(labels_a) | set(labels_b))
    category_index = {category: index for index, category in enumerate(categories)}
    n_samples = len(labels_a)

    confusion = np.zeros((len(categories), len(categories)), dtype=float)
    for label_a, label_b in zip(labels_a, labels_b, strict=True):
        confusion[category_index[label_a], category_index[label_b]] += 1

    observed_agreement = float(np.trace(confusion) / n_samples)
    row_marginals = confusion.sum(axis=1) / n_samples
    col_marginals = confusion.sum(axis=0) / n_samples
    expected_agreement = float(np.sum(row_marginals * col_marginals))

    if expected_agreement >= 1.0:
        return 1.0
    return (observed_agreement - expected_agreement) / (1.0 - expected_agreement)


def _validate_reliability_data(reliability_data: Sequence[Sequence[str | None]]) -> None:
    """Valida o formato da matriz de confiabilidade do Alpha de Krippendorff.

    Parameters
    ----------
    reliability_data : Sequence[Sequence[str | None]]
        Matriz de confiabilidade a validar (ver
        :func:`calculate_krippendorff_alpha`).

    Raises
    ------
    ValueError
        Se houver menos de 2 avaliadores ou linhas de tamanhos diferentes.
    """
    n_raters = len(reliability_data)
    if n_raters < 2:
        raise ValueError(
            f"Krippendorff's alpha requer ao menos 2 avaliadores, recebido: {n_raters}"
        )

    n_units = len(reliability_data[0])
    for rater_values in reliability_data:
        if len(rater_values) != n_units:
            raise ValueError("Todos os avaliadores devem ter o mesmo número de unidades avaliadas.")


def _build_coincidence_matrix(
    reliability_data: Sequence[Sequence[str | None]], category_index: dict[str, int]
) -> tuple[np.ndarray, float]:
    """Constrói a matriz de coincidências do Alpha de Krippendorff, por unidade avaliada.

    Parameters
    ----------
    reliability_data : Sequence[Sequence[str | None]]
        Matriz de confiabilidade (ver :func:`calculate_krippendorff_alpha`).
    category_index : dict[str, int]
        Mapeamento de cada categoria observada para seu índice na matriz.

    Returns
    -------
    tuple[np.ndarray, float]
        Matriz de coincidências ``(n_categorias, n_categorias)`` e o total
        de avaliações pareáveis (unidades com ao menos 2 avaliadores).
    """
    n_raters = len(reliability_data)
    n_units = len(reliability_data[0])
    n_categories = len(category_index)

    coincidence = np.zeros((n_categories, n_categories), dtype=float)
    total_pairable_values = 0.0

    for unit_index in range(n_units):
        unit_values = [
            value
            for value in (
                reliability_data[rater_index][unit_index] for rater_index in range(n_raters)
            )
            if value is not None
        ]
        n_values_in_unit = len(unit_values)
        if n_values_in_unit < 2:
            continue

        counts = np.zeros(n_categories, dtype=float)
        for value in unit_values:
            counts[category_index[value]] += 1

        unit_coincidence = np.outer(counts, counts)
        np.fill_diagonal(unit_coincidence, counts * (counts - 1))
        coincidence += unit_coincidence / (n_values_in_unit - 1)
        total_pairable_values += n_values_in_unit

    return coincidence, total_pairable_values


def calculate_krippendorff_alpha(reliability_data: Sequence[Sequence[str | None]]) -> float:
    """Calcula o Alpha de Krippendorff (métrica nominal) para múltiplos avaliadores.

    Implementação via matriz de coincidências (Hayes & Krippendorff, 2007),
    suportando avaliações ausentes (``None``) e número desigual de
    avaliações por unidade.

    Parameters
    ----------
    reliability_data : Sequence[Sequence[str | None]]
        Matriz de confiabilidade com uma linha por avaliador e uma coluna
        por unidade avaliada (amostra); ``None`` indica ausência de
        avaliação daquele avaliador para aquela unidade. Todas as linhas
        devem ter o mesmo número de colunas.

    Returns
    -------
    float
        Alpha de Krippendorff. 1.0 indica concordância perfeita; 0.0
        indica concordância igual à esperada ao acaso.

    Raises
    ------
    ValueError
        Se houver menos de 2 avaliadores ou linhas de tamanhos diferentes.
    EmptyDatasetError
        Se não houver nenhuma categoria observada ou nenhuma unidade com
        ao menos 2 avaliações (unidades não pareáveis são ignoradas).

    Examples
    --------
    >>> dados = [
    ...     ["positivo", "positivo", "negativo"],
    ...     ["positivo", "negativo", "negativo"],
    ... ]
    >>> round(calculate_krippendorff_alpha(dados), 4)
    0.4444
    """
    _validate_reliability_data(reliability_data)

    categories = sorted(
        {value for rater_values in reliability_data for value in rater_values if value is not None}
    )
    if not categories:
        raise EmptyDatasetError("reliability_data")
    category_index = {category: index for index, category in enumerate(categories)}

    coincidence, total_pairable_values = _build_coincidence_matrix(reliability_data, category_index)
    if total_pairable_values == 0:
        raise EmptyDatasetError("unidades com ao menos 2 avaliações em reliability_data")

    n = total_pairable_values
    category_totals = coincidence.sum(axis=0)
    observed_disagreement = (n - np.trace(coincidence)) / n
    expected_disagreement = (n**2 - np.sum(category_totals**2)) / (n * (n - 1))

    if expected_disagreement == 0.0:
        return 1.0
    return float(1.0 - observed_disagreement / expected_disagreement)


@dataclass(frozen=True)
class GoldSetValidationResult:
    """Resultado da validação dos rótulos de consenso contra um gold set de referência.

    Parameters
    ----------
    cohen_kappa : float
        Concordância (Kappa de Cohen) entre os rótulos preditos e o gold set.
    n_samples : int
        Número de amostras em comum entre os dois conjuntos, usadas no cálculo.
    meets_minimum_agreement : bool
        Se ``cohen_kappa`` atinge o limiar mínimo exigido.
    """

    cohen_kappa: float
    n_samples: int
    meets_minimum_agreement: bool


def evaluate_against_gold_set(
    predicted: pl.DataFrame,
    gold: pl.DataFrame,
    *,
    id_column: str = "id",
    label_column: str = "sentiment_label",
    minimum_kappa: float = 0.6,
) -> GoldSetValidationResult:
    """Avalia os rótulos de consenso contra um gold set de referência (TweetSentBR/RePro).

    Limiar padrão reflete ``configs/labeling.yaml`` (``validation.minimum_agreement``).

    Parameters
    ----------
    predicted : pl.DataFrame
        Rótulos produzidos pelo pipeline, contendo ``id_column`` e
        ``label_column``.
    gold : pl.DataFrame
        Rótulos de referência (TweetSentBR/RePro), contendo ``id_column``
        e ``label_column``.
    id_column : str, optional
        Nome da coluna identificadora comum, by default "id".
    label_column : str, optional
        Nome da coluna de rótulo em ambos os DataFrames, by default
        "sentiment_label".
    minimum_kappa : float, optional
        Limiar mínimo de Kappa de Cohen exigido, by default 0.6.

    Returns
    -------
    GoldSetValidationResult
        Métricas de validação contra o gold set.

    Raises
    ------
    EmptyDatasetError
        Se não houver amostras em comum entre ``predicted`` e ``gold``.

    Examples
    --------
    >>> predicted = pl.DataFrame(
    ...     {"id": ["1", "2", "3"], "sentiment_label": ["positivo", "negativo", "positivo"]}
    ... )
    >>> gold = pl.DataFrame(
    ...     {"id": ["1", "2", "3"], "sentiment_label": ["positivo", "negativo", "positivo"]}
    ... )
    >>> resultado = evaluate_against_gold_set(predicted, gold)
    >>> resultado.n_samples
    3
    >>> resultado.meets_minimum_agreement
    True
    """
    joined = predicted.select([id_column, label_column]).join(
        gold.select([id_column, label_column]).rename({label_column: "gold_label"}),
        on=id_column,
        how="inner",
    )
    validate_not_empty_collection(joined, collection_name="amostra_comum_com_gold_set")

    kappa = calculate_cohen_kappa(joined[label_column].to_list(), joined["gold_label"].to_list())
    result = GoldSetValidationResult(
        cohen_kappa=kappa,
        n_samples=joined.height,
        meets_minimum_agreement=kappa >= minimum_kappa,
    )
    logger.info(
        "Validação contra gold set: kappa=%.4f (n=%d, limiar=%.2f, atende=%s).",
        result.cohen_kappa,
        result.n_samples,
        minimum_kappa,
        result.meets_minimum_agreement,
    )
    return result
