"""Estatísticas descritivas das representações produzidas por ``src/features/``.

Implementa a Seção 4.10 do documento mestre (apoio a visualizações como
UMAP/t-SNE dos embeddings) e a Seção 4.9 (análises estatísticas): resumos
por feature e por documento das matrizes produzidas por ``lexical.py``,
``static_embeddings.py``, ``contextual_embeddings.py`` e ``reduction.py``,
usados tanto em notebooks exploratórios quanto em relatórios
(``reports/statistics/``).
"""

import logging

import numpy as np
import polars as pl

from utils.validation import validate_not_empty_collection

logger = logging.getLogger(__name__)


def calculate_descriptive_statistics(
    feature_matrix: pl.DataFrame, *, id_column: str = "id"
) -> pl.DataFrame:
    """Calcula estatísticas descritivas por coluna de feature.

    Parameters
    ----------
    feature_matrix : pl.DataFrame
        Matriz de features no formato largo. Não vazia.
    id_column : str, optional
        Nome da coluna identificadora, excluída do cálculo, by default
        "id".

    Returns
    -------
    pl.DataFrame
        DataFrame com uma linha por feature e as colunas ``feature``,
        ``mean``, ``std``, ``min``, ``max`` e ``zero_ratio`` (fração de
        valores exatamente zero, indicador de esparsidade).

    Raises
    ------
    EmptyDatasetError
        Se ``feature_matrix`` estiver vazia.

    Examples
    --------
    >>> df = pl.DataFrame({"id": ["1", "2", "3"], "a": [0.0, 0.0, 3.0]})
    >>> result = calculate_descriptive_statistics(df)
    >>> result["feature"].to_list()
    ['a']
    >>> round(result["zero_ratio"].to_list()[0], 4)
    0.6667
    """
    validate_not_empty_collection(feature_matrix, collection_name="feature_matrix")
    feature_columns = [column for column in feature_matrix.columns if column != id_column]

    statistics_rows = [
        {
            "feature": column,
            "mean": feature_matrix[column].mean(),
            "std": feature_matrix[column].std(ddof=0),
            "min": feature_matrix[column].min(),
            "max": feature_matrix[column].max(),
            "zero_ratio": (feature_matrix[column] == 0).sum() / feature_matrix.height,
        }
        for column in feature_columns
    ]
    return pl.DataFrame(statistics_rows)


def calculate_embedding_norms(
    feature_matrix: pl.DataFrame, *, id_column: str = "id"
) -> pl.DataFrame:
    """Calcula a norma L2 de cada documento (linha) de uma matriz de features.

    Diagnóstico típico de embeddings densos: normas muito distantes da
    média podem indicar amostras degeneradas (ex.: texto vazio após
    pré-processamento) ou atípicas.

    Parameters
    ----------
    feature_matrix : pl.DataFrame
        Matriz de features no formato largo. Não vazia.
    id_column : str, optional
        Nome da coluna identificadora, by default "id".

    Returns
    -------
    pl.DataFrame
        DataFrame com ``id_column`` e a coluna ``l2_norm``.

    Raises
    ------
    EmptyDatasetError
        Se ``feature_matrix`` estiver vazia.

    Examples
    --------
    >>> df = pl.DataFrame({"id": ["1", "2"], "a": [3.0, 0.0], "b": [4.0, 0.0]})
    >>> calculate_embedding_norms(df)["l2_norm"].to_list()
    [5.0, 0.0]
    """
    validate_not_empty_collection(feature_matrix, collection_name="feature_matrix")
    feature_columns = [column for column in feature_matrix.columns if column != id_column]
    feature_values = feature_matrix.select(feature_columns).to_numpy()
    norms = np.linalg.norm(feature_values, axis=1)
    return pl.DataFrame({id_column: feature_matrix[id_column], "l2_norm": norms})


def calculate_feature_sparsity_ratio(
    feature_matrix: pl.DataFrame, *, id_column: str = "id"
) -> float:
    """Calcula a fração global de valores exatamente zero em uma matriz de features.

    Parameters
    ----------
    feature_matrix : pl.DataFrame
        Matriz de features no formato largo. Não vazia.
    id_column : str, optional
        Nome da coluna identificadora, excluída do cálculo, by default
        "id".

    Returns
    -------
    float
        Fração de valores zero sobre o total de células numéricas (entre
        0.0 e 1.0), típica de matrizes TF-IDF esparsas.

    Raises
    ------
    EmptyDatasetError
        Se ``feature_matrix`` estiver vazia.

    Examples
    --------
    >>> df = pl.DataFrame({"id": ["1", "2"], "a": [0.0, 1.0], "b": [0.0, 0.0]})
    >>> calculate_feature_sparsity_ratio(df)
    0.75
    """
    validate_not_empty_collection(feature_matrix, collection_name="feature_matrix")
    feature_columns = [column for column in feature_matrix.columns if column != id_column]
    feature_values = feature_matrix.select(feature_columns).to_numpy()
    return float(np.mean(feature_values == 0))


def summarize_feature_matrix(
    feature_matrix: pl.DataFrame, *, id_column: str = "id", matrix_name: str
) -> dict[str, float | int]:
    """Resume uma matriz de features em um dicionário de métricas agregadas.

    Combina :func:`calculate_feature_sparsity_ratio` e
    :func:`calculate_embedding_norms` em um único resumo, registrado via
    ``logging`` e adequado para comparação lado a lado entre diferentes
    representações (ex.: TF-IDF vs. embeddings estáticos vs. contextuais).

    Parameters
    ----------
    feature_matrix : pl.DataFrame
        Matriz de features no formato largo. Não vazia.
    id_column : str, optional
        Nome da coluna identificadora, excluída do cálculo, by default
        "id".
    matrix_name : str
        Nome descritivo da representação (ex.: ``"tfidf"``,
        ``"embeddings_contextuais"``), usado na mensagem de log.

    Returns
    -------
    dict[str, float | int]
        Dicionário com as chaves ``n_documents``, ``n_features``,
        ``sparsity_ratio``, ``mean_l2_norm`` e ``std_l2_norm``.

    Raises
    ------
    EmptyDatasetError
        Se ``feature_matrix`` estiver vazia.

    Examples
    --------
    >>> df = pl.DataFrame({"id": ["1", "2"], "a": [3.0, 0.0], "b": [4.0, 0.0]})
    >>> summary = summarize_feature_matrix(df, matrix_name="exemplo")
    >>> summary["n_documents"], summary["n_features"]
    (2, 2)
    """
    validate_not_empty_collection(feature_matrix, collection_name="feature_matrix")
    feature_columns = [column for column in feature_matrix.columns if column != id_column]
    norms = calculate_embedding_norms(feature_matrix, id_column=id_column)["l2_norm"].to_numpy()

    summary: dict[str, float | int] = {
        "n_documents": feature_matrix.height,
        "n_features": len(feature_columns),
        "sparsity_ratio": calculate_feature_sparsity_ratio(feature_matrix, id_column=id_column),
        "mean_l2_norm": float(np.mean(norms)),
        "std_l2_norm": float(np.std(norms)),
    }
    logger.info(
        "Resumo de '%s': %d documento(s), %d feature(s), esparsidade %.2f%%.",
        matrix_name,
        summary["n_documents"],
        summary["n_features"],
        summary["sparsity_ratio"] * 100,
    )
    return summary
