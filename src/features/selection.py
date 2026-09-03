"""Seleção de features relevantes para os classificadores clássicos.

Implementa a Seção 4.9 do documento mestre (análise de importância de
variáveis) e dá suporte ao *ablation study* de ``configs/evaluation.yaml
-> ablation.components``: permite descartar features de baixa variância ou
redundantes e habilitar/desabilitar grupos nomeados de features (ex.:
TF-IDF, embeddings estáticos, embeddings contextuais) sem reprocessar a
extração. Implementado com ``numpy``/``polars`` puros, sem depender de
``scikit-learn`` (ainda não presente no projeto).
"""

import logging
from collections.abc import Mapping, Sequence

import numpy as np
import polars as pl

from utils.validation import validate_not_empty_collection

logger = logging.getLogger(__name__)


def calculate_feature_variance(
    feature_matrix: pl.DataFrame, *, id_column: str = "id"
) -> pl.DataFrame:
    """Calcula a variância populacional de cada coluna de feature de uma matriz.

    Parameters
    ----------
    feature_matrix : pl.DataFrame
        Matriz de features no formato largo (uma linha por documento),
        contendo ``id_column`` e as demais colunas como features
        numéricas. Não vazia.
    id_column : str, optional
        Nome da coluna identificadora, excluída do cálculo, by default
        "id".

    Returns
    -------
    pl.DataFrame
        DataFrame com as colunas ``feature`` e ``variance``, uma linha por
        feature.

    Raises
    ------
    EmptyDatasetError
        Se ``feature_matrix`` estiver vazia.

    Examples
    --------
    >>> df = pl.DataFrame({"id": ["1", "2", "3"], "a": [1.0, 1.0, 1.0], "b": [1.0, 2.0, 3.0]})
    >>> result = calculate_feature_variance(df).sort("feature")
    >>> result["feature"].to_list()
    ['a', 'b']
    >>> [round(value, 4) for value in result["variance"].to_list()]
    [0.0, 0.6667]
    """
    validate_not_empty_collection(feature_matrix, collection_name="feature_matrix")
    feature_columns = [column for column in feature_matrix.columns if column != id_column]
    variances = [feature_matrix[column].var(ddof=0) for column in feature_columns]
    return pl.DataFrame({"feature": feature_columns, "variance": variances})


def select_features_by_variance_threshold(
    feature_matrix: pl.DataFrame, *, id_column: str = "id", minimum_variance: float = 0.0
) -> pl.DataFrame:
    """Remove features com variância menor ou igual a um limiar.

    Descarta features quase constantes (ex.: dimensões de embedding
    "mortas" para o corpus em questão), que não contribuem para distinguir
    amostras e apenas aumentam a dimensionalidade do problema.

    Parameters
    ----------
    feature_matrix : pl.DataFrame
        Matriz de features no formato largo. Não vazia.
    id_column : str, optional
        Nome da coluna identificadora, sempre preservada, by default "id".
    minimum_variance : float, optional
        Variância mínima (exclusive) para uma feature ser mantida, by
        default 0.0 (remove apenas features constantes).

    Returns
    -------
    pl.DataFrame
        ``feature_matrix`` contendo apenas ``id_column`` e as features com
        variância acima de ``minimum_variance``.

    Raises
    ------
    EmptyDatasetError
        Se ``feature_matrix`` estiver vazia.

    Examples
    --------
    >>> df = pl.DataFrame({"id": ["1", "2", "3"], "a": [1.0, 1.0, 1.0], "b": [1.0, 2.0, 3.0]})
    >>> select_features_by_variance_threshold(df).columns
    ['id', 'b']
    """
    variances = calculate_feature_variance(feature_matrix, id_column=id_column)
    selected_features = variances.filter(pl.col("variance") > minimum_variance)["feature"].to_list()
    return feature_matrix.select([id_column, *selected_features])


def calculate_feature_correlation_matrix(
    feature_matrix: pl.DataFrame, *, id_column: str = "id"
) -> tuple[list[str], np.ndarray]:
    """Calcula a matriz de correlação de Pearson entre as colunas de feature.

    Parameters
    ----------
    feature_matrix : pl.DataFrame
        Matriz de features no formato largo. Não vazia.
    id_column : str, optional
        Nome da coluna identificadora, excluída do cálculo, by default
        "id".

    Returns
    -------
    tuple[list[str], np.ndarray]
        Par ``(nomes_das_features, matriz_de_correlacao)``, com a matriz na
        mesma ordem dos nomes retornados.

    Raises
    ------
    EmptyDatasetError
        Se ``feature_matrix`` estiver vazia.

    Examples
    --------
    >>> df = pl.DataFrame({"id": ["1", "2", "3"], "a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
    >>> feature_names, correlation = calculate_feature_correlation_matrix(df)
    >>> feature_names
    ['a', 'b']
    >>> round(float(correlation[0, 1]), 4)
    1.0
    """
    validate_not_empty_collection(feature_matrix, collection_name="feature_matrix")
    feature_columns = [column for column in feature_matrix.columns if column != id_column]
    feature_values = feature_matrix.select(feature_columns).to_numpy()
    correlation_matrix = np.atleast_2d(np.corrcoef(feature_values, rowvar=False))
    return feature_columns, correlation_matrix


def select_features_by_redundancy(
    feature_matrix: pl.DataFrame, *, id_column: str = "id", correlation_threshold: float = 0.95
) -> pl.DataFrame:
    """Remove features redundantes (altamente correlacionadas) entre si.

    Percorre as features na ordem em que aparecem em ``feature_matrix``: ao
    encontrar um par com correlação absoluta acima de
    ``correlation_threshold`` em relação a uma feature já selecionada,
    descarta a feature atual, preservando a primeira do par.

    Parameters
    ----------
    feature_matrix : pl.DataFrame
        Matriz de features no formato largo. Não vazia.
    id_column : str, optional
        Nome da coluna identificadora, sempre preservada, by default "id".
    correlation_threshold : float, optional
        Limiar de correlação absoluta (exclusive) acima do qual uma
        feature é considerada redundante, by default 0.95.

    Returns
    -------
    pl.DataFrame
        ``feature_matrix`` sem as features redundantes.

    Raises
    ------
    EmptyDatasetError
        Se ``feature_matrix`` estiver vazia.

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "id": ["1", "2", "3"],
    ...         "a": [1.0, 2.0, 3.0],
    ...         "b": [2.0, 4.0, 6.0],
    ...         "c": [3.0, 1.0, 2.0],
    ...     }
    ... )
    >>> select_features_by_redundancy(df).columns
    ['id', 'a', 'c']
    """
    feature_names, correlation_matrix = calculate_feature_correlation_matrix(
        feature_matrix, id_column=id_column
    )
    kept_feature_indices: list[int] = []
    for feature_index in range(len(feature_names)):
        is_redundant = any(
            abs(correlation_matrix[feature_index, kept_index]) > correlation_threshold
            for kept_index in kept_feature_indices
        )
        if not is_redundant:
            kept_feature_indices.append(feature_index)
    kept_features = [feature_names[index] for index in kept_feature_indices]
    return feature_matrix.select([id_column, *kept_features])


def select_k_best_features_by_target_correlation(
    feature_matrix: pl.DataFrame,
    target: Sequence[float],
    *,
    id_column: str = "id",
    k: int = 100,
) -> pl.DataFrame:
    """Seleciona as ``k`` features mais correlacionadas (em módulo) com o alvo.

    Seleção supervisionada simples, alternativa a
    ``sklearn.feature_selection.SelectKBest`` (não instalado no projeto):
    ordena as features pela correlação de Pearson absoluta com ``target``
    (ex.: rótulo de sentimento codificado via
    ``constants.labels.transform_label_to_id``) e mantém as ``k`` mais
    correlacionadas.

    Parameters
    ----------
    feature_matrix : pl.DataFrame
        Matriz de features no formato largo. Não vazia.
    target : Sequence[float]
        Alvo numérico, um valor por linha de ``feature_matrix``, na mesma
        ordem.
    id_column : str, optional
        Nome da coluna identificadora, sempre preservada, by default "id".
    k : int, optional
        Quantidade de features a manter, by default 100.

    Returns
    -------
    pl.DataFrame
        ``feature_matrix`` contendo apenas ``id_column`` e as ``k``
        features selecionadas, na ordem decrescente de correlação absoluta
        com o alvo.

    Raises
    ------
    EmptyDatasetError
        Se ``feature_matrix`` estiver vazia.

    Examples
    --------
    >>> df = pl.DataFrame({"id": ["1", "2", "3"], "a": [1.0, 2.0, 3.0], "b": [3.0, 1.0, 2.0]})
    >>> select_k_best_features_by_target_correlation(df, [1.0, 2.0, 3.0], k=1).columns
    ['id', 'a']
    """
    validate_not_empty_collection(feature_matrix, collection_name="feature_matrix")
    feature_columns = [column for column in feature_matrix.columns if column != id_column]
    target_array = np.asarray(target, dtype=float)

    correlations = {
        column: abs(np.corrcoef(feature_matrix[column].to_numpy(), target_array)[0, 1])
        for column in feature_columns
    }
    selected_features = sorted(correlations, key=lambda column: correlations[column], reverse=True)[
        :k
    ]
    return feature_matrix.select([id_column, *selected_features])


def build_feature_group_mask(
    feature_matrix: pl.DataFrame,
    feature_groups: Mapping[str, Sequence[str]],
    *,
    enabled_groups: Sequence[str],
    id_column: str = "id",
) -> pl.DataFrame:
    """Restringe uma matriz de features aos grupos nomeados habilitados.

    Dá suporte ao *ablation study* de ``configs/evaluation.yaml ->
    ablation.components``: cada grupo nomeado (ex.: ``"tfidf"``,
    ``"embeddings_estaticos"``, ``"embeddings_contextuais"``) corresponde a
    um subconjunto de colunas de ``feature_matrix``, permitindo comparar o
    desempenho do modelo com e sem cada grupo sem reprocessar a extração.

    Parameters
    ----------
    feature_matrix : pl.DataFrame
        Matriz de features no formato largo, contendo todas as colunas
        referenciadas em ``feature_groups``. Não vazia.
    feature_groups : Mapping[str, Sequence[str]]
        Nome do grupo -> colunas de ``feature_matrix`` que o compõem.
    enabled_groups : Sequence[str]
        Nomes dos grupos (chaves de ``feature_groups``) a manter.
    id_column : str, optional
        Nome da coluna identificadora, sempre preservada, by default "id".

    Returns
    -------
    pl.DataFrame
        ``feature_matrix`` contendo ``id_column`` e apenas as colunas dos
        grupos habilitados.

    Raises
    ------
    EmptyDatasetError
        Se ``feature_matrix`` estiver vazia.
    KeyError
        Se algum nome em ``enabled_groups`` não existir em
        ``feature_groups``.

    Examples
    --------
    >>> df = pl.DataFrame({"id": ["1"], "tfidf_a": [0.5], "emb_0": [0.1], "emb_1": [0.2]})
    >>> groups = {"tfidf": ["tfidf_a"], "embeddings": ["emb_0", "emb_1"]}
    >>> build_feature_group_mask(df, groups, enabled_groups=["embeddings"]).columns
    ['id', 'emb_0', 'emb_1']
    """
    validate_not_empty_collection(feature_matrix, collection_name="feature_matrix")
    selected_columns = [
        column for group_name in enabled_groups for column in feature_groups[group_name]
    ]
    logger.info(
        "Máscara de ablation aplicada: %d grupo(s) habilitado(s) de %d, %d coluna(s) mantida(s).",
        len(enabled_groups),
        len(feature_groups),
        len(selected_columns),
    )
    return feature_matrix.select([id_column, *selected_columns])
