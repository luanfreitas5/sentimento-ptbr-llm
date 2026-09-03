"""Amostragem para validação humana do pipeline de rotulagem.

Implementa a Seção 4.3 do documento mestre: amostragem de subconjuntos do
corpus (aleatória ou estratificada por classe) para estimar a taxa de erro
da rotulagem automática antes do uso em treinamento.
"""

import logging

import polars as pl

from constants.defaults import DEFAULT_RANDOM_SEED
from utils.validation import validate_not_empty_collection

logger = logging.getLogger(__name__)


def sample_random_subset(
    dataframe: pl.DataFrame,
    *,
    sample_size: int,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pl.DataFrame:
    """Amostra aleatoriamente um subconjunto do DataFrame, com semente fixa.

    Parameters
    ----------
    dataframe : pl.DataFrame
        DataFrame de origem, não vazio.
    sample_size : int
        Número de exemplos desejados na amostra. Se maior que o número de
        linhas disponíveis, a amostra é limitada ao tamanho do DataFrame.
    random_seed : int, optional
        Semente do gerador aleatório, garantindo reprodutibilidade, by
        default :data:`constants.defaults.DEFAULT_RANDOM_SEED`.

    Returns
    -------
    pl.DataFrame
        Subconjunto amostrado, sem reposição.

    Raises
    ------
    EmptyDatasetError
        Se ``dataframe`` estiver vazio.
    ValueError
        Se ``sample_size`` não for positivo.

    Examples
    --------
    >>> df = pl.DataFrame({"id": [str(i) for i in range(10)]})
    >>> sample_random_subset(df, sample_size=3).height
    3
    """
    validate_not_empty_collection(dataframe, collection_name="dataframe")
    if sample_size <= 0:
        raise ValueError(f"sample_size deve ser positivo, recebido: {sample_size}")

    sampled_size = min(sample_size, dataframe.height)
    sampled_df = dataframe.sample(n=sampled_size, seed=random_seed, shuffle=True)
    logger.info(
        "Amostra aleatória de %d/%d exemplo(s) selecionada.", sampled_size, dataframe.height
    )
    return sampled_df


def sample_stratified_subset(
    dataframe: pl.DataFrame,
    *,
    stratify_column: str,
    sample_size: int,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pl.DataFrame:
    """Amostra um subconjunto do DataFrame, mantendo a proporção entre classes.

    Cada valor distinto de ``stratify_column`` contribui com uma fração da
    amostra proporcional ao seu tamanho no DataFrame original, garantindo
    pelo menos um exemplo por classe presente.

    Parameters
    ----------
    dataframe : pl.DataFrame
        DataFrame de origem, não vazio.
    stratify_column : str
        Nome da coluna usada para estratificação (ex.: rótulo de sentimento).
    sample_size : int
        Número aproximado de exemplos desejados na amostra total. O
        tamanho final pode diferir levemente devido ao arredondamento por
        estrato.
    random_seed : int, optional
        Semente do gerador aleatório, garantindo reprodutibilidade, by
        default :data:`constants.defaults.DEFAULT_RANDOM_SEED`.

    Returns
    -------
    pl.DataFrame
        Subconjunto amostrado, com todas as classes de ``stratify_column``
        representadas.

    Raises
    ------
    EmptyDatasetError
        Se ``dataframe`` estiver vazio.
    ValueError
        Se ``sample_size`` não for positivo.

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "id": [str(i) for i in range(20)],
    ...         "sentiment_label": ["positivo"] * 10 + ["negativo"] * 10,
    ...     }
    ... )
    >>> resultado = sample_stratified_subset(df, stratify_column="sentiment_label", sample_size=10)
    >>> sorted(resultado["sentiment_label"].unique().to_list())
    ['negativo', 'positivo']
    """
    validate_not_empty_collection(dataframe, collection_name="dataframe")
    if sample_size <= 0:
        raise ValueError(f"sample_size deve ser positivo, recebido: {sample_size}")

    sampled_size = min(sample_size, dataframe.height)
    fraction = sampled_size / dataframe.height

    sampled_groups: list[pl.DataFrame] = []
    for grupo in dataframe.partition_by(stratify_column, maintain_order=True):
        grupo_amostra_size = min(max(1, round(grupo.height * fraction)), grupo.height)
        sampled_groups.append(grupo.sample(n=grupo_amostra_size, seed=random_seed, shuffle=True))

    stratified_subset = pl.concat(sampled_groups)
    logger.info(
        "Amostra estratificada de %d exemplo(s) selecionada (alvo: %d).",
        stratified_subset.height,
        sample_size,
    )
    return stratified_subset
