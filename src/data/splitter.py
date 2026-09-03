"""Particionamento estratificado de dados em treino/validação/teste.

Implementa a etapa de particionamento da Fase 5 do plano de elaboração:
cada classe da coluna de estratificação é dividida nas mesmas proporções
entre os três conjuntos, usando uma semente aleatória fixa para garantir
reprodutibilidade fim a fim (ver CLAUDE.md, "Reprodutibilidade &
Determinismo").
"""

import logging

import numpy as np
import polars as pl

from constants.defaults import DEFAULT_RANDOM_SEED, DEFAULT_TEST_SIZE, DEFAULT_VALIDATION_SIZE
from schemas.training import DATA_SPLITS

logger = logging.getLogger(__name__)

_TRAIN_SPLIT, _VALIDATION_SPLIT, _TEST_SPLIT = DATA_SPLITS


def _assign_split_labels_for_group(
    group_size: int, *, test_size: float, validation_size: float
) -> list[str]:
    """Calcula a contagem de exemplos por split e retorna os rótulos correspondentes.

    Parameters
    ----------
    group_size : int
        Número de exemplos no grupo (estrato).
    test_size : float
        Proporção do grupo destinada ao conjunto de teste.
    validation_size : float
        Proporção do grupo destinada ao conjunto de validação.

    Returns
    -------
    list[str]
        Um rótulo de split por posição do grupo, na ordem teste -> validação
        -> treino. A aleatoriedade da atribuição é responsabilidade do
        chamador, que deve embaralhar previamente os índices do grupo.
    """
    test_count = min(round(group_size * test_size), group_size)
    validation_count = min(round(group_size * validation_size), group_size - test_count)
    train_count = group_size - test_count - validation_count
    return (
        [_TEST_SPLIT] * test_count
        + [_VALIDATION_SPLIT] * validation_count
        + [_TRAIN_SPLIT] * train_count
    )


def create_stratified_split(
    dataframe: pl.DataFrame,
    *,
    label_column: str,
    split_column: str = "split",
    test_size: float = DEFAULT_TEST_SIZE,
    validation_size: float = DEFAULT_VALIDATION_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pl.DataFrame:
    """Particiona um DataFrame em treino/validação/teste, estratificado por classe.

    Cada valor distinto de ``label_column`` é particionado independentemente
    nas mesmas proporções, preservando a distribuição original das classes
    em cada conjunto (ver CLAUDE.md, seção "Modeling Workflow").

    Parameters
    ----------
    dataframe : pl.DataFrame
        DataFrame de entrada, contendo ao menos a coluna ``label_column``.
    label_column : str
        Nome da coluna usada para estratificação (ex.: rótulo de sentimento).
    split_column : str, optional
        Nome da coluna de split a ser criada (ou sobrescrita, se já
        existir), by default "split".
    test_size : float, optional
        Proporção de cada estrato destinada ao teste, by default
        :data:`constants.defaults.DEFAULT_TEST_SIZE`.
    validation_size : float, optional
        Proporção de cada estrato destinada à validação, by default
        :data:`constants.defaults.DEFAULT_VALIDATION_SIZE`.
    random_seed : int, optional
        Semente do gerador aleatório, garantindo reprodutibilidade, by
        default :data:`constants.defaults.DEFAULT_RANDOM_SEED`.

    Returns
    -------
    pl.DataFrame
        O DataFrame original acrescido (ou com sobrescrita) da coluna
        ``split_column``, com valores em :data:`schemas.training.DATA_SPLITS`.

    Raises
    ------
    ValueError
        Se ``test_size`` e ``validation_size`` não somarem menos que 1.

    Examples
    --------
    >>> df = pl.DataFrame({"id": [str(i) for i in range(10)], "sentiment_label": ["positivo"] * 10})
    >>> resultado = create_stratified_split(df, label_column="sentiment_label")
    >>> sorted(resultado["split"].unique().to_list())
    ['teste', 'treino', 'validacao']
    """
    if test_size + validation_size >= 1:
        raise ValueError(
            "A soma de test_size e validation_size deve ser menor que 1, "
            f"recebido: {test_size + validation_size}"
        )

    rng = np.random.default_rng(random_seed)
    row_indices_by_label: dict[object, list[int]] = {}
    for row_index, label in enumerate(dataframe[label_column].to_list()):
        row_indices_by_label.setdefault(label, []).append(row_index)

    split_labels_by_row_index: list[str] = [""] * dataframe.height
    for row_indices in row_indices_by_label.values():
        shuffled_row_indices = row_indices.copy()
        rng.shuffle(shuffled_row_indices)
        split_labels = _assign_split_labels_for_group(
            len(shuffled_row_indices), test_size=test_size, validation_size=validation_size
        )
        for row_index, split_label in zip(shuffled_row_indices, split_labels, strict=True):
            split_labels_by_row_index[row_index] = split_label

    result = dataframe.with_columns(pl.Series(split_column, split_labels_by_row_index))
    logger.info(
        "Split estratificado concluído: %d treino, %d validação, %d teste",
        split_labels_by_row_index.count(_TRAIN_SPLIT),
        split_labels_by_row_index.count(_VALIDATION_SPLIT),
        split_labels_by_row_index.count(_TEST_SPLIT),
    )
    return result
