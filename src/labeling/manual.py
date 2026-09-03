"""Fluxo de amostragem e incorporação de validação humana na rotulagem em cascata.

Implementa a seção ``human_validation`` de ``configs/labeling.yaml``:
seleciona, dentre as amostras sinalizadas por
``src/labeling/confidence.py``, um subconjunto estratificado por faixa de
confiança para revisão humana, incorpora os rótulos revisados de volta ao
corpus de consenso e estima a taxa de erro da rotulagem automática a
partir da amostra revisada.
"""

import logging

import polars as pl

from constants.defaults import DEFAULT_RANDOM_SEED
from data.sampler import sample_random_subset, sample_stratified_subset
from utils.validation import validate_not_empty_collection

logger = logging.getLogger(__name__)


def _bucket_confidence_level(agreement_ratio: float) -> str:
    """Classifica uma razão de concordância em uma faixa discreta de confiança.

    Usado para estratificar a amostragem de validação humana
    (``sampling_strategy: "stratified_by_confidence"`` em
    ``configs/labeling.yaml``) sem depender de quantis, que exigiriam um
    número mínimo de amostras distintas por faixa.

    Parameters
    ----------
    agreement_ratio : float
        Razão de concordância da amostra, em ``[0.0, 1.0]``.

    Returns
    -------
    str
        Uma das faixas ``"baixa"`` (< 0.3), ``"media"`` (0.3 a 0.5) ou
        ``"moderada"`` (>= 0.5).

    Examples
    --------
    >>> _bucket_confidence_level(0.2)
    'baixa'
    >>> _bucket_confidence_level(0.4)
    'media'
    >>> _bucket_confidence_level(0.6)
    'moderada'
    """
    if agreement_ratio < 0.3:
        return "baixa"
    if agreement_ratio < 0.5:
        return "media"
    return "moderada"


def select_samples_for_human_validation(
    flagged_consensus: pl.DataFrame,
    *,
    sample_size: int = 500,
    stratify_by_confidence: bool = True,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pl.DataFrame:
    """Seleciona amostras de baixa confiança/alta discordância para validação humana.

    Parameters
    ----------
    flagged_consensus : pl.DataFrame
        Saída de ``src/labeling/confidence.py``'s
        ``flag_low_confidence_samples``, contendo ``agreement_ratio`` e
        ``requires_human_validation``.
    sample_size : int, optional
        Número de amostras desejado, repassado a
        :func:`data.sampler.sample_random_subset`/
        :func:`data.sampler.sample_stratified_subset`, by default 500
        (``human_validation.sample_size`` em ``configs/labeling.yaml``).
    stratify_by_confidence : bool, optional
        Se ``True``, estratifica a amostra por faixa de confiança (ver
        :func:`_bucket_confidence_level`); caso contrário, amostra
        aleatoriamente, by default True.
    random_seed : int, optional
        Semente do gerador aleatório, by default
        :data:`constants.defaults.DEFAULT_RANDOM_SEED`.

    Returns
    -------
    pl.DataFrame
        Subconjunto de ``flagged_consensus`` selecionado para revisão
        humana.

    Raises
    ------
    EmptyDatasetError
        Se nenhuma amostra estiver sinalizada para validação humana.

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "id": [str(i) for i in range(6)],
    ...         "consensus_label": ["positivo"] * 3 + ["negativo"] * 3,
    ...         "agreement_ratio": [0.2, 0.25, 0.28, 0.35, 0.4, 0.45],
    ...         "requires_human_validation": [True] * 6,
    ...     }
    ... )
    >>> select_samples_for_human_validation(df, sample_size=6).height
    6
    """
    candidates = flagged_consensus.filter(pl.col("requires_human_validation"))
    validate_not_empty_collection(candidates, collection_name="candidatos_validacao_humana")

    if stratify_by_confidence:
        candidates = candidates.with_columns(
            pl.col("agreement_ratio")
            .map_elements(_bucket_confidence_level, return_dtype=pl.Utf8)
            .alias("confidence_bucket")
        )
        selected = sample_stratified_subset(
            candidates,
            stratify_column="confidence_bucket",
            sample_size=sample_size,
            random_seed=random_seed,
        ).drop("confidence_bucket")
    else:
        selected = sample_random_subset(
            candidates, sample_size=sample_size, random_seed=random_seed
        )

    logger.info("%d amostra(s) selecionada(s) para validação humana.", selected.height)
    return selected


def apply_human_validation_labels(
    consensus: pl.DataFrame,
    human_labels: pl.DataFrame,
    *,
    id_column: str = "id",
    label_column: str = "sentiment_label",
) -> pl.DataFrame:
    """Sobrescreve o rótulo de consenso pelo rótulo humano, quando disponível.

    Parameters
    ----------
    consensus : pl.DataFrame
        Corpus com rótulo de consenso (ver
        ``src/labeling/consensus.py``'s ``aggregate_by_weighted_majority_vote``),
        contendo ``id_column`` e ``label_column``.
    human_labels : pl.DataFrame
        Rótulos revisados por humanos, contendo ``id_column`` e
        ``label_column``, para o subconjunto amostrado por
        :func:`select_samples_for_human_validation`.
    id_column : str, optional
        Nome da coluna identificadora comum, by default "id".
    label_column : str, optional
        Nome da coluna de rótulo em ambos os DataFrames, by default
        "sentiment_label".

    Returns
    -------
    pl.DataFrame
        ``consensus`` com ``label_column`` substituído pelo rótulo humano
        onde disponível, acrescido da coluna booleana
        ``is_human_validated``.

    Examples
    --------
    >>> consensus = pl.DataFrame({"id": ["1", "2"], "sentiment_label": ["positivo", "negativo"]})
    >>> humano = pl.DataFrame({"id": ["1"], "sentiment_label": ["neutro"]})
    >>> resultado = apply_human_validation_labels(consensus, humano).sort("id")
    >>> resultado["sentiment_label"].to_list()
    ['neutro', 'negativo']
    >>> resultado["is_human_validated"].to_list()
    [True, False]
    """
    human_labels_renamed = human_labels.select([id_column, label_column]).rename(
        {label_column: "human_sentiment_label"}
    )
    merged = consensus.join(human_labels_renamed, on=id_column, how="left")
    merged = merged.with_columns(
        pl.coalesce(["human_sentiment_label", label_column]).alias(label_column),
        pl.col("human_sentiment_label").is_not_null().alias("is_human_validated"),
    ).drop("human_sentiment_label")
    n_validated = merged.filter(pl.col("is_human_validated")).height
    logger.info("%d rótulo(s) substituído(s) por validação humana.", n_validated)
    return merged


def calculate_labeling_error_rate(
    consensus: pl.DataFrame,
    human_labels: pl.DataFrame,
    *,
    id_column: str = "id",
    label_column: str = "sentiment_label",
) -> float:
    """Estima a taxa de erro da rotulagem automática a partir da amostra validada por humanos.

    Parameters
    ----------
    consensus : pl.DataFrame
        Corpus com rótulo de consenso (antes da substituição por rótulos
        humanos), contendo ``id_column`` e ``label_column``.
    human_labels : pl.DataFrame
        Rótulos revisados por humanos, contendo ``id_column`` e
        ``label_column``, para o subconjunto amostrado.
    id_column : str, optional
        Nome da coluna identificadora comum, by default "id".
    label_column : str, optional
        Nome da coluna de rótulo em ambos os DataFrames, by default
        "sentiment_label".

    Returns
    -------
    float
        Proporção de amostras revisadas em que o rótulo automático diverge
        do rótulo humano, em ``[0.0, 1.0]``.

    Raises
    ------
    EmptyDatasetError
        Se não houver amostras em comum entre ``consensus`` e
        ``human_labels``.

    Examples
    --------
    >>> consensus = pl.DataFrame({"id": ["1", "2"], "sentiment_label": ["positivo", "negativo"]})
    >>> humano = pl.DataFrame({"id": ["1", "2"], "sentiment_label": ["positivo", "neutro"]})
    >>> calculate_labeling_error_rate(consensus, humano)
    0.5
    """
    joined = consensus.select([id_column, label_column]).join(
        human_labels.select([id_column, label_column]).rename({label_column: "human_label"}),
        on=id_column,
        how="inner",
    )
    validate_not_empty_collection(joined, collection_name="amostra_validada_por_humanos")
    disagreement_count = joined.filter(pl.col(label_column) != pl.col("human_label")).height
    error_rate = disagreement_count / joined.height
    logger.info(
        "Taxa de erro estimada da rotulagem automática: %.4f (%d/%d discordância(s)).",
        error_rate,
        disagreement_count,
        joined.height,
    )
    return error_rate
