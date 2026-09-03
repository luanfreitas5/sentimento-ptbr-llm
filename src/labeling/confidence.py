"""Pontuação de confiança e discordância por amostra na rotulagem em cascata.

Agrega os candidatos produzidos por ``src/labeling/automatic.py`` (formato
longo de :class:`schemas.labeling.LabelingResultSchema`) em uma pontuação
ponderada por rótulo e amostra, calcula a razão de concordância entre
rotuladores e sinaliza amostras de baixa confiança/alta discordância para
validação humana, conforme os limiares de ``configs/labeling.yaml``
(seção ``confidence``).
"""

import logging

import polars as pl

from schemas.labeling import validate_labeling_result
from utils.validation import validate_not_empty_collection

logger = logging.getLogger(__name__)


def calculate_weighted_label_scores(labeling_results: pl.DataFrame) -> pl.DataFrame:
    """Soma, por amostra e rótulo candidato, o score ponderado (confiança x peso).

    Parameters
    ----------
    labeling_results : pl.DataFrame
        Resultados da cascata no formato longo (ver
        :func:`labeling.automatic.run_cascade_labeling`), validado contra
        :class:`schemas.labeling.LabelingResultSchema`. Não vazio.

    Returns
    -------
    pl.DataFrame
        DataFrame com colunas ``id``, ``sentiment_label`` e
        ``weighted_score`` (soma de ``confidence * weight`` de todos os
        rotuladores que atribuíram aquele rótulo à amostra).

    Raises
    ------
    EmptyDatasetError
        Se ``labeling_results`` estiver vazio.
    DataValidationError
        Se ``labeling_results`` violar o contrato de dados.

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "id": ["1", "1", "1"],
    ...         "tagger": ["heuristica", "llm", "modelo"],
    ...         "sentiment_label": ["positivo", "positivo", "negativo"],
    ...         "confidence": [0.8, 0.6, 0.9],
    ...         "weight": [1.0, 2.0, 2.0],
    ...     }
    ... )
    >>> calculate_weighted_label_scores(df).sort("sentiment_label")["weighted_score"].to_list()
    [1.8, 2.0]
    """
    validate_not_empty_collection(labeling_results, collection_name="labeling_results")
    validated = validate_labeling_result(labeling_results)
    return (
        validated.with_columns((pl.col("confidence") * pl.col("weight")).alias("weighted_score"))
        .group_by(["id", "sentiment_label"])
        .agg(pl.col("weighted_score").sum())
    )


def calculate_agreement_ratio(labeling_results: pl.DataFrame) -> pl.DataFrame:
    """Calcula, por amostra, o rótulo vencedor e a razão de concordância ponderada.

    A razão de concordância é o score ponderado do rótulo vencedor dividido
    pela soma dos scores ponderados de todos os rótulos candidatos daquela
    amostra — quanto mais próxima de 1.0, maior o consenso entre
    rotuladores.

    Parameters
    ----------
    labeling_results : pl.DataFrame
        Resultados da cascata no formato longo (ver
        :func:`calculate_weighted_label_scores`).

    Returns
    -------
    pl.DataFrame
        DataFrame com colunas ``id``, ``consensus_label`` (rótulo com maior
        score ponderado) e ``agreement_ratio`` (em ``[0.0, 1.0]``).

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "id": ["1", "1", "1"],
    ...         "tagger": ["heuristica", "llm", "modelo"],
    ...         "sentiment_label": ["positivo", "positivo", "negativo"],
    ...         "confidence": [0.8, 0.6, 0.9],
    ...         "weight": [1.0, 2.0, 2.0],
    ...     }
    ... )
    >>> resultado = calculate_agreement_ratio(df)
    >>> resultado["consensus_label"].to_list()
    ['positivo']
    >>> round(resultado["agreement_ratio"].to_list()[0], 4)
    0.5263
    """
    scores = calculate_weighted_label_scores(labeling_results)
    totals = scores.group_by("id").agg(pl.col("weighted_score").sum().alias("total_score"))
    winners = scores.group_by("id").agg(
        pl.col("sentiment_label")
        .sort_by("weighted_score", descending=True)
        .first()
        .alias("consensus_label"),
        pl.col("weighted_score").max().alias("winning_score"),
    )
    result = winners.join(totals, on="id").with_columns(
        (pl.col("winning_score") / pl.col("total_score")).alias("agreement_ratio")
    )
    logger.info("Razão de concordância calculada para %d amostra(s).", result.height)
    return result.select(["id", "consensus_label", "agreement_ratio"])


def calculate_discordance_score(labeling_results: pl.DataFrame) -> pl.DataFrame:
    """Calcula a discordância (complemento da concordância) por amostra.

    Parameters
    ----------
    labeling_results : pl.DataFrame
        Resultados da cascata no formato longo (ver
        :func:`calculate_weighted_label_scores`).

    Returns
    -------
    pl.DataFrame
        DataFrame com as colunas de :func:`calculate_agreement_ratio`
        acrescidas de ``discordance_score`` (``1.0 - agreement_ratio``).

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "id": ["1", "1", "1"],
    ...         "tagger": ["heuristica", "llm", "modelo"],
    ...         "sentiment_label": ["positivo", "positivo", "negativo"],
    ...         "confidence": [0.8, 0.6, 0.9],
    ...         "weight": [1.0, 2.0, 2.0],
    ...     }
    ... )
    >>> round(calculate_discordance_score(df)["discordance_score"].to_list()[0], 4)
    0.4737
    """
    agreement = calculate_agreement_ratio(labeling_results)
    return agreement.with_columns((1.0 - pl.col("agreement_ratio")).alias("discordance_score"))


def flag_low_confidence_samples(
    discordance_scores: pl.DataFrame,
    *,
    low_confidence_threshold: float = 0.5,
    discordance_threshold: float = 0.3,
) -> pl.DataFrame:
    """Sinaliza amostras candidatas à validação humana por baixa concordância ou alta discordância.

    Limiares padrão refletem ``configs/labeling.yaml`` (seção
    ``confidence``): ``low_confidence_threshold`` corresponde a
    ``low_confidence_threshold`` e ``discordance_threshold`` a
    ``discordance_threshold``.

    Parameters
    ----------
    discordance_scores : pl.DataFrame
        Saída de :func:`calculate_discordance_score`, contendo
        ``agreement_ratio`` e ``discordance_score``.
    low_confidence_threshold : float, optional
        Abaixo deste valor de ``agreement_ratio``, a amostra é sinalizada,
        by default 0.5.
    discordance_threshold : float, optional
        Acima deste valor de ``discordance_score``, a amostra é sinalizada,
        by default 0.3.

    Returns
    -------
    pl.DataFrame
        DataFrame de entrada acrescido da coluna booleana
        ``requires_human_validation``.

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "id": ["1", "1", "1"],
    ...         "tagger": ["heuristica", "llm", "modelo"],
    ...         "sentiment_label": ["positivo", "positivo", "negativo"],
    ...         "confidence": [0.8, 0.6, 0.9],
    ...         "weight": [1.0, 2.0, 2.0],
    ...     }
    ... )
    >>> discordancia = calculate_discordance_score(df)
    >>> flag_low_confidence_samples(discordancia)["requires_human_validation"].to_list()
    [True]
    """
    flagged = discordance_scores.with_columns(
        (
            (pl.col("agreement_ratio") < low_confidence_threshold)
            | (pl.col("discordance_score") > discordance_threshold)
        ).alias("requires_human_validation")
    )
    n_flagged = flagged.filter(pl.col("requires_human_validation")).height
    logger.info(
        "%d/%d amostra(s) sinalizada(s) para validação humana (baixa confiança/alta discordância).",
        n_flagged,
        flagged.height,
    )
    return flagged
