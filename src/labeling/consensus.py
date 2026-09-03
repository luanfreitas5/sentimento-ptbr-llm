"""Agregação por concordância entre rotuladores da cascata de rotulagem.

Implementa a estratégia ``aggregation_strategy: "weighted_majority_vote"``
de ``configs/labeling.yaml``: combina os candidatos produzidos por
``src/labeling/automatic.py`` em um único rótulo de consenso por amostra,
usando as pontuações ponderadas de ``src/labeling/confidence.py``, e mescla
o resultado de volta ao corpus original.
"""

import logging

import polars as pl

from labeling.confidence import calculate_agreement_ratio

logger = logging.getLogger(__name__)


def aggregate_by_weighted_majority_vote(labeling_results: pl.DataFrame) -> pl.DataFrame:
    """Agrega os candidatos da cascata em um rótulo de consenso por votação majoritária ponderada.

    Parameters
    ----------
    labeling_results : pl.DataFrame
        Resultados da cascata no formato longo, validados contra
        :class:`schemas.labeling.LabelingResultSchema` (ver
        :func:`labeling.automatic.run_cascade_labeling`).

    Returns
    -------
    pl.DataFrame
        DataFrame com colunas ``id``, ``sentiment_label`` (rótulo de
        consenso) e ``confidence`` (razão de concordância entre
        rotuladores, em ``[0.0, 1.0]``).

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "id": ["1", "1"],
    ...         "tagger": ["heuristica", "llm"],
    ...         "sentiment_label": ["positivo", "positivo"],
    ...         "confidence": [0.8, 0.9],
    ...         "weight": [1.0, 2.0],
    ...     }
    ... )
    >>> aggregate_by_weighted_majority_vote(df)["sentiment_label"].to_list()
    ['positivo']
    """
    agreement = calculate_agreement_ratio(labeling_results)
    result = agreement.rename(
        {"consensus_label": "sentiment_label", "agreement_ratio": "confidence"}
    )
    logger.info(
        "Consenso por votação majoritária ponderada calculado para %d amostra(s).", result.height
    )
    return result


def merge_consensus_into_corpus(
    corpus: pl.DataFrame, consensus: pl.DataFrame, *, id_column: str = "id"
) -> pl.DataFrame:
    """Junta os rótulos de consenso ao corpus original pelo identificador da amostra.

    Parameters
    ----------
    corpus : pl.DataFrame
        Corpus original, contendo ao menos ``id_column``.
    consensus : pl.DataFrame
        Saída de :func:`aggregate_by_weighted_majority_vote`, contendo
        ``id_column``, ``sentiment_label`` e ``confidence``.
    id_column : str, optional
        Nome da coluna identificadora comum aos dois DataFrames,
        by default "id".

    Returns
    -------
    pl.DataFrame
        ``corpus`` acrescido das colunas ``sentiment_label`` e
        ``confidence``. Amostras sem consenso correspondente recebem
        valores nulos nessas colunas (junção à esquerda).

    Examples
    --------
    >>> corpus = pl.DataFrame({"id": ["1", "2"], "text": ["ótimo", "sem opinião"]})
    >>> consensus = pl.DataFrame(
    ...     {"id": ["1"], "sentiment_label": ["positivo"], "confidence": [0.9]}
    ... )
    >>> merge_consensus_into_corpus(corpus, consensus).sort("id")["sentiment_label"].to_list()
    ['positivo', None]
    """
    merged = corpus.join(
        consensus.select([id_column, "sentiment_label", "confidence"]), on=id_column, how="left"
    )
    logger.info("Rótulos de consenso mesclados ao corpus: %d linha(s).", merged.height)
    return merged
