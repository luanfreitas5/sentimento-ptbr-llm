"""Consenso de rotulagem entre vários rotuladores (modelos do Hugging Face e gold sets).

Funções puras sobre as bases rotuladas de cada fonte (``LabeledSourceSchema``), sem
E/S nem gráficos: a etapa ``label_consensus`` (``src/pipelines/label_consensus.py``)
as grava em disco. Cada fonte é um "rotulador" (``huggingface``,
``huggingface_cardiff``, ``gold_openai``, ``gold_unb``) e todos rotulam os MESMOS
tweets (a amostra dos gold sets).

Para cada tweet, conta-se quantos rotuladores votaram em cada classe e classifica-se o
nível de consenso:

* ``unanimidade``: todos votaram na mesma classe;
* ``maioria``: uma classe tem mais da metade dos votos, mas não todos;
* ``pluralidade``: uma classe lidera de forma única, sem chegar à metade dos votos;
* ``empate``: duas ou mais classes empatam na liderança (sem rótulo de consenso).

Os gold sets são rotulados por LLMs, não por humanos: o consenso mede **concordância
entre rotuladores automáticos**, nunca "acerto" absoluto. Tweets com sentimento
``indefinido`` em qualquer fonte ficam de fora (a matriz de votos precisa estar
completa). Só o texto normalizado (sem menções/URLs) é exposto.
"""

import logging
import warnings
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

import numpy as np
import polars as pl
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa

from constants.labels import SENTIMENT_CLASSES, UNDEFINED_LABEL
from evaluation.llm_comparison import _bootstrap_interval, _calculate_wilson_interval
from exceptions.data import DataValidationError, EmptyDatasetError
from labeling.validation import calculate_krippendorff_alpha
from schemas.labeling import validate_labeled_source

logger = logging.getLogger(__name__)

CONSENSUS_LEVELS: tuple[str, ...] = ("unanimidade", "maioria", "pluralidade", "empate")
NO_CONSENSUS_LABEL = "sem_consenso"
_MIN_RATERS = 2
_MAX_ID_SAMPLE_IN_ERROR = 5
_VOTES_PREFIX = "votes_"


def _label_column(source_name: str) -> str:
    """Nome da coluna de rótulo de um rotulador no quadro de consenso."""
    return f"label_{source_name}"


def build_consensus_frame(sources: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    """Valida as bases e as une pelo ``id``, com a contagem de votos e o nível de consenso.

    Parameters
    ----------
    sources : Mapping[str, pl.DataFrame]
        Base rotulada de cada rotulador, indexada pelo nome da fonte (pelo menos duas;
        contrato :class:`schemas.labeling.LabeledSourceSchema`). Todas devem conter
        exatamente os mesmos ``id``.

    Returns
    -------
    pl.DataFrame
        Uma linha por tweet, ordenada por ``id``: ``text_normalized``, ``label_<fonte>`` e
        ``confidence_<fonte>`` de cada rotulador, ``votes_<classe>`` (votos por classe),
        ``n_raters``, ``consensus_label`` (classe líder ou ``sem_consenso`` em caso de
        empate), ``consensus_level`` (um de :data:`CONSENSUS_LEVELS`) e
        ``polarity_conflict`` (houve voto ``negativo`` e ``positivo`` no mesmo tweet).

    Raises
    ------
    DataValidationError
        Se houver menos de dois rotuladores, alguma base violar o contrato ou as bases
        tiverem tweets diferentes.
    EmptyDatasetError
        Se alguma base estiver vazia ou nenhum tweet restar após remover os indefinidos.

    Examples
    --------
    >>> base = pl.DataFrame(
    ...     {
    ...         "id": ["1"],
    ...         "text": ["Adorei!"],
    ...         "text_normalized": ["adorei"],
    ...         "sentiment_label": ["positivo"],
    ...         "confidence_score": [0.9],
    ...     }
    ... )
    >>> frame = build_consensus_frame({"a": base, "b": base})
    >>> frame["consensus_level"].to_list()
    ['unanimidade']
    """
    if len(sources) < _MIN_RATERS:
        raise DataValidationError(
            schema_name="label_consensus",
            detail=f"o consenso exige ao menos {_MIN_RATERS} rotuladores; recebido: {len(sources)}",
        )
    id_sets: dict[str, set[str]] = {}
    for name, dataframe in sources.items():
        if dataframe.is_empty():
            raise EmptyDatasetError(f"base rotulada '{name}'")
        validate_labeled_source(dataframe)
        id_sets[name] = set(dataframe["id"].to_list())

    reference_name, reference_ids = next(iter(id_sets.items()))
    for name, ids in id_sets.items():
        if ids != reference_ids:
            raise DataValidationError(
                schema_name="label_consensus_id_sets",
                detail=(
                    "as bases devem conter exatamente os mesmos tweets: "
                    f"'{name}' difere de '{reference_name}' em {len(ids ^ reference_ids)} id(s) "
                    f"(ex.: {sorted(ids ^ reference_ids)[:_MAX_ID_SAMPLE_IN_ERROR]})"
                ),
            )

    names = list(sources)
    first = sources[names[0]].select("id", "text_normalized")
    joined = first
    for name in names:
        joined = joined.join(
            sources[name].select(
                "id",
                pl.col("sentiment_label").alias(_label_column(name)),
                pl.col("confidence_score").alias(f"confidence_{name}"),
            ),
            on="id",
            how="inner",
        )
    label_columns = [_label_column(name) for name in names]
    defined = joined.filter(pl.all_horizontal(pl.col(c) != UNDEFINED_LABEL for c in label_columns))
    if defined.is_empty():
        raise EmptyDatasetError("tweets com rótulo definido em todos os rotuladores")
    if defined.height < joined.height:
        logger.warning(
            "%d tweet(s) com sentimento indefinido em algum rotulador ficaram fora do consenso.",
            joined.height - defined.height,
        )
    return _add_consensus_columns(defined.sort("id"), label_columns)


def _add_consensus_columns(frame: pl.DataFrame, label_columns: Sequence[str]) -> pl.DataFrame:
    """Acrescenta contagem de votos, rótulo e nível de consenso ao quadro unido."""
    n_raters = len(label_columns)
    vote_columns = {
        label: pl.sum_horizontal(pl.col(c) == label for c in label_columns).alias(
            f"{_VOTES_PREFIX}{label}"
        )
        for label in SENTIMENT_CLASSES
    }
    with_votes = frame.with_columns(*vote_columns.values()).with_columns(
        pl.lit(n_raters).alias("n_raters"),
        pl.max_horizontal(f"{_VOTES_PREFIX}{label}" for label in SENTIMENT_CLASSES).alias(
            "_top_votes"
        ),
    )
    n_leaders = pl.sum_horizontal(
        pl.col(f"{_VOTES_PREFIX}{label}") == pl.col("_top_votes") for label in SENTIMENT_CLASSES
    )
    leader_label = pl.coalesce(
        *(
            pl.when(pl.col(f"{_VOTES_PREFIX}{label}") == pl.col("_top_votes")).then(pl.lit(label))
            for label in SENTIMENT_CLASSES
        )
    )
    return (
        with_votes.with_columns(
            n_leaders.alias("_n_leaders"),
            leader_label.alias("_leader"),
        )
        .with_columns(
            pl.when(pl.col("_n_leaders") > 1)
            .then(pl.lit(NO_CONSENSUS_LABEL))
            .otherwise(pl.col("_leader"))
            .alias("consensus_label"),
            pl.when(pl.col("_top_votes") == n_raters)
            .then(pl.lit("unanimidade"))
            .when(pl.col("_n_leaders") > 1)
            .then(pl.lit("empate"))
            .when(pl.col("_top_votes") * 2 > n_raters)
            .then(pl.lit("maioria"))
            .otherwise(pl.lit("pluralidade"))
            .alias("consensus_level"),
            (
                (pl.col(f"{_VOTES_PREFIX}negativo") > 0) & (pl.col(f"{_VOTES_PREFIX}positivo") > 0)
            ).alias("polarity_conflict"),
        )
        .drop("_top_votes", "_n_leaders", "_leader")
    )


def summarize_consensus_levels(
    frame: pl.DataFrame, *, confidence_level: float = 0.95
) -> pl.DataFrame:
    """Quantos tweets há em cada nível de consenso, com intervalo de Wilson.

    Parameters
    ----------
    frame : pl.DataFrame
        Saída de :func:`build_consensus_frame`.
    confidence_level : float, optional
        Nível de confiança do intervalo de Wilson, by default 0.95.

    Returns
    -------
    pl.DataFrame
        ``consensus_level``, ``n_tweets``, ``proportion``, ``ci_lower`` e ``ci_upper``,
        com uma linha por nível de :data:`CONSENSUS_LEVELS` (inclusive os de contagem zero).

    Examples
    --------
    >>> summarize_consensus_levels(build_consensus_frame({"a": base, "b": base}))  # doctest: +SKIP
    """
    total = frame.height
    counts = dict(frame.group_by("consensus_level").len().iter_rows())
    rows: list[dict[str, Any]] = []
    for level in CONSENSUS_LEVELS:
        n_tweets = counts.get(level, 0)
        lower, upper = _calculate_wilson_interval(n_tweets, total, confidence_level)
        rows.append(
            {
                "consensus_level": level,
                "n_tweets": n_tweets,
                "proportion": n_tweets / total,
                "ci_lower": lower,
                "ci_upper": upper,
            }
        )
    return pl.DataFrame(rows)


def summarize_consensus_by_class(frame: pl.DataFrame) -> pl.DataFrame:
    """Cruza a classe de consenso com o nível de consenso (quais classes têm consenso forte).

    Parameters
    ----------
    frame : pl.DataFrame
        Saída de :func:`build_consensus_frame`.

    Returns
    -------
    pl.DataFrame
        ``consensus_label``, ``n_tweets``, uma coluna ``n_<nível>`` por nível de
        :data:`CONSENSUS_LEVELS` e ``unanimity_rate`` (fração de unanimidade na classe), com
        uma linha por classe de sentimento mais ``sem_consenso``.

    Examples
    --------
    >>> summarize_consensus_by_class(frame)  # doctest: +SKIP
    """
    rows: list[dict[str, Any]] = []
    for label in (*SENTIMENT_CLASSES, NO_CONSENSUS_LABEL):
        subset = frame.filter(pl.col("consensus_label") == label)
        row: dict[str, Any] = {"consensus_label": label, "n_tweets": subset.height}
        for level in CONSENSUS_LEVELS:
            row[f"n_{level}"] = subset.filter(pl.col("consensus_level") == level).height
        row["unanimity_rate"] = row["n_unanimidade"] / subset.height if subset.height else None
        rows.append(row)
    return pl.DataFrame(rows)


def _safe_cohen_kappa(labels_a: np.ndarray, labels_b: np.ndarray) -> float | None:
    """Kappa de Cohen sobre as três classes; ``None`` quando indefinido (sem variação)."""
    with warnings.catch_warnings():
        # reamostras com uma única classe em comum são esperadas no bootstrap
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        kappa = float(cohen_kappa_score(labels_a, labels_b, labels=list(SENTIMENT_CLASSES)))
    return None if np.isnan(kappa) else kappa


def calculate_pairwise_agreement(
    frame: pl.DataFrame,
    source_names: Sequence[str],
    *,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> pl.DataFrame:
    """Concordância e Kappa de Cohen entre cada par de rotuladores, com IC por bootstrap.

    Parameters
    ----------
    frame : pl.DataFrame
        Saída de :func:`build_consensus_frame`.
    source_names : Sequence[str]
        Rotuladores a comparar par a par.
    n_bootstrap : int, optional
        Reamostragens de tweets, by default 1000.
    confidence_level : float, optional
        Nível de confiança dos intervalos, by default 0.95.
    random_seed : int, optional
        Semente do bootstrap, by default 42.

    Returns
    -------
    pl.DataFrame
        Uma linha por par: ``source_a``, ``source_b``, ``n_tweets``, ``agreement_rate``,
        ``agreement_ci_lower``/``agreement_ci_upper``, ``cohen_kappa`` e
        ``kappa_ci_lower``/``kappa_ci_upper`` (nulos quando o Kappa é indefinido).

    Examples
    --------
    >>> calculate_pairwise_agreement(frame, ["a", "b"], n_bootstrap=50)  # doctest: +SKIP
    """
    rng = np.random.default_rng(random_seed)
    n_tweets = frame.height
    resample_indices = rng.integers(0, n_tweets, size=(n_bootstrap, n_tweets))
    rows: list[dict[str, Any]] = []
    for name_a, name_b in combinations(source_names, 2):
        labels_a = frame[_label_column(name_a)].to_numpy()
        labels_b = frame[_label_column(name_b)].to_numpy()
        agree = labels_a == labels_b
        agreement_boot = agree[resample_indices].mean(axis=1)
        kappa_boot = [
            _safe_cohen_kappa(labels_a[indices], labels_b[indices]) for indices in resample_indices
        ]
        valid_kappa = np.array([k for k in kappa_boot if k is not None])
        kappa = _safe_cohen_kappa(labels_a, labels_b)
        agreement_ci = _bootstrap_interval(agreement_boot, confidence_level)
        kappa_ci = (
            _bootstrap_interval(valid_kappa, confidence_level) if valid_kappa.size else (None, None)
        )
        rows.append(
            {
                "source_a": name_a,
                "source_b": name_b,
                "n_tweets": n_tweets,
                "agreement_rate": float(agree.mean()),
                "agreement_ci_lower": agreement_ci[0],
                "agreement_ci_upper": agreement_ci[1],
                "cohen_kappa": kappa,
                "kappa_ci_lower": kappa_ci[0],
                "kappa_ci_upper": kappa_ci[1],
            }
        )
    return pl.DataFrame(rows, schema_overrides={"cohen_kappa": pl.Float64})


def _calculate_fleiss(votes: np.ndarray) -> float | None:
    """Kappa de Fleiss da tabela tweets x classes; ``None`` quando indefinido."""
    kappa = float(fleiss_kappa(votes, method="fleiss"))
    return None if np.isnan(kappa) else kappa


def calculate_multi_rater_agreement(
    frame: pl.DataFrame,
    source_names: Sequence[str],
    *,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Concordância global entre todos os rotuladores: Kappa de Fleiss e Alpha de Krippendorff.

    Parameters
    ----------
    frame : pl.DataFrame
        Saída de :func:`build_consensus_frame`.
    source_names : Sequence[str]
        Rotuladores considerados.
    n_bootstrap : int, optional
        Reamostragens de tweets para o IC do Kappa de Fleiss, by default 1000.
    confidence_level : float, optional
        Nível de confiança do intervalo, by default 0.95.
    random_seed : int, optional
        Semente do bootstrap, by default 42.

    Returns
    -------
    dict[str, Any]
        ``n_tweets``, ``n_raters``, ``fleiss_kappa`` (+ ``fleiss_kappa_ci``),
        ``krippendorff_alpha`` e ``unanimity_rate``.

    Examples
    --------
    >>> calculate_multi_rater_agreement(frame, ["a", "b"], n_bootstrap=50)  # doctest: +SKIP
    """
    votes = frame.select(f"{_VOTES_PREFIX}{label}" for label in SENTIMENT_CLASSES).to_numpy()
    rng = np.random.default_rng(random_seed)
    resample_indices = rng.integers(0, frame.height, size=(n_bootstrap, frame.height))
    boot = [_calculate_fleiss(votes[indices]) for indices in resample_indices]
    valid_boot = np.array([value for value in boot if value is not None])
    reliability_data = [frame[_label_column(name)].to_list() for name in source_names]
    return {
        "n_tweets": frame.height,
        "n_raters": len(source_names),
        "fleiss_kappa": _calculate_fleiss(votes),
        "fleiss_kappa_ci": (
            list(_bootstrap_interval(valid_boot, confidence_level)) if valid_boot.size else None
        ),
        "krippendorff_alpha": calculate_krippendorff_alpha(reliability_data),
        "unanimity_rate": float((frame["consensus_level"] == "unanimidade").mean()),
        "confidence_level": confidence_level,
    }


def find_disputed_tweets(frame: pl.DataFrame, *, top_n: int = 50) -> pl.DataFrame:
    """Tweets com menor consenso, para inspeção manual (polaridade oposta primeiro).

    Parameters
    ----------
    frame : pl.DataFrame
        Saída de :func:`build_consensus_frame`.
    top_n : int, optional
        Máximo de tweets devolvidos, by default 50.

    Returns
    -------
    pl.DataFrame
        Tweets que não são unânimes, ordenados por conflito de polaridade e por menor número
        de votos na classe líder; só o texto normalizado é exposto.

    Examples
    --------
    >>> find_disputed_tweets(frame, top_n=10)  # doctest: +SKIP
    """
    vote_columns = [f"{_VOTES_PREFIX}{label}" for label in SENTIMENT_CLASSES]
    label_columns = [c for c in frame.columns if c.startswith("label_")]
    return (
        frame.filter(pl.col("consensus_level") != "unanimidade")
        .with_columns(pl.max_horizontal(vote_columns).alias("_top_votes"))
        .sort(["polarity_conflict", "_top_votes", "id"], descending=[True, False, False])
        .select(
            "id",
            "text_normalized",
            *label_columns,
            *vote_columns,
            "consensus_label",
            "consensus_level",
            "polarity_conflict",
        )
        .head(top_n)
    )
