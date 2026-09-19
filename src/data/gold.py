"""Loaders de gold sets (TweetSentBR, RePro) e divisão descoberta/avaliação.

Os arquivos NÃO são baixados aqui: fonte e licença de cada gold set devem ser
confirmadas por quem executa, e o arquivo colocado em ``data/external/``
(``paths.tweetsentbr_file``/``paths.repro_file``). O contrato esperado é uma
tabela com identificador, texto e rótulo de sentimento.

O texto é sanitizado (URLs e @menções substituídas por tokens) antes de
qualquer uso com embeddings ou LLM, conforme a política de LGPD do projeto.

O gold é dividido em ``G_disc`` (descoberta de hipóteses do alvo ``gold_error``)
e ``G_eval`` (reporte v1 vs v2). Nenhum tweet de ``G_eval`` entra em SAE,
seleção de neurônios ou geração de hipóteses.
"""

import logging
from pathlib import Path

import numpy as np
import polars as pl

from constants.labels import NEGATIVE_LABEL, NEUTRAL_LABEL, POSITIVE_LABEL, SENTIMENT_CLASSES
from data.loader import read_dataset_file
from exceptions.data import DataNotFoundError, DataValidationError, EmptyDatasetError
from preprocessing.text import normalize_mentions, normalize_urls

logger = logging.getLogger(__name__)

LABEL_ALIASES: dict[str, str] = {
    "positivo": POSITIVE_LABEL,
    "positive": POSITIVE_LABEL,
    "pos": POSITIVE_LABEL,
    "negativo": NEGATIVE_LABEL,
    "negative": NEGATIVE_LABEL,
    "neg": NEGATIVE_LABEL,
    "neutro": NEUTRAL_LABEL,
    "neutral": NEUTRAL_LABEL,
    "neu": NEUTRAL_LABEL,
}
GOLD_COLUMNS: tuple[str, ...] = ("id", "text_normalized", "gold_label")


def normalize_gold_label(value: object) -> str | None:
    """Converte um rótulo de gold set para uma classe de ``SENTIMENT_CLASSES``.

    Parameters
    ----------
    value : object
        Rótulo original (ex.: ``"Positive"``, ``"neg"``).

    Returns
    -------
    str | None
        Classe em pt-BR, ou ``None`` se o rótulo não for reconhecido.

    Examples
    --------
    >>> normalize_gold_label("Positive")
    'positivo'
    >>> normalize_gold_label("misto") is None
    True
    """
    return LABEL_ALIASES.get(str(value).strip().lower())


def sanitize_tweet_text(text: str) -> str:
    """Substitui URLs e @menções por tokens (LGPD) e normaliza espaços.

    Parameters
    ----------
    text : str
        Texto bruto.

    Returns
    -------
    str
        Texto sanitizado.

    Examples
    --------
    >>> sanitize_tweet_text("oi @fulano veja http://x.com")
    'oi [MENCAO] veja [URL]'
    """
    return " ".join(normalize_mentions(normalize_urls(str(text))).split())


def load_gold_set(
    file_path: Path,
    *,
    text_column: str = "text",
    label_column: str = "sentiment_label",
    id_column: str = "id",
    max_tweets: int | None = None,
    random_seed: int = 42,
) -> pl.DataFrame:
    """Lê um gold set e o converte para ``id``/``text_normalized``/``gold_label``.

    Parameters
    ----------
    file_path : Path
        Parquet ou CSV do gold set (ex.: ``paths.tweetsentbr_file``).
    text_column, label_column, id_column : str, optional
        Nomes das colunas de origem.
    max_tweets : int | None, optional
        Se informado, subamostra (estratificada por rótulo) até este total, by default None.
    random_seed : int, optional
        Semente da subamostra, by default 42.

    Returns
    -------
    pl.DataFrame
        Colunas :data:`GOLD_COLUMNS`, sem texto bruto.

    Raises
    ------
    DataNotFoundError
        Se o arquivo não existir (mensagem orienta a colocá-lo em ``data/external/``).
    DataValidationError
        Se faltarem colunas ou houver rótulos não reconhecidos.
    EmptyDatasetError
        Se nenhuma linha restar após a limpeza.

    Examples
    --------
    >>> load_gold_set(Path("data/external/tweetsentbr.parquet"))  # doctest: +SKIP
    """
    if not file_path.is_file():
        raise DataNotFoundError(
            f"{file_path} — coloque o gold set em data/external/ (confirme fonte e licença)"
        )
    raw = read_dataset_file(file_path)
    missing = [c for c in (text_column, label_column, id_column) if c not in raw.columns]
    if missing:
        raise DataValidationError(schema_name="GoldSet", detail=f"colunas ausentes: {missing}")

    gold = raw.select(
        pl.col(id_column).cast(pl.String).alias("id"),
        pl.col(text_column)
        .map_elements(sanitize_tweet_text, return_dtype=pl.String)
        .alias("text_normalized"),
        pl.col(label_column)
        .map_elements(normalize_gold_label, return_dtype=pl.String)
        .alias("gold_label"),
    )
    unknown = gold.filter(pl.col("gold_label").is_null()).height
    if unknown:
        raise DataValidationError(
            schema_name="GoldSet",
            detail=(
                f"{unknown} rótulo(s) não reconhecidos; esperado um de {list(SENTIMENT_CLASSES)}"
            ),
        )
    gold = gold.filter(pl.col("text_normalized") != "").unique(subset=["id"], keep="first")
    if gold.is_empty():
        raise EmptyDatasetError(str(file_path))
    if max_tweets is not None and gold.height > max_tweets:
        gold = _stratified_head(gold, max_tweets, random_seed)
    return gold


def _stratified_head(gold: pl.DataFrame, n_rows: int, random_seed: int) -> pl.DataFrame:
    """Subamostra estratificada por ``gold_label`` mantendo a proporção de classes."""
    fraction = n_rows / gold.height
    parts = [
        group.sample(n=max(1, round(group.height * fraction)), seed=random_seed)
        for _, group in gold.group_by("gold_label", maintain_order=True)
    ]
    return pl.concat(parts).sort("id")


def split_gold_discovery_eval(
    gold: pl.DataFrame, *, eval_fraction: float, random_seed: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Divide o gold em ``G_disc`` (descoberta) e ``G_eval`` (avaliação), estratificado por rótulo.

    Parameters
    ----------
    gold : pl.DataFrame
        Gold no formato de :data:`GOLD_COLUMNS`.
    eval_fraction : float
        Fração destinada a ``G_eval`` (0 < fração < 1).
    random_seed : int
        Semente da divisão.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        ``(G_disc, G_eval)``, disjuntos por ``id``.

    Raises
    ------
    DataValidationError
        Se ``eval_fraction`` estiver fora de (0, 1).

    Examples
    --------
    >>> g = pl.DataFrame(
    ...     {
    ...         "id": ["1", "2"],
    ...         "text_normalized": ["a", "b"],
    ...         "gold_label": ["positivo", "positivo"],
    ...     }
    ... )
    >>> disc, ev = split_gold_discovery_eval(g, eval_fraction=0.5, random_seed=0)
    >>> disc.height + ev.height
    2
    """
    if not 0.0 < eval_fraction < 1.0:
        raise DataValidationError(
            schema_name="GoldSet", detail="eval_fraction deve estar em (0, 1)"
        )
    rng = np.random.default_rng(random_seed)
    eval_ids: set[str] = set()
    for _, group in gold.group_by("gold_label", maintain_order=True):
        ids = group["id"].to_list()
        rng.shuffle(ids)
        eval_ids.update(ids[: round(len(ids) * eval_fraction)])
    is_eval = pl.col("id").is_in(list(eval_ids))
    return gold.filter(~is_eval), gold.filter(is_eval)
