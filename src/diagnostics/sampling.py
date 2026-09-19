"""Amostra estratificada por conceito para rotulagem humana (sem identificadores).

Gera ``para_rotular.csv`` com ``sample_id`` (pseudônimo SHA-256 com sal), o
``concept`` do estrato, o ``text`` (já sanitizado) e uma coluna vazia
``rotulo_humano`` para preenchimento. O mapeamento ``sample_id -> id`` fica em
um arquivo SEPARADO (``chave``), fora do git, para permitir juntar os rótulos
humanos às predições depois. O sal vem de variável de ambiente, nunca do YAML.
"""

import logging
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import polars as pl

from exceptions.configuration import MissingEnvironmentVariableError
from io_utils.csv import write_csv
from io_utils.parquet import write_parquet
from utils.hashing import calculate_text_hash

logger = logging.getLogger(__name__)

HUMAN_LABEL_COLUMN = "rotulo_humano"
SAMPLE_COLUMNS: tuple[str, ...] = ("sample_id", "concept", "text", HUMAN_LABEL_COLUMN)


def resolve_sample_salt(env_var: str) -> str:
    """Lê o sal do pseudônimo da variável de ambiente.

    Parameters
    ----------
    env_var : str
        Nome da variável (``sampling.salt_env_var``).

    Returns
    -------
    str
        Sal.

    Raises
    ------
    MissingEnvironmentVariableError
        Se a variável não estiver definida ou estiver vazia.
    """
    salt = os.environ.get(env_var, "")
    if not salt:
        raise MissingEnvironmentVariableError(env_var)
    return salt


def pseudonymize_id(tweet_id: str, salt: str) -> str:
    """Gera um pseudônimo curto e estável para um identificador de tweet.

    Parameters
    ----------
    tweet_id : str
        Identificador original.
    salt : str
        Sal secreto (irreversível sem ele).

    Returns
    -------
    str
        16 primeiros caracteres hexadecimais do SHA-256 de ``salt + id``.

    Examples
    --------
    >>> len(pseudonymize_id("123", "sal"))
    16
    >>> pseudonymize_id("123", "a") == pseudonymize_id("123", "a")
    True
    """
    return calculate_text_hash(f"{salt}{tweet_id}")[:16]


def sample_tweets_by_concept(
    tweets: pl.DataFrame,
    annotations: Mapping[str, np.ndarray],
    *,
    per_concept: int,
    salt: str,
    random_seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Amostra até ``per_concept`` tweets por conceito (anotação = 1), sem repetir tweets.

    Os conceitos mais raros são amostrados primeiro, para que não fiquem sem
    tweets por causa dos mais frequentes.

    Parameters
    ----------
    tweets : pl.DataFrame
        Tweets anotados, com ``id`` e ``text_normalized``, na mesma ordem das anotações.
    annotations : Mapping[str, np.ndarray]
        Conceito -> vetor 0/1 alinhado a ``tweets``.
    per_concept : int
        Tweets por conceito.
    salt : str
        Sal do pseudônimo.
    random_seed : int
        Semente da amostragem.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        ``(amostra, chave)``: a amostra tem :data:`SAMPLE_COLUMNS` (sem ``id``); a
        chave tem ``sample_id``/``id``/``concept``.

    Examples
    --------
    >>> t = pl.DataFrame({"id": ["1", "2"], "text_normalized": ["a", "b"]})
    >>> s, k = sample_tweets_by_concept(
    ...     t, {"c": np.array([1, 1])}, per_concept=1, salt="x", random_seed=0
    ... )
    >>> s.columns
    ['sample_id', 'concept', 'text', 'rotulo_humano']
    """
    rng = np.random.default_rng(random_seed)
    ids = tweets["id"].to_list()
    texts = tweets["text_normalized"].to_list()
    used: set[int] = set()
    rows: list[dict[str, str | None]] = []
    keys: list[dict[str, str]] = []
    for concept in sorted(annotations, key=lambda name: int(np.sum(annotations[name]))):
        candidates = [i for i in np.flatnonzero(annotations[concept] == 1) if int(i) not in used]
        chosen = rng.choice(candidates, size=min(per_concept, len(candidates)), replace=False)
        if len(candidates) < per_concept:
            logger.warning(
                "Conceito '%s': só %d tweet(s) disponíveis (< %d).",
                concept,
                len(candidates),
                per_concept,
            )
        for index in (int(i) for i in chosen):
            used.add(index)
            sample_id = pseudonymize_id(str(ids[index]), salt)
            rows.append(
                {
                    "sample_id": sample_id,
                    "concept": concept,
                    "text": texts[index],
                    HUMAN_LABEL_COLUMN: None,
                }
            )
            keys.append({"sample_id": sample_id, "id": str(ids[index]), "concept": concept})
    sample = pl.DataFrame(rows, schema=dict.fromkeys(SAMPLE_COLUMNS, pl.String))
    key = pl.DataFrame(keys, schema={"sample_id": pl.String, "id": pl.String, "concept": pl.String})
    return sample, key


def write_labeling_sample(
    sample: pl.DataFrame, key: pl.DataFrame, *, sample_csv: Path, key_parquet: Path
) -> None:
    """Grava ``para_rotular.csv`` (sem identificadores) e a chave em arquivo separado.

    Parameters
    ----------
    sample : pl.DataFrame
        Amostra (sem ``id``).
    key : pl.DataFrame
        Mapeamento ``sample_id -> id`` (manter fora do git).
    sample_csv : Path
        Destino do CSV a ser rotulado.
    key_parquet : Path
        Destino da chave.

    Examples
    --------
    >>> write_labeling_sample(sample, key, sample_csv=Path("a.csv"), key_parquet=Path("k.parquet"))
    ... # doctest: +SKIP
    """
    write_csv(sample, sample_csv)
    write_parquet(key, key_parquet)
    logger.info(
        "Amostra de %d tweets gravada em '%s' (chave em '%s').",
        sample.height,
        sample_csv,
        key_parquet,
    )
