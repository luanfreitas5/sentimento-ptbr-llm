"""Auditoria de vazamento entre treino, validação e teste.

Dois vazamentos são relevantes em corpora de tweets:

* **Autor repetido:** o mesmo perfil aparece em partições diferentes. O modelo pode
  aprender o estilo e o vocabulário do autor em vez do sentimento, e o teste fica
  otimista.
* **Texto duplicado:** retweets e cópias quase idênticas cruzam a fronteira entre
  partições. A comparação usa o texto normalizado (minúsculas, sem pontuação, espaços
  colapsados), de modo que pequenas variações de caixa e pontuação também contam.

Os resultados são proporções e contagens; nenhum texto ou identificador é devolvido.
"""

import logging
from typing import Any

import polars as pl

from schemas.training import DATA_SPLITS

logger = logging.getLogger(__name__)

_TRAIN_SPLIT, _VALIDATION_SPLIT, _TEST_SPLIT = DATA_SPLITS
SPLIT_PAIRS: tuple[tuple[str, str], ...] = (
    (_TRAIN_SPLIT, _TEST_SPLIT),
    (_TRAIN_SPLIT, _VALIDATION_SPLIT),
    (_VALIDATION_SPLIT, _TEST_SPLIT),
)
DUPLICATE_KEY_COLUMN = "_duplicate_key"


def build_duplicate_key_expression(text_column: str) -> pl.Expr:
    """Cria a expressão que normaliza o texto para a comparação de duplicatas.

    Parameters
    ----------
    text_column : str
        Nome da coluna de texto.

    Returns
    -------
    pl.Expr
        Expressão que minusculiza, troca pontuação por espaço e colapsa espaços.

    Examples
    --------
    >>> df = pl.DataFrame({"t": ["Oi, Mundo!", "oi mundo"]})
    >>> df.select(build_duplicate_key_expression("t").n_unique()).item()
    1
    """
    return (
        pl.col(text_column)
        .str.to_lowercase()
        .str.replace_all(r"[^\p{L}\p{N}\s]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .alias(DUPLICATE_KEY_COLUMN)
    )


def _share_of_target_in_source(
    target: pl.DataFrame, source: pl.DataFrame, column: str
) -> tuple[int, float]:
    """Conta as linhas do alvo cujo valor de ``column`` aparece na origem."""
    if target.is_empty():
        return 0, 0.0
    source_values = source[column].drop_nulls().unique()
    overlapping = int(target[column].is_in(source_values.implode()).sum())
    return overlapping, overlapping / target.height


def audit_split_leakage(
    frame: pl.DataFrame,
    *,
    text_column: str = "text",
    group_column: str | None = None,
    split_column: str = "split",
) -> dict[str, Any]:
    """Mede o vazamento de autores e de textos duplicados entre as partições.

    Parameters
    ----------
    frame : pl.DataFrame
        Dados com a coluna de partição (``treino``/``validacao``/``teste``) e a coluna
        de texto; opcionalmente a de autor (pseudônimo).
    text_column : str, optional
        Coluna de texto, by default "text".
    group_column : str | None, optional
        Coluna do autor (pseudônimo). ``None`` dispensa a medida de autores, by default None.
    split_column : str, optional
        Coluna da partição, by default "split".

    Returns
    -------
    dict[str, Any]
        Um item por par ``origem->destino`` com ``n_target``, ``duplicate_text_count``,
        ``duplicate_text_share`` e, se houver ``group_column``, ``author_overlap_count``
        e ``author_overlap_share``: a fração das linhas do destino cujo autor também
        aparece na origem.

    Raises
    ------
    KeyError
        Se faltar a coluna de partição, de texto ou de autor informada.

    Examples
    --------
    >>> df = pl.DataFrame({"text": ["a", "a", "b"], "split": ["treino", "teste", "teste"]})
    >>> audit_split_leakage(df)["treino->teste"]["duplicate_text_count"]
    1
    """
    required = [split_column, text_column] + ([group_column] if group_column else [])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Colunas ausentes para a auditoria de vazamento: {missing}")

    keyed = frame.with_columns(build_duplicate_key_expression(text_column))
    report: dict[str, Any] = {}
    for source_name, target_name in SPLIT_PAIRS:
        source = keyed.filter(pl.col(split_column) == source_name)
        target = keyed.filter(pl.col(split_column) == target_name)
        duplicate_count, duplicate_share = _share_of_target_in_source(
            target, source, DUPLICATE_KEY_COLUMN
        )
        entry: dict[str, Any] = {
            "n_target": target.height,
            "duplicate_text_count": duplicate_count,
            "duplicate_text_share": duplicate_share,
        }
        if group_column:
            author_count, author_share = _share_of_target_in_source(target, source, group_column)
            entry["author_overlap_count"] = author_count
            entry["author_overlap_share"] = author_share
        report[f"{source_name}->{target_name}"] = entry
        logger.info(
            "Vazamento %s->%s: %.2f%% de textos duplicados%s.",
            source_name,
            target_name,
            100 * duplicate_share,
            f", {100 * entry['author_overlap_share']:.2f}% de autores em comum"
            if group_column
            else "",
        )
    return report
