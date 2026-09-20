"""Construção dos alvos de diagnóstico do HypotheSAEs a partir dos rótulos por modelo.

O HypotheSAEs não classifica: ele explica um alvo derivado. Este módulo
constrói os quatro alvos permitidos (ver ``CLAUDE.md``, "Alvos permitidos")
como funções puras sobre o contrato de :mod:`schemas.diagnostics`:

* ``disagreement``: discordância entre modelos (``lab_a != lab_b``).
* ``uncertainty``: incerteza (``1 - agreement_score``), alvo contínuo.
* ``pseudo_label``: pseudo-rótulo one-vs-rest de um modelo.
* ``gold_error``: erro contra o gold (``pred != gold``).

Hipóteses sobre pseudo-rótulo descrevem o comportamento do modelo, nunca a
verdade: somente o gold set mede acerto.
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Literal

import polars as pl

from constants.labels import SENTIMENT_CLASSES
from exceptions.data import DataValidationError, EmptyDatasetError
from schemas.diagnostics import (
    GOLD_LABEL_COLUMN,
    MODEL_LABEL_PREFIX,
    list_model_label_columns,
    validate_binary_target,
    validate_continuous_target,
    validate_diagnostic_corpus,
)

logger = logging.getLogger(__name__)

TargetName = Literal["disagreement", "uncertainty", "pseudo_label", "gold_error"]
TARGET_NAMES: tuple[str, ...] = ("disagreement", "uncertainty", "pseudo_label", "gold_error")

# Mapeia as colunas de ``corpus_rotulado.parquet`` para o contrato: o LLM do Hugging Face e a
# API OpenAI (etapa ``labeling``) fazem o papel de "modelos".
DEFAULT_MODEL_COLUMNS: Mapping[str, str] = {
    "sentiment_label_huggingface": f"{MODEL_LABEL_PREFIX}huggingface",
    "sentiment_label_openai": f"{MODEL_LABEL_PREFIX}openai",
}
DEFAULT_AGREEMENT_COLUMN = "confidence_score"
DEFAULT_TEXT_COLUMN = "text_normalized"


def adapt_labeled_corpus(
    labeled_corpus: pl.DataFrame,
    *,
    model_columns: Mapping[str, str] = DEFAULT_MODEL_COLUMNS,
    agreement_column: str = DEFAULT_AGREEMENT_COLUMN,
    text_column: str = DEFAULT_TEXT_COLUMN,
    gold_column: str | None = None,
) -> pl.DataFrame:
    """Converte o corpus rotulado atual para o contrato de diagnóstico.

    Mantém apenas as colunas do contrato (minimização de dados, LGPD): ``user_id``,
    texto bruto e demais metadados são descartados. Usa o texto já normalizado
    (``[MENCAO]``/``[URL]``), nunca o texto bruto.

    Parameters
    ----------
    labeled_corpus : pl.DataFrame
        Corpus rotulado (ex.: ``data/processed/corpus_rotulado.parquet``).
    model_columns : Mapping[str, str], optional
        Mapa ``coluna_original -> lab_<modelo>``, by default
        :data:`DEFAULT_MODEL_COLUMNS`.
    agreement_column : str, optional
        Coluna usada como ``agreement_score``, by default "confidence_score".
    text_column : str, optional
        Coluna de texto sanitizado, by default "text_normalized".
    gold_column : str | None, optional
        Coluna com o rótulo gold, se houver, by default None.

    Returns
    -------
    pl.DataFrame
        DataFrame com ``id``, ``text_normalized``, ``agreement_score``, uma
        coluna ``lab_<modelo>`` por modelo e, opcionalmente, ``gold_label``.

    Raises
    ------
    DataValidationError
        Se faltar alguma coluna de origem ou o resultado violar o contrato.

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "id": ["1"],
    ...         "user_id": ["u"],
    ...         "text_normalized": ["ótimo"],
    ...         "confidence_score": [0.9],
    ...         "sentiment_label_huggingface": ["positivo"],
    ...         "sentiment_label_openai": ["neutro"],
    ...     }
    ... )
    >>> adapt_labeled_corpus(df).columns
    ['id', 'text_normalized', 'agreement_score', 'lab_huggingface', 'lab_openai']
    """
    source_columns = [text_column, agreement_column, *model_columns, "id"]
    if gold_column is not None:
        source_columns.append(gold_column)
    missing = [name for name in source_columns if name not in labeled_corpus.columns]
    if missing:
        raise DataValidationError(
            schema_name="DiagnosticCorpusSchema",
            detail=f"colunas ausentes no corpus de origem: {missing}",
        )

    expressions = [
        pl.col("id").cast(pl.String),
        pl.col(text_column).alias("text_normalized"),
        pl.col(agreement_column).cast(pl.Float64).alias("agreement_score"),
        *[pl.col(source).alias(target) for source, target in model_columns.items()],
    ]
    if gold_column is not None:
        expressions.append(pl.col(gold_column).alias(GOLD_LABEL_COLUMN))
    adapted = labeled_corpus.select(expressions)
    return validate_diagnostic_corpus(adapted)


def _require_model_columns(corpus: pl.DataFrame, columns: Sequence[str]) -> None:
    """Garante que todas as colunas ``lab_*`` pedidas existem no corpus."""
    known = list_model_label_columns(corpus)
    unknown = [name for name in columns if name not in known]
    if unknown:
        raise DataValidationError(
            schema_name="DiagnosticCorpusSchema",
            detail=f"colunas de modelo desconhecidas: {unknown}; disponíveis: {known}",
        )


def _finalize_binary_target(target: pl.DataFrame, *, name: str) -> pl.DataFrame:
    """Valida o alvo binário e falha cedo se o resultado for vazio."""
    if target.is_empty():
        raise EmptyDatasetError(f"alvo '{name}' sem nenhuma linha elegível")
    result = validate_binary_target(target)
    logger.info(
        "Alvo '%s': %d linhas, prevalência do positivo = %.4f.",
        name,
        result.height,
        result["target"].mean(),
    )
    return result


def build_disagreement_target(
    corpus: pl.DataFrame, *, model_columns: Sequence[str] | None = None
) -> pl.DataFrame:
    """Constrói o alvo de discordância entre modelos.

    O alvo vale 1 quando os modelos atribuem mais de um rótulo distinto ao
    tweet. Só entram tweets em que **todos** os modelos escolhidos rotularam
    (com dois modelos, equivale a ``lab_a != lab_b``).

    Parameters
    ----------
    corpus : pl.DataFrame
        Corpus no formato do contrato de diagnóstico.
    model_columns : Sequence[str] | None, optional
        Colunas ``lab_<modelo>`` a comparar (mínimo 2). Se None, usa todas,
        by default None.

    Returns
    -------
    pl.DataFrame
        DataFrame ``id``/``target`` (0 = concordam, 1 = discordam).

    Raises
    ------
    DataValidationError
        Se houver menos de dois modelos ou colunas desconhecidas.
    EmptyDatasetError
        Se nenhum tweet tiver rótulo de todos os modelos.

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "id": ["1", "2"],
    ...         "text_normalized": ["a", "b"],
    ...         "agreement_score": [0.9, 0.4],
    ...         "lab_a": ["positivo", "negativo"],
    ...         "lab_b": ["positivo", "neutro"],
    ...     }
    ... )
    >>> build_disagreement_target(df)["target"].to_list()
    [0, 1]
    """
    columns = list(model_columns) if model_columns else list_model_label_columns(corpus)
    if len(columns) < 2:
        raise DataValidationError(
            schema_name="DiagnosticCorpusSchema",
            detail="o alvo de discordância exige ao menos dois modelos (colunas lab_*)",
        )
    _require_model_columns(corpus, columns)
    eligible = corpus.drop_nulls(subset=columns)
    disagree = pl.concat_list(columns).list.n_unique() > 1
    target = eligible.select(
        pl.col("id"),
        disagree.cast(pl.Int64).alias("target"),
    )
    return _finalize_binary_target(target, name="disagreement")


def build_uncertainty_target(corpus: pl.DataFrame) -> pl.DataFrame:
    """Constrói o alvo contínuo de incerteza, ``1 - agreement_score``.

    Parameters
    ----------
    corpus : pl.DataFrame
        Corpus no formato do contrato de diagnóstico.

    Returns
    -------
    pl.DataFrame
        DataFrame ``id``/``target`` com valores em ``[0, 1]``.

    Raises
    ------
    EmptyDatasetError
        Se o corpus estiver vazio.

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "id": ["1"],
    ...         "text_normalized": ["a"],
    ...         "agreement_score": [0.75],
    ...         "lab_a": ["positivo"],
    ...     }
    ... )
    >>> build_uncertainty_target(df)["target"].to_list()
    [0.25]
    """
    if corpus.is_empty():
        raise EmptyDatasetError("corpus vazio para o alvo 'uncertainty'")
    target = corpus.select(
        pl.col("id"),
        (1.0 - pl.col("agreement_score")).alias("target"),
    )
    return validate_continuous_target(target)


def build_pseudo_label_target(
    corpus: pl.DataFrame, *, model_column: str, label: str
) -> pl.DataFrame:
    """Constrói o pseudo-rótulo one-vs-rest de um modelo (``lab_modelo == label``).

    Hipóteses sobre este alvo descrevem o comportamento do modelo, nunca a verdade.

    Parameters
    ----------
    corpus : pl.DataFrame
        Corpus no formato do contrato de diagnóstico.
    model_column : str
        Coluna ``lab_<modelo>`` do modelo analisado.
    label : str
        Classe positiva do one-vs-rest (uma de ``SENTIMENT_CLASSES``).

    Returns
    -------
    pl.DataFrame
        DataFrame ``id``/``target`` (1 = o modelo previu ``label``).

    Raises
    ------
    DataValidationError
        Se ``label`` ou ``model_column`` forem inválidos.
    EmptyDatasetError
        Se o modelo não rotulou nenhum tweet.

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "id": ["1", "2"],
    ...         "text_normalized": ["a", "b"],
    ...         "agreement_score": [0.9, 0.9],
    ...         "lab_a": ["negativo", None],
    ...     }
    ... )
    >>> build_pseudo_label_target(df, model_column="lab_a", label="negativo")["target"].to_list()
    [1]
    """
    if label not in SENTIMENT_CLASSES:
        raise DataValidationError(
            schema_name="DiagnosticCorpusSchema",
            detail=f"rótulo '{label}' fora de {list(SENTIMENT_CLASSES)}",
        )
    _require_model_columns(corpus, [model_column])
    target = corpus.drop_nulls(subset=[model_column]).select(
        pl.col("id"),
        (pl.col(model_column) == label).cast(pl.Int64).alias("target"),
    )
    return _finalize_binary_target(target, name="pseudo_label")


def build_gold_error_target(corpus: pl.DataFrame, *, model_column: str) -> pl.DataFrame:
    """Constrói o alvo de erro contra o gold (``pred != gold``).

    Parameters
    ----------
    corpus : pl.DataFrame
        Corpus no formato do contrato, com a coluna ``gold_label``.
    model_column : str
        Coluna ``lab_<modelo>`` avaliada contra o gold.

    Returns
    -------
    pl.DataFrame
        DataFrame ``id``/``target`` (1 = o modelo errou).

    Raises
    ------
    DataValidationError
        Se não houver ``gold_label`` ou a coluna do modelo for desconhecida.
    EmptyDatasetError
        Se nenhum tweet tiver predição e gold simultaneamente.

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "id": ["1", "2"],
    ...         "text_normalized": ["a", "b"],
    ...         "agreement_score": [0.9, 0.9],
    ...         "lab_a": ["positivo", "positivo"],
    ...         "gold_label": ["positivo", "negativo"],
    ...     }
    ... )
    >>> build_gold_error_target(df, model_column="lab_a")["target"].to_list()
    [0, 1]
    """
    if GOLD_LABEL_COLUMN not in corpus.columns:
        raise DataValidationError(
            schema_name="DiagnosticCorpusSchema",
            detail=f"o alvo 'gold_error' exige a coluna '{GOLD_LABEL_COLUMN}'",
        )
    _require_model_columns(corpus, [model_column])
    target = corpus.drop_nulls(subset=[model_column, GOLD_LABEL_COLUMN]).select(
        pl.col("id"),
        (pl.col(model_column) != pl.col(GOLD_LABEL_COLUMN)).cast(pl.Int64).alias("target"),
    )
    return _finalize_binary_target(target, name="gold_error")


def build_target(
    corpus: pl.DataFrame,
    target_name: TargetName,
    *,
    model_column: str | None = None,
    model_columns: Sequence[str] | None = None,
    label: str | None = None,
) -> pl.DataFrame:
    """Despacha para o construtor do alvo escolhido, validando o corpus antes.

    Parameters
    ----------
    corpus : pl.DataFrame
        Corpus no formato do contrato de diagnóstico.
    target_name : {"disagreement", "uncertainty", "pseudo_label", "gold_error"}
        Alvo a construir.
    model_column : str | None, optional
        Modelo analisado (``pseudo_label`` e ``gold_error``), by default None.
    model_columns : Sequence[str] | None, optional
        Modelos comparados (``disagreement``), by default None.
    label : str | None, optional
        Classe positiva (``pseudo_label``), by default None.

    Returns
    -------
    pl.DataFrame
        DataFrame ``id``/``target``.

    Raises
    ------
    DataValidationError
        Se o alvo for desconhecido ou faltarem argumentos obrigatórios.

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "id": ["1"],
    ...         "text_normalized": ["a"],
    ...         "agreement_score": [0.5],
    ...         "lab_a": ["positivo"],
    ...     }
    ... )
    >>> build_target(df, "uncertainty")["target"].to_list()
    [0.5]
    """
    validate_diagnostic_corpus(corpus)
    if target_name == "disagreement":
        return build_disagreement_target(corpus, model_columns=model_columns)
    if target_name == "uncertainty":
        return build_uncertainty_target(corpus)
    if target_name in ("pseudo_label", "gold_error") and model_column is None:
        raise DataValidationError(
            schema_name="DiagnosticCorpusSchema",
            detail=f"o alvo '{target_name}' exige 'model_column'",
        )
    if target_name == "pseudo_label":
        if label is None:
            raise DataValidationError(
                schema_name="DiagnosticCorpusSchema",
                detail="o alvo 'pseudo_label' exige 'label'",
            )
        return build_pseudo_label_target(corpus, model_column=str(model_column), label=label)
    if target_name == "gold_error":
        return build_gold_error_target(corpus, model_column=str(model_column))
    raise DataValidationError(
        schema_name="DiagnosticCorpusSchema",
        detail=f"alvo desconhecido '{target_name}'; disponíveis: {list(TARGET_NAMES)}",
    )
