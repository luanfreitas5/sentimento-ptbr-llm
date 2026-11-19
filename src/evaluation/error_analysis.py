"""Avaliação por fatias com incerteza e análise de erros do conjunto de teste.

A macro-F1 agregada esconde onde o modelo falha. Este módulo complementa
:mod:`evaluation.slice_evaluation` com:

* fatias linguísticas detectadas por padrões lexicais (:func:`build_slice_flags`):
  comprimento, emoji, hashtag, negação, ironia aparente e gíria. São *proxies*: a
  "ironia aparente" é a presença de marcadores típicos, não a detecção de ironia;
* macro-F1 por fatia com IC de bootstrap e diferença contra o restante do teste
  (:func:`evaluate_slices_with_uncertainty`), sinalizando fatias pequenas;
* tabela dos maiores erros por paradigma, com categoria sugerida e coluna vazia para a
  classificação manual (:func:`build_top_error_table`);
* decomposição dos erros em erro do modelo e erro do rótulo de treino, nos tweets que
  têm referência *gold* (:func:`summarize_error_sources`).

O texto dos tweets sai sempre mascarado (:func:`utils.privacy.mask_personal_data`).
Os rótulos *gold* também são automáticos (LLMs), então "erro do rótulo de treino" é
uma estimativa e não verdade humana.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from constants.defaults import (
    DEFAULT_BOOTSTRAP_ITERATIONS,
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_RANDOM_SEED,
)
from constants.labels import SENTIMENT_CLASSES
from constants.regex import (
    EMOJI_PATTERN,
    HASHTAG_PATTERN,
    IRONY_PATTERN,
    NEGATION_PATTERN,
    SLANG_PATTERN,
)
from evaluation.gold_reference import CONSENSUS_REFERENCE
from evaluation.predictions import ModelPredictions
from evaluation.significance import calculate_macro_f1_from_codes
from exceptions.data import EmptyDatasetError
from utils.privacy import mask_personal_data

logger = logging.getLogger(__name__)

DEFAULT_MIN_SLICE_SIZE = 30
DEFAULT_TOP_N_ERRORS = 50
SHORT_TEXT_MAX_WORDS = 4
LOW_CONFIDENCE_THRESHOLD = 0.5

DIMENSION_LENGTH = "comprimento_texto"
DIMENSION_EMOJI = "emoji"
DIMENSION_HASHTAG = "hashtag"
DIMENSION_NEGATION = "negacao"
DIMENSION_IRONY = "ironia_aparente"
DIMENSION_SLANG = "giria"
BINARY_DIMENSIONS: tuple[str, ...] = (
    DIMENSION_EMOJI,
    DIMENSION_HASHTAG,
    DIMENSION_NEGATION,
    DIMENSION_IRONY,
    DIMENSION_SLANG,
)
ALL_DIMENSIONS: tuple[str, ...] = (DIMENSION_LENGTH, *BINARY_DIMENSIONS)

_BINARY_PATTERNS = {
    DIMENSION_EMOJI: EMOJI_PATTERN,
    DIMENSION_HASHTAG: HASHTAG_PATTERN,
    DIMENSION_NEGATION: NEGATION_PATTERN,
    DIMENSION_IRONY: IRONY_PATTERN,
    DIMENSION_SLANG: SLANG_PATTERN,
}
_OPTIONAL_SLICE_COLUMNS: tuple[str, ...] = (
    "accuracy",
    "f1_macro",
    "f1_macro_ci_low",
    "f1_macro_ci_high",
    "f1_macro_rest",
    "f1_macro_difference",
    "difference_ci_low",
    "difference_ci_high",
    "significantly_worse",
)
_LENGTH_LEVELS: tuple[str, ...] = ("1_curto", "2_medio", "3_longo")

CATEGORY_LABEL_NOISE = "ruido_de_rotulo"
CATEGORY_IRONY = "ironia_aparente"
CATEGORY_NO_CONTEXT = "contexto_ausente"
CATEGORY_NEUTRAL = "neutro_ambiguo"
CATEGORY_NEGATION = "negacao"
CATEGORY_OTHER = "outro"

SOURCE_CORRECT = "acerto"
SOURCE_MODEL_ERROR = "erro_do_modelo"
SOURCE_LABEL_ERROR = "erro_do_rotulo_de_treino"
SOURCE_REPRODUCES_LABEL = "modelo_reproduz_rotulo_errado"
SOURCE_BOTH_WRONG = "ambos_errados_e_discordam"
ERROR_SOURCES: tuple[str, ...] = (
    SOURCE_CORRECT,
    SOURCE_MODEL_ERROR,
    SOURCE_LABEL_ERROR,
    SOURCE_REPRODUCES_LABEL,
    SOURCE_BOTH_WRONG,
)


def build_slice_flags(texts: Sequence[str]) -> pl.DataFrame:
    """Marca, para cada texto, o valor em cada dimensão de fatia.

    Parameters
    ----------
    texts : Sequence[str]
        Textos dos tweets (sem mascaramento é suficiente: só se buscam padrões).

    Returns
    -------
    pl.DataFrame
        Uma linha por texto e uma coluna por dimensão de :data:`ALL_DIMENSIONS`.
        ``comprimento_texto`` vale ``1_curto``, ``2_medio`` ou ``3_longo`` (tercis do
        número de palavras); as demais valem ``com_<dimensão>`` ou ``sem_<dimensão>``.

    Raises
    ------
    EmptyDatasetError
        Se ``texts`` estiver vazio.

    Examples
    --------
    >>> build_slice_flags(["não gostei #ruim", "adorei 😍"])["emoji"].to_list()
    ['sem_emoji', 'com_emoji']
    """
    if not texts:
        raise EmptyDatasetError("texts")
    lengths = np.array([len(text.split()) for text in texts])
    lower, upper = np.quantile(lengths, [1 / 3, 2 / 3])
    columns: dict[str, list[str]] = {
        DIMENSION_LENGTH: [
            _LENGTH_LEVELS[0]
            if length <= lower
            else _LENGTH_LEVELS[1]
            if length <= upper
            else _LENGTH_LEVELS[2]
            for length in lengths
        ]
    }
    for dimension, pattern in _BINARY_PATTERNS.items():
        columns[dimension] = [
            f"com_{dimension}" if pattern.search(text) else f"sem_{dimension}" for text in texts
        ]
    return pl.DataFrame(columns)


def _bootstrap_macro_f1(
    true_codes: np.ndarray,
    pred_codes: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Distribuição bootstrap da macro-F1 (reamostragem de pares verdadeiro/predito)."""
    n_classes = len(SENTIMENT_CLASSES)
    size = len(true_codes)
    values = np.empty(n_bootstrap)
    for iteration in range(n_bootstrap):
        sample = rng.integers(0, size, size)
        values[iteration] = calculate_macro_f1_from_codes(
            true_codes[sample], pred_codes[sample], n_classes
        )
    return values


def _encode(labels: Sequence[str]) -> np.ndarray:
    """Converte rótulos em códigos inteiros na ordem de ``SENTIMENT_CLASSES``."""
    index = {label: position for position, label in enumerate(SENTIMENT_CLASSES)}
    return np.array([index[label] for label in labels], dtype=np.int64)


def _summarize_slice(
    true_codes: np.ndarray,
    pred_codes: np.ndarray,
    in_slice: np.ndarray,
    *,
    n_bootstrap: int,
    confidence_level: float,
    rng: np.random.Generator,
    min_slice_size: int,
) -> dict[str, Any]:
    """Métricas de uma fatia, IC da macro-F1 e diferença contra o restante do teste."""
    alpha = (1 - confidence_level) / 2
    quantiles = [100 * alpha, 100 * (1 - alpha)]
    n_classes = len(SENTIMENT_CLASSES)
    inside = (true_codes[in_slice], pred_codes[in_slice])
    outside = (true_codes[~in_slice], pred_codes[~in_slice])
    row: dict[str, Any] = {
        "n": int(in_slice.sum()),
        "share": float(in_slice.mean()),
        "small_sample": bool(in_slice.sum() < min_slice_size),
        **dict.fromkeys(_OPTIONAL_SLICE_COLUMNS),
    }
    if row["n"] == 0:
        return row
    row["accuracy"] = float(np.mean(inside[0] == inside[1]))
    row["f1_macro"] = calculate_macro_f1_from_codes(*inside, n_classes)
    inside_boot = _bootstrap_macro_f1(*inside, n_bootstrap, rng)
    row["f1_macro_ci_low"], row["f1_macro_ci_high"] = (
        float(value) for value in np.percentile(inside_boot, quantiles)
    )
    if len(outside[0]) == 0:
        return row
    row["f1_macro_rest"] = calculate_macro_f1_from_codes(*outside, n_classes)
    row["f1_macro_difference"] = row["f1_macro"] - row["f1_macro_rest"]
    difference = inside_boot - _bootstrap_macro_f1(*outside, n_bootstrap, rng)
    row["difference_ci_low"], row["difference_ci_high"] = (
        float(value) for value in np.percentile(difference, quantiles)
    )
    row["significantly_worse"] = bool(not row["small_sample"] and row["difference_ci_high"] < 0)
    return row


def evaluate_slices_with_uncertainty(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    slice_flags: pl.DataFrame,
    *,
    dimensions: Sequence[str] = ALL_DIMENSIONS,
    min_slice_size: int = DEFAULT_MIN_SLICE_SIZE,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pl.DataFrame:
    """Calcula a macro-F1 de cada fatia com IC e a diferença contra o restante.

    Parameters
    ----------
    y_true : Sequence[str]
        Rótulos verdadeiros dos tweets avaliados.
    y_pred : Sequence[str]
        Rótulos preditos, alinhados a ``y_true``.
    slice_flags : pl.DataFrame
        Saída de :func:`build_slice_flags` restrita aos mesmos tweets, na mesma ordem.
    dimensions : Sequence[str], optional
        Dimensões de ``slice_flags`` a avaliar, by default :data:`ALL_DIMENSIONS`.
    min_slice_size : int, optional
        Abaixo deste ``n`` a fatia é marcada ``small_sample`` e nunca como pior de forma
        significativa, by default 30.
    n_bootstrap : int, optional
        Reamostragens do bootstrap, by default 1000.
    confidence_level : float, optional
        Nível de confiança dos ICs, by default 0.95.
    random_seed : int, optional
        Semente do bootstrap, by default 42.

    Returns
    -------
    pl.DataFrame
        Uma linha por (dimensão, fatia) com ``n``, ``share``, ``accuracy``, ``f1_macro``
        e IC, ``f1_macro_rest`` (fora da fatia), ``f1_macro_difference`` com IC e
        ``significantly_worse`` (IC da diferença inteiro abaixo de zero).

    Raises
    ------
    EmptyDatasetError
        Se ``y_true`` estiver vazio.
    ValueError
        Se os tamanhos de ``y_true``, ``y_pred`` e ``slice_flags`` diferirem ou se uma
        dimensão não existir em ``slice_flags``.

    Examples
    --------
    >>> flags = build_slice_flags(["a b c", "d", "e f"] * 20)
    >>> y = ["positivo", "negativo", "neutro"] * 20
    >>> tabela = evaluate_slices_with_uncertainty(y, y, flags, n_bootstrap=20)
    >>> float(tabela["f1_macro"].max())
    1.0
    """
    if not y_true:
        raise EmptyDatasetError("y_true")
    if not (len(y_true) == len(y_pred) == slice_flags.height):
        raise ValueError(
            "y_true, y_pred e slice_flags devem ter o mesmo tamanho, recebido "
            f"{len(y_true)}, {len(y_pred)} e {slice_flags.height}"
        )
    missing = [name for name in dimensions if name not in slice_flags.columns]
    if missing:
        raise ValueError(f"Dimensões ausentes em slice_flags: {missing}")

    rng = np.random.default_rng(random_seed)
    true_codes, pred_codes = _encode(y_true), _encode(y_pred)
    rows: list[dict[str, Any]] = []
    for dimension in dimensions:
        values = slice_flags[dimension].to_numpy()
        for slice_value in sorted(set(values.tolist())):
            summary = _summarize_slice(
                true_codes,
                pred_codes,
                values == slice_value,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                rng=rng,
                min_slice_size=min_slice_size,
            )
            rows.append({"dimension": dimension, "slice": slice_value} | summary)
    return pl.DataFrame(rows, infer_schema_length=None)


def classify_error_source(label: str, predicted: str, reference: str) -> str:
    """Atribui um tweet à origem do erro, dado o rótulo de treino e a referência *gold*.

    Parameters
    ----------
    label : str
        Rótulo de treino (do rotulador automático).
    predicted : str
        Rótulo predito pelo modelo.
    reference : str
        Rótulo da referência *gold* (a tratada como mais confiável).

    Returns
    -------
    str
        Um valor de :data:`ERROR_SOURCES`: acerto (modelo e rótulo coincidem com a
        referência); ``erro_do_modelo`` (rótulo certo, modelo errado);
        ``erro_do_rotulo_de_treino`` (modelo acerta a referência, rótulo não);
        ``modelo_reproduz_rotulo_errado`` (modelo copia o rótulo errado) ou
        ``ambos_errados_e_discordam``.

    Examples
    --------
    >>> classify_error_source("negativo", "positivo", "positivo")
    'erro_do_rotulo_de_treino'
    """
    if predicted == reference:
        return SOURCE_CORRECT if label == reference else SOURCE_LABEL_ERROR
    if label == reference:
        return SOURCE_MODEL_ERROR
    return SOURCE_REPRODUCES_LABEL if predicted == label else SOURCE_BOTH_WRONG


def summarize_error_sources(
    predictions: Sequence[ModelPredictions],
    y_true_all: Sequence[str],
    test_ids: Sequence[str],
    references: Mapping[str, pl.DataFrame],
) -> pl.DataFrame:
    """Decompõe os erros de cada modelo em erro do modelo e erro do rótulo de treino.

    Só entram os tweets de teste cobertos pela referência (e preditos pelo modelo).

    Parameters
    ----------
    predictions : Sequence[ModelPredictions]
        Predições por modelo.
    y_true_all : Sequence[str]
        Rótulo de treino de todo o teste.
    test_ids : Sequence[str]
        ``id`` de todo o teste, alinhado a ``y_true_all``.
    references : Mapping[str, pl.DataFrame]
        Referências *gold* (``id`` e ``label``), como em
        :func:`evaluation.gold_reference.build_gold_references`.

    Returns
    -------
    pl.DataFrame
        Uma linha por (modelo, referência) com ``n`` e, para cada origem de
        :data:`ERROR_SOURCES`, a contagem e a fração (``<origem>_share``). Vazia se não
        houver referência.

    Examples
    --------
    >>> summarize_error_sources([], [], [], {}).is_empty()
    True
    """
    rows: list[dict[str, Any]] = []
    for reference_name, reference in references.items():
        reference_by_id = dict(
            zip(reference["id"].to_list(), reference["label"].to_list(), strict=True)
        )
        for item in predictions:
            counts = dict.fromkeys(ERROR_SOURCES, 0)
            for index, predicted in zip(item.indices, item.y_pred, strict=True):
                gold = reference_by_id.get(test_ids[index])
                if gold is not None:
                    counts[classify_error_source(y_true_all[index], predicted, gold)] += 1
            total = sum(counts.values())
            if total == 0:
                continue
            row: dict[str, Any] = {
                "model": item.name,
                "category": item.category,
                "reference": reference_name,
                "n": total,
            }
            for source, count in counts.items():
                row[source] = count
                row[f"{source}_share"] = count / total
            rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def suggest_error_category(
    text: str,
    label: str,
    predicted: str,
    *,
    confidence: float | None,
    error_source: str | None,
    flags: Mapping[str, str],
) -> str:
    """Sugere a categoria de um erro por regras simples (a revisão manual confirma).

    A ordem das regras é a prioridade: ruído de rótulo, ironia aparente, contexto
    ausente (texto curto), neutro ambíguo, negação e, por fim, ``outro``.

    Parameters
    ----------
    text : str
        Texto do tweet.
    label : str
        Rótulo de treino.
    predicted : str
        Rótulo predito.
    confidence : float | None
        Probabilidade da classe predita, ou ``None`` se o modelo não a expõe.
    error_source : str | None
        Origem do erro vs. a referência *gold*, ou ``None`` se o tweet não tem gold.
    flags : Mapping[str, str]
        Valores das dimensões de fatia do tweet.

    Returns
    -------
    str
        Uma das categorias ``ruido_de_rotulo``, ``ironia_aparente``,
        ``contexto_ausente``, ``neutro_ambiguo``, ``negacao`` ou ``outro``.

    Examples
    --------
    >>> suggest_error_category(
    ...     "oi", "neutro", "positivo", confidence=0.9, error_source=None, flags={}
    ... )
    'contexto_ausente'
    """
    if error_source == SOURCE_LABEL_ERROR:
        return CATEGORY_LABEL_NOISE
    if flags.get(DIMENSION_IRONY) == f"com_{DIMENSION_IRONY}":
        return CATEGORY_IRONY
    if len(text.split()) <= SHORT_TEXT_MAX_WORDS:
        return CATEGORY_NO_CONTEXT
    boundary_with_neutral = "neutro" in (label, predicted)
    if boundary_with_neutral and (confidence is None or confidence < LOW_CONFIDENCE_THRESHOLD):
        return CATEGORY_NEUTRAL
    if flags.get(DIMENSION_NEGATION) == f"com_{DIMENSION_NEGATION}":
        return CATEGORY_NEGATION
    return CATEGORY_OTHER


def select_best_model_by_category(
    metrics_table: pl.DataFrame, *, metric: str = "f1_macro"
) -> dict[str, str]:
    """Escolhe o melhor modelo de cada paradigma pela métrica principal.

    Parameters
    ----------
    metrics_table : pl.DataFrame
        Tabela com ``model``, ``category`` e ``metric``.
    metric : str, optional
        Métrica de ordenação, by default "f1_macro".

    Returns
    -------
    dict[str, str]
        Categoria -> nome do melhor modelo.

    Examples
    --------
    >>> t = pl.DataFrame({"model": ["a", "b"], "category": ["x", "x"], "f1_macro": [0.5, 0.7]})
    >>> select_best_model_by_category(t)
    {'x': 'b'}
    """
    best = metrics_table.sort(metric, descending=True).unique(
        subset="category", keep="first", maintain_order=True
    )
    return dict(zip(best["category"].to_list(), best["model"].to_list(), strict=True))


@dataclass(frozen=True)
class _ErrorContext:
    """Dados do teste compartilhados na montagem da tabela de maiores erros."""

    test_ids: list[Any]
    test_texts: list[str]
    y_true: list[str]
    wrong_count: np.ndarray
    gold_by_id: dict[Any, str]
    slice_flags: pl.DataFrame


def _build_gold_by_id(
    references: Mapping[str, pl.DataFrame], reference_name: str | None
) -> dict[Any, str]:
    """Mapeia ``id`` -> rótulo da referência (``reference_name``, consenso ou a primeira)."""
    if not references:
        return {}
    if reference_name is not None and reference_name in references:
        chosen = reference_name
    elif CONSENSUS_REFERENCE in references:
        chosen = CONSENSUS_REFERENCE
    else:
        chosen = next(iter(references))
    reference = references[chosen]
    return dict(zip(reference["id"].to_list(), reference["label"].to_list(), strict=True))


def _count_wrong_models(predictions: Sequence[ModelPredictions], y_true: list[str]) -> np.ndarray:
    """Conta, por tweet do teste, quantos modelos erram a classe."""
    wrong_count = np.zeros(len(y_true), dtype=int)
    for item in predictions:
        for index, predicted in zip(item.indices, item.y_pred, strict=True):
            wrong_count[index] += int(predicted != y_true[index])
    return wrong_count


def _build_error_row(
    category: str,
    model_name: str,
    index: int,
    predicted: str,
    confidence: float | None,
    context: _ErrorContext,
) -> dict[str, Any]:
    """Monta a linha de um erro (texto mascarado, origem do erro e categoria sugerida)."""
    y_true = context.y_true[index]
    gold = context.gold_by_id.get(context.test_ids[index])
    source = classify_error_source(y_true, predicted, gold) if gold else None
    flags = context.slice_flags.row(index, named=True)
    return {
        "category": category,
        "model": model_name,
        "id": context.test_ids[index],
        "text": mask_personal_data(context.test_texts[index]),
        "y_true": y_true,
        "y_pred": predicted,
        "confidence": confidence,
        "n_models_wrong": int(context.wrong_count[index]),
        "gold": gold,
        "error_source": source,
        **flags,
        "suggested_category": suggest_error_category(
            context.test_texts[index],
            y_true,
            predicted,
            confidence=confidence,
            error_source=source,
            flags=flags,
        ),
        "manual_category": "",
    }


def _collect_model_errors(
    item: ModelPredictions, category: str, model_name: str, context: _ErrorContext
) -> list[dict[str, Any]]:
    """Lista os erros de um modelo, ordenados do mais confiante e mais compartilhado ao menos."""
    confidences = None if item.y_score is None else item.y_score.max(axis=1)
    candidates: list[dict[str, Any]] = []
    for position, (index, predicted) in enumerate(zip(item.indices, item.y_pred, strict=True)):
        y_true = context.y_true[index]
        if predicted == y_true:
            continue
        confidence = None if confidences is None else float(confidences[position])
        candidates.append(
            _build_error_row(category, model_name, index, predicted, confidence, context)
        )
    candidates.sort(
        key=lambda row: (
            -(row["confidence"] if row["confidence"] is not None else -1.0),
            -row["n_models_wrong"],
            row["id"],
        )
    )
    return candidates


def build_top_error_table(
    predictions: Sequence[ModelPredictions],
    best_models: Mapping[str, str],
    test: pl.DataFrame,
    slice_flags: pl.DataFrame,
    references: Mapping[str, pl.DataFrame],
    *,
    reference_name: str | None = None,
    top_n: int = DEFAULT_TOP_N_ERRORS,
) -> pl.DataFrame:
    """Monta a tabela dos maiores erros do melhor modelo de cada paradigma.

    Os erros são ordenados pela confiança na classe errada (maior primeiro) e, em
    empate ou sem probabilidades (LLMs), pelo número de modelos que também erram o tweet.

    Parameters
    ----------
    predictions : Sequence[ModelPredictions]
        Predições por modelo.
    best_models : Mapping[str, str]
        Categoria -> modelo cujos erros serão listados.
    test : pl.DataFrame
        Teste com ``id``, ``text`` e ``sentiment_label``.
    slice_flags : pl.DataFrame
        Saída de :func:`build_slice_flags` para o teste inteiro.
    references : Mapping[str, pl.DataFrame]
        Referências *gold*; a ``reference_name`` (ou a única) define a origem do erro.
    reference_name : str | None, optional
        Referência usada na origem do erro, by default None (a de consenso se existir,
        senão a primeira).
    top_n : int, optional
        Erros por paradigma, by default 50.

    Returns
    -------
    pl.DataFrame
        Colunas ``category``, ``model``, ``rank``, ``id``, ``text`` (mascarado),
        ``y_true``, ``y_pred``, ``confidence``, ``n_models_wrong``, ``gold``,
        ``error_source``, as dimensões de fatia, ``suggested_category`` e
        ``manual_category`` (vazia, para a revisão manual).

    Examples
    --------
    >>> build_top_error_table([], {}, pl.DataFrame(), pl.DataFrame(), {}).is_empty()
    True
    """
    if not best_models:
        return pl.DataFrame()
    y_true_all = test["sentiment_label"].to_list()
    wrong_count = _count_wrong_models(predictions, y_true_all)
    context = _ErrorContext(
        test_ids=test["id"].to_list(),
        test_texts=test["text"].to_list(),
        y_true=y_true_all,
        wrong_count=wrong_count,
        gold_by_id=_build_gold_by_id(references, reference_name),
        slice_flags=slice_flags,
    )
    by_name = {item.name: item for item in predictions}
    rows: list[dict[str, Any]] = []
    for category, model_name in best_models.items():
        candidates = _collect_model_errors(by_name[model_name], category, model_name, context)
        for rank, row in enumerate(candidates[:top_n], start=1):
            rows.append(row | {"rank": rank})
    if not rows:
        return pl.DataFrame()
    table = pl.DataFrame(rows, infer_schema_length=None)
    leading = ["category", "model", "rank", "id", "text"]
    return table.select(leading + [name for name in table.columns if name not in leading])


def summarize_error_patterns(
    predictions: Sequence[ModelPredictions],
    best_models: Mapping[str, str],
    test: pl.DataFrame,
    slice_flags: pl.DataFrame,
    references: Mapping[str, pl.DataFrame],
    *,
    reference_name: str | None = None,
) -> pl.DataFrame:
    """Conta todos os erros do melhor modelo de cada paradigma por categoria sugerida.

    Diferente de :func:`build_top_error_table`, cobre todos os erros, não só os 50 de
    maior confiança, para que o padrão não dependa do recorte.

    Parameters
    ----------
    predictions : Sequence[ModelPredictions]
        Predições por modelo.
    best_models : Mapping[str, str]
        Categoria -> modelo analisado.
    test : pl.DataFrame
        Teste com ``id``, ``text`` e ``sentiment_label``.
    slice_flags : pl.DataFrame
        Saída de :func:`build_slice_flags` para o teste inteiro.
    references : Mapping[str, pl.DataFrame]
        Referências *gold*.
    reference_name : str | None, optional
        Referência da origem do erro, by default None.

    Returns
    -------
    pl.DataFrame
        ``category``, ``model``, ``suggested_category``, ``n_errors`` e ``share``
        (fração dos erros do modelo). Vazia se não houver erros.

    Examples
    --------
    >>> summarize_error_patterns([], {}, pl.DataFrame(), pl.DataFrame(), {}).is_empty()
    True
    """
    full = build_top_error_table(
        predictions,
        best_models,
        test,
        slice_flags,
        references,
        reference_name=reference_name,
        top_n=test.height or 1,
    )
    if full.is_empty():
        return full
    counts = full.group_by("category", "model", "suggested_category").agg(
        pl.len().alias("n_errors")
    )
    totals = counts.group_by("category").agg(pl.col("n_errors").sum().alias("_total"))
    return (
        counts.join(totals, on="category")
        .with_columns((pl.col("n_errors") / pl.col("_total")).alias("share"))
        .drop("_total")
        .sort(["category", "n_errors"], descending=[False, True])
    )
