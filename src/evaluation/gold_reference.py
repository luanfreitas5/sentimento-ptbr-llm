"""Avaliação dos modelos contra referências independentes do rotulador de treino.

Os modelos de ``training_*`` aprendem o rótulo do classificador do Hugging Face e, no
teste, são medidos contra esse mesmo rótulo. Esse número mostra o quanto o modelo
reproduz o rotulador, não o quanto acerta o sentimento. Este módulo mede os mesmos
modelos contra os *gold sets* (rotulados por LLMs de API) e contra o consenso deles,
e quantifica o ruído do próprio rótulo de treino:

* :func:`build_gold_references` normaliza cada *gold set* e cria a referência
  ``consenso_gold`` (tweets em que todos os gold sets concordam);
* :func:`summarize_training_label_noise` mede a concordância do rótulo de treino com
  cada referência (acurácia, Kappa de Cohen, macro-F1, alfa de Krippendorff);
* :func:`evaluate_models_against_references` calcula macro-F1 e acurácia com IC de
  bootstrap de cada modelo em cada referência, só nos tweets de teste que a referência
  cobre (nenhum deles foi usado no treino).

Os gold sets também são rótulos automáticos (de LLMs), não verdade humana: servem como
referência independente do rotulador de treino, não como substitutos da validação humana.
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef

from constants.defaults import (
    DEFAULT_BOOTSTRAP_ITERATIONS,
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_RANDOM_SEED,
)
from constants.labels import SENTIMENT_CLASSES
from evaluation.significance import calculate_macro_f1_from_codes
from exceptions.data import EmptyDatasetError

logger = logging.getLogger(__name__)

CONSENSUS_REFERENCE = "consenso_gold"
TRAINING_LABELER_NAME = "rotulador_hf"
_ID_COLUMN = "id"
_LABEL_COLUMN = "label"


def build_gold_references(
    gold_sets: Mapping[str, pl.DataFrame], *, label_column: str = "sentiment_label"
) -> dict[str, pl.DataFrame]:
    """Normaliza os gold sets e acrescenta a referência de consenso entre eles.

    Parameters
    ----------
    gold_sets : Mapping[str, pl.DataFrame]
        Nome do gold set -> DataFrame com ``id`` e ``label_column``.
    label_column : str, optional
        Coluna do rótulo nos gold sets, by default "sentiment_label".

    Returns
    -------
    dict[str, pl.DataFrame]
        Cada referência com as colunas ``id`` e ``label``, só com rótulos em
        :data:`constants.labels.SENTIMENT_CLASSES` (outros, como "indefinido", são
        descartados). Com dois ou mais gold sets,
        inclui :data:`CONSENSUS_REFERENCE`: os tweets presentes em todos e com o mesmo
        rótulo em todos.

    Raises
    ------
    EmptyDatasetError
        Se ``gold_sets`` estiver vazio.

    Examples
    --------
    >>> a = pl.DataFrame({"id": ["1", "2"], "sentiment_label": ["positivo", "negativo"]})
    >>> b = pl.DataFrame({"id": ["1", "2"], "sentiment_label": ["positivo", "neutro"]})
    >>> refs = build_gold_references({"a": a, "b": b})
    >>> refs["consenso_gold"]["id"].to_list()
    ['1']
    """
    if not gold_sets:
        raise EmptyDatasetError("gold_sets")
    references: dict[str, pl.DataFrame] = {}
    for name, frame in gold_sets.items():
        normalized = frame.select(
            pl.col(_ID_COLUMN).cast(pl.String), pl.col(label_column).alias(_LABEL_COLUMN)
        ).drop_nulls()
        valid = normalized.filter(pl.col(_LABEL_COLUMN).is_in(list(SENTIMENT_CLASSES)))
        if valid.height < normalized.height:
            # ex.: o gold da UnB tem rótulos "indefinido"; como classe extra, sem suporte nas
            # predições, eles derrubariam o macro-F1 sem ter relação com o desempenho do modelo.
            logger.info(
                "Gold %s: %d rótulo(s) fora de %s descartado(s).",
                name,
                normalized.height - valid.height,
                list(SENTIMENT_CLASSES),
            )
        references[name] = valid
    if len(references) >= 2:
        frames = list(references.values())
        joined = frames[0].rename({_LABEL_COLUMN: "label_0"})
        for index, frame in enumerate(frames[1:], start=1):
            joined = joined.join(
                frame.rename({_LABEL_COLUMN: f"label_{index}"}), on=_ID_COLUMN, how="inner"
            )
        label_columns = [column for column in joined.columns if column.startswith("label_")]
        agreeing = joined.filter(pl.concat_list(label_columns).list.n_unique() == 1).select(
            _ID_COLUMN, pl.col(label_columns[0]).alias(_LABEL_COLUMN)
        )
        references[CONSENSUS_REFERENCE] = agreeing
    return references


def calculate_krippendorff_alpha_nominal(ratings: np.ndarray) -> float:
    """Calcula o alfa de Krippendorff para dados nominais completos.

    Diferente do Kappa de Cohen (dois avaliadores), o alfa vale para qualquer número de
    avaliadores. 1 é concordância perfeita; 0, a esperada por acaso; valores negativos
    indicam discordância sistemática.

    Parameters
    ----------
    ratings : np.ndarray
        Matriz ``(n_itens, n_avaliadores)`` de rótulos (qualquer dtype comparável), sem
        valores ausentes e com ao menos dois avaliadores.

    Returns
    -------
    float
        Alfa de Krippendorff; ``nan`` se só houver uma categoria observada.

    Raises
    ------
    ValueError
        Se ``ratings`` não for 2D ou tiver menos de 2 avaliadores ou itens.

    Examples
    --------
    >>> import numpy as np
    >>> calculate_krippendorff_alpha_nominal(np.array([["a", "a"], ["b", "b"], ["a", "a"]]))
    1.0
    """
    if ratings.ndim != 2 or ratings.shape[1] < 2 or ratings.shape[0] < 2:
        raise ValueError("ratings deve ter forma (n_itens >= 2, n_avaliadores >= 2).")
    categories, codes = np.unique(ratings, return_inverse=True)
    codes = codes.reshape(ratings.shape)
    n_categories = len(categories)
    n_raters = ratings.shape[1]
    coincidence = np.zeros((n_categories, n_categories))
    for row in codes:
        counts = np.bincount(row, minlength=n_categories).astype(float)
        coincidence += (np.outer(counts, counts) - np.diag(counts)) / (n_raters - 1)
    category_totals = coincidence.sum(axis=1)
    total = category_totals.sum()
    observed = coincidence.sum() - np.trace(coincidence)
    expected = (np.outer(category_totals, category_totals).sum() - (category_totals**2).sum()) / (
        total - 1
    )
    if expected == 0:
        return float("nan")
    return float(1 - (observed / total) / (expected / total))


def _encode(labels: Sequence[str], classes: Sequence[str]) -> np.ndarray:
    index_by_class = {label: index for index, label in enumerate(classes)}
    return np.array([index_by_class[label] for label in labels])


def _bootstrap_interval(
    true_codes: np.ndarray,
    pred_codes: np.ndarray,
    n_classes: int,
    *,
    statistic: str,
    n_bootstrap: int,
    confidence_level: float,
    random_seed: int,
) -> tuple[float, float]:
    """IC percentil de ``f1_macro`` ou ``accuracy`` por bootstrap simples."""
    generator = np.random.default_rng(random_seed)
    n_samples = len(true_codes)
    values = np.empty(n_bootstrap)
    for iteration in range(n_bootstrap):
        sample = generator.integers(0, n_samples, size=n_samples)
        if statistic == "f1_macro":
            values[iteration] = calculate_macro_f1_from_codes(
                true_codes[sample], pred_codes[sample], n_classes
            )
        else:
            values[iteration] = float(np.mean(true_codes[sample] == pred_codes[sample]))
    tail = 100 * (1 - confidence_level) / 2
    return float(np.percentile(values, tail)), float(np.percentile(values, 100 - tail))


def _score_predictions(
    reference_labels: Sequence[str],
    predicted_labels: Sequence[str],
    classes: Sequence[str],
    *,
    n_bootstrap: int,
    confidence_level: float,
    random_seed: int,
) -> dict[str, float]:
    """Macro-F1 e acurácia (com IC), Kappa e MCC de uma predição contra uma referência."""
    true_codes = _encode(reference_labels, classes)
    pred_codes = _encode(predicted_labels, classes)
    n_classes = len(classes)
    f1_low, f1_high = _bootstrap_interval(
        true_codes,
        pred_codes,
        n_classes,
        statistic="f1_macro",
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_seed=random_seed,
    )
    accuracy_low, accuracy_high = _bootstrap_interval(
        true_codes,
        pred_codes,
        n_classes,
        statistic="accuracy",
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_seed=random_seed,
    )
    return {
        "f1_macro": calculate_macro_f1_from_codes(true_codes, pred_codes, n_classes),
        "f1_macro_ci_low": f1_low,
        "f1_macro_ci_high": f1_high,
        "accuracy": float(np.mean(true_codes == pred_codes)),
        "accuracy_ci_low": accuracy_low,
        "accuracy_ci_high": accuracy_high,
        "cohen_kappa": float(cohen_kappa_score(true_codes, pred_codes)),
        "mcc": float(matthews_corrcoef(true_codes, pred_codes)),
    }


def evaluate_models_against_references(
    predictions: pl.DataFrame,
    references: Mapping[str, pl.DataFrame],
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pl.DataFrame:
    """Mede cada modelo contra cada referência independente, com IC de bootstrap.

    Parameters
    ----------
    predictions : pl.DataFrame
        Tabela de ``predicoes_teste.parquet``: ``id``, ``y_true`` (rótulo de treino, do
        Hugging Face) e uma coluna ``pred_<modelo>`` por modelo (nula onde não prediz).
    references : Mapping[str, pl.DataFrame]
        Referências de :func:`build_gold_references` (``id`` e ``label``).
    n_bootstrap : int, optional
        Reamostragens, by default :data:`constants.defaults.DEFAULT_BOOTSTRAP_ITERATIONS`.
    confidence_level : float, optional
        Nível dos intervalos, by default :data:`constants.defaults.DEFAULT_CONFIDENCE_LEVEL`.
    random_seed : int, optional
        Semente, by default :data:`constants.defaults.DEFAULT_RANDOM_SEED`.

    Returns
    -------
    pl.DataFrame
        Uma linha por referência e modelo, com ``n``, ``f1_macro`` (+IC), ``accuracy``
        (+IC), ``cohen_kappa`` e ``mcc``. O rótulo de treino entra como o "modelo"
        :data:`TRAINING_LABELER_NAME`, o teto de um modelo que apenas o reproduza. Vazia se
        nenhuma referência cobrir tweets do teste.

    Raises
    ------
    KeyError
        Se ``predictions`` não tiver ``id`` e ``y_true``.

    Examples
    --------
    >>> pred = pl.DataFrame(
    ...     {
    ...         "id": ["1", "2", "3"],
    ...         "y_true": ["positivo", "negativo", "neutro"],
    ...         "pred_m": ["positivo", "negativo", "negativo"],
    ...     }
    ... )
    >>> ref = {
    ...     "g": pl.DataFrame({"id": ["1", "2", "3"], "label": ["positivo", "negativo", "neutro"]})
    ... }
    >>> out = evaluate_models_against_references(pred, ref, n_bootstrap=10)
    >>> sorted(out["model"].unique().to_list())
    ['m', 'rotulador_hf']
    """
    for required in (_ID_COLUMN, "y_true"):
        if required not in predictions.columns:
            raise KeyError(f"Coluna ausente na tabela de predições: {required!r}")
    model_columns = [column for column in predictions.columns if column.startswith("pred_")]
    candidates = predictions.select(
        pl.col(_ID_COLUMN).cast(pl.String),
        pl.col("y_true").alias(f"pred_{TRAINING_LABELER_NAME}"),
        *model_columns,
    )
    candidate_columns = [f"pred_{TRAINING_LABELER_NAME}", *model_columns]

    rows: list[dict[str, Any]] = []
    for reference_name, reference in references.items():
        joined = candidates.join(reference, on=_ID_COLUMN, how="inner")
        if joined.is_empty():
            logger.warning("Referência %s não cobre nenhum tweet do teste.", reference_name)
            continue
        for column in candidate_columns:
            valid = joined.filter(pl.col(column).is_not_null())
            if valid.height < 2:
                continue
            classes = sorted(set(valid[_LABEL_COLUMN].to_list()) | set(valid[column].to_list()))
            scores = _score_predictions(
                valid[_LABEL_COLUMN].to_list(),
                valid[column].to_list(),
                classes,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                random_seed=random_seed,
            )
            rows.append(
                {
                    "reference": reference_name,
                    "model": column.removeprefix("pred_"),
                    "n": valid.height,
                }
                | scores
            )
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).sort(["reference", "f1_macro"], descending=[False, True])


def summarize_training_label_noise(
    training_labels: pl.DataFrame,
    references: Mapping[str, pl.DataFrame],
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pl.DataFrame:
    """Quantifica o quanto o rótulo de treino diverge de cada referência independente.

    Parameters
    ----------
    training_labels : pl.DataFrame
        Colunas ``id`` e ``label``: o rótulo usado no treino, para todos os tweets que se
        queira comparar (normalmente o corpus inteiro, não só o teste).
    references : Mapping[str, pl.DataFrame]
        Referências de :func:`build_gold_references`.
    n_bootstrap : int, optional
        Reamostragens, by default :data:`constants.defaults.DEFAULT_BOOTSTRAP_ITERATIONS`.
    confidence_level : float, optional
        Nível dos intervalos, by default :data:`constants.defaults.DEFAULT_CONFIDENCE_LEVEL`.
    random_seed : int, optional
        Semente, by default :data:`constants.defaults.DEFAULT_RANDOM_SEED`.

    Returns
    -------
    pl.DataFrame
        Uma linha por referência: ``n``, acurácia (+IC), ``cohen_kappa``, ``f1_macro``
        (+IC), ``mcc`` e ``krippendorff_alpha`` (rótulo de treino × referência). A
        concordância máxima que um modelo treinado nesse rótulo pode esperar com a
        referência é da ordem da acurácia do próprio rótulo de treino contra ela.

    Examples
    --------
    >>> treino = pl.DataFrame({"id": ["1", "2"], "label": ["positivo", "negativo"]})
    >>> ref = {"g": pl.DataFrame({"id": ["1", "2"], "label": ["positivo", "neutro"]})}
    >>> summarize_training_label_noise(treino, ref, n_bootstrap=10)["n"].to_list()
    [2]
    """
    rows: list[dict[str, Any]] = []
    labels = training_labels.select(pl.col(_ID_COLUMN).cast(pl.String), pl.col(_LABEL_COLUMN))
    for reference_name, reference in references.items():
        joined = labels.join(reference, on=_ID_COLUMN, how="inner", suffix="_reference")
        if joined.height < 2:
            continue
        training = joined[_LABEL_COLUMN].to_list()
        gold = joined[f"{_LABEL_COLUMN}_reference"].to_list()
        classes = sorted(set(training) | set(gold))
        scores = _score_predictions(
            gold,
            training,
            classes,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_seed=random_seed,
        )
        alpha = calculate_krippendorff_alpha_nominal(np.column_stack([training, gold]))
        rows.append(
            {"reference": reference_name, "n": joined.height, **scores, "krippendorff_alpha": alpha}
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()
