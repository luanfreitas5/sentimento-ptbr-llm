"""Comparação entre as bases rotuladas pelo LLM do Hugging Face e pela API OpenAI.

Funções puras sobre o DataFrame unificado por ``id`` (ver
:func:`build_comparison_frame`), sem E/S nem gráficos: cada uma devolve uma
tabela (ou um dicionário de métricas) que a etapa ``comparative_evaluation``
(``src/pipelines/comparative_evaluation.py``) grava em disco e que
``src/visualization/comparison.py`` desenha.

Como não há verdade de referência, a comparação mede **concordância** e
descreve o comportamento de cada modelo — nunca "acerto". Toda métrica de
concordância é reportada com intervalo de confiança (bootstrap por tweet).

Colunas do DataFrame unificado: ``id``, ``text_normalized``,
``label_huggingface``/``confidence_huggingface``,
``label_openai``/``confidence_openai``, ``word_count``, ``agree`` e
``label_distance`` (0 = mesma classe; 1 = uma classe é ``neutro``; 2 =
polaridades opostas). O texto original nunca é usado aqui: as tabelas
derivadas só expõem o texto normalizado (sem menções/URLs).
"""

import logging
import warnings
from typing import Any

import numpy as np
import polars as pl
from scipy import stats
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.contingency_tables import SquareTable

from constants.defaults import DEFAULT_BOOTSTRAP_ITERATIONS, DEFAULT_CONFIDENCE_LEVEL
from constants.labels import LABEL_TO_ID, SENTIMENT_CLASSES
from exceptions.data import DataValidationError, EmptyDatasetError
from labeling.validation import calculate_cohen_kappa
from metrics.classification import calculate_confusion_matrix
from schemas.labeling import validate_labeled_source

logger = logging.getLogger(__name__)

MODEL_NAMES: tuple[str, str] = ("huggingface", "openai")
_HF, _OA = MODEL_NAMES
_DEFAULT_SEED = 42
_MAX_ID_SAMPLE_IN_ERROR = 5


def build_comparison_frame(huggingface: pl.DataFrame, openai: pl.DataFrame) -> pl.DataFrame:
    """Valida as duas bases e as une pelo ``id`` do tweet.

    Parameters
    ----------
    huggingface : pl.DataFrame
        Base ``tweets_data_huggingface`` (contrato :class:`schemas.labeling.LabeledSourceSchema`).
    openai : pl.DataFrame
        Base ``tweets_data_openai`` (mesmo contrato).

    Returns
    -------
    pl.DataFrame
        Uma linha por tweet, com os rótulos e confianças dos dois modelos, ``word_count``,
        ``agree`` e ``label_distance``, na ordem da base do Hugging Face.

    Raises
    ------
    EmptyDatasetError
        Se alguma base estiver vazia.
    DataValidationError
        Se alguma base violar o contrato, ou se as duas não tiverem exatamente os mesmos ``id``.

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
    >>> build_comparison_frame(base, base)["agree"].to_list()
    [True]
    """
    for name, dataframe in ((_HF, huggingface), (_OA, openai)):
        if dataframe.is_empty():
            raise EmptyDatasetError(f"base rotulada '{name}'")
        validate_labeled_source(dataframe)

    hf_ids, oa_ids = set(huggingface["id"].to_list()), set(openai["id"].to_list())
    if hf_ids != oa_ids:
        only_hf, only_oa = sorted(hf_ids - oa_ids), sorted(oa_ids - hf_ids)
        raise DataValidationError(
            schema_name="comparison_id_sets",
            detail=(
                "as bases devem conter exatamente os mesmos tweets: "
                f"{len(only_hf)} só no huggingface (ex.: {only_hf[:_MAX_ID_SAMPLE_IN_ERROR]}), "
                f"{len(only_oa)} só no openai (ex.: {only_oa[:_MAX_ID_SAMPLE_IN_ERROR]})"
            ),
        )

    hf_columns = huggingface.select(
        "id",
        "text_normalized",
        pl.col("sentiment_label").alias(f"label_{_HF}"),
        pl.col("confidence_score").alias(f"confidence_{_HF}"),
    )
    oa_columns = openai.select(
        "id",
        pl.col("sentiment_label").alias(f"label_{_OA}"),
        pl.col("confidence_score").alias(f"confidence_{_OA}"),
    )
    return hf_columns.join(oa_columns, on="id", how="inner").with_columns(
        pl.col("text_normalized").str.count_matches(r"\S+").alias("word_count"),
        (pl.col(f"label_{_HF}") == pl.col(f"label_{_OA}")).alias("agree"),
        (
            pl.col(f"label_{_HF}").replace_strict(LABEL_TO_ID, return_dtype=pl.Int64)
            - pl.col(f"label_{_OA}").replace_strict(LABEL_TO_ID, return_dtype=pl.Int64)
        )
        .abs()
        .alias("label_distance"),
    )


def calculate_class_distribution(frame: pl.DataFrame) -> pl.DataFrame:
    """Contagem e proporção de cada classe de sentimento, por modelo.

    Parameters
    ----------
    frame : pl.DataFrame
        Saída de :func:`build_comparison_frame`.

    Returns
    -------
    pl.DataFrame
        Colunas ``model``, ``sentiment_label``, ``count`` e ``proportion``; todas as
        classes aparecem para os dois modelos (contagem 0 quando ausentes).

    Examples
    --------
    >>> frame = pl.DataFrame({"label_huggingface": ["positivo"], "label_openai": ["negativo"]})
    >>> calculate_class_distribution(frame).height
    6
    """
    rows: list[dict[str, Any]] = []
    for model in MODEL_NAMES:
        counts = frame[f"label_{model}"].value_counts().to_dicts()
        count_by_label = {row[f"label_{model}"]: row["count"] for row in counts}
        for label in SENTIMENT_CLASSES:
            count = count_by_label.get(label, 0)
            rows.append(
                {
                    "model": model,
                    "sentiment_label": label,
                    "count": count,
                    "proportion": count / frame.height,
                }
            )
    return pl.DataFrame(rows)


def build_agreement_matrix(frame: pl.DataFrame) -> np.ndarray:
    """Matriz de concordância (linhas = Hugging Face, colunas = OpenAI).

    Parameters
    ----------
    frame : pl.DataFrame
        Saída de :func:`build_comparison_frame`.

    Returns
    -------
    np.ndarray
        Matriz ``(3, 3)`` na ordem de :data:`constants.labels.SENTIMENT_CLASSES`; a diagonal
        são os tweets em que os modelos concordam.

    Examples
    --------
    >>> frame = pl.DataFrame({"label_huggingface": ["positivo"], "label_openai": ["positivo"]})
    >>> int(build_agreement_matrix(frame).trace())
    1
    """
    return calculate_confusion_matrix(
        frame[f"label_{_HF}"].to_list(), frame[f"label_{_OA}"].to_list()
    )


def _bootstrap_interval(values: np.ndarray, confidence_level: float) -> tuple[float, float]:
    """Intervalo percentil de uma distribuição bootstrap."""
    alpha = 1 - confidence_level
    return (
        float(np.percentile(values, 100 * alpha / 2)),
        float(np.percentile(values, 100 * (1 - alpha / 2))),
    )


def _calculate_marginal_homogeneity(matrix: np.ndarray) -> dict[str, float] | None:
    """Teste de Stuart-Maxwell: os dois modelos têm a mesma distribuição de classes?

    Devolve ``None`` quando o teste não é aplicável (ex.: matriz singular).
    """
    try:
        result: Any = SquareTable(matrix, shift_zeros=True).homogeneity(method="stuart_maxwell")
    except (ValueError, np.linalg.LinAlgError):
        logger.warning("Teste de homogeneidade marginal (Stuart-Maxwell) não aplicável.")
        return None
    return {
        "statistic": float(result.statistic),
        "df": float(result.df),
        "p_value": float(result.pvalue),
    }


def calculate_agreement_summary(
    frame: pl.DataFrame,
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    random_seed: int = _DEFAULT_SEED,
) -> dict[str, Any]:
    """Concordância entre os modelos, com intervalos de confiança por bootstrap.

    Parameters
    ----------
    frame : pl.DataFrame
        Saída de :func:`build_comparison_frame`.
    n_bootstrap : int, optional
        Reamostragens (de tweets, com reposição), by default
        :data:`constants.defaults.DEFAULT_BOOTSTRAP_ITERATIONS`.
    confidence_level : float, optional
        Nível de confiança dos intervalos, by default
        :data:`constants.defaults.DEFAULT_CONFIDENCE_LEVEL`.
    random_seed : int, optional
        Semente do bootstrap, by default 42.

    Returns
    -------
    dict[str, Any]
        ``n_tweets``, ``n_agree``, ``n_diverge``, ``agreement_rate``/``divergence_rate``,
        ``cohen_kappa`` e ``weighted_kappa_quadratic`` (as classes são ordinais), cada
        taxa/kappa com ``*_ci`` = ``(inferior, superior)``, ``opposite_polarity_rate`` (fração
        de divergências positivo-vs-negativo) e ``marginal_homogeneity`` (Stuart-Maxwell).

    Raises
    ------
    EmptyDatasetError
        Se ``frame`` estiver vazio.

    Examples
    --------
    >>> frame = pl.DataFrame(
    ...     {
    ...         "label_huggingface": ["positivo", "negativo", "neutro", "positivo"],
    ...         "label_openai": ["positivo", "negativo", "positivo", "positivo"],
    ...     }
    ... )
    >>> calculate_agreement_summary(frame, n_bootstrap=20)["agreement_rate"]
    0.75
    """
    if frame.is_empty():
        raise EmptyDatasetError("frame de comparação")

    hf_labels = np.asarray(frame[f"label_{_HF}"].to_list())
    oa_labels = np.asarray(frame[f"label_{_OA}"].to_list())
    n_tweets = len(hf_labels)
    classes = list(SENTIMENT_CLASSES)

    def weighted_kappa(a: np.ndarray, b: np.ndarray) -> float:
        # Kappa indefinido (ex.: reamostra com uma única classe): 1.0 se idênticos, senão 0.0,
        # o mesmo critério de labeling.validation.calculate_cohen_kappa.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            value = float(cohen_kappa_score(a, b, labels=classes, weights="quadratic"))
        if np.isnan(value):
            return 1.0 if np.array_equal(a, b) else 0.0
        return value

    generator = np.random.default_rng(random_seed)
    agreement_samples = np.empty(n_bootstrap)
    kappa_samples = np.empty(n_bootstrap)
    weighted_kappa_samples = np.empty(n_bootstrap)
    for iteration in range(n_bootstrap):
        indices = generator.integers(0, n_tweets, size=n_tweets)
        sample_a, sample_b = hf_labels[indices], oa_labels[indices]
        agreement_samples[iteration] = float(np.mean(sample_a == sample_b))
        kappa_samples[iteration] = calculate_cohen_kappa(sample_a.tolist(), sample_b.tolist())
        weighted_kappa_samples[iteration] = weighted_kappa(sample_a, sample_b)

    n_agree = int(np.sum(hf_labels == oa_labels))
    n_diverge = n_tweets - n_agree
    agreement_rate = n_agree / n_tweets
    agreement_interval = _bootstrap_interval(agreement_samples, confidence_level)
    opposite = int(frame.filter(pl.col("label_distance") == 2).height)

    return {
        "n_tweets": n_tweets,
        "n_agree": n_agree,
        "n_diverge": n_diverge,
        "agreement_rate": agreement_rate,
        "agreement_rate_ci": agreement_interval,
        "divergence_rate": n_diverge / n_tweets,
        "divergence_rate_ci": (1 - agreement_interval[1], 1 - agreement_interval[0]),
        "cohen_kappa": calculate_cohen_kappa(hf_labels.tolist(), oa_labels.tolist()),
        "cohen_kappa_ci": _bootstrap_interval(kappa_samples, confidence_level),
        "weighted_kappa_quadratic": weighted_kappa(hf_labels, oa_labels),
        "weighted_kappa_quadratic_ci": _bootstrap_interval(
            weighted_kappa_samples, confidence_level
        ),
        "opposite_polarity_rate": opposite / n_diverge if n_diverge else 0.0,
        "confidence_level": confidence_level,
        "n_bootstrap": n_bootstrap,
        "marginal_homogeneity": _calculate_marginal_homogeneity(build_agreement_matrix(frame)),
    }


def compare_confidence_scores(frame: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Compara as distribuições de confiança dos dois modelos.

    Parameters
    ----------
    frame : pl.DataFrame
        Saída de :func:`build_comparison_frame`.

    Returns
    -------
    tuple[pl.DataFrame, dict[str, Any]]
        Tabela com estatísticas descritivas por modelo (``model``, ``mean``, ``std``,
        ``median``, ``p10``, ``p90``, ``mean_when_agree``, ``mean_when_diverge``) e dicionário
        com a correlação de Spearman entre as confianças, o teste de Wilcoxon pareado
        (``None`` se todas as diferenças forem nulas) e a diferença média absoluta.

    Raises
    ------
    EmptyDatasetError
        Se ``frame`` estiver vazio.

    Examples
    --------
    >>> frame = pl.DataFrame(
    ...     {
    ...         "confidence_huggingface": [0.9, 0.6, 0.8],
    ...         "confidence_openai": [0.8, 0.7, 0.9],
    ...         "agree": [True, False, True],
    ...     }
    ... )
    >>> table, _ = compare_confidence_scores(frame)
    >>> table["model"].to_list()
    ['huggingface', 'openai']
    """
    if frame.is_empty():
        raise EmptyDatasetError("frame de comparação")

    rows: list[dict[str, Any]] = []
    for model in MODEL_NAMES:
        column = pl.col(f"confidence_{model}")
        stats_row = frame.select(
            column.mean().alias("mean"),
            column.std().alias("std"),
            column.median().alias("median"),
            column.quantile(0.1).alias("p10"),
            column.quantile(0.9).alias("p90"),
            column.filter(pl.col("agree")).mean().alias("mean_when_agree"),
            column.filter(~pl.col("agree")).mean().alias("mean_when_diverge"),
        ).to_dicts()[0]
        rows.append({"model": model} | stats_row)

    hf_confidence = frame[f"confidence_{_HF}"].to_numpy()
    oa_confidence = frame[f"confidence_{_OA}"].to_numpy()
    differences = hf_confidence - oa_confidence
    spearman: Any = stats.spearmanr(hf_confidence, oa_confidence)
    wilcoxon: dict[str, float] | None = None
    if np.any(differences != 0):
        result: Any = stats.wilcoxon(hf_confidence, oa_confidence)
        wilcoxon = {"statistic": float(result.statistic), "p_value": float(result.pvalue)}

    summary = {
        "spearman_rho": float(spearman.statistic),
        "spearman_p_value": float(spearman.pvalue),
        "wilcoxon_paired": wilcoxon,
        "mean_absolute_difference": float(np.mean(np.abs(differences))),
        "mean_difference_huggingface_minus_openai": float(np.mean(differences)),
    }
    return pl.DataFrame(rows), summary


_TWEET_COLUMNS: tuple[str, ...] = (
    "id",
    "text_normalized",
    f"label_{_HF}",
    f"confidence_{_HF}",
    f"label_{_OA}",
    f"confidence_{_OA}",
)


def find_top_divergences(frame: pl.DataFrame, *, top_n: int = 50) -> pl.DataFrame:
    """Tweets divergentes mais graves: polaridade oposta e ambos os modelos confiantes.

    A gravidade ordena primeiro pela distância entre as classes (positivo vs negativo antes de
    neutro vs polar) e depois pela menor das duas confianças (dois modelos convictos e
    discordantes indicam um tweet realmente difícil, não um palpite fraco).

    Parameters
    ----------
    frame : pl.DataFrame
        Saída de :func:`build_comparison_frame`.
    top_n : int, optional
        Máximo de tweets devolvidos, by default 50.

    Returns
    -------
    pl.DataFrame
        Tweets divergentes ordenados, com ``label_distance`` e ``min_confidence``.

    Examples
    --------
    >>> frame = pl.DataFrame(
    ...     {
    ...         "id": ["1"],
    ...         "text_normalized": ["t"],
    ...         "label_huggingface": ["positivo"],
    ...         "confidence_huggingface": [0.9],
    ...         "label_openai": ["negativo"],
    ...         "confidence_openai": [0.8],
    ...         "agree": [False],
    ...         "label_distance": [2],
    ...     }
    ... )
    >>> find_top_divergences(frame)["min_confidence"].to_list()
    [0.8]
    """
    return (
        frame.filter(~pl.col("agree"))
        .with_columns(
            pl.min_horizontal(f"confidence_{_HF}", f"confidence_{_OA}").alias("min_confidence")
        )
        .sort(["label_distance", "min_confidence"], descending=True)
        .select([*_TWEET_COLUMNS, "label_distance", "min_confidence"])
        .head(top_n)
    )


def find_confidence_conflicts(
    frame: pl.DataFrame, *, high_threshold: float = 0.8, low_threshold: float = 0.5
) -> pl.DataFrame:
    """Tweets em que um modelo tem alta confiança e o outro, baixa.

    Parameters
    ----------
    frame : pl.DataFrame
        Saída de :func:`build_comparison_frame`.
    high_threshold : float, optional
        Confiança mínima considerada alta, by default 0.8.
    low_threshold : float, optional
        Confiança máxima considerada baixa, by default 0.5.

    Returns
    -------
    pl.DataFrame
        Tweets em conflito, ordenados pela diferença de confiança, com ``more_confident_model``,
        ``confidence_gap`` e ``agree``.

    Examples
    --------
    >>> frame = pl.DataFrame(
    ...     {
    ...         "id": ["1"],
    ...         "text_normalized": ["t"],
    ...         "label_huggingface": ["positivo"],
    ...         "confidence_huggingface": [0.95],
    ...         "label_openai": ["neutro"],
    ...         "confidence_openai": [0.4],
    ...         "agree": [False],
    ...     }
    ... )
    >>> find_confidence_conflicts(frame)["more_confident_model"].to_list()
    ['huggingface']
    """
    hf_confidence, oa_confidence = pl.col(f"confidence_{_HF}"), pl.col(f"confidence_{_OA}")
    return (
        frame.filter(
            ((hf_confidence >= high_threshold) & (oa_confidence <= low_threshold))
            | ((oa_confidence >= high_threshold) & (hf_confidence <= low_threshold))
        )
        .with_columns(
            pl.when(hf_confidence > oa_confidence)
            .then(pl.lit(_HF))
            .otherwise(pl.lit(_OA))
            .alias("more_confident_model"),
            (hf_confidence - oa_confidence).abs().alias("confidence_gap"),
        )
        .sort("confidence_gap", descending=True)
        .select([*_TWEET_COLUMNS, "agree", "more_confident_model", "confidence_gap"])
    )


def _calculate_wilson_interval(
    successes: int, total: int, confidence_level: float
) -> tuple[float, float]:
    """Intervalo de Wilson para uma proporção (estável para proporções extremas e n pequeno)."""
    z_value = float(stats.norm.ppf(1 - (1 - confidence_level) / 2))
    proportion = successes / total
    denominator = 1 + z_value**2 / total
    center = (proportion + z_value**2 / (2 * total)) / denominator
    margin = (
        z_value
        * np.sqrt(proportion * (1 - proportion) / total + z_value**2 / (4 * total**2))
        / denominator
    )
    return float(center - margin), float(center + margin)


def analyze_by_text_length(
    frame: pl.DataFrame, *, n_bins: int = 5, confidence_level: float = DEFAULT_CONFIDENCE_LEVEL
) -> pl.DataFrame:
    """Concordância e confiança por faixa de tamanho do texto (quantis de ``word_count``).

    Parameters
    ----------
    frame : pl.DataFrame
        Saída de :func:`build_comparison_frame`.
    n_bins : int, optional
        Número de faixas (quantis); faixas com limites repetidos são fundidas, by default 5.
    confidence_level : float, optional
        Nível do intervalo de Wilson da taxa de concordância, by default
        :data:`constants.defaults.DEFAULT_CONFIDENCE_LEVEL`.

    Returns
    -------
    pl.DataFrame
        Uma linha por faixa: ``length_bin``, ``min_words``, ``max_words``, ``n_tweets``,
        ``agreement_rate`` (+ ``_ci_lower``/``_ci_upper``) e a confiança média de cada modelo.

    Raises
    ------
    EmptyDatasetError
        Se ``frame`` estiver vazio.

    Examples
    --------
    >>> frame = pl.DataFrame(
    ...     {
    ...         "word_count": [1, 2, 3, 4],
    ...         "agree": [True, True, False, True],
    ...         "confidence_huggingface": [0.9] * 4,
    ...         "confidence_openai": [0.8] * 4,
    ...     }
    ... )
    >>> analyze_by_text_length(frame, n_bins=2)["n_tweets"].to_list()
    [2, 2]
    """
    if frame.is_empty():
        raise EmptyDatasetError("frame de comparação")

    word_counts = frame["word_count"].to_numpy()
    edges = np.unique(np.quantile(word_counts, np.linspace(0, 1, n_bins + 1)))
    bin_index = np.digitize(word_counts, edges[1:-1], right=True)
    binned = frame.with_columns(pl.Series("_bin", bin_index))

    rows: list[dict[str, Any]] = []
    for bin_id in sorted(set(bin_index.tolist())):
        group = binned.filter(pl.col("_bin") == bin_id)
        n_agree = int(group["agree"].sum())
        lower, upper = _calculate_wilson_interval(n_agree, group.height, confidence_level)
        min_words, max_words = int(group["word_count"].min()), int(group["word_count"].max())  # type: ignore[arg-type]
        rows.append(
            {
                "length_bin": f"{min_words}-{max_words} palavras",
                "min_words": min_words,
                "max_words": max_words,
                "n_tweets": group.height,
                "agreement_rate": n_agree / group.height,
                "agreement_rate_ci_lower": lower,
                "agreement_rate_ci_upper": upper,
                f"mean_confidence_{_HF}": float(group[f"confidence_{_HF}"].mean()),  # type: ignore[arg-type]
                f"mean_confidence_{_OA}": float(group[f"confidence_{_OA}"].mean()),  # type: ignore[arg-type]
            }
        )
    return pl.DataFrame(rows)


def find_ambiguous_cases(
    frame: pl.DataFrame, *, low_threshold: float = 0.5
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Classifica os tweets ambíguos: divergência e/ou baixa confiança.

    Categorias de ``ambiguity_type`` (excludentes, da mais à menos grave):
    ``divergente_baixa_confianca`` (divergem e ao menos um está pouco confiante),
    ``divergente_alta_confianca`` (divergem, ambos confiantes),
    ``concordante_baixa_confianca_ambos`` e ``concordante_baixa_confianca_um``.

    Parameters
    ----------
    frame : pl.DataFrame
        Saída de :func:`build_comparison_frame`.
    low_threshold : float, optional
        Confiança abaixo da qual o modelo é considerado pouco confiante, by default 0.5.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        Os tweets ambíguos (com ``ambiguity_type``) e o resumo por categoria
        (``ambiguity_type``, ``n_tweets``, ``proportion`` sobre todo o corpus).

    Examples
    --------
    >>> frame = pl.DataFrame(
    ...     {
    ...         "id": ["1"],
    ...         "text_normalized": ["t"],
    ...         "label_huggingface": ["positivo"],
    ...         "confidence_huggingface": [0.4],
    ...         "label_openai": ["positivo"],
    ...         "confidence_openai": [0.45],
    ...         "agree": [True],
    ...     }
    ... )
    >>> find_ambiguous_cases(frame)[1]["ambiguity_type"].to_list()
    ['concordante_baixa_confianca_ambos']
    """
    hf_low = pl.col(f"confidence_{_HF}") < low_threshold
    oa_low = pl.col(f"confidence_{_OA}") < low_threshold
    ambiguity_type = (
        pl.when(~pl.col("agree") & (hf_low | oa_low))
        .then(pl.lit("divergente_baixa_confianca"))
        .when(~pl.col("agree"))
        .then(pl.lit("divergente_alta_confianca"))
        .when(hf_low & oa_low)
        .then(pl.lit("concordante_baixa_confianca_ambos"))
        .when(hf_low | oa_low)
        .then(pl.lit("concordante_baixa_confianca_um"))
        .otherwise(None)
        .alias("ambiguity_type")
    )
    cases = (
        frame.with_columns(ambiguity_type)
        .filter(pl.col("ambiguity_type").is_not_null())
        .select([*_TWEET_COLUMNS, "agree", "ambiguity_type"])
    )
    summary = (
        cases.group_by("ambiguity_type")
        .agg(pl.len().alias("n_tweets"))
        .with_columns((pl.col("n_tweets") / frame.height).alias("proportion"))
        .sort("n_tweets", descending=True)
    )
    return cases, summary


def summarize_classification_differences(
    frame: pl.DataFrame, *, examples_per_transition: int = 3
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Principais diferenças de classificação: para que classe cada divergência migra.

    Parameters
    ----------
    frame : pl.DataFrame
        Saída de :func:`build_comparison_frame`.
    examples_per_transition : int, optional
        Exemplos exportados por par (Hugging Face → OpenAI), by default 3.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        Tabela de transições (``label_huggingface``, ``label_openai``, ``n_tweets``,
        ``share_of_divergences``, confianças médias e ``label_distance``) e tabela de exemplos
        (os tweets de maior confiança mínima de cada par, com o texto normalizado).

    Examples
    --------
    >>> frame = pl.DataFrame(
    ...     {
    ...         "id": ["1"],
    ...         "text_normalized": ["t"],
    ...         "label_huggingface": ["positivo"],
    ...         "confidence_huggingface": [0.9],
    ...         "label_openai": ["neutro"],
    ...         "confidence_openai": [0.7],
    ...         "agree": [False],
    ...         "label_distance": [1],
    ...     }
    ... )
    >>> summarize_classification_differences(frame)[0]["n_tweets"].to_list()
    [1]
    """
    diverging = frame.filter(~pl.col("agree")).with_columns(
        pl.min_horizontal(f"confidence_{_HF}", f"confidence_{_OA}").alias("min_confidence")
    )
    transitions = (
        diverging.group_by([f"label_{_HF}", f"label_{_OA}"])
        .agg(
            pl.len().alias("n_tweets"),
            pl.col(f"confidence_{_HF}").mean().alias(f"mean_confidence_{_HF}"),
            pl.col(f"confidence_{_OA}").mean().alias(f"mean_confidence_{_OA}"),
            pl.col("label_distance").first(),
        )
        .with_columns((pl.col("n_tweets") / max(diverging.height, 1)).alias("share_of_divergences"))
        .sort("n_tweets", descending=True)
    )
    examples = (
        diverging.sort("min_confidence", descending=True)
        .group_by([f"label_{_HF}", f"label_{_OA}"], maintain_order=True)
        .head(examples_per_transition)
        .select([*_TWEET_COLUMNS, "min_confidence"])
        .sort([f"label_{_HF}", f"label_{_OA}", "min_confidence"], descending=[False, False, True])
    )
    return transitions, examples
