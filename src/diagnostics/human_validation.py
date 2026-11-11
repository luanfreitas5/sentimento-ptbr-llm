"""Validação humana (n=500) da rotulagem automática, estratificada por confiança.

Duas etapas, ambas sobre a amostra comum dos *gold sets* (cada post tem o rótulo do
rotulador primário Hugging Face e dos dois LLMs):

1. ``prepare``: sorteia ``n`` posts com alocação igual entre três faixas de confiança
   do rotulador primário (baixa, média, alta) e grava ``para_rotular_500.csv`` (sem
   identificadores e **sem** os rótulos automáticos, para não ancorar o anotador) e a
   chave ``sample_id -> id`` (fora do git, em ``data/silver``) com a faixa e o peso.
2. ``evaluate``: lê o CSV preenchido (coluna ``rotulo_humano``) e estima, por
   rotulador, a concordância com o humano (ponderada pelo desenho amostral), o Kappa de
   Cohen e o Macro F1, com IC de 95% por *bootstrap* estratificado.

Uso::

    PYTHONPATH=src python -m diagnostics.human_validation prepare
    PYTHONPATH=src python -m diagnostics.human_validation evaluate

Os rótulos humanos medem acerto; os rótulos dos modelos medem apenas concordância.
"""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import cohen_kappa_score, f1_score

from config.environment import configure_environment_variables
from config.logging import configure_logging
from config.paths import ProjectPaths, load_project_paths
from constants.defaults import DEFAULT_BOOTSTRAP_ITERATIONS, DEFAULT_RANDOM_SEED
from constants.labels import SENTIMENT_CLASSES
from diagnostics.gold_eval import load_human_labels
from diagnostics.sampling import HUMAN_LABEL_COLUMN, pseudonymize_id, resolve_sample_salt
from evaluation.significance import run_mcnemar_test
from exceptions.data import DataValidationError, EmptyDatasetError
from io_utils.csv import write_csv
from io_utils.parquet import write_parquet

# Executado com `python -m`, `__name__` vira "__main__" e o nível de log do projeto não se aplica.
logger = logging.getLogger("diagnostics.human_validation" if __name__ == "__main__" else __name__)

SAMPLE_FILE_NAME = "para_rotular_500.csv"
KEY_FILE_NAME = "chave_validacao_humana.parquet"
RESULTS_FILE_NAME = "validacao_humana_resultados.csv"
MCNEMAR_FILE_NAME = "validacao_humana_mcnemar.csv"
OUTPUT_DIR_NAME = "validacao_humana"
CORPUS_FILE_NAME = "corpus_rotulado.parquet"
SALT_ENV_VAR = "DIAGNOSTICS_SAMPLE_SALT"
STRATA: tuple[str, ...] = ("baixa", "media", "alta")
CONCEPT_PLACEHOLDER = "validacao_humana"
RATER_COLUMNS: dict[str, str] = {
    "huggingface_primario": "lab_huggingface",
    "gold_openai": "lab_openai",
    "gold_unb": "lab_unb",
}


def build_rater_table(paths: ProjectPaths) -> pl.DataFrame:
    """Une os rótulos do primário HF e dos dois gold sets na amostra comum.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos do projeto (camada gold).

    Returns
    -------
    pl.DataFrame
        Colunas ``id``, ``text_normalized``, ``conf_hf``, ``lab_huggingface``,
        ``lab_openai`` e ``lab_unb`` (apenas posts com rótulo válido nos três).

    Raises
    ------
    EmptyDatasetError
        Se a interseção entre as três fontes for vazia.
    """
    logger.info("Lendo o corpus rotulado e os gold sets da camada gold...")
    corpus = pl.read_parquet(paths.data_gold_dir / CORPUS_FILE_NAME).select(
        "id",
        "text_normalized",
        pl.col("confidence_score_huggingface").alias("conf_hf"),
        pl.col("sentiment_label_huggingface").alias("lab_huggingface"),
    )
    openai = pl.read_parquet(paths.gold_openai_file).select(
        "id", pl.col("sentiment_label").alias("lab_openai")
    )
    unb = pl.read_parquet(paths.gold_unb_file).select(
        "id", pl.col("sentiment_label").alias("lab_unb")
    )
    table = corpus.join(openai, on="id").join(unb, on="id")
    logger.info(
        "Interseção das três fontes: %d post(s) (corpus=%d, gold_openai=%d, gold_unb=%d).",
        table.height,
        corpus.height,
        openai.height,
        unb.height,
    )
    valid_classes = list(SENTIMENT_CLASSES)
    table = table.filter(
        pl.all_horizontal(pl.col(column).is_in(valid_classes) for column in RATER_COLUMNS.values())
    )
    logger.info("%d post(s) com rótulo válido nos três rotuladores.", table.height)
    if table.is_empty():
        raise EmptyDatasetError("amostra comum dos rotuladores (HF + gold sets)")
    return table


def assign_confidence_strata(table: pl.DataFrame) -> pl.DataFrame:
    """Atribui a faixa de confiança (tercis do primário HF) a cada post.

    Parameters
    ----------
    table : pl.DataFrame
        Saída de :func:`build_rater_table`.

    Returns
    -------
    pl.DataFrame
        ``table`` com a coluna ``stratum`` (``baixa``, ``media`` ou ``alta``).

    Examples
    --------
    >>> t = pl.DataFrame({"conf_hf": [0.1, 0.5, 0.9]})
    >>> assign_confidence_strata(t)["stratum"].to_list()
    ['baixa', 'media', 'alta']
    """
    low, high = table["conf_hf"].quantile(1 / 3), table["conf_hf"].quantile(2 / 3)
    return table.with_columns(
        pl.when(pl.col("conf_hf") <= low)
        .then(pl.lit(STRATA[0]))
        .when(pl.col("conf_hf") <= high)
        .then(pl.lit(STRATA[1]))
        .otherwise(pl.lit(STRATA[2]))
        .alias("stratum")
    )


def draw_stratified_sample(
    table: pl.DataFrame, *, sample_size: int, random_seed: int
) -> pl.DataFrame:
    """Sorteia a amostra com alocação igual entre as faixas e calcula o peso de cada post.

    O peso é ``N_faixa / n_faixa`` (inverso da probabilidade de inclusão), usado na
    estimativa ponderada para refletir a população dos gold sets.

    Parameters
    ----------
    table : pl.DataFrame
        Saída de :func:`assign_confidence_strata`.
    sample_size : int
        Tamanho total desejado.
    random_seed : int
        Semente do sorteio.

    Returns
    -------
    pl.DataFrame
        Linhas sorteadas, com a coluna ``weight``.

    Raises
    ------
    EmptyDatasetError
        Se não houver posts para sortear.

    Examples
    --------
    >>> draw_stratified_sample(table, sample_size=500, random_seed=42)  # doctest: +SKIP
    """
    per_stratum = sample_size // len(STRATA)
    parts: list[pl.DataFrame] = []
    for index, stratum in enumerate(STRATA):
        pool = table.filter(pl.col("stratum") == stratum)
        n_take = min(per_stratum + (1 if index < sample_size % len(STRATA) else 0), pool.height)
        if n_take == 0:
            continue
        parts.append(
            pool.sample(n=n_take, seed=random_seed).with_columns(
                pl.lit(pool.height / n_take).alias("weight")
            )
        )
    if not parts:
        raise EmptyDatasetError("faixas de confiança sem posts")
    return pl.concat(parts)


def prepare_sample(paths: ProjectPaths, *, sample_size: int, random_seed: int) -> Path:
    """Gera o CSV para rotulagem humana e a chave de junção.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos do projeto.
    sample_size : int
        Número de posts (padrão do projeto: 500).
    random_seed : int
        Semente do sorteio.

    Returns
    -------
    Path
        Caminho do CSV a ser preenchido.

    Raises
    ------
    MissingEnvironmentVariableError
        Se ``DIAGNOSTICS_SAMPLE_SALT`` não estiver definida.
    """
    logger.info(
        "Preparando a amostra de validação humana (n=%d, semente=%d).", sample_size, random_seed
    )
    salt = resolve_sample_salt(SALT_ENV_VAR)
    table = assign_confidence_strata(build_rater_table(paths))
    logger.info(
        "Posts por faixa de confiança: %s",
        table["stratum"].value_counts().sort("stratum").to_dicts(),
    )
    sample = draw_stratified_sample(table, sample_size=sample_size, random_seed=random_seed)
    logger.info(
        "Amostra sorteada por faixa: %s",
        sample["stratum"].value_counts().sort("stratum").to_dicts(),
    )
    sample = sample.with_columns(
        pl.col("id")
        .map_elements(lambda value: pseudonymize_id(str(value), salt), return_dtype=pl.Utf8)
        .alias("sample_id")
    ).sort("sample_id")

    output_dir = paths.reports_tables_dir / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / SAMPLE_FILE_NAME
    write_csv(
        sample.select(
            "sample_id",
            pl.lit(CONCEPT_PLACEHOLDER).alias("concept"),
            pl.col("text_normalized").alias("text"),
            pl.lit("").alias(HUMAN_LABEL_COLUMN),
        ),
        csv_path,
    )
    key_dir = paths.data_silver_dir / OUTPUT_DIR_NAME
    key_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(sample.drop("text_normalized"), key_dir / KEY_FILE_NAME)
    logger.info("Chave sample_id -> id gravada em '%s' (fora do git).", key_dir / KEY_FILE_NAME)
    logger.info(
        "Amostra de %d post(s) gravada em '%s'. Preencha '%s' com negativo|neutro|positivo.",
        sample.height,
        csv_path,
        HUMAN_LABEL_COLUMN,
    )
    return csv_path


def _weighted_agreement(
    hits: np.ndarray, weights: np.ndarray, strata: np.ndarray, rng: np.random.Generator | None
) -> float:
    """Concordância ponderada; com ``rng``, reamostra dentro de cada faixa (bootstrap)."""
    if rng is None:
        return float(np.average(hits, weights=weights))
    indices = np.concatenate(
        [
            rng.choice(np.flatnonzero(strata == s), size=int((strata == s).sum()))
            for s in np.unique(strata)
        ]
    )
    return float(np.average(hits[indices], weights=weights[indices]))


def evaluate_raters(
    joined: pl.DataFrame, *, n_bootstrap: int, confidence_level: float, random_seed: int
) -> pl.DataFrame:
    """Compara cada rotulador ao rótulo humano.

    Parameters
    ----------
    joined : pl.DataFrame
        Chave com ``stratum``, ``weight``, colunas dos rotuladores e ``rotulo_humano``.
    n_bootstrap : int
        Reamostragens (padrão do projeto: 1.000).
    confidence_level : float
        Nível do IC (0,95).
    random_seed : int
        Semente do *bootstrap*.

    Returns
    -------
    pl.DataFrame
        Uma linha por rotulador: concordância ponderada com IC, concordância por faixa,
        Kappa de Cohen e Macro F1 (estes dois sem ponderação).

    Raises
    ------
    EmptyDatasetError
        Se ``joined`` estiver vazio.
    """
    if joined.is_empty():
        raise EmptyDatasetError("amostra rotulada por humanos")
    human = joined[HUMAN_LABEL_COLUMN].to_numpy()
    weights = joined["weight"].to_numpy()
    strata = joined["stratum"].to_numpy()
    alpha = (1 - confidence_level) / 2
    rows: list[dict[str, float | str | int]] = []
    for rater, column in RATER_COLUMNS.items():
        predicted = joined[column].to_numpy()
        hits = (predicted == human).astype(float)
        rng = np.random.default_rng(random_seed)
        boots = [_weighted_agreement(hits, weights, strata, rng) for _ in range(n_bootstrap)]
        row: dict[str, float | str | int] = {
            "rotulador": rater,
            "n": joined.height,
            "concordancia_ponderada": _weighted_agreement(hits, weights, strata, None),
            "ic_inferior": float(np.quantile(boots, alpha)),
            "ic_superior": float(np.quantile(boots, 1 - alpha)),
            "kappa_cohen": float(cohen_kappa_score(human, predicted)),
            "f1_macro": float(
                f1_score(
                    human,
                    predicted,
                    average="macro",
                    zero_division=0,  # pyright: ignore[reportArgumentType]
                )
            ),
        }
        for stratum in STRATA:
            mask = strata == stratum
            row[f"concordancia_{stratum}"] = (
                float(hits[mask].mean()) if mask.any() else float("nan")
            )
        rows.append(row)
    return pl.DataFrame(rows)


def run_mcnemar_comparisons(joined: pl.DataFrame) -> pl.DataFrame:
    """Compara os acertos de cada par de rotuladores contra o humano (McNemar exato).

    O teste usa apenas os posts em que os dois rotuladores divergem quanto a acertar. A
    amostra tem alocação igual entre faixas de confiança (sem ponderação), então o resultado
    vale para esta amostra, não diretamente para a população dos *gold sets*.

    Parameters
    ----------
    joined : pl.DataFrame
        Chave com as colunas dos rotuladores e ``rotulo_humano``.

    Returns
    -------
    pl.DataFrame
        Uma linha por par: acertos exclusivos de cada um, estatística e p-valor.

    Raises
    ------
    EmptyDatasetError
        Se ``joined`` estiver vazio.

    Examples
    --------
    >>> run_mcnemar_comparisons(joined)  # doctest: +SKIP
    """
    if joined.is_empty():
        raise EmptyDatasetError("amostra rotulada por humanos")
    human = joined[HUMAN_LABEL_COLUMN].to_list()
    names = list(RATER_COLUMNS)
    rows: list[dict[str, float | str | int]] = []
    for index, name_a in enumerate(names):
        for name_b in names[index + 1 :]:
            pred_a = joined[RATER_COLUMNS[name_a]].to_list()
            pred_b = joined[RATER_COLUMNS[name_b]].to_list()
            hit_a = [a == h for a, h in zip(pred_a, human, strict=True)]
            hit_b = [b == h for b, h in zip(pred_b, human, strict=True)]
            result = run_mcnemar_test(human, pred_a, pred_b, exact=True)
            rows.append(
                {
                    "rotulador_a": name_a,
                    "rotulador_b": name_b,
                    "so_a_acerta": sum(a and not b for a, b in zip(hit_a, hit_b, strict=True)),
                    "so_b_acerta": sum(b and not a for a, b in zip(hit_a, hit_b, strict=True)),
                    "estatistica": result["statistic"],
                    "p_valor": result["p_value"],
                }
            )
    return pl.DataFrame(rows)


def run_evaluation(
    paths: ProjectPaths, *, csv_path: Path | None, n_bootstrap: int, random_seed: int
) -> Path:
    """Lê o CSV preenchido, junta à chave e grava a tabela de resultados.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos do projeto.
    csv_path : Path | None
        CSV preenchido; ``None`` usa o caminho padrão de :func:`prepare_sample`.
    n_bootstrap : int
        Reamostragens do *bootstrap*.
    random_seed : int
        Semente.

    Returns
    -------
    Path
        Caminho da tabela de resultados.

    Raises
    ------
    DataValidationError
        Se houver ``sample_id`` sem correspondência na chave, ou a amostra rotulada tiver
        menos de dois rótulos distintos (sinal de preenchimento padrão).
    """
    output_dir = paths.reports_tables_dir / OUTPUT_DIR_NAME
    logger.info(
        "Avaliando a validação humana (bootstrap=%d, semente=%d)...", n_bootstrap, random_seed
    )
    labels = load_human_labels(csv_path or output_dir / SAMPLE_FILE_NAME).select(
        "sample_id", HUMAN_LABEL_COLUMN
    )
    logger.info("%d post(s) rotulado(s) pelo humano; lendo a chave de junção...", labels.height)
    key = pl.read_parquet(paths.data_silver_dir / OUTPUT_DIR_NAME / KEY_FILE_NAME)
    joined = labels.join(key, on="sample_id", how="inner")
    if joined.height != labels.height:
        raise DataValidationError(
            schema_name="ValidacaoHumana",
            detail=f"{labels.height - joined.height} sample_id sem correspondência na chave",
        )
    distribution = joined[HUMAN_LABEL_COLUMN].value_counts().sort("count", descending=True)
    logger.info("Distribuição dos rótulos humanos: %s", distribution.to_dicts())
    if distribution.height < 2:
        raise DataValidationError(
            schema_name="ValidacaoHumana",
            detail="apenas uma classe rotulada; confira se o preenchimento não foi padrão",
        )
    if joined.height < len(key):
        logger.warning("Apenas %d de %d posts rotulados.", joined.height, len(key))
    logger.info("Calculando concordância, Kappa e F1 por rotulador...")
    results = evaluate_raters(
        joined, n_bootstrap=n_bootstrap, confidence_level=0.95, random_seed=random_seed
    )
    results_path = output_dir / RESULTS_FILE_NAME
    write_csv(results, results_path)
    logger.info("Resultados da validação humana:\n%s", results)
    logger.info("Tabela de resultados gravada em '%s'.", results_path)
    logger.info("Comparando os rotuladores par a par com o teste de McNemar (acertos vs humano)...")
    mcnemar = run_mcnemar_comparisons(joined)
    mcnemar_path = output_dir / MCNEMAR_FILE_NAME
    write_csv(mcnemar, mcnemar_path)
    logger.info("McNemar (a vs b; acertos exclusivos e p-valor):\n%s", mcnemar)
    logger.info("Tabela do McNemar gravada em '%s'.", mcnemar_path)
    return results_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Interpreta os argumentos de linha de comando.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Argumentos (``None`` usa ``sys.argv``), by default None.

    Returns
    -------
    argparse.Namespace
        Argumentos validados.

    Examples
    --------
    >>> parse_arguments(["prepare", "--sample-size", "10"]).sample_size
    10
    """
    parser = argparse.ArgumentParser(
        prog="python -m diagnostics.human_validation", description=(__doc__ or "").split("\n")[0]
    )
    parser.add_argument("action", choices=("prepare", "evaluate"))
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--csv", type=Path, help="CSV preenchido (evaluate).")
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_BOOTSTRAP_ITERATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Ponto de entrada da CLI.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Argumentos (``None`` usa ``sys.argv``), by default None.

    Returns
    -------
    int
        Código de saída (0 = sucesso).

    Examples
    --------
    >>> main(["prepare"])  # doctest: +SKIP
    0
    """
    args = parse_arguments(argv)
    configure_environment_variables()
    configure_logging()
    paths = load_project_paths()
    if args.action == "prepare":
        prepare_sample(paths, sample_size=args.sample_size, random_seed=args.seed)
    else:
        run_evaluation(
            paths, csv_path=args.csv, n_bootstrap=args.n_bootstrap, random_seed=args.seed
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
