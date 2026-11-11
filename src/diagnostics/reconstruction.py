"""Erro de reconstrução do autoencoder (posts atípicos) e projeção UMAP do espaço latente.

Reaproveita o cache de embeddings do HypotheSAEs (mesmo modelo de
``configs/diagnostics.yaml``), treina o autoencoder de ``configs/model_params.yaml``
(``autoencoder``) sobre o corpus silver, calcula o erro de reconstrução (MSE) por post e
marca como atípicos os acima do percentil ``diagnostics.autoencoder_reconstruction_error``
de ``configs/labeling.yaml`` (95). Em seguida relaciona o erro à confiança e à classe do
rotulador primário e projeta o espaço latente em 2D com UMAP.

Uso::

    PYTHONPATH=src python -m diagnostics.reconstruction
    PYTHONPATH=src python -m diagnostics.reconstruction --umap-sample 5000

Posts atípicos são candidatos a ambiguidade, não erro comprovado.
"""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import mannwhitneyu

from config.environment import configure_environment_variables
from config.logging import configure_logging
from config.paths import CONFIGS_DIR, ProjectPaths, load_project_paths
from constants.defaults import DEFAULT_RANDOM_SEED
from diagnostics.sae_runner import build_embedding_cache_name, extract_local_embeddings
from diagnostics.settings import DEFAULT_DIAGNOSTICS_CONFIG_FILE, load_diagnostics_settings
from exceptions.data import EmptyDatasetError
from features.reduction import (
    compute_reconstruction_error,
    encode_with_autoencoder,
    train_autoencoder,
)
from io_utils.csv import write_csv
from io_utils.parquet import write_parquet
from io_utils.yaml import read_yaml
from visualization.embeddings import plot_embedding_scatter
from visualization.theme import save_figure

logger = logging.getLogger(__name__)

SILVER_FILE_NAME = "corpus_normalizado.parquet"
LABELED_FILE_NAME = "corpus_rotulado.parquet"
OUTPUT_DIR_NAME = "reconstrucao"
ANOMALY_PERCENTILE_DEFAULT = 95
LOW_CONFIDENCE_THRESHOLD = 0.5
TOP_ANOMALIES = 50


def flag_anomalies(errors: np.ndarray, percentile: float) -> tuple[np.ndarray, float]:
    """Marca os posts com erro de reconstrução acima do percentil.

    Parameters
    ----------
    errors : np.ndarray
        Erro de reconstrução por post.
    percentile : float
        Percentil de corte (0 a 100).

    Returns
    -------
    tuple[np.ndarray, float]
        Máscara booleana de anomalias e o limiar numérico.

    Raises
    ------
    EmptyDatasetError
        Se ``errors`` estiver vazio.

    Examples
    --------
    >>> mask, limiar = flag_anomalies(np.arange(100, dtype=float), 95)
    >>> int(mask.sum())
    5
    """
    if errors.size == 0:
        raise EmptyDatasetError("erros de reconstrução")
    threshold = float(np.percentile(errors, percentile))
    return errors > threshold, threshold


def summarize_error_by_group(scored: pl.DataFrame) -> pl.DataFrame:
    """Resume o erro e a taxa de anomalias por classe do primário e faixa de confiança.

    Parameters
    ----------
    scored : pl.DataFrame
        Com ``reconstruction_error``, ``is_anomaly``, ``lab_huggingface`` e ``conf_hf``
        (posts sem rótulo ficam fora).

    Returns
    -------
    pl.DataFrame
        Colunas ``grupo``, ``valor``, ``n``, ``erro_medio`` e ``taxa_anomalia``.

    Examples
    --------
    >>> summarize_error_by_group(scored)  # doctest: +SKIP
    """
    labeled = scored.drop_nulls(["lab_huggingface", "conf_hf"]).with_columns(
        pl.when(pl.col("conf_hf") < LOW_CONFIDENCE_THRESHOLD)
        .then(pl.lit("baixa (<0,5)"))
        .otherwise(pl.lit("alta (>=0,5)"))
        .alias("faixa_confianca")
    )
    parts = [
        labeled.group_by(column)
        .agg(
            pl.len().alias("n"),
            pl.col("reconstruction_error").mean().alias("erro_medio"),
            pl.col("is_anomaly").mean().alias("taxa_anomalia"),
        )
        .rename({column: "valor"})
        .with_columns(pl.lit(name).alias("grupo"))
        .select("grupo", "valor", "n", "erro_medio", "taxa_anomalia")
        for name, column in (
            ("classe_primario", "lab_huggingface"),
            ("confianca_primario", "faixa_confianca"),
        )
    ]
    return pl.concat(parts).sort("grupo", "valor")


def run_reconstruction(
    paths: ProjectPaths, *, config_path: Path, umap_sample: int, random_seed: int
) -> Path:
    """Executa o diagnóstico completo e grava tabelas e figuras.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos do projeto.
    config_path : Path
        ``configs/diagnostics.yaml`` (modelo de embedding).
    umap_sample : int
        Posts usados na projeção UMAP (subamostra; o UMAP completo é lento em CPU).
    random_seed : int
        Semente do autoencoder, da subamostra e do UMAP.

    Returns
    -------
    Path
        Diretório com os arquivos gerados.

    Raises
    ------
    EmptyDatasetError
        Se o corpus silver estiver vazio.
    """
    settings = load_diagnostics_settings(config_path)
    config_dir = CONFIGS_DIR
    ae_params = read_yaml(config_dir / "model_params.yaml")["autoencoder"]
    percentile = float(
        read_yaml(config_dir / "labeling.yaml")["diagnostics"][
            "autoencoder_reconstruction_error"
        ].get("anomaly_percentile", ANOMALY_PERCENTILE_DEFAULT)
    )

    corpus = pl.read_parquet(paths.data_silver_dir / SILVER_FILE_NAME).unique(
        subset=["text_normalized"], keep="first", maintain_order=True
    )
    if corpus.is_empty():
        raise EmptyDatasetError(SILVER_FILE_NAME)
    texts = corpus["text_normalized"].to_list()
    logger.info("Carregando embeddings de %d post(s) (cache do HypotheSAEs)...", len(texts))
    mapping = extract_local_embeddings(
        texts,
        model=settings.embedding.model_name,
        batch_size=settings.embedding.batch_size,
        cache_name=build_embedding_cache_name(settings.embedding.model_name, len(texts)),
    )
    embeddings = np.vstack([np.asarray(mapping[text], dtype=np.float32) for text in texts])

    logger.info("Treinando o autoencoder (latente=%d)...", ae_params["latent_dim"])
    artifacts = train_autoencoder(
        embeddings,
        input_dim=embeddings.shape[1],
        latent_dim=ae_params["latent_dim"],
        hidden_layers=ae_params["hidden_layers"],
        activation=ae_params["activation"],
        dropout=ae_params["dropout"],
        learning_rate=ae_params["learning_rate"],
        batch_size=ae_params["batch_size"],
        epochs=ae_params["epochs"],
        early_stopping_patience=ae_params["early_stopping_patience"],
        random_state=random_seed,
    )
    errors = compute_reconstruction_error(embeddings, artifacts)
    is_anomaly, threshold = flag_anomalies(errors, percentile)
    logger.info(
        "Limiar (p%.0f) = %.6f; %d post(s) atípico(s) de %d.",
        percentile,
        threshold,
        int(is_anomaly.sum()),
        len(errors),
    )

    labeled = pl.read_parquet(paths.data_gold_dir / LABELED_FILE_NAME).select(
        "id",
        pl.col("sentiment_label_huggingface").alias("lab_huggingface"),
        pl.col("confidence_score_huggingface").alias("conf_hf"),
    )
    scored = (
        corpus.select("id", "text_normalized")
        .with_columns(
            pl.Series("reconstruction_error", errors), pl.Series("is_anomaly", is_anomaly)
        )
        .join(labeled, on="id", how="left")
    )
    output_dir = paths.reports_tables_dir / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(scored.drop("text_normalized"), output_dir / "erro_reconstrucao.parquet")
    write_csv(
        scored.filter(pl.col("is_anomaly"))
        .sort("reconstruction_error", descending=True)
        .head(TOP_ANOMALIES)
        .select("id", "text_normalized", "reconstruction_error", "lab_huggingface", "conf_hf"),
        output_dir / "posts_mais_atipicos.csv",
    )
    summary = summarize_error_by_group(scored)
    write_csv(summary, output_dir / "resumo_erro_reconstrucao.csv")
    _log_confidence_test(scored)
    _plot_umap(
        artifacts, embeddings, scored, paths, sample_size=umap_sample, random_seed=random_seed
    )
    return output_dir


def _log_confidence_test(scored: pl.DataFrame) -> None:
    """Testa (Mann-Whitney) se o erro difere entre baixa e alta confiança do primário."""
    labeled = scored.drop_nulls("conf_hf")
    low = labeled.filter(pl.col("conf_hf") < LOW_CONFIDENCE_THRESHOLD)[
        "reconstruction_error"
    ].to_numpy()
    high = labeled.filter(pl.col("conf_hf") >= LOW_CONFIDENCE_THRESHOLD)[
        "reconstruction_error"
    ].to_numpy()
    if low.size == 0 or high.size == 0:
        return
    result = mannwhitneyu(low, high, alternative="two-sided")
    logger.info(
        "Erro de reconstrução, baixa vs alta confiança: mediana %.6f vs %.6f "
        "(Mann-Whitney p=%.3g).",
        float(np.median(low)),
        float(np.median(high)),
        float(result.pvalue),
    )


def _plot_umap(
    artifacts: object,
    embeddings: np.ndarray,
    scored: pl.DataFrame,
    paths: ProjectPaths,
    *,
    sample_size: int,
    random_seed: int,
) -> None:
    """Projeta o espaço latente em 2D (UMAP) numa subamostra rotulada e salva a figura."""
    import umap  # type: ignore[import-untyped]

    rows = np.flatnonzero(scored["lab_huggingface"].is_not_null().to_numpy())
    if rows.size == 0:
        logger.warning("Nenhum post rotulado; UMAP ignorado.")
        return
    rng = np.random.default_rng(random_seed)
    chosen = np.sort(rng.choice(rows, size=min(sample_size, rows.size), replace=False))
    latent = encode_with_autoencoder(embeddings[chosen], artifacts)  # type: ignore[arg-type]
    logger.info("Projetando %d ponto(s) com UMAP...", len(chosen))
    coordinates = umap.UMAP(n_components=2, random_state=random_seed).fit_transform(latent)
    labels = scored["lab_huggingface"].to_numpy()[chosen].tolist()
    figure = plot_embedding_scatter(
        coordinates, labels, title="Espaço latente do autoencoder (UMAP) por rótulo do primário"
    )
    save_figure(
        figure, "umap_espaco_latente", directory=paths.reports_figures_dir / OUTPUT_DIR_NAME
    )


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
    >>> parse_arguments(["--umap-sample", "100"]).umap_sample
    100
    """
    parser = argparse.ArgumentParser(
        prog="python -m diagnostics.reconstruction", description=__doc__.split("\n")[0]
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_DIAGNOSTICS_CONFIG_FILE)
    parser.add_argument("--umap-sample", type=int, default=10000)
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
    >>> main([])  # doctest: +SKIP
    0
    """
    args = parse_arguments(argv)
    configure_environment_variables()
    configure_logging()
    run_reconstruction(
        load_project_paths(),
        config_path=args.config,
        umap_sample=args.umap_sample,
        random_seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
