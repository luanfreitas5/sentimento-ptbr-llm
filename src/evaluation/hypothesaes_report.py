"""Consolidação de hipóteses HypotheSAEs em uma tabela comparável e legível.

Prepara a saída de :func:`hypothesaes.quickstart.generate_hypotheses` (ver
``pipelines.hypothesaes_analysis``) para leitura humana e para o gráfico de
barras divergente de ``visualization.hypothesaes``: limpa aspas/hifens
residuais deixados pelo LLM na interpretação gerada, classifica o sinal de
cada hipótese (baixa vs. alta confiança) e ordena por poder preditivo.
"""

import logging
from pathlib import Path

import polars as pl

from exceptions.data import EmptyDatasetError
from io_utils.csv import write_csv

logger = logging.getLogger(__name__)

_INTERPRETATION_COLUMN = "interpretation"
_SIGNAL_COLUMN = "signal"
LOW_CONFIDENCE_SIGNAL = "baixa confiança"
HIGH_CONFIDENCE_SIGNAL = "alta confiança"


def build_top_hypotheses_table(hypotheses: pl.DataFrame, *, target_column: str) -> pl.DataFrame:
    """Limpa e classifica as hipóteses geradas, ordenando por poder preditivo.

    Parameters
    ----------
    hypotheses : pl.DataFrame
        Saída de :func:`hypothesaes.quickstart.generate_hypotheses`
        (colunas ``neuron_idx``, ``target_column``, ``interpretation`` e,
        opcionalmente, ``f1_fidelity_score``).
    target_column : str
        Nome da coluna de poder preditivo (``f"target_{selection_method}"``).

    Returns
    -------
    pl.DataFrame
        ``hypotheses`` com ``interpretation`` limpa (sem aspas/hifens nas
        bordas) e uma nova coluna ``signal`` (:data:`LOW_CONFIDENCE_SIGNAL`
        quando ``target_column >= 0``, :data:`HIGH_CONFIDENCE_SIGNAL` caso
        contrário), ordenado por ``target_column`` decrescente.

    Raises
    ------
    EmptyDatasetError
        Se ``hypotheses`` estiver vazio.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame(
    ...     {
    ...         "neuron_idx": [1, 2],
    ...         "target_separation_score": [0.3, -0.1],
    ...         "interpretation": [' "usa ironia" ', "é muito curto"],
    ...     }
    ... )
    >>> build_top_hypotheses_table(df, target_column="target_separation_score")["signal"].to_list()
    ['baixa confiança', 'alta confiança']
    """
    if hypotheses.height == 0:
        raise EmptyDatasetError("hypotheses")

    return (
        hypotheses.drop_nulls(subset=[_INTERPRETATION_COLUMN])
        .with_columns(
            pl.col(_INTERPRETATION_COLUMN)
            .str.replace(r'^[\s"\'-]+', "")
            .str.replace(r'[\s"\']+$', "")
            .str.strip_chars()
        )
        .with_columns(
            pl.when(pl.col(target_column) >= 0)
            .then(pl.lit(LOW_CONFIDENCE_SIGNAL))
            .otherwise(pl.lit(HIGH_CONFIDENCE_SIGNAL))
            .alias(_SIGNAL_COLUMN)
        )
        .sort(target_column, descending=True)
    )


def save_top_hypotheses_table(table: pl.DataFrame, output_path: Path) -> None:
    """Salva a tabela de hipóteses em CSV, criando diretórios pais se necessário.

    Parameters
    ----------
    table : pl.DataFrame
        Tabela produzida por :func:`build_top_hypotheses_table`.
    output_path : Path
        Caminho do arquivo CSV de destino (ver
        ``config.paths.ProjectPaths.reports_interpretability_dir``).

    Returns
    -------
    None

    Examples
    --------
    >>> import polars as pl
    >>> save_top_hypotheses_table(
    ...     pl.DataFrame({"a": [1]}), Path("reports/interpretability/exemplo.csv")
    ... )  # doctest: +SKIP
    """
    write_csv(table, output_path)
    logger.info("Tabela de hipóteses salva em '%s' (%d linha(s)).", output_path, table.height)
