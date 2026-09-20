"""HypotheSAEs sobre a divergência e a incerteza entre os dois rotuladores.

Aplica a camada de diagnóstico (:mod:`diagnostics`) às bases
``tweets_data_huggingface`` e ``tweets_data_openai`` já unidas por ``id``
(:func:`evaluation.llm_comparison.build_comparison_frame`), para gerar
hipóteses em linguagem natural sobre **o que dificulta a rotulagem**:

* alvo ``disagreement``: o LLM do Hugging Face e a API OpenAI atribuem classes
  diferentes ao tweet (``lab_huggingface != lab_openai``);
* alvo ``uncertainty``: ``1 - agreement_score``, com ``agreement_score`` igual à
  **menor** das duas confianças (o tweet é incerto se ao menos um modelo hesita).

Entrada do HypotheSAEs: o texto normalizado (sem menções/URLs) e o alvo
derivado — nunca o texto original. Processamento: embeddings BERTimbau →
autoencoder esparso treinado uma vez e reutilizado entre alvos → gate de
sanidade (ridge sobre os embeddings; se o alvo não for previsível acima do
acaso, o alvo é abortado e **registrado como resultado**, não como falha) →
neurônios mais associados ao alvo → hipóteses interpretadas por LLM. As
hipóteses descrevem o comportamento dos modelos, nunca a verdade: sem gold
set, apenas a discordância e a incerteza são observáveis.

Como evidência, cada hipótese vem acompanhada dos tweets em que o neurônio
correspondente mais ativa (:func:`build_hypothesis_evidence`), com o rótulo e a
confiança de cada modelo, para validação posterior.

As importações pesadas (``torch``, cliente de LLM) ficam dentro das funções,
mantendo a camada opt-in.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from config.paths import ProjectPaths
from diagnostics.hypotheses import run_target_diagnostics
from diagnostics.sae_runner import DiscoveryData, prepare_discovery_data
from diagnostics.settings import (
    DEFAULT_DIAGNOSTICS_CONFIG_FILE,
    DiagnosticsSettings,
    load_diagnostics_settings,
)
from exceptions.data import DataValidationError
from exceptions.pipeline import SanityGateFailedError
from io_utils.csv import write_csv
from io_utils.parquet import write_parquet
from schemas.diagnostics import MODEL_LABEL_PREFIX, validate_diagnostic_corpus

logger = logging.getLogger(__name__)

_HF, _OA = "huggingface", "openai"
DIAGNOSTIC_CORPUS_FILE_NAME = "corpus_diagnostico.parquet"
_EVIDENCE_COLUMNS: tuple[str, ...] = (
    "target",
    "hypothesis",
    "neuron_idx",
    "rank",
    "activation",
    "id",
    "text_normalized",
    f"label_{_HF}",
    f"confidence_{_HF}",
    f"label_{_OA}",
    f"confidence_{_OA}",
)


@dataclass(frozen=True)
class HypothesisTargetOutcome:
    """Resultado do diagnóstico de um alvo (``disagreement`` ou ``uncertainty``).

    Attributes
    ----------
    target : str
        Nome do alvo.
    status : str
        ``"concluido"`` ou ``"gate_reprovado"`` (os embeddings não preveem o alvo acima
        do acaso — resultado científico válido, sem hipóteses).
    detail : str
        Motivo do gate (quando reprovado) ou resumo das hipóteses geradas.
    n_hypotheses : int
        Quantidade de hipóteses geradas.
    hypotheses_path : Path | None
        Tabela de hipóteses (parquet/CSV ao lado), ``None`` se o gate reprovou.
    evidence_path : Path | None
        CSV com os tweets mais ativados por hipótese, ``None`` se não houver hipóteses.
    """

    target: str
    status: str
    detail: str
    n_hypotheses: int
    hypotheses_path: Path | None
    evidence_path: Path | None


def build_diagnostic_corpus(comparison_frame: pl.DataFrame) -> pl.DataFrame:
    """Converte o DataFrame unificado no contrato de diagnóstico (``lab_*`` + ``agreement_score``).

    Parameters
    ----------
    comparison_frame : pl.DataFrame
        Saída de :func:`evaluation.llm_comparison.build_comparison_frame`.

    Returns
    -------
    pl.DataFrame
        Colunas ``id``, ``text_normalized``, ``agreement_score`` (menor das duas confianças),
        ``lab_huggingface`` e ``lab_openai``; apenas o texto normalizado é mantido (LGPD).

    Raises
    ------
    DataValidationError
        Se o resultado violar :class:`schemas.diagnostics.DiagnosticCorpusSchema`.

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
    ...     }
    ... )
    >>> build_diagnostic_corpus(frame)["agreement_score"].to_list()
    [0.7]
    """
    corpus = comparison_frame.select(
        "id",
        "text_normalized",
        pl.min_horizontal(f"confidence_{_HF}", f"confidence_{_OA}").alias("agreement_score"),
        pl.col(f"label_{_HF}").alias(f"{MODEL_LABEL_PREFIX}{_HF}"),
        pl.col(f"label_{_OA}").alias(f"{MODEL_LABEL_PREFIX}{_OA}"),
    )
    return validate_diagnostic_corpus(corpus)


def build_hypothesis_evidence(
    hypotheses: pl.DataFrame,
    discovery: DiscoveryData,
    comparison_frame: pl.DataFrame,
    *,
    target: str,
    top_tweets_per_hypothesis: int,
) -> pl.DataFrame:
    """Lista, por hipótese, os tweets em que o neurônio correspondente mais ativa.

    Parameters
    ----------
    hypotheses : pl.DataFrame
        Tabela de hipóteses (colunas ``hypothesis`` e ``neuron_idx``).
    discovery : DiscoveryData
        Corpus particionado, embeddings alinhados e SAE treinado.
    comparison_frame : pl.DataFrame
        Saída de :func:`evaluation.llm_comparison.build_comparison_frame` (rótulos/confianças).
    target : str
        Nome do alvo, gravado na coluna ``target``.
    top_tweets_per_hypothesis : int
        Quantidade de tweets por hipótese.

    Returns
    -------
    pl.DataFrame
        Colunas de ``_EVIDENCE_COLUMNS``: uma linha por (hipótese, tweet), ``rank`` 1 = maior
        ativação; apenas ativações positivas entram.

    Examples
    --------
    >>> build_hypothesis_evidence(  # doctest: +SKIP
    ...     hypotheses, discovery, frame, target="disagreement", top_tweets_per_hypothesis=10
    ... )
    """
    activations = np.asarray(discovery.sae.compute_activations(discovery.embeddings))
    row_ids = discovery.partitioned["id"].to_list()
    tweets = comparison_frame.select(
        "id",
        "text_normalized",
        f"label_{_HF}",
        f"confidence_{_HF}",
        f"label_{_OA}",
        f"confidence_{_OA}",
    )

    evidence_parts: list[pl.DataFrame] = []
    for row in hypotheses.select("hypothesis", "neuron_idx").iter_rows(named=True):
        neuron_activations = activations[:, int(row["neuron_idx"])]
        top_rows = np.argsort(-neuron_activations)[:top_tweets_per_hypothesis]
        top_rows = top_rows[neuron_activations[top_rows] > 0]
        if top_rows.size == 0:
            continue
        evidence_parts.append(
            pl.DataFrame(
                {
                    "id": [row_ids[index] for index in top_rows],
                    "activation": neuron_activations[top_rows].astype(float),
                    "rank": np.arange(1, top_rows.size + 1),
                }
            ).with_columns(
                pl.lit(target).alias("target"),
                pl.lit(row["hypothesis"]).alias("hypothesis"),
                pl.lit(int(row["neuron_idx"])).alias("neuron_idx"),
            )
        )
    if not evidence_parts:
        return pl.DataFrame(schema=dict.fromkeys(_EVIDENCE_COLUMNS, pl.String))
    return (
        pl.concat(evidence_parts)
        .join(tweets, on="id", how="left")
        .select(list(_EVIDENCE_COLUMNS))
        .sort(["hypothesis", "rank"])
    )


def _run_single_target(
    target: str,
    corpus: pl.DataFrame,
    discovery: DiscoveryData,
    comparison_frame: pl.DataFrame,
    settings: DiagnosticsSettings,
    paths: ProjectPaths,
    *,
    output_dir: Path,
    top_tweets_per_hypothesis: int,
    track: bool,
) -> HypothesisTargetOutcome:
    """Executa um alvo, converte a reprovação do gate em resultado e grava a evidência."""
    target_arguments: dict[str, Any] = (
        {"model_columns": [f"{MODEL_LABEL_PREFIX}{_HF}", f"{MODEL_LABEL_PREFIX}{_OA}"]}
        if target == "disagreement"
        else {}
    )
    try:
        result = run_target_diagnostics(
            target, corpus, discovery, settings, paths, track=track, **target_arguments
        )
    except SanityGateFailedError as exception:
        logger.warning("Alvo '%s' abortado pelo gate de sanidade: %s", target, exception.message)
        return HypothesisTargetOutcome(target, "gate_reprovado", exception.message, 0, None, None)

    hypotheses = result.hypotheses if result.hypotheses is not None else pl.DataFrame()
    evidence_path: Path | None = None
    if not hypotheses.is_empty():
        evidence = build_hypothesis_evidence(
            hypotheses,
            discovery,
            comparison_frame,
            target=target,
            top_tweets_per_hypothesis=top_tweets_per_hypothesis,
        )
        evidence_path = output_dir / f"hipoteses_{target}_evidencias.csv"
        write_csv(evidence, evidence_path)
    return HypothesisTargetOutcome(
        target,
        "concluido",
        f"{hypotheses.height} hipótese(s) gerada(s)",
        hypotheses.height,
        result.output_path,
        evidence_path,
    )


def run_disagreement_hypotheses(
    comparison_frame: pl.DataFrame,
    paths: ProjectPaths,
    *,
    output_dir: Path,
    targets: Sequence[str] = ("disagreement", "uncertainty"),
    top_tweets_per_hypothesis: int = 10,
    config_file: Path | None = None,
    random_seed: int | None = None,
    track: bool = True,
) -> list[HypothesisTargetOutcome]:
    """Gera hipóteses do HypotheSAEs sobre a divergência/incerteza entre os rotuladores.

    O SAE é treinado uma única vez (:func:`diagnostics.sae_runner.prepare_discovery_data`) e
    reutilizado entre os alvos; descoberta e validação usam partições disjuntas com semente fixa
    (``configs/diagnostics.yaml``). O corpus de diagnóstico (texto normalizado) é gravado em
    ``output_dir`` para auditoria.

    Parameters
    ----------
    comparison_frame : pl.DataFrame
        Saída de :func:`evaluation.llm_comparison.build_comparison_frame`.
    paths : ProjectPaths
        Caminhos resolvidos do projeto.
    output_dir : Path
        Diretório das evidências (CSV) e do corpus de diagnóstico.
    targets : Sequence[str], optional
        Alvos a diagnosticar (``disagreement`` e/ou ``uncertainty``), by default os dois.
    top_tweets_per_hypothesis : int, optional
        Tweets de evidência por hipótese, by default 10.
    config_file : Path | None, optional
        ``configs/diagnostics.yaml`` alternativo, by default None.
    random_seed : int | None, optional
        Sobrescreve a semente do YAML, by default None.
    track : bool, optional
        Se registra cada alvo no MLflow, by default True.

    Returns
    -------
    list[HypothesisTargetOutcome]
        Um resultado por alvo, na ordem de ``targets``.

    Raises
    ------
    DataValidationError
        Se algum alvo não for ``disagreement`` ou ``uncertainty``, ou o corpus violar o contrato.
    ModelError
        Se as dependências pesadas (torch, cliente de LLM) não estiverem instaladas.

    Examples
    --------
    >>> run_disagreement_hypotheses(
    ...     frame, paths, output_dir=Path("reports/tables/x")
    ... )  # doctest: +SKIP
    """
    unknown = [target for target in targets if target not in ("disagreement", "uncertainty")]
    if unknown:
        raise DataValidationError(
            schema_name="comparative_evaluation_hypotheses",
            detail=f"alvos inválidos {unknown}; use 'disagreement' e/ou 'uncertainty'",
        )
    settings = load_diagnostics_settings(config_file or DEFAULT_DIAGNOSTICS_CONFIG_FILE)
    if random_seed is not None:
        settings = settings.model_copy(update={"random_seed": random_seed})

    corpus = build_diagnostic_corpus(comparison_frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(corpus, output_dir / DIAGNOSTIC_CORPUS_FILE_NAME)
    discovery = prepare_discovery_data(corpus, settings, paths)

    outcomes = [
        _run_single_target(
            target,
            corpus,
            discovery,
            comparison_frame,
            settings,
            paths,
            output_dir=output_dir,
            top_tweets_per_hypothesis=top_tweets_per_hypothesis,
            track=track,
        )
        for target in targets
    ]
    logger.info(
        "HypotheSAEs concluído: %s.",
        "; ".join(f"{outcome.target}={outcome.status}" for outcome in outcomes),
    )
    return outcomes
