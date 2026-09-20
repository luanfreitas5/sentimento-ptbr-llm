"""Ponto de entrada único do pipeline de análise de sentimentos pt-BR.

Composição raiz do projeto (ver CLAUDE.md, "Clean Architecture"): monta a
infraestrutura de configuração (``src/config/``) e resolve, para cada
estágio registrado em ``src/pipelines/workflow.py``, os argumentos que o
próprio estágio não pode assumir sozinho por design (ver docstrings de
``src/pipelines/ingestion.py``/``labeling.py``, que injetam a coleta de
dados e os rotuladores por parâmetro para permanecerem testáveis sem rede
nem credenciais). Cada etapa lê seus dados de entrada dos artefatos já
gravados em disco pela etapa anterior (``configs/paths.yaml``).

Uso
---
    uv run python src/main.py --stage <nome_da_etapa>
    uv run python src/main.py --stage all

Ver ``make help`` para os alvos pré-configurados (um por etapa) e
``configs/config.yaml -> stages`` para a lista/ordem canônica de estágios.

Lacunas conhecidas (ver ``_build_ingestion_stage_kwargs``): a etapa ``ingestion``
exige um adaptador de scraping que este projeto não implementa como módulo
próprio (``src/data/collector.py``); por isso, recebe a função de coleta via
caminho pontilhado (``--scrape-func``), informado pelo operador da execução.
"""

from __future__ import annotations

import argparse
import importlib
import logging
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from config.constants import CONFIG_FILE_NAMES
from config.environment import configure_environment_variables, configure_reproducibility
from config.logging import configure_logging
from config.paths import CONFIGS_DIR, ProjectPaths, load_project_paths
from config.settings import GeneralConfig, Settings, create_settings, load_general_config
from data.loader import load_labeled_corpus, load_training_example_dataset, read_dataset_file
from diagnostics.targets import TARGET_NAMES
from exceptions.configuration import InvalidConfigurationError
from features.lexical import pivot_tfidf_features_to_wide
from hypothesaes.llm_api import LLM_PROVIDER_NAMES
from hypothesaes.utils import load_prompt_template
from io_utils.yaml import read_yaml
from labeling.huggingface import open_huggingface_classifier
from labeling.openai_labeler import create_openai_batch_classifier
from pipelines.diagnostics_analysis import DIAGNOSTICS_GOLD_CHOICES, DIAGNOSTICS_STEPS
from pipelines.labeling import LABEL_SOURCE_NAMES, LabelingSource
from pipelines.training_classical import DEFAULT_CLASSICAL_MODEL_NAMES
from pipelines.training_deep_learning import DEFAULT_DEEP_LEARNING_MODEL_NAMES
from pipelines.workflow import STAGE_REGISTRY, run_pipeline_stage

logger = logging.getLogger(__name__)

_ALL_STAGES_OPTION = "all"
_STAGE_CHOICES: tuple[str, ...] = (*STAGE_REGISTRY, _ALL_STAGES_OPTION)
_ALL_SOURCES_OPTION = "all"
_LOG_LEVEL_CHOICES: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

# Espelha `pipelines.features._TFIDF_FEATURES_FILE_NAME` (constante privada
# do módulo), já que apenas o caminho dos três conjuntos particionados é
# exposto publicamente via `pipelines.features.FeatureArtifacts`.
_TFIDF_FEATURES_FILE_NAME = "tfidf_features.parquet"

# Mapeia cada modelo de deep learning/Transformer para a seção/subseção
# correspondente em `configs/model_params.yaml` (os nomes não coincidem:
# "lstm"/"cnn" ficam em `deep_learning.recurrent`/`deep_learning.convolutional`).
_DEEP_LEARNING_MODEL_PARAM_KEYS: dict[str, tuple[str, str]] = {
    "lstm": ("deep_learning", "recurrent"),
    "cnn": ("deep_learning", "convolutional"),
    "bertimbau": ("transformers", "bertimbau"),
    "roberta": ("transformers", "roberta"),
    "distilbert": ("transformers", "distilbert"),
}


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Constrói e interpreta os argumentos de linha de comando do pipeline.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Argumentos a interpretar, by default None (usa ``sys.argv[1:]``).

    Returns
    -------
    argparse.Namespace
        Argumentos interpretados.

    Examples
    --------
    >>> parse_arguments(["--stage", "preprocessing"]).stage
    'preprocessing'
    """
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Orquestra as etapas do pipeline de análise de sentimentos pt-BR.",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=_STAGE_CHOICES,
        help=(
            "Nome da etapa a executar (ver `configs/config.yaml -> stages`), "
            "ou 'all' para o workflow completo, na ordem configurada."
        ),
    )
    parser.add_argument(
        "--log-level",
        default=None,
        type=str.upper,
        choices=_LOG_LEVEL_CHOICES,
        help="Sobrescreve o nível de log de `configs/logging.yaml`.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Sobrescreve a semente de reprodutibilidade de `configs/config.yaml`.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help=(
            "Número máximo de threads/processos paralelos (etapas "
            "`ingestion`/`preprocessing`/`hypothesaes_analysis`; na etapa `labeling`, "
            "sobrescreve `configs/labeling.yaml -> openai.n_workers`)."
        ),
    )
    parser.add_argument(
        "--model-names",
        default=None,
        metavar="MODELO1,MODELO2,...",
        help=(
            "Lista de modelos separados por vírgula (etapas `training_classical`/"
            "`training_deep_learning`); usa o padrão da etapa quando omitido."
        ),
    )
    parser.add_argument(
        "--track-with-mlflow",
        action="store_true",
        help="Habilita o rastreamento MLflow nas etapas de treino.",
    )
    parser.add_argument(
        "--label-source",
        default=_ALL_SOURCES_OPTION,
        choices=[*LABEL_SOURCE_NAMES, _ALL_SOURCES_OPTION],
        help=(
            "Fonte(s) de rotulagem da etapa `labeling`: `huggingface` (LLM local), "
            "`openai` (API) ou `all` (as duas bases, em sequência)."
        ),
    )
    parser.add_argument(
        "--scrape-func",
        default=None,
        metavar="MODULO:FUNCAO",
        help=(
            "Caminho pontilhado para a função de coleta por consulta (etapa "
            "`ingestion`), ex.: 'meu_pacote.coleta:coletar_por_termo'."
        ),
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        default=None,
        metavar="CONSULTA",
        help="Consultas de coleta (etapa `ingestion`), ex.: --queries termo1 termo2.",
    )
    parser.add_argument(
        "--skip-hypotheses",
        action="store_true",
        help=(
            "Não executa o HypotheSAEs ao final da etapa `comparative_evaluation` "
            "(sobrescreve `configs/evaluation.yaml -> llm_comparison.hypotheses.enabled` "
            "apenas para desligar)."
        ),
    )
    parser.add_argument(
        "--hypothesaes-evaluate",
        action="store_true",
        help=(
            "Habilita a avaliação das hipóteses num holdout via LLM (etapa "
            "`hypothesaes_analysis`); sobrescreve `configs/hypothesaes.yaml -> "
            "evaluation.enabled` apenas para ligar (nunca desliga)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Apenas estima nº de chamadas de LLM, tokens e custo, sem rede, embeddings, SAE "
            "ou MLflow (etapa `diagnostics`)."
        ),
    )
    parser.add_argument(
        "--diagnostics-step",
        default="hypotheses",
        choices=DIAGNOSTICS_STEPS,
        help=(
            "Passo da etapa `diagnostics`: `hypotheses` (gate de sanidade + hipóteses), "
            "`validation` (holdout com Bonferroni + amostra para rotulagem) ou "
            "`comparison` (prompts v1 vs v2 no gold)."
        ),
    )
    parser.add_argument(
        "--diagnostics-target",
        default="disagreement",
        choices=TARGET_NAMES,
        help="Alvo de diagnóstico (etapa `diagnostics`, passos `hypotheses`/`validation`).",
    )
    parser.add_argument(
        "--diagnostics-model-column",
        default=None,
        metavar="LAB_MODELO",
        help=(
            "Coluna `lab_<modelo>` dos alvos `pseudo_label`/`gold_error`; por padrão, a de "
            "`configs/diagnostics.yaml -> targets`."
        ),
    )
    parser.add_argument(
        "--diagnostics-label",
        default=None,
        metavar="CLASSE",
        help="Classe positiva do alvo `pseudo_label` (one-vs-rest); por padrão, a do YAML.",
    )
    parser.add_argument(
        "--diagnostics-corpus",
        type=Path,
        default=None,
        help=(
            "Parquet já no contrato de diagnóstico (obrigatório para `gold_error`: predições "
            "sobre o gold); por padrão adapta o corpus rotulado."
        ),
    )
    parser.add_argument(
        "--diagnostics-gold",
        default="tweetsentbr",
        choices=DIAGNOSTICS_GOLD_CHOICES,
        help="Gold set do passo `comparison` (etapa `diagnostics`).",
    )
    return parser.parse_args(argv)


def _import_callable_from_dotted_path(dotted_path: str) -> Callable[..., Any]:
    """Importa um objeto chamável a partir de um caminho ``modulo.submodulo:funcao``.

    Parameters
    ----------
    dotted_path : str
        Caminho no formato ``modulo.submodulo:atributo``.

    Returns
    -------
    Callable[..., Any]
        Objeto chamável importado.

    Raises
    ------
    InvalidConfigurationError
        Se ``dotted_path`` não seguir o formato esperado.

    Examples
    --------
    >>> _import_callable_from_dotted_path("json:dumps")  # doctest: +SKIP
    """
    module_path, separator, attribute_name = dotted_path.partition(":")
    if not separator or not attribute_name:
        raise InvalidConfigurationError(
            f"caminho pontilhado inválido (esperado 'modulo:funcao'): '{dotted_path}'"
        )
    module = importlib.import_module(module_path)
    return getattr(module, attribute_name)


def _parse_model_names(raw_value: str | None) -> tuple[str, ...] | None:
    """Converte a lista de modelos separada por vírgula de ``--model-names``.

    Parameters
    ----------
    raw_value : str | None
        Valor bruto de ``--model-names`` (ex.: ``"svm,naive_bayes"``).

    Returns
    -------
    tuple[str, ...] | None
        Nomes de modelo, ou ``None`` se ``raw_value`` for ``None``.

    Examples
    --------
    >>> _parse_model_names("svm, naive_bayes")
    ('svm', 'naive_bayes')
    """
    if raw_value is None:
        return None
    return tuple(name.strip() for name in raw_value.split(",") if name.strip())


def _load_classical_training_arrays(paths: ProjectPaths) -> tuple[np.ndarray, list[str]]:
    """Carrega a matriz TF-IDF de treino (etapa ``features``) como array denso.

    Converte o formato longo produzido por
    :func:`features.lexical.compute_tfidf_features` para uma matriz densa
    (:func:`features.lexical.pivot_tfidf_features_to_wide`) e alinha cada
    linha ao rótulo correspondente em ``paths.training_corpus_file`` pelo
    ``id`` (documentos sem nenhum peso TF-IDF não nulo não aparecem no
    formato longo, então o corpus de treino é restrito aos ``id`` presentes).

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto (``configs/paths.yaml``).

    Returns
    -------
    tuple[np.ndarray, list[str]]
        Matriz de features de treino (``X_train``) e rótulos correspondentes
        (``y_train``), na mesma ordem.

    Examples
    --------
    >>> _load_classical_training_arrays(paths)  # doctest: +SKIP
    """
    tfidf_features_path = paths.data_processed_dir / _TFIDF_FEATURES_FILE_NAME
    tfidf_wide = pivot_tfidf_features_to_wide(read_dataset_file(tfidf_features_path)).sort("id")
    training_corpus = (
        load_training_example_dataset(paths.training_corpus_file)
        .filter(pl.col("id").is_in(tfidf_wide["id"]))
        .sort("id")
    )
    feature_columns = [column for column in tfidf_wide.columns if column != "id"]
    X_train = tfidf_wide.select(feature_columns).to_numpy()  # noqa: N806
    y_train = training_corpus["sentiment_label"].to_list()
    return X_train, y_train


def _build_ingestion_stage_kwargs(
    paths: ProjectPaths, general_config: GeneralConfig, settings: Settings, args: argparse.Namespace
) -> dict[str, Any]:
    """Monta os argumentos de :func:`pipelines.ingestion.run_ingestion_stage`.

    A coleta em si (chamadas de rede) é responsabilidade do chamador por
    design (ver ``src/data/downloader.py``), então ``--scrape-func`` e
    ``--queries`` são obrigatórios nesta etapa.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto.
    general_config : GeneralConfig
        Configuração geral validada (``configs/config.yaml``), não utilizada
        diretamente nesta etapa.
    settings : Settings
        Configurações sensíveis ao ambiente, não utilizadas diretamente
        nesta etapa.
    args : argparse.Namespace
        Argumentos de linha de comando.

    Returns
    -------
    dict[str, Any]
        Argumentos nomeados para :func:`pipelines.ingestion.run_ingestion_stage`.

    Raises
    ------
    InvalidConfigurationError
        Se ``--scrape-func`` ou ``--queries`` não forem informados.
    """
    del general_config, settings
    if args.scrape_func is None or not args.queries:
        raise InvalidConfigurationError(
            "a etapa 'ingestion' exige '--scrape-func' e '--queries': o projeto não "
            "acopla a coleta a um provedor específico (ver src/data/downloader.py); "
            "informe uma função de coleta própria (ex.: um adaptador twscrape)."
        )
    return {
        "paths": paths,
        "scrape_func": _import_callable_from_dotted_path(args.scrape_func),
        "queries": args.queries,
        "max_workers": args.max_workers,
    }


def _build_preprocessing_stage_kwargs(
    paths: ProjectPaths, general_config: GeneralConfig, settings: Settings, args: argparse.Namespace
) -> dict[str, Any]:
    """Monta os argumentos de :func:`pipelines.preprocessing.run_preprocessing_stage`.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto.
    general_config : GeneralConfig
        Configuração geral validada, não utilizada diretamente nesta etapa.
    settings : Settings
        Configurações sensíveis ao ambiente, não utilizadas diretamente
        nesta etapa.
    args : argparse.Namespace
        Argumentos de linha de comando (``--max-workers``).

    Returns
    -------
    dict[str, Any]
        Argumentos nomeados para :func:`pipelines.preprocessing.run_preprocessing_stage`.
    """
    del general_config, settings
    return {"paths": paths, "max_workers": args.max_workers}


def _resolve_active_llm_provider(llm_config: dict[str, Any]) -> str:
    """Valida e retorna ``configs/llm.yaml -> active_provider``.

    Provedor usado pelo HypotheSAEs (etapa ``hypothesaes_analysis``); a
    rotulagem dos tweets (etapa ``labeling``) tem configuração própria em
    ``configs/labeling.yaml``.

    Parameters
    ----------
    llm_config : dict[str, Any]
        Conteúdo de ``configs/llm.yaml``.

    Returns
    -------
    str
        ``"openai"`` ou ``"ollama"``.

    Raises
    ------
    InvalidConfigurationError
        Se ``active_provider`` não for um de :data:`hypothesaes.llm_api.LLM_PROVIDER_NAMES`.
    """
    provider = llm_config["active_provider"]
    if provider not in LLM_PROVIDER_NAMES:
        raise InvalidConfigurationError(
            f"configs/llm.yaml -> active_provider inválido: '{provider}'. "
            f"Valores aceitos: {list(LLM_PROVIDER_NAMES)}"
        )
    return provider


def _build_labeling_sources(
    labeling_config: dict[str, Any], settings: Settings, args: argparse.Namespace
) -> list[LabelingSource]:
    """Monta as fontes de rotulagem pedidas em ``--label-source``.

    O modelo do Hugging Face só é carregado quando a fonte é executada (e a GPU é liberada ao
    fim dela); o cliente OpenAI usa a chave de ``OPENAI_KEY`` (``.env``), lida sob demanda.

    Parameters
    ----------
    labeling_config : dict[str, Any]
        Conteúdo de ``configs/labeling.yaml``.
    settings : Settings
        Configurações sensíveis ao ambiente (``huggingface_token``).
    args : argparse.Namespace
        Argumentos de linha de comando (``--label-source``, ``--max-workers``).

    Returns
    -------
    list[LabelingSource]
        Fontes a executar, na ordem ``huggingface`` -> ``openai``.
    """
    requested = (
        LABEL_SOURCE_NAMES if args.label_source == _ALL_SOURCES_OPTION else (args.label_source,)
    )
    prompt_name = labeling_config["prompt_name"]
    prompt_template = load_prompt_template(prompt_name)
    sources: list[LabelingSource] = []

    if "huggingface" in requested:
        hf_config = labeling_config["huggingface"]
        sources.append(
            LabelingSource(
                name="huggingface",
                model_name=f"{hf_config['model']}@{hf_config['revision']}",
                prompt_name=prompt_name,
                prompt_template=prompt_template,
                temperature=hf_config["temperature"],
                batch_size=hf_config["batch_size"],
                open_classifier=partial(
                    open_huggingface_classifier,
                    prompt_template,
                    model_name=hf_config["model"],
                    device=hf_config["device"],
                    dtype=hf_config["dtype"],
                    load_in_4bit=hf_config["load_in_4bit"],
                    token=settings.huggingface_token,
                    revision=hf_config["revision"],
                    max_new_tokens=hf_config["max_new_tokens"],
                    max_input_tokens=hf_config["max_input_tokens"],
                    max_retries=hf_config["max_retries"],
                    retry_temperature=hf_config["retry_temperature"],
                ),
            )
        )

    if "openai" in requested:
        oa_config = labeling_config["openai"]
        classifier = create_openai_batch_classifier(
            prompt_template,
            model=oa_config["model"],
            temperature=oa_config["temperature"],
            max_retries=oa_config["max_retries"],
            n_workers=args.max_workers or oa_config["n_workers"],
            request_interval_seconds=oa_config["request_interval_seconds"],
            request_timeout_seconds=oa_config["request_timeout_seconds"],
        )
        sources.append(
            LabelingSource(
                name="openai",
                model_name=oa_config["model"],
                prompt_name=prompt_name,
                prompt_template=prompt_template,
                temperature=oa_config["temperature"],
                batch_size=oa_config["batch_size"],
                open_classifier=lambda: nullcontext(classifier),
            )
        )
    return sources


def _build_labeling_stage_kwargs(
    paths: ProjectPaths, general_config: GeneralConfig, settings: Settings, args: argparse.Namespace
) -> dict[str, Any]:
    """Monta os argumentos de :func:`pipelines.labeling.run_labeling_stage`.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto.
    general_config : GeneralConfig
        Configuração geral validada, não utilizada diretamente nesta etapa.
    settings : Settings
        Configurações sensíveis ao ambiente (``huggingface_token``).
    args : argparse.Namespace
        Argumentos de linha de comando (``--label-source``, ``--max-workers``).

    Returns
    -------
    dict[str, Any]
        Argumentos nomeados para :func:`pipelines.labeling.run_labeling_stage`.
    """
    del general_config
    labeling_config = read_yaml(CONFIGS_DIR / CONFIG_FILE_NAMES["labeling"])
    return {
        "paths": paths,
        "sources": _build_labeling_sources(labeling_config, settings, args),
        "downstream_source": labeling_config["downstream_source"],
        "human_validation_sample_size": labeling_config["human_validation"]["sample_size"],
        "low_confidence_threshold": labeling_config["confidence"]["low_confidence_threshold"],
        "minimum_kappa": labeling_config["validation"]["minimum_agreement"],
    }


def _build_features_stage_kwargs(
    paths: ProjectPaths, general_config: GeneralConfig, settings: Settings, args: argparse.Namespace
) -> dict[str, Any]:
    """Monta os argumentos de :func:`pipelines.features.run_features_stage`.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto.
    general_config : GeneralConfig
        Configuração geral validada (``configs/config.yaml -> data_split``).
    settings : Settings
        Configurações sensíveis ao ambiente, não utilizadas diretamente
        nesta etapa.
    args : argparse.Namespace
        Argumentos de linha de comando (``--random-seed``).

    Returns
    -------
    dict[str, Any]
        Argumentos nomeados para :func:`pipelines.features.run_features_stage`.
    """
    del settings
    random_seed = (
        args.random_seed if args.random_seed is not None else general_config.data_split.random_state
    )
    return {
        "paths": paths,
        "test_size": general_config.data_split.test_size,
        "validation_size": general_config.data_split.validation_size,
        "random_seed": random_seed,
    }


def _build_training_classical_stage_kwargs(
    paths: ProjectPaths, general_config: GeneralConfig, settings: Settings, args: argparse.Namespace
) -> dict[str, Any]:
    """Monta os argumentos de :func:`pipelines.training_classical.run_training_classical_stage`.

    ``X_val``/``y_val`` são sempre ``None``: a etapa ``features`` só calcula
    TF-IDF para o conjunto de treino (ver ``pipelines.features.run_features_stage``),
    então nenhuma matriz de validação está disponível em disco.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto.
    general_config : GeneralConfig
        Configuração geral validada, não utilizada diretamente nesta etapa.
    settings : Settings
        Configurações sensíveis ao ambiente, não utilizadas diretamente
        nesta etapa.
    args : argparse.Namespace
        Argumentos de linha de comando (``--model-names``, ``--track-with-mlflow``).

    Returns
    -------
    dict[str, Any]
        Argumentos nomeados para
        :func:`pipelines.training_classical.run_training_classical_stage`.
    """
    del general_config, settings
    X_train, y_train = _load_classical_training_arrays(paths)  # noqa: N806
    model_params = read_yaml(CONFIGS_DIR / CONFIG_FILE_NAMES["model_params"])["classical"]
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": None,
        "y_val": None,
        "model_names": _parse_model_names(args.model_names) or DEFAULT_CLASSICAL_MODEL_NAMES,
        "model_params": model_params,
        "checkpoints_dir": paths.models_checkpoints_dir,
        "track_with_mlflow": args.track_with_mlflow,
    }


def _build_training_deep_learning_stage_kwargs(
    paths: ProjectPaths, general_config: GeneralConfig, settings: Settings, args: argparse.Namespace
) -> dict[str, Any]:
    """Monta os argumentos de
    :func:`pipelines.training_deep_learning.run_training_deep_learning_stage`.

    ``X_train`` é o texto (já tokenizado/normalizado) do conjunto de treino:
    cada classificador (LSTM/CNN/Transformer) é responsável por sua própria
    tensorização/tokenização específica (ver ``src/models/``). Assim como em
    :func:`_build_training_classical_stage_kwargs`, ``X_val``/``y_val`` são
    ``None`` por falta de um conjunto de validação já processado em disco.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto.
    general_config : GeneralConfig
        Configuração geral validada, não utilizada diretamente nesta etapa.
    settings : Settings
        Configurações sensíveis ao ambiente, não utilizadas diretamente
        nesta etapa.
    args : argparse.Namespace
        Argumentos de linha de comando (``--model-names``, ``--track-with-mlflow``).

    Returns
    -------
    dict[str, Any]
        Argumentos nomeados para
        :func:`pipelines.training_deep_learning.run_training_deep_learning_stage`.
    """
    del general_config, settings
    training_corpus = load_training_example_dataset(paths.training_corpus_file)
    model_params_by_section = read_yaml(CONFIGS_DIR / CONFIG_FILE_NAMES["model_params"])
    model_params = {
        model_name: model_params_by_section[section][subsection]
        for model_name, (section, subsection) in _DEEP_LEARNING_MODEL_PARAM_KEYS.items()
    }
    return {
        "X_train": training_corpus["text"].to_list(),
        "y_train": training_corpus["sentiment_label"].to_list(),
        "X_val": None,
        "y_val": None,
        "model_names": _parse_model_names(args.model_names) or DEFAULT_DEEP_LEARNING_MODEL_NAMES,
        "model_params": model_params,
        "checkpoints_dir": paths.models_checkpoints_dir,
        "track_with_mlflow": args.track_with_mlflow,
    }


def _build_comparative_evaluation_stage_kwargs(
    paths: ProjectPaths, general_config: GeneralConfig, settings: Settings, args: argparse.Namespace
) -> dict[str, Any]:
    """Monta os argumentos de
    :func:`pipelines.comparative_evaluation.run_comparative_evaluation_stage`.

    A etapa lê as bases ``tweets_data_huggingface``/``tweets_data_openai`` direto do disco e
    não exige nenhum argumento manual: limiares, bootstrap e HypotheSAEs vêm de
    ``configs/evaluation.yaml -> llm_comparison``.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto.
    general_config : GeneralConfig
        Configuração geral validada, não utilizada diretamente nesta etapa.
    settings : Settings
        Configurações sensíveis ao ambiente, não utilizadas diretamente nesta etapa.
    args : argparse.Namespace
        Argumentos de linha de comando (``--random-seed``, ``--skip-hypotheses``).

    Returns
    -------
    dict[str, Any]
        Argumentos nomeados para
        :func:`pipelines.comparative_evaluation.run_comparative_evaluation_stage`.
    """
    del general_config, settings
    config = read_yaml(CONFIGS_DIR / CONFIG_FILE_NAMES["evaluation"])["llm_comparison"]
    hypotheses_config = config["hypotheses"]
    return {
        "paths": paths,
        "output_subdir": config["output_subdir"],
        "random_seed": args.random_seed if args.random_seed is not None else config["random_seed"],
        "n_bootstrap": config["n_bootstrap"],
        "confidence_level": config["confidence_level"],
        "top_n_divergences": config["top_n_divergences"],
        "high_confidence_threshold": config["high_confidence_threshold"],
        "low_confidence_threshold": config["low_confidence_threshold"],
        "n_length_bins": config["n_length_bins"],
        "examples_per_transition": config["examples_per_transition"],
        "run_hypotheses": hypotheses_config["enabled"] and not args.skip_hypotheses,
        "hypotheses_targets": tuple(hypotheses_config["targets"]),
        "top_tweets_per_hypothesis": hypotheses_config["top_tweets_per_hypothesis"],
        "track_with_mlflow": True,
    }


def _build_hypothesaes_analysis_stage_kwargs(
    paths: ProjectPaths, general_config: GeneralConfig, settings: Settings, args: argparse.Namespace
) -> dict[str, Any]:
    """Monta os argumentos de :func:`pipelines.hypothesaes_analysis.run_hypothesaes_analysis_stage`.

    O endpoint LLM (OpenAI-compatível, modelo ``UnB-Llama-3.3-70B-Instruct``
    por padrão) é resolvido exclusivamente via ``OPENAI_BASE_URL``/
    ``OPENAI_KEY`` (``.env`` — ver ``.env.example``), lidas em tempo de
    chamada por ``hypothesaes.llm_api.create_client``: este é o único
    estágio do projeto (além da re-rotulagem via LLM da etapa ``labeling``)
    que depende desse cliente.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto.
    general_config : GeneralConfig
        Configuração geral validada, não utilizada diretamente nesta etapa.
    settings : Settings
        Configurações sensíveis ao ambiente, não utilizadas diretamente
        nesta etapa. O provedor de LLM (``configs/llm.yaml ->
        active_provider``) escolhe entre o endpoint OpenAI-compatível
        (``OPENAI_BASE_URL``/``OPENAI_KEY``, ``.env``) e o servidor Ollama
        local (``configs/llm.yaml -> backends.ollama.base_url``) — nenhum
        dos dois vem de ``settings.ollama_base_url``.
    args : argparse.Namespace
        Argumentos de linha de comando (``--hypothesaes-evaluate``,
        ``--max-workers``, ``--random-seed``).

    Returns
    -------
    dict[str, Any]
        Argumentos nomeados para
        :func:`pipelines.hypothesaes_analysis.run_hypothesaes_analysis_stage`.
    """
    del general_config, settings
    hypothesaes_config = read_yaml(CONFIGS_DIR / CONFIG_FILE_NAMES["hypothesaes"])
    task_specific_instructions = load_prompt_template(
        hypothesaes_config["llm"]["task_specific_instructions_prompt_name"]
    )
    llm_config = read_yaml(CONFIGS_DIR / CONFIG_FILE_NAMES["llm"])
    llm_provider = _resolve_active_llm_provider(llm_config)

    return {
        "labeled_corpus": load_labeled_corpus(paths.labeled_corpus_file),
        "paths": paths,
        "score_threshold": hypothesaes_config["low_confidence"]["score_threshold"],
        "embedder_model_name": hypothesaes_config["embedding"]["model_name"],
        "embedding_batch_size": hypothesaes_config["embedding"]["batch_size"],
        "m_total_neurons": hypothesaes_config["sae"]["m_total_neurons"],
        "k_active_neurons": hypothesaes_config["sae"]["k_active_neurons"],
        "matryoshka_prefix_lengths": hypothesaes_config["sae"]["matryoshka_prefix_lengths"],
        "n_random_neurons": hypothesaes_config["discovery"]["n_random_neurons"],
        "selection_method": hypothesaes_config["hypotheses"]["selection_method"],
        "n_selected_neurons": hypothesaes_config["hypotheses"]["n_selected_neurons"],
        "n_scoring_examples": hypothesaes_config["hypotheses"]["n_scoring_examples"],
        "interpreter_model": hypothesaes_config["llm"][f"interpreter_model_{llm_provider}"],
        "annotator_model": hypothesaes_config["llm"][f"annotator_model_{llm_provider}"],
        "n_examples_for_interpretation": hypothesaes_config["llm"]["n_examples_for_interpretation"],
        "max_words_per_example": hypothesaes_config["llm"]["max_words_per_example"],
        "max_interpretation_tokens": hypothesaes_config["llm"]["max_interpretation_tokens"],
        "task_specific_instructions": task_specific_instructions,
        "llm_provider": llm_provider,
        "llm_ollama_base_url": llm_config["backends"]["ollama"]["base_url"],
        "n_workers": args.max_workers or hypothesaes_config["llm"]["n_workers"],
        "evaluate_on_holdout": args.hypothesaes_evaluate
        or hypothesaes_config["evaluation"]["enabled"],
        "holdout_size": hypothesaes_config["evaluation"]["holdout_size"],
        "validation_size": hypothesaes_config["evaluation"]["validation_size"],
        "random_seed": (
            args.random_seed if args.random_seed is not None else hypothesaes_config["random_seed"]
        ),
    }


def _build_diagnostics_stage_kwargs(
    paths: ProjectPaths, general_config: GeneralConfig, settings: Settings, args: argparse.Namespace
) -> dict[str, Any]:
    """Monta os argumentos de :func:`pipelines.diagnostics_analysis.run_diagnostics_stage`.

    Etapa opt-in: não faz parte de ``configs/config.yaml -> stages`` (``--stage all`` não a
    executa). Parâmetros de modelo, SAE, LLM e custo vêm de ``configs/diagnostics.yaml``
    (validado por Pydantic); a chave da OpenAI, quando usada, vem de ``OPENAI_KEY`` (``.env``).

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto.
    general_config : GeneralConfig
        Configuração geral validada, não utilizada diretamente nesta etapa.
    settings : Settings
        Configurações sensíveis ao ambiente, não utilizadas diretamente nesta etapa.
    args : argparse.Namespace
        Argumentos de linha de comando (``--diagnostics-*``, ``--dry-run``, ``--random-seed``).

    Returns
    -------
    dict[str, Any]
        Argumentos nomeados para
        :func:`pipelines.diagnostics_analysis.run_diagnostics_stage`.
    """
    del general_config, settings
    return {
        "paths": paths,
        "step": args.diagnostics_step,
        "target_name": args.diagnostics_target,
        "model_column": args.diagnostics_model_column,
        "label": args.diagnostics_label,
        "corpus_path": args.diagnostics_corpus,
        "gold": args.diagnostics_gold,
        "dry_run": args.dry_run,
        "random_seed": args.random_seed,
    }


_STAGE_KWARGS_BUILDERS: dict[
    str, Callable[[ProjectPaths, GeneralConfig, Settings, argparse.Namespace], dict[str, Any]]
] = {
    "ingestion": _build_ingestion_stage_kwargs,
    "preprocessing": _build_preprocessing_stage_kwargs,
    "labeling": _build_labeling_stage_kwargs,
    "features": _build_features_stage_kwargs,
    "training_classical": _build_training_classical_stage_kwargs,
    "training_deep_learning": _build_training_deep_learning_stage_kwargs,
    "comparative_evaluation": _build_comparative_evaluation_stage_kwargs,
    "hypothesaes_analysis": _build_hypothesaes_analysis_stage_kwargs,
    "diagnostics": _build_diagnostics_stage_kwargs,
}


def run_stages(
    stage_names: Sequence[str],
    paths: ProjectPaths,
    general_config: GeneralConfig,
    settings: Settings,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Executa uma sequência de estágios, montando os argumentos de cada um sob demanda.

    Os argumentos de cada estágio só são montados imediatamente antes de sua
    execução (não antecipadamente para toda a sequência), pois estágios
    tardios (ex.: ``training_classical``) leem artefatos que só existem em
    disco após a execução dos estágios anteriores (ex.: ``features``).

    Parameters
    ----------
    stage_names : Sequence[str]
        Nomes dos estágios a executar, na ordem de execução.
    paths : ProjectPaths
        Caminhos resolvidos do projeto.
    general_config : GeneralConfig
        Configuração geral validada.
    settings : Settings
        Configurações sensíveis ao ambiente.
    args : argparse.Namespace
        Argumentos de linha de comando.

    Returns
    -------
    dict[str, Any]
        Resultado de cada estágio executado com sucesso, indexado pelo nome
        do estágio, na ordem de ``stage_names``.

    Examples
    --------
    >>> run_stages(["preprocessing"], paths, general_config, settings, args)  # doctest: +SKIP
    """
    results: dict[str, Any] = {}
    for stage_name in stage_names:
        stage_kwargs = _STAGE_KWARGS_BUILDERS[stage_name](paths, general_config, settings, args)
        results[stage_name] = run_pipeline_stage(stage_name, **stage_kwargs)
    logger.info(
        "Execução via CLI concluída: %d etapa(s) executada(s) (%s).",
        len(results),
        ", ".join(results),
    )
    return results


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Ponto de entrada do pipeline: configura o ambiente e despacha a(s) etapa(s) escolhida(s).

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Argumentos de linha de comando, by default None (usa ``sys.argv[1:]``).

    Returns
    -------
    dict[str, Any]
        Resultado de cada estágio executado, indexado pelo nome do estágio.

    Examples
    --------
    >>> main(["--stage", "preprocessing"])  # doctest: +SKIP
    """
    args = parse_arguments(argv)

    configure_environment_variables()
    configure_logging()
    settings = create_settings()
    if args.log_level is not None:
        logging.getLogger().setLevel(args.log_level)
    elif settings.log_level:
        logging.getLogger().setLevel(settings.log_level.upper())

    general_config = load_general_config()
    paths = load_project_paths()

    random_seed = (
        args.random_seed
        if args.random_seed is not None
        else general_config.reproducibility.random_seed
    )
    configure_reproducibility(
        random_seed,
        deterministic_algorithms=general_config.reproducibility.deterministic_algorithms,
    )

    stage_names = list(general_config.stages) if args.stage == _ALL_STAGES_OPTION else [args.stage]
    return run_stages(stage_names, paths, general_config, settings, args)


if __name__ == "__main__":
    main()
