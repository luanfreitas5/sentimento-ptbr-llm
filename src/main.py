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

Lacunas conhecidas (ver comentários nas funções ``_build_*_stage_kwargs``
abaixo): as etapas ``ingestion`` e ``comparative_evaluation`` exigem
componentes que este projeto ainda não implementa como módulos próprios
(um adaptador de scraping - ``src/data/collector.py`` - e uma
transformação TF-IDF reutilizável para conjuntos fora do treino -
``src/features/lexical.py``); por isso, recebem a função necessária via
caminho pontilhado (``--scrape-func``/``--predictions-func``), informado
pelo operador da execução.
"""

from __future__ import annotations

import argparse
import importlib
import logging
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import polars as pl

from config.constants import CONFIG_FILE_NAMES
from config.environment import configure_environment_variables, configure_reproducibility
from config.logging import configure_logging
from config.paths import CONFIGS_DIR, ProjectPaths, load_project_paths
from config.settings import GeneralConfig, Settings, create_settings, load_general_config
from data.loader import load_training_example_dataset, read_dataset_file
from exceptions.configuration import InvalidConfigurationError
from features.lexical import pivot_tfidf_features_to_wide
from io_utils.yaml import read_yaml
from labeling.automatic import LexicalHeuristicLabeler, SentimentLabeler
from pipelines.training_classical import DEFAULT_CLASSICAL_MODEL_NAMES
from pipelines.training_deep_learning import DEFAULT_DEEP_LEARNING_MODEL_NAMES
from pipelines.workflow import STAGE_REGISTRY, run_pipeline_stage

logger = logging.getLogger(__name__)

_ALL_STAGES_OPTION = "all"
_STAGE_CHOICES: tuple[str, ...] = (*STAGE_REGISTRY, _ALL_STAGES_OPTION)
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
        help="Número máximo de threads paralelas (etapas `ingestion`/`llm_evaluation`).",
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
        "--llm-backend",
        default=None,
        choices=["ollama", "huggingface"],
        help="Sobrescreve o backend LLM de `configs/llm.yaml` (etapa `llm_evaluation`).",
    )
    parser.add_argument(
        "--llm-strategy",
        default=None,
        choices=["zero_shot", "few_shot", "chain_of_thought"],
        help="Sobrescreve a estratégia de prompt de `configs/llm.yaml` (etapa `llm_evaluation`).",
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
        "--predictions-func",
        default=None,
        metavar="MODULO:FUNCAO",
        help=(
            "Caminho pontilhado para uma função sem argumentos que retorna a "
            "tupla `(model_predictions, y_true)` (etapa `comparative_evaluation`)."
        ),
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
    X_train = tfidf_wide.select(feature_columns).to_numpy()
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
        Argumentos de linha de comando, não utilizados diretamente nesta
        etapa.

    Returns
    -------
    dict[str, Any]
        Argumentos nomeados para :func:`pipelines.preprocessing.run_preprocessing_stage`.
    """
    del general_config, settings, args
    return {"paths": paths}


def _build_labeling_stage_kwargs(
    paths: ProjectPaths, general_config: GeneralConfig, settings: Settings, args: argparse.Namespace
) -> dict[str, Any]:
    """Monta os argumentos de :func:`pipelines.labeling.run_labeling_stage`.

    Apenas o rotulador heurístico-lexical (:class:`labeling.automatic.LexicalHeuristicLabeler`)
    é injetado nesta composição: os demais rotuladores de ``configs/labeling.yaml
    -> cascade.labelers`` (``llm_zero_shot``, ``modelo_referencia``) dependem de
    módulos ainda não implementados neste projeto (um rotulador via LLM e um
    classificador de referência já treinado, respectivamente).

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
        Argumentos de linha de comando, não utilizados diretamente nesta
        etapa.

    Returns
    -------
    dict[str, Any]
        Argumentos nomeados para :func:`pipelines.labeling.run_labeling_stage`.
    """
    del general_config, settings, args
    labeling_config = read_yaml(CONFIGS_DIR / CONFIG_FILE_NAMES["labeling"])
    labelers: dict[str, SentimentLabeler] = {"heuristica_lexica": LexicalHeuristicLabeler()}
    weights = {
        labeler["name"]: labeler["weight"]
        for labeler in labeling_config["cascade"]["labelers"]
        if labeler["name"] in labelers
    }
    logger.warning(
        "Rotuladores 'llm_zero_shot'/'modelo_referencia' de configs/labeling.yaml "
        "ainda não estão implementados neste projeto; usando apenas "
        "'heuristica_lexica' nesta execução."
    )
    return {
        "paths": paths,
        "labelers": labelers,
        "weights": weights,
        "human_validation_sample_size": labeling_config["human_validation"]["sample_size"],
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
    X_train, y_train = _load_classical_training_arrays(paths)
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
    """Monta os argumentos de :func:`pipelines.training_deep_learning.run_training_deep_learning_stage`.

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


def _build_llm_evaluation_stage_kwargs(
    paths: ProjectPaths, general_config: GeneralConfig, settings: Settings, args: argparse.Namespace
) -> dict[str, Any]:
    """Monta os argumentos de :func:`pipelines.llm_evaluation.run_llm_evaluation_stage`.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto.
    general_config : GeneralConfig
        Configuração geral validada, não utilizada diretamente nesta etapa.
    settings : Settings
        Configurações sensíveis ao ambiente (``settings.ollama_base_url``,
        quando definida, sobrescreve `configs/llm.yaml -> backends.ollama.base_url`).
    args : argparse.Namespace
        Argumentos de linha de comando (``--llm-backend``, ``--llm-strategy``,
        ``--max-workers``).

    Returns
    -------
    dict[str, Any]
        Argumentos nomeados para :func:`pipelines.llm_evaluation.run_llm_evaluation_stage`.
    """
    del general_config
    test_dataframe = load_training_example_dataset(paths.test_corpus_file)
    llm_config = read_yaml(CONFIGS_DIR / CONFIG_FILE_NAMES["llm"])
    backend_name = args.llm_backend or (
        "ollama" if llm_config["backends"]["ollama"]["enabled"] else "huggingface"
    )
    backend_overrides: dict[str, Any] = {}
    if backend_name == "ollama" and settings.ollama_base_url is not None:
        backend_overrides["base_url"] = settings.ollama_base_url
    return {
        "test_dataframe": test_dataframe,
        "backend_name": backend_name,
        "backend_overrides": backend_overrides,
        "strategy": args.llm_strategy or llm_config["prompting"]["default_strategy"],
        "prompt_version": llm_config["orchestration"]["prompt_template_version"],
        "max_workers": args.max_workers,
    }


def _build_comparative_evaluation_stage_kwargs(
    paths: ProjectPaths, general_config: GeneralConfig, settings: Settings, args: argparse.Namespace
) -> dict[str, Any]:
    """Monta os argumentos de :func:`pipelines.comparative_evaluation.run_comparative_evaluation_stage`.

    A coleta das predições de cada modelo treinado não pode ser reconstruída
    apenas a partir de artefatos em disco: a etapa ``features`` só calcula
    TF-IDF para o conjunto de treino (ver ``_build_training_classical_stage_kwargs``),
    e o projeto ainda não expõe uma transformação TF-IDF reutilizável para o
    conjunto de teste com o vocabulário de treino (``src/features/lexical.py``).
    Por isso, ``--predictions-func`` é obrigatório nesta etapa.

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
        Argumentos de linha de comando (``--predictions-func``).

    Returns
    -------
    dict[str, Any]
        Argumentos nomeados para
        :func:`pipelines.comparative_evaluation.run_comparative_evaluation_stage`.

    Raises
    ------
    InvalidConfigurationError
        Se ``--predictions-func`` não for informado.
    """
    del general_config, settings
    if args.predictions_func is None:
        raise InvalidConfigurationError(
            "a etapa 'comparative_evaluation' exige '--predictions-func': informe o "
            "caminho pontilhado de uma função sem argumentos que retorne a tupla "
            "'(model_predictions, y_true)' com as predições de cada modelo já "
            "treinado sobre o conjunto de teste."
        )
    predictions_func = _import_callable_from_dotted_path(args.predictions_func)
    model_predictions, y_true = predictions_func()
    return {
        "model_predictions": model_predictions,
        "y_true": y_true,
        "output_path": paths.reports_metrics_dir / "comparativo_modelos.csv",
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
    "llm_evaluation": _build_llm_evaluation_stage_kwargs,
    "comparative_evaluation": _build_comparative_evaluation_stage_kwargs,
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
