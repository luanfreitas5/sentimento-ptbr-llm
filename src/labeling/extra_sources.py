"""Descoberta das bases rotuladas por modelos adicionais do Hugging Face.

Além dos modelos configurados em ``configs/labeling.yaml`` (``huggingface`` e
``huggingface_secondary``), o diretório ``data/gold`` pode conter outras bases
``tweets_data_<modelo>.parquet`` (geradas por
``scripts/adjust_labeled_hf_datasets_to_tweets_data.py``), cada uma com o seu
``tweets_data_<modelo>.meta.json``. Este módulo as descobre pelo nome do arquivo e
lê o modelo do ``meta.json``, para que a avaliação comparativa trate todos os modelos
do Hugging Face como uma lista, sem cadastrá-los um a um na configuração.

Os modelos adicionais são configurados em ``configs/labeling.yaml -> huggingface_extra`` e
rotulados pela etapa ``labeling`` (``--label-source huggingface_extra``); bases já existentes
em ``data/gold`` entram na avaliação mesmo sem constar na configuração.
"""

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from config.paths import ProjectPaths
from exceptions.configuration import InvalidConfigurationError
from io_utils.json import read_json
from labeling.model_names import build_display_name

logger = logging.getLogger(__name__)

EXTRA_SOURCE_PREFIX = "huggingface_extra:"
EXTRA_FILE_GLOB = "tweets_data_*.parquet"
_FILE_STEM_PREFIX = "tweets_data_"


def is_extra_source(source_name: str) -> bool:
    """Indica se a fonte é uma base de modelo adicional do Hugging Face.

    Parameters
    ----------
    source_name : str
        Identificador da fonte (ex.: ``huggingface_extra:FinBERT-PT-BR``).

    Returns
    -------
    bool
        ``True`` se o identificador usa o prefixo das fontes adicionais.

    Examples
    --------
    >>> is_extra_source("huggingface_extra:FinBERT-PT-BR")
    True
    >>> is_extra_source("gold_unb")
    False
    """
    return source_name.startswith(EXTRA_SOURCE_PREFIX)


def map_extra_source_paths(paths: ProjectPaths) -> dict[str, Path]:
    """Arquivos das bases adicionais do Hugging Face existentes em ``data/gold``.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto.

    Returns
    -------
    dict[str, Path]
        ``huggingface_extra:<sufixo do arquivo>`` -> arquivo, em ordem alfabética. As bases
        primária e secundária ficam de fora (já são fontes configuradas).

    Examples
    --------
    >>> map_extra_source_paths(paths)  # doctest: +SKIP
    {'huggingface_extra:FinBERT-PT-BR': PosixPath('.../tweets_data_FinBERT-PT-BR.parquet')}
    """
    configured = {paths.huggingface_labeled_file, paths.huggingface_secondary_file}
    return {
        f"{EXTRA_SOURCE_PREFIX}{path.stem.removeprefix(_FILE_STEM_PREFIX)}": path
        for path in sorted(paths.data_gold_dir.glob(EXTRA_FILE_GLOB))
        if path not in configured
    }


def load_extra_model_names(paths: ProjectPaths) -> dict[str, str]:
    """Nome do modelo de cada base adicional, para tabelas, gráficos e pastas.

    O nome vem do campo ``model`` do ``meta.json`` ao lado da base (sem o prefixo da
    organização); na ausência do arquivo, usa o sufixo do nome da base.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto.

    Returns
    -------
    dict[str, str]
        Nome do modelo, indexado pelo identificador da fonte.

    Examples
    --------
    >>> load_extra_model_names(paths)  # doctest: +SKIP
    {'huggingface_extra:FinBERT-PT-BR': 'FinBERT-PT-BR'}
    """
    names: dict[str, str] = {}
    for source_name, path in map_extra_source_paths(paths).items():
        meta_path = path.with_suffix(".meta.json")
        model_id = source_name.removeprefix(EXTRA_SOURCE_PREFIX)
        if meta_path.is_file():
            model_id = str(read_json(meta_path).get("model") or model_id)
        else:
            logger.warning("Metadados ausentes para %s: usando o nome do arquivo.", path.name)
        names[source_name] = build_display_name(model_id)
    return names


def build_extra_source_name(model_id: str) -> str:
    """Identificador da fonte de um modelo adicional do Hugging Face.

    Parameters
    ----------
    model_id : str
        Identificador do modelo no Hub (ex.: ``lucas-leme/FinBERT-PT-BR``).

    Returns
    -------
    str
        ``huggingface_extra:<nome do modelo sem a organização>``.

    Examples
    --------
    >>> build_extra_source_name("lucas-leme/FinBERT-PT-BR")
    'huggingface_extra:FinBERT-PT-BR'
    """
    return f"{EXTRA_SOURCE_PREFIX}{build_display_name(model_id)}"


def build_extra_source_path(paths: ProjectPaths, source_name: str) -> Path:
    """Arquivo ``tweets_data_<modelo>.parquet`` da base de uma fonte adicional.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto.
    source_name : str
        Identificador da fonte (``huggingface_extra:<modelo>``).

    Returns
    -------
    Path
        Caminho da base em ``data/gold`` (pode ainda não existir).

    Examples
    --------
    >>> build_extra_source_path(paths, "huggingface_extra:FinBERT-PT-BR").name  # doctest: +SKIP
    'tweets_data_FinBERT-PT-BR.parquet'
    """
    suffix = source_name.removeprefix(EXTRA_SOURCE_PREFIX)
    return paths.data_gold_dir / f"{_FILE_STEM_PREFIX}{suffix}.parquet"


def read_extra_model_configs(labeling_config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Configuração de cada modelo adicional, com os padrões comuns já aplicados.

    Parameters
    ----------
    labeling_config : Mapping[str, Any]
        Conteúdo de ``configs/labeling.yaml``.

    Returns
    -------
    dict[str, dict[str, Any]]
        ``device``, ``dtype``, ``batch_size``, ``max_input_tokens``, ``model`` e ``revision`` de
        cada modelo, indexados pelo identificador da fonte; vazio se não houver
        ``huggingface_extra``.

    Raises
    ------
    InvalidConfigurationError
        Se um modelo repetir o nome de outro ou faltar ``model``/``revision``.

    Examples
    --------
    >>> config = {
    ...     "huggingface_extra": {"batch_size": 8, "models": [{"model": "o/m", "revision": "x"}]}
    ... }
    >>> read_extra_model_configs(config)["huggingface_extra:m"]["batch_size"]
    8
    """
    block = labeling_config.get("huggingface_extra") or {}
    defaults = {key: value for key, value in block.items() if key != "models"}
    configs: dict[str, dict[str, Any]] = {}
    for entry in block.get("models", []):
        merged = {**defaults, **entry}
        if "model" not in merged or "revision" not in merged:
            raise InvalidConfigurationError(
                f"huggingface_extra.models exige 'model' e 'revision' em cada item: {entry}"
            )
        source_name = build_extra_source_name(str(merged["model"]))
        if source_name in configs:
            raise InvalidConfigurationError(f"modelo adicional repetido: '{merged['model']}'")
        configs[source_name] = merged
    return configs
