"""Nomes dos modelos de cada fonte de rotulagem, para tabelas, gráficos e arquivos.

As fontes (``huggingface``, ``huggingface_secondary``, ``gold_openai``, ``gold_unb``) são
identificadores internos; nos relatórios o que importa é QUEM rotulou. Este módulo lê os modelos
configurados em ``configs/labeling.yaml`` e os expõe de duas formas:

* nome de exibição, para tabelas e gráficos, sem o prefixo da organização
  (ex.: ``twitter-xlm-roberta-base-sentiment-multilingual``);
* ``slug``, seguro para nomes de arquivo e pastas (sem espaços nem caracteres especiais;
  ex.: ``gpt-6-luna``).
"""

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from config.constants import CONFIG_FILE_NAMES
from config.paths import CONFIGS_DIR
from constants.labels import HUGGINGFACE_SECONDARY_SOURCE, HUGGINGFACE_SOURCE
from exceptions.data import DataValidationError
from io_utils.yaml import read_yaml

_UNSAFE_FILE_CHARACTERS = re.compile(r"[^A-Za-z0-9._-]+")


def build_display_name(model_id: str) -> str:
    """Nome curto de um modelo: o trecho após a última ``/`` do identificador.

    Parameters
    ----------
    model_id : str
        Identificador do modelo
        (ex.: ``cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual``).

    Returns
    -------
    str
        Nome sem o prefixo da organização.

    Examples
    --------
    >>> build_display_name("pysentimiento/bertweet-pt-sentiment")
    'bertweet-pt-sentiment'
    """
    return model_id.rstrip("/").rsplit("/", maxsplit=1)[-1]


def build_file_slug(name: str) -> str:
    """Converte um nome de modelo em texto seguro para nomes de arquivo e pastas.

    Parameters
    ----------
    name : str
        Nome de exibição do modelo.

    Returns
    -------
    str
        Caracteres fora de ``[A-Za-z0-9._-]`` trocados por ``-``; a caixa é preservada.

    Examples
    --------
    >>> build_file_slug("UnB Llama/3.3")
    'UnB-Llama-3.3'
    """
    return _UNSAFE_FILE_CHARACTERS.sub("-", name).strip("-")


def resolve_source_model_names(labeling_config: Mapping[str, Any]) -> dict[str, str]:
    """Nome do modelo de cada fonte de rotulagem, a partir de ``configs/labeling.yaml``.

    Parameters
    ----------
    labeling_config : Mapping[str, Any]
        Conteúdo de ``configs/labeling.yaml``.

    Returns
    -------
    dict[str, str]
        Nome de exibição indexado pela fonte (``huggingface``, ``huggingface_secondary``,
        ``gold_openai`` e ``gold_unb``).

    Raises
    ------
    DataValidationError
        Se a configuração não definir o modelo de alguma fonte.

    Examples
    --------
    >>> config = {
    ...     "huggingface": {"model": "org/a"},
    ...     "huggingface_secondary": {"model": "org/b"},
    ...     "gold_sets": {"openai": {"model": "c"}, "unb": {"model": "d"}},
    ... }
    >>> resolve_source_model_names(config)["gold_unb"]
    'd'
    """
    try:
        model_ids = {
            HUGGINGFACE_SOURCE: labeling_config["huggingface"]["model"],
            HUGGINGFACE_SECONDARY_SOURCE: labeling_config["huggingface_secondary"]["model"],
            "gold_openai": labeling_config["gold_sets"]["openai"]["model"],
            "gold_unb": labeling_config["gold_sets"]["unb"]["model"],
        }
    except (KeyError, TypeError) as error:
        raise DataValidationError(
            schema_name="labeling_config",
            detail=f"modelo das fontes de rotulagem ausente em labeling.yaml: {error}",
        ) from error
    return {source: build_display_name(str(model_id)) for source, model_id in model_ids.items()}


def load_source_model_names(config_file: Path | None = None) -> dict[str, str]:
    """Lê ``configs/labeling.yaml`` e devolve o nome do modelo de cada fonte.

    Parameters
    ----------
    config_file : Path | None, optional
        Caminho alternativo do YAML de rotulagem, by default ``configs/labeling.yaml``.

    Returns
    -------
    dict[str, str]
        Ver :func:`resolve_source_model_names`.

    Examples
    --------
    >>> load_source_model_names()["gold_openai"]  # doctest: +SKIP
    'gpt-6-luna'
    """
    path = config_file or CONFIGS_DIR / CONFIG_FILE_NAMES["labeling"]
    return resolve_source_model_names(read_yaml(path))
