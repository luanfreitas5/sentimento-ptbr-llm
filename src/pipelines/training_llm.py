"""Preparo e validação dos LLMs open-source (Ollama) como classificadores de sentimento.

Implementa o estágio ``training_llm`` de ``configs/config.yaml -> stages``. LLMs
open-source não passam por otimização de pesos aqui: o "treino" seleciona os
exemplos balanceados do prompt few-shot a partir do conjunto de treino
(:meth:`models.llm.LLMSentimentClassifier.fit`) e mede o desempenho no conjunto
de validação, com um limite configurável de amostras para conter o custo de
inferência. Cada modelo gera uma especificação reproduzível em
``models/checkpoints/<modelo>.llm.json`` (hiperparâmetros + métricas de
validação); a avaliação final no teste (``evaluate``) reconstrói o classificador
a partir dessa especificação. Exige o servidor Ollama ativo
(``make install-llm`` e ``make ollama``).
"""

import logging
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from io_utils.json import write_json
from models.factory import create_classifier
from training.trainer import Trainer, TrainingResult

logger = logging.getLogger(__name__)

DEFAULT_LLM_MODEL_NAMES: tuple[str, ...] = ("llama3_2",)
LLM_SPEC_SUFFIX = ".llm.json"


def select_sample_indices(n_total: int, max_samples: int | None, random_seed: int) -> list[int]:
    """Sorteia, de forma determinística, os índices de uma subamostra ordenada.

    Parameters
    ----------
    n_total : int
        Tamanho do conjunto completo.
    max_samples : int | None
        Tamanho máximo da subamostra; ``None`` (ou ``>= n_total``) usa todos os índices.
    random_seed : int
        Semente do sorteio.

    Returns
    -------
    list[int]
        Índices ordenados.

    Examples
    --------
    >>> select_sample_indices(5, None, 42)
    [0, 1, 2, 3, 4]
    >>> len(select_sample_indices(100, 10, 42))
    10
    """
    if max_samples is None or max_samples >= n_total:
        return list(range(n_total))
    generator = np.random.default_rng(random_seed)
    return sorted(generator.choice(n_total, size=max_samples, replace=False).tolist())


def _select_validation_subset(
    X_val: Sequence[str] | None,  # noqa: N803
    y_val: Sequence[str] | None,
    max_samples: int | None,
    random_seed: int,
) -> tuple[list[str] | None, list[str] | None]:
    """Subamostra determinística da validação enviada ao LLM (``None`` se não houver validação)."""
    if X_val is None or y_val is None:
        return None, None
    indices = select_sample_indices(len(X_val), max_samples, random_seed)
    return [X_val[index] for index in indices], [y_val[index] for index in indices]


def _write_llm_specification(
    path: Path,
    model_name: str,
    overrides: Mapping[str, Any],
    n_validation: int,
    result: TrainingResult,
) -> None:
    """Grava a especificação reproduzível do LLM (hiperparâmetros + métricas de validação)."""
    write_json(
        {
            "model_name": model_name,
            "overrides": dict(overrides),
            "n_validation_samples": n_validation,
            "validation_metrics": result.metrics,
            "elapsed_seconds": result.elapsed_seconds,
        },
        path,
    )


def run_training_llm_stage(
    X_train: Sequence[str],  # noqa: N803
    y_train: Sequence[str],
    X_val: Sequence[str] | None,  # noqa: N803
    y_val: Sequence[str] | None,
    *,
    model_names: Sequence[str] = DEFAULT_LLM_MODEL_NAMES,
    model_params: Mapping[str, Mapping[str, Any]] | None = None,
    checkpoints_dir: Path,
    max_validation_samples: int | None = None,
    random_seed: int = 42,
    track_with_mlflow: bool = False,
) -> dict[str, TrainingResult]:
    """Seleciona os exemplos few-shot e valida cada LLM open-source configurado.

    Parameters
    ----------
    X_train : Sequence[str]
        Textos de treino, de onde saem os exemplos few-shot.
    y_train : Sequence[str]
        Rótulos de sentimento de treino, mesmo tamanho de ``X_train``.
    X_val : Sequence[str] | None
        Textos de validação; ``None`` dispensa a medição de métricas.
    y_val : Sequence[str] | None
        Rótulos de validação, mesmo tamanho de ``X_val``.
    model_names : Sequence[str], optional
        Chaves de ``configs/model_params.yaml -> llm``, by default
        :data:`DEFAULT_LLM_MODEL_NAMES`.
    model_params : Mapping[str, Mapping[str, Any]] | None, optional
        Hiperparâmetros por modelo (``configs/model_params.yaml -> llm``), by default None.
    checkpoints_dir : Path
        Diretório das especificações (``paths.models_checkpoints_dir``).
    max_validation_samples : int | None, optional
        Máximo de textos de validação enviados ao LLM (custo de inferência), by default None
        (todos).
    random_seed : int, optional
        Semente da subamostragem de validação, by default 42.
    track_with_mlflow : bool, optional
        Repassado a :class:`training.trainer.Trainer`, by default False.

    Returns
    -------
    dict[str, TrainingResult]
        Resultado de cada modelo (métricas na subamostra de validação), indexado pelo nome.

    Examples
    --------
    >>> run_training_llm_stage(
    ...     X_train, y_train, X_val, y_val, checkpoints_dir=Path("models/checkpoints")
    ... )  # doctest: +SKIP
    """
    resolved_model_params = model_params or {}
    results: dict[str, TrainingResult] = {}

    val_texts, val_labels = _select_validation_subset(
        X_val, y_val, max_validation_samples, random_seed
    )

    for model_name in model_names:
        overrides = dict(resolved_model_params.get(model_name, {}))
        trainer = Trainer(
            partial(create_classifier, "llm", **overrides),
            random_state=random_seed,
            track_with_mlflow=track_with_mlflow,
        )
        result = trainer.fit(X_train, y_train, val_texts, val_labels)
        results[model_name] = result
        _write_llm_specification(
            checkpoints_dir / f"{model_name}{LLM_SPEC_SUFFIX}",
            model_name,
            overrides,
            len(val_texts or []),
            result,
        )
        logger.info(
            "LLM '%s' preparado em %.2fs (métricas de validação=%s).",
            model_name,
            result.elapsed_seconds,
            result.metrics,
        )

    return results
