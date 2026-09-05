"""Rastreamento, registro e reprodutibilidade de experimentos.

Implementa CLAUDE.md, "Model & Data Versioning" e "Reproducibility &
Determinism": ciclo de vida de execuções do MLflow Tracking, promoção de
modelos no MLflow Model Registry e o manifesto de reprodutibilidade
(SHA do Git + hash do dataset + versões de bibliotecas) que rastreia
exatamente o que produziu cada resultado.

Modules
-------
tracker
    Abertura/encerramento de execuções MLflow, registro de parâmetros,
    métricas e artefatos, e montagem do DataFrame validado por
    :mod:`schemas.experiment`.
registry
    Registro de versões de modelo e promoção entre estágios (``Staging``
    -> ``Production``) no MLflow Model Registry.
reproducibility
    Manifesto de reprodutibilidade por execução (SHA do Git, hash do
    dataset, versões de bibliotecas) e comparação entre manifestos.
"""

from experiment.registry import (
    MODEL_REGISTRY_STAGES,
    get_latest_model_version,
    register_model_version,
    transition_model_stage,
)
from experiment.reproducibility import (
    TRACKED_LIBRARY_NAMES,
    build_reproducibility_manifest,
    collect_library_versions,
    compare_reproducibility_manifests,
    get_current_git_sha,
)
from experiment.tracker import (
    build_experiment_run_metrics_dataframe,
    log_run_artifact,
    log_run_metrics,
    log_run_parameters,
    track_experiment_run,
)

__all__: list[str] = [
    "MODEL_REGISTRY_STAGES",
    "TRACKED_LIBRARY_NAMES",
    "build_experiment_run_metrics_dataframe",
    "build_reproducibility_manifest",
    "collect_library_versions",
    "compare_reproducibility_manifests",
    "get_current_git_sha",
    "get_latest_model_version",
    "log_run_artifact",
    "log_run_metrics",
    "log_run_parameters",
    "register_model_version",
    "track_experiment_run",
    "transition_model_stage",
]
