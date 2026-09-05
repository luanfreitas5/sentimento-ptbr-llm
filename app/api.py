"""API FastAPI para inferência de sentimento sobre texto livre.

Implementa a Fase 21 do plano de elaboração (ver ``PLANO-ELABORACAO.md``):
expõe o :class:`inference.predictor.Predictor` já treinado via HTTP.
Apenas abordagens cujo classificador aceita texto cru diretamente são
servidas (``deep_learning``, ``transformer``, ``llm``) — a abordagem
``classical`` depende de uma matriz TF-IDF ajustada ao vocabulário de
treino que o projeto ainda não persiste como transformação reutilizável
para textos fora do treino (ver docstring de ``src/main.py``), então não é
servida aqui.

Resolução do modelo (na subida da API, ver :func:`_lifespan`): tenta
primeiro o MLflow Model Registry (``configs/deploy.yaml ->
mlflow_serving``, coerente com ``CLAUDE.md`` -> "Model & Data
Versioning"); se o registro estiver desabilitado ou indisponível, recorre
ao melhor checkpoint local salvo em disco pelo estágio de treino
correspondente (``src/pipelines/training_classical.py``/
``training_deep_learning.py``).

Uso
---
    uv run uvicorn app.api:app --host 0.0.0.0 --port 8000

Ver ``configs/deploy.yaml -> api`` para host/porta/abordagem padrão.
"""

from __future__ import annotations

import sys
from pathlib import Path

# `app/` não fica dentro de `src/` (raiz de importação do projeto — ver
# CLAUDE.md, "Import style"), então o diretório precisa ser inserido em
# `sys.path` manualmente antes de qualquer import próprio (ver
# `pyproject.toml -> [tool.ruff.lint.per-file-ignores] "app/*"`).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import logging
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config.constants import CONFIG_FILE_NAMES
from config.logging import configure_logging
from config.paths import CONFIGS_DIR, ProjectPaths, load_project_paths
from data.loader import load_training_example_dataset
from exceptions.base import ProjectError
from exceptions.configuration import InvalidConfigurationError
from inference.online import OnlinePredictor
from inference.predictor import Predictor
from io_utils.yaml import read_yaml
from models.persistence import load_classifier

logger = logging.getLogger(__name__)

# Convenção própria desta camada de API para resolver qual checkpoint local
# carregar por abordagem: `configs/deploy.yaml` decide apenas a *abordagem*
# padrão (`api.default_approach`), não o modelo específico dentro dela (ver
# comentário "decisão final... em aberto" no próprio arquivo). Escolhe-se
# aqui o modelo mais representativo de cada paradigma, coerente com
# `src/pipelines/training_deep_learning.py -> DEFAULT_DEEP_LEARNING_MODEL_NAMES`.
_DEFAULT_MODEL_NAME_BY_APPROACH: dict[str, str] = {
    "deep_learning": "lstm",
    "transformer": "bertimbau",
}
_CHECKPOINT_BACKEND_BY_APPROACH: dict[str, str] = {
    "deep_learning": "torch",
    "transformer": "torch",
}
_UNSUPPORTED_APPROACH = "classical"


class PredictionRequest(BaseModel):
    """Corpo da requisição de predição de sentimento para um único texto."""

    text: str = Field(..., min_length=1, description="Texto livre em português a classificar.")


class BatchPredictionRequest(BaseModel):
    """Corpo da requisição de predição de sentimento em lote."""

    texts: list[str] = Field(..., min_length=1, description="Lista de textos a classificar.")


class PredictionResponse(BaseModel):
    """Corpo da resposta de uma predição de sentimento para um único texto."""

    sentiment_label: str
    confidence: float
    probabilities: dict[str, float]


class BatchPredictionItem(BaseModel):
    """Uma predição individual dentro de uma resposta de predição em lote."""

    id: str
    text: str
    sentiment_label: str
    confidence: float


def _read_deploy_config() -> dict[str, Any]:
    """Lê ``configs/deploy.yaml`` como dicionário bruto.

    Returns
    -------
    dict[str, Any]
        Conteúdo de ``configs/deploy.yaml``. Não existe modelo Pydantic
        próprio para este arquivo no projeto (ver ``src/config/settings.py``),
        então a leitura é feita diretamente aqui.
    """
    return read_yaml(CONFIGS_DIR / CONFIG_FILE_NAMES["deploy"])


def _load_model_from_registry(model_name: str, stage: str) -> Any | None:
    """Tenta carregar o modelo de produção a partir do MLflow Model Registry.

    Parameters
    ----------
    model_name : str
        Nome do modelo registrado (``configs/deploy.yaml ->
        mlflow_serving.model_name``).
    stage : str
        Estágio do Model Registry a carregar (``configs/deploy.yaml ->
        api.default_model_stage``).

    Returns
    -------
    Any | None
        Modelo carregado, ou ``None`` quando o registro está indisponível
        ou nenhum modelo foi promovido ainda ao estágio solicitado — nesse
        caso o chamador deve recorrer ao checkpoint local (ver
        :func:`_load_model_from_local_checkpoint`).
    """
    try:
        import mlflow

        return mlflow.pyfunc.load_model(f"models:/{model_name}/{stage}")
    except Exception as exception:
        logger.warning(
            "Não foi possível carregar o modelo do MLflow Model Registry "
            "('%s', estágio '%s'): %s. Tentando checkpoint local.",
            model_name,
            stage,
            exception,
        )
        return None


def _load_model_from_local_checkpoint(approach: str, paths: ProjectPaths) -> Any:
    """Carrega o melhor checkpoint local salvo para a abordagem solicitada.

    Parameters
    ----------
    approach : str
        Abordagem servida (uma das chaves de
        :data:`_DEFAULT_MODEL_NAME_BY_APPROACH`).
    paths : ProjectPaths
        Caminhos resolvidos do projeto.

    Returns
    -------
    Any
        Modelo desserializado, pronto para ``predict``/``predict_proba``.
    """
    model_name = _DEFAULT_MODEL_NAME_BY_APPROACH[approach]
    backend = _CHECKPOINT_BACKEND_BY_APPROACH[approach]
    extension = "joblib" if backend == "joblib" else "pt"
    checkpoint_path = paths.models_checkpoints_dir / f"{model_name}.{extension}"
    return load_classifier(checkpoint_path, backend=backend)


def _build_llm_classifier(paths: ProjectPaths) -> Any:
    """Monta um classificador LLM pronto para uso, sem depender de checkpoint em disco.

    O classificador LLM não possui pesos treináveis: seu único estado é a
    seleção de exemplos few-shot (ver ``src/models/llm.py``), reconstruída
    aqui a partir do corpus de treino já processado, quando disponível.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto.

    Returns
    -------
    Any
        Classificador LLM pronto para ``predict``/``predict_proba``.
    """
    from models.llm import LLMSentimentClassifier, load_ollama_backend

    classifier = LLMSentimentClassifier(load_ollama_backend())
    if paths.training_corpus_file.exists():
        training_corpus = load_training_example_dataset(paths.training_corpus_file)
        classifier.fit(
            training_corpus["text"].to_list(), training_corpus["sentiment_label"].to_list()
        )
    else:
        logger.warning(
            "Corpus de treino ausente em '%s': classificador LLM iniciado sem "
            "exemplos few-shot (modo zero-shot efetivo).",
            paths.training_corpus_file,
        )
    return classifier


def _resolve_classifier(approach: str, paths: ProjectPaths, deploy_config: dict[str, Any]) -> Any:
    """Resolve o classificador a servir para a abordagem configurada.

    Parameters
    ----------
    approach : str
        Abordagem a servir (``configs/deploy.yaml -> api.default_approach``).
    paths : ProjectPaths
        Caminhos resolvidos do projeto.
    deploy_config : dict[str, Any]
        Conteúdo de ``configs/deploy.yaml``.

    Returns
    -------
    Any
        Classificador pronto para ``predict``/``predict_proba``.

    Raises
    ------
    InvalidConfigurationError
        Se ``approach`` for ``"classical"`` ou não for uma abordagem
        conhecida.
    """
    if approach == _UNSUPPORTED_APPROACH:
        raise InvalidConfigurationError(
            "a abordagem 'classical' não é servida por esta API: depende de uma "
            "matriz TF-IDF ajustada ao vocabulário de treino que o projeto ainda "
            "não persiste como transformação reutilizável para textos fora do "
            "treino (ver docstring de 'src/main.py'); escolha 'deep_learning', "
            "'transformer' ou 'llm' em 'configs/deploy.yaml -> api.default_approach'."
        )
    if approach == "llm":
        return _build_llm_classifier(paths)
    if approach not in _DEFAULT_MODEL_NAME_BY_APPROACH:
        raise InvalidConfigurationError(
            f"abordagem '{approach}' desconhecida em 'configs/deploy.yaml -> "
            f"api.default_approach'; use uma de: "
            f"{('deep_learning', 'transformer', 'llm')}."
        )

    mlflow_serving_config = deploy_config.get("mlflow_serving", {})
    if mlflow_serving_config.get("enabled", False):
        registry_model = _load_model_from_registry(
            mlflow_serving_config["model_name"], deploy_config["api"]["default_model_stage"]
        )
        if registry_model is not None:
            return registry_model
    return _load_model_from_local_checkpoint(approach, paths)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> Iterator[None]:
    """Configura logging/ambiente e carrega o modelo de produção na subida da API.

    Parameters
    ----------
    app : FastAPI
        Instância da aplicação, usada para guardar o preditor carregado em
        ``app.state`` (evita recarregar o modelo a cada requisição).
    """
    configure_logging()
    deploy_config = _read_deploy_config()
    paths = load_project_paths()
    approach = deploy_config["api"]["default_approach"]

    model = _resolve_classifier(approach, paths, deploy_config)
    predictor = Predictor(model)
    app.state.online_predictor = OnlinePredictor(predictor)
    app.state.batch_predictor = predictor
    app.state.approach = approach
    logger.info("API de inferência de sentimento pronta (abordagem='%s').", approach)

    yield

    logger.info("Encerrando API de inferência de sentimento.")


app = FastAPI(
    title="API de Análise de Sentimentos pt-BR",
    description="Inferência de sentimento (positivo/negativo/neutro) sobre texto livre em português.",
    version="0.1.0",
    lifespan=_lifespan,
)


@app.get("/health")
def get_health_status() -> dict[str, str]:
    """Verifica se a API está no ar e qual abordagem de modelo está sendo servida.

    Returns
    -------
    dict[str, str]
        Dicionário com as chaves ``"status"`` e ``"approach"``.
    """
    return {"status": "ok", "approach": app.state.approach}


@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: PredictionRequest) -> PredictionResponse:
    """Classifica o sentimento de um único texto.

    Parameters
    ----------
    request : PredictionRequest
        Corpo da requisição, contendo o texto a classificar.

    Returns
    -------
    PredictionResponse
        Rótulo de sentimento predito, confiança e probabilidades por classe.

    Raises
    ------
    HTTPException
        Com status 422 se ``request.text`` for inválido, ou 500 se a
        inferência falhar por um erro conhecido do projeto.
    """
    try:
        record = app.state.online_predictor.predict(request.text)
    except ValueError as exception:
        raise HTTPException(status_code=422, detail=str(exception)) from exception
    except ProjectError as exception:
        raise HTTPException(status_code=500, detail=str(exception)) from exception
    return PredictionResponse(**record)


@app.post("/predict/batch", response_model=list[BatchPredictionItem])
def predict_sentiment_batch(request: BatchPredictionRequest) -> list[BatchPredictionItem]:
    """Classifica o sentimento de um lote de textos.

    Parameters
    ----------
    request : BatchPredictionRequest
        Corpo da requisição, contendo a lista de textos a classificar.

    Returns
    -------
    list[BatchPredictionItem]
        Uma predição por texto de entrada, na mesma ordem.

    Raises
    ------
    HTTPException
        Com status 500 se a inferência falhar por um erro conhecido do
        projeto.
    """
    try:
        predictions = app.state.batch_predictor.predict(request.texts)
    except ProjectError as exception:
        raise HTTPException(status_code=500, detail=str(exception)) from exception
    return [BatchPredictionItem(**row) for row in predictions.to_dicts()]
