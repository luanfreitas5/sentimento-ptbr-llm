"""Descoberta de checkpoints e predição sobre o conjunto de teste, por categoria de modelo.

Reúne o que a etapa ``evaluate`` precisa para tratar de forma uniforme as quatro
categorias de modelo do estudo: Baseline + ML tradicional (``.joblib``, sobre
TF-IDF), Deep Learning e Transformers (``.pt``, sobre texto) e LLMs open-source
(``.llm.json``, reconstruídos a partir da especificação gravada por
``training_llm``). As predições são sempre devolvidas como rótulos de sentimento
(``str``) e, quando o modelo expõe probabilidades, como matriz de scores na ordem
de :data:`constants.labels.SENTIMENT_CLASSES`.
"""

import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from constants.labels import ID_TO_LABEL, SENTIMENT_CLASSES
from exceptions.model import ModelError
from models.factory import create_classifier
from models.persistence import load_classifier

logger = logging.getLogger(__name__)

CATEGORY_CLASSICAL = "classical"
CATEGORY_DEEP_LEARNING = "deep_learning"
CATEGORY_TRANSFORMER = "transformer"
CATEGORY_LLM = "llm"

# Rótulo legível de cada categoria (nomes dos "Modelos avaliados" do estudo).
CATEGORY_LABELS: dict[str, str] = {
    CATEGORY_CLASSICAL: "Baseline + ML tradicional",
    CATEGORY_DEEP_LEARNING: "Deep Learning",
    CATEGORY_TRANSFORMER: "Transformer",
    CATEGORY_LLM: "LLM open-source",
}

TRANSFORMER_MODEL_NAMES: frozenset[str] = frozenset({"bertimbau", "roberta", "distilbert"})
LLM_SPEC_SUFFIX = ".llm.json"
TFIDF_VECTORIZER_STEM = "tfidf_vectorizer"


@dataclass(frozen=True)
class ModelCheckpoint:
    """Um artefato de modelo encontrado em ``models/checkpoints``.

    Parameters
    ----------
    name : str
        Nome do modelo (nome do arquivo sem extensão).
    category : str
        Uma das chaves de :data:`CATEGORY_LABELS`.
    path : Path
        Caminho do artefato.
    """

    name: str
    category: str
    path: Path


@dataclass(frozen=True)
class ModelPredictions:
    """Predições de um modelo sobre (parte de) o conjunto de teste.

    Parameters
    ----------
    name : str
        Nome do modelo.
    category : str
        Categoria do modelo (chave de :data:`CATEGORY_LABELS`).
    indices : list[int]
        Posições, no conjunto de teste, dos textos efetivamente preditos (todos, exceto para
        LLMs com limite de amostras).
    y_pred : list[str]
        Rótulos preditos, alinhados a ``indices``.
    y_score : np.ndarray | None
        Probabilidades ``(n, 3)`` na ordem de ``SENTIMENT_CLASSES``; ``None`` se indisponível.
    inference_ms_per_sample : float
        Tempo médio de inferência por texto, em milissegundos.
    """

    name: str
    category: str
    indices: list[int]
    y_pred: list[str]
    y_score: np.ndarray | None
    inference_ms_per_sample: float


def _classify_checkpoint(path: Path) -> tuple[str, str] | None:
    """Devolve ``(nome, categoria)`` de um arquivo de ``models/checkpoints``, ou ``None``.

    Arquivos que não são modelo (vetorizador TF-IDF, subpastas, notas) retornam ``None``.
    """
    if path.name.endswith(LLM_SPEC_SUFFIX):
        return path.name.removesuffix(LLM_SPEC_SUFFIX), CATEGORY_LLM
    if path.suffix == ".pt":
        is_transformer = path.stem in TRANSFORMER_MODEL_NAMES
        return path.stem, CATEGORY_TRANSFORMER if is_transformer else CATEGORY_DEEP_LEARNING
    if path.suffix == ".joblib" and path.stem != TFIDF_VECTORIZER_STEM:
        return path.stem, CATEGORY_CLASSICAL
    return None


def discover_checkpoints(
    checkpoints_dir: Path, model_names: Sequence[str] | None = None
) -> list[ModelCheckpoint]:
    """Lista os modelos persistidos, ordenados por categoria e nome.

    Parameters
    ----------
    checkpoints_dir : Path
        Diretório de checkpoints (``paths.models_checkpoints_dir``).
    model_names : Sequence[str] | None, optional
        Restringe a descoberta a estes modelos, by default None (todos).

    Returns
    -------
    list[ModelCheckpoint]
        Checkpoints encontrados. Vazio se o diretório não existir.

    Examples
    --------
    >>> discover_checkpoints(Path("inexistente"))
    []
    """
    if not checkpoints_dir.is_dir():
        return []

    classified = ((path, _classify_checkpoint(path)) for path in sorted(checkpoints_dir.iterdir()))
    found = [
        ModelCheckpoint(name=info[0], category=info[1], path=path)
        for path, info in classified
        if path.is_file() and info is not None and (model_names is None or info[0] in model_names)
    ]
    order = list(CATEGORY_LABELS)
    return sorted(found, key=lambda item: (order.index(item.category), item.name))


def _decode_label(value: Any) -> str:
    """Converte um rótulo (texto ou inteiro codificado, como no XGBoost) em texto."""
    if isinstance(value, (int, np.integer)):
        return ID_TO_LABEL[int(value)]
    return str(value)


def normalize_predictions(
    model: Any, raw_predictions: Any, raw_scores: np.ndarray | None
) -> tuple[list[str], np.ndarray | None]:
    """Converte a saída bruta de um modelo em rótulos de sentimento e scores ordenados.

    Modelos treinados com rótulos inteiros (XGBoost) têm ``ID_TO_LABEL`` aplicado; as colunas de
    probabilidade são reordenadas para ``SENTIMENT_CLASSES`` (classes ausentes recebem 0).

    Parameters
    ----------
    model : Any
        Modelo treinado (usa ``classes_`` quando disponível).
    raw_predictions : Any
        Saída de ``model.predict``.
    raw_scores : np.ndarray | None
        Saída de ``model.predict_proba``, se houver.

    Returns
    -------
    tuple[list[str], np.ndarray | None]
        Rótulos preditos e scores ``(n, 3)`` (ou ``None``).

    Examples
    --------
    >>> normalize_predictions(None, np.array([0, 2]), None)
    (['negativo', 'positivo'], None)
    """
    labels = [_decode_label(value) for value in np.asarray(raw_predictions)]
    if raw_scores is None:
        return labels, None

    classes = [_decode_label(value) for value in getattr(model, "classes_", SENTIMENT_CLASSES)]
    scores = np.zeros((len(labels), len(SENTIMENT_CLASSES)), dtype=float)
    for column, label in enumerate(classes):
        if label in SENTIMENT_CLASSES:
            scores[:, SENTIMENT_CLASSES.index(label)] = np.asarray(raw_scores)[:, column]
    return labels, scores


def _predict_texts(
    model: Any, features: Any, *, with_scores: bool
) -> tuple[list[str], np.ndarray | None, float]:
    """Prediz e mede o tempo médio por amostra (ms)."""
    started = time.perf_counter()
    raw_predictions = model.predict(features)
    elapsed_ms = (time.perf_counter() - started) * 1000
    raw_scores = (
        model.predict_proba(features) if with_scores and hasattr(model, "predict_proba") else None
    )
    labels, scores = normalize_predictions(model, raw_predictions, raw_scores)
    return labels, scores, elapsed_ms / max(len(labels), 1)


def _predict_with_llm(
    checkpoint: ModelCheckpoint,
    test_texts: Sequence[str],
    train_texts: Sequence[str],
    train_labels: Sequence[str],
    llm_indices: Sequence[int] | None,
) -> ModelPredictions:
    """Reconstrói o LLM da especificação, seleciona os exemplos few-shot e prediz o subconjunto."""
    specification = json.loads(checkpoint.path.read_text(encoding="utf-8"))
    model = create_classifier("llm", **specification["overrides"])
    model.fit(list(train_texts), list(train_labels))
    indices = list(llm_indices) if llm_indices is not None else list(range(len(test_texts)))
    labels, scores, per_sample = _predict_texts(
        model, [test_texts[index] for index in indices], with_scores=False
    )
    return ModelPredictions(
        checkpoint.name, checkpoint.category, indices, labels, scores, per_sample
    )


def _predict_with_supervised_model(
    checkpoint: ModelCheckpoint, test_texts: Sequence[str], vectorizer: Any | None
) -> ModelPredictions:
    """Carrega um modelo clássico (sobre TF-IDF) ou neural (sobre texto) e prediz todo o teste."""
    if checkpoint.category == CATEGORY_CLASSICAL:
        if vectorizer is None:
            raise ModelError("O vetorizador TF-IDF é obrigatório para avaliar modelos clássicos.")
        model = load_classifier(checkpoint.path)
        features: Any = vectorizer.transform(list(test_texts))
    else:
        model = load_classifier(checkpoint.path, backend="torch")
        features = list(test_texts)
    labels, scores, per_sample = _predict_texts(model, features, with_scores=True)
    return ModelPredictions(
        checkpoint.name,
        checkpoint.category,
        list(range(len(test_texts))),
        labels,
        scores,
        per_sample,
    )


def predict_with_checkpoint(
    checkpoint: ModelCheckpoint,
    test_texts: Sequence[str],
    *,
    vectorizer: Any | None = None,
    train_texts: Sequence[str] = (),
    train_labels: Sequence[str] = (),
    llm_indices: Sequence[int] | None = None,
) -> ModelPredictions:
    """Carrega um modelo e o aplica ao conjunto de teste.

    Parameters
    ----------
    checkpoint : ModelCheckpoint
        Modelo a aplicar.
    test_texts : Sequence[str]
        Textos do conjunto de teste.
    vectorizer : Any | None, optional
        Vetorizador TF-IDF ajustado no treino (obrigatório para a categoria clássica),
        by default None.
    train_texts : Sequence[str], optional
        Textos de treino, de onde o LLM seleciona os exemplos few-shot, by default ().
    train_labels : Sequence[str], optional
        Rótulos de treino alinhados a ``train_texts``, by default ().
    llm_indices : Sequence[int] | None, optional
        Subconjunto do teste enviado aos LLMs (custo de inferência), by default None (todo o teste).

    Returns
    -------
    ModelPredictions
        Predições do modelo.

    Raises
    ------
    ModelError
        Se um modelo clássico for pedido sem ``vectorizer``.
    """
    if checkpoint.category == CATEGORY_LLM:
        return _predict_with_llm(checkpoint, test_texts, train_texts, train_labels, llm_indices)
    return _predict_with_supervised_model(checkpoint, test_texts, vectorizer)
