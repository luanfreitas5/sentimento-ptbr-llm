"""Rotulagem de sentimento via modelo do Hugging Face (``transformers.pipeline``).

Implementa a seção ``huggingface`` de ``configs/labeling.yaml``: substitui a
rotulagem em cascata (``src/labeling/automatic.py``) por um único modelo de
linguagem pré-treinado em português brasileiro (por padrão,
``pysentimiento/bertweet-pt-sentiment``), classificado em lote via
``transformers.pipeline``. Produz diretamente ``sentiment_label`` e
``confidence_score`` para cada tweet do corpus — nenhum dos dois pode ficar
vazio: um rótulo bruto do pipeline fora de ``label_mapping`` é tratado como
falha (``DataValidationError``), nunca gravado silenciosamente, e
``confidence_score`` é sempre a probabilidade do rótulo previsto,
arredondada a 4 casas decimais.

``transformers``/``torch`` são dependências pesadas: o import ocorre de
forma tardia, dentro de :func:`load_huggingface_sentiment_pipeline`, mesmo
padrão de ``src/features/contextual_embeddings.py``.
"""

import logging
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from typing import Any, Protocol

import polars as pl
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from constants.labels import NEGATIVE_LABEL, NEUTRAL_LABEL, POSITIVE_LABEL
from exceptions.data import DataValidationError
from exceptions.model import ModelError
from utils.validation import validate_not_empty_collection

logger = logging.getLogger(__name__)

DEFAULT_HUGGINGFACE_MODEL = "pysentimiento/bertweet-pt-sentiment"

# Mapeamento padrão do rótulo bruto do pipeline (saída do modelo
# pysentimiento/bertweet-pt-sentiment: "POS"/"NEG"/"NEU") para as classes
# de sentimento em pt-BR usadas no projeto
# (``constants.labels.SENTIMENT_CLASSES``). Sobrescrito por
# ``configs/labeling.yaml -> huggingface.label_mapping``.
DEFAULT_LABEL_MAPPING: dict[str, str] = {
    "POS": POSITIVE_LABEL,
    "NEG": NEGATIVE_LABEL,
    "NEU": NEUTRAL_LABEL,
}


class SentimentPipeline(Protocol):
    """Interface mínima de um pipeline de classificação de sentimento em lote.

    Espelha a assinatura de ``transformers.pipeline("sentiment-analysis")``
    quando chamado com uma lista de textos: devolve, para cada um, o rótulo
    bruto previsto e sua pontuação. Permite injetar um dublê de teste em
    :func:`label_corpus_with_huggingface_pipeline` sem depender de
    ``transformers``/``torch`` instalados (ver ``tests/test_labeling.py``).
    """

    def __call__(self, texts: list[str], /) -> Sequence[Mapping[str, Any]]:
        """Classifica um lote de textos.

        Parameters
        ----------
        texts : list[str]
            Lote de textos de entrada, sempre repassado posicionalmente
            (parâmetro somente-posicional: ``transformers.pipeline`` nomeia
            o parâmetro equivalente ``inputs``, não ``texts``).

        Returns
        -------
        Sequence[Mapping[str, Any]]
            Um dicionário ``{"label": ..., "score": ...}`` por texto, na
            mesma ordem de entrada.
        """
        ...


def _resolve_pipeline_device(device: str | None, torch_module: Any) -> str:
    """Resolve o dispositivo do pipeline, escolhendo GPU automaticamente quando disponível.

    Parameters
    ----------
    device : str | None
        Dispositivo solicitado (``"cpu"``, ``"cuda"``, ``"cuda:0"``, ...).
        ``None`` ou ``"auto"`` delega a escolha a
        ``torch.cuda.is_available()``.
    torch_module : Any
        Módulo ``torch`` já importado (evita reimportar).

    Returns
    -------
    str
        Dispositivo resolvido, repassado a ``transformers.pipeline``.

    Examples
    --------
    >>> class _FakeTorchCuda:
    ...     @staticmethod
    ...     def is_available():
    ...         return False
    >>> class _FakeTorch:
    ...     cuda = _FakeTorchCuda()
    >>> _resolve_pipeline_device(None, _FakeTorch())
    'cpu'
    >>> _resolve_pipeline_device("cuda:1", _FakeTorch())
    'cuda:1'
    """
    if device is None or device == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    return device


def load_huggingface_sentiment_pipeline(
    model_name: str = DEFAULT_HUGGINGFACE_MODEL,
    *,
    device: str | None = None,
    batch_size: int = 32,
    max_length: int = 128,
) -> SentimentPipeline:
    """Carrega o pipeline de classificação de sentimento do Hugging Face Hub.

    Parameters
    ----------
    model_name : str, optional
        Nome do modelo no Hugging Face Hub, by default
        :data:`DEFAULT_HUGGINGFACE_MODEL` (``configs/labeling.yaml ->
        huggingface.model``).
    device : str | None, optional
        Dispositivo PyTorch (``"cpu"``, ``"cuda"``, ``"cuda:0"``, ...). Se
        ``None`` ou ``"auto"``, usa ``"cuda"`` quando disponível e ``"cpu"``
        caso contrário, by default None.
    batch_size : int, optional
        Tamanho do lote usado internamente pelo pipeline a cada chamada,
        by default 32.
    max_length : int, optional
        Comprimento máximo de subtokens por texto, truncando o excedente,
        by default 128.

    Returns
    -------
    SentimentPipeline
        Pipeline pronto para uso em
        :func:`label_corpus_with_huggingface_pipeline`.

    Raises
    ------
    ModelError
        Se as bibliotecas ``transformers``/``torch`` não estiverem
        instaladas.

    Examples
    --------
    >>> load_huggingface_sentiment_pipeline()  # doctest: +SKIP
    """
    try:
        import torch  # type: ignore[reportMissingImports]
        from transformers import (  # type: ignore[reportMissingImports]
            pipeline as build_transformers_pipeline,
        )
    except ImportError as exception:
        raise ModelError(
            "As bibliotecas 'transformers'/'torch' não estão instaladas. Instale com "
            "`uv add transformers torch` para rotular sentimento via Hugging Face."
        ) from exception

    resolved_device = _resolve_pipeline_device(device, torch)
    hf_pipeline = build_transformers_pipeline(
        task="text-classification",
        model=model_name,
        tokenizer=model_name,
        device=resolved_device,
        truncation=True,
        max_length=max_length,
        batch_size=batch_size,
    )
    logger.info(
        "Pipeline de sentimento Hugging Face '%s' carregado no dispositivo '%s'.",
        model_name,
        resolved_device,
    )
    return hf_pipeline


def _map_prediction_to_label(
    prediction: Mapping[str, Any], *, label_mapping: Mapping[str, str]
) -> tuple[str, float]:
    """Converte a predição bruta do pipeline em ``(sentiment_label, confidence_score)``.

    Parameters
    ----------
    prediction : Mapping[str, Any]
        Item de saída do pipeline (``{"label": ..., "score": ...}``).
    label_mapping : Mapping[str, str]
        Mapeamento do rótulo bruto do modelo para as classes de sentimento
        em pt-BR (``constants.labels.SENTIMENT_CLASSES``).

    Returns
    -------
    tuple[str, float]
        Par ``(sentiment_label, confidence_score)``, com a confiança
        arredondada a 4 casas decimais.

    Raises
    ------
    DataValidationError
        Se o rótulo bruto não constar em ``label_mapping`` — garante que
        ``sentiment_label`` nunca seja gravado vazio ou não classificado.

    Examples
    --------
    >>> _map_prediction_to_label(
    ...     {"label": "POS", "score": 0.987654}, label_mapping={"POS": "positivo"}
    ... )
    ('positivo', 0.9877)
    >>> _map_prediction_to_label(
    ...     {"label": "DESCONHECIDO", "score": 0.5}, label_mapping={"POS": "positivo"}
    ... )
    Traceback (most recent call last):
        ...
    exceptions.data.DataValidationError: ...
    """
    raw_label = str(prediction.get("label", ""))
    if raw_label not in label_mapping:
        raise DataValidationError(
            schema_name="huggingface_label_mapping",
            detail=(
                f"rótulo bruto '{raw_label}' retornado pelo pipeline não consta em "
                f"label_mapping {sorted(label_mapping)} (ver configs/labeling.yaml -> "
                "huggingface.label_mapping)"
            ),
        )
    return label_mapping[raw_label], round(float(prediction["score"]), 4)


def label_corpus_with_huggingface_pipeline(
    dataframe: pl.DataFrame,
    pipeline: SentimentPipeline,
    *,
    id_column: str = "id",
    text_column: str = "text_normalized",
    label_mapping: Mapping[str, str] | None = None,
    batch_size: int = 32,
    show_progress: bool = True,
) -> pl.DataFrame:
    """Classifica o sentimento de todo o corpus em lote, via pipeline Hugging Face.

    Substitui a rotulagem em cascata (``src/labeling/automatic.py`` +
    ``src/labeling/consensus.py``) como única fonte de ``sentiment_label``/
    ``confidence_score``: com um único modelo, não há concordância entre
    rotuladores a agregar — a confiança gravada é a probabilidade que o
    próprio modelo atribuiu ao rótulo previsto.

    Parameters
    ----------
    dataframe : pl.DataFrame
        Corpus de entrada, contendo ao menos ``id_column`` e
        ``text_column``. Não vazio.
    pipeline : SentimentPipeline
        Pipeline de classificação, via
        :func:`load_huggingface_sentiment_pipeline` (ou um dublê de teste
        que implemente :class:`SentimentPipeline`).
    id_column : str, optional
        Nome da coluna identificadora de cada amostra, by default "id".
    text_column : str, optional
        Nome da coluna de texto classificada pelo pipeline, by default
        "text_normalized" (produzida por
        ``src/pipelines/preprocessing.py``).
    label_mapping : Mapping[str, str] | None, optional
        Mapeamento do rótulo bruto do modelo para as classes de sentimento
        em pt-BR, by default None (usa :data:`DEFAULT_LABEL_MAPPING`).
    batch_size : int, optional
        Quantidade de textos classificados por chamada a ``pipeline``,
        by default 32 (``configs/labeling.yaml -> huggingface.batch_size``).
    show_progress : bool, optional
        Se ``True``, exibe uma barra de progresso no console, by default
        True.

    Returns
    -------
    pl.DataFrame
        DataFrame largo com ``id_column``, ``sentiment_label`` e
        ``confidence_score`` (em ``[0.0, 1.0]``, 4 casas decimais) — uma
        linha por amostra de ``dataframe``, nenhum valor nulo.

    Raises
    ------
    EmptyDatasetError
        Se ``dataframe`` estiver vazio.
    DataValidationError
        Se o pipeline devolver, para algum texto, um rótulo bruto fora de
        ``label_mapping`` (ver :func:`_map_prediction_to_label`).

    Examples
    --------
    >>> def _fake_pipeline(texts):
    ...     return [{"label": "POS", "score": 0.9} for _ in texts]
    >>> df = pl.DataFrame({"id": ["1"], "text_normalized": ["adorei o produto"]})
    >>> resultado = label_corpus_with_huggingface_pipeline(df, _fake_pipeline)
    >>> resultado["sentiment_label"].to_list()
    ['positivo']
    >>> resultado["confidence_score"].to_list()
    [0.9]
    """
    validate_not_empty_collection(dataframe, collection_name="dataframe")
    resolved_label_mapping = label_mapping or DEFAULT_LABEL_MAPPING

    row_ids = dataframe[id_column].to_list()
    texts = dataframe[text_column].to_list()
    batches = [texts[start : start + batch_size] for start in range(0, len(texts), batch_size)]

    sentiment_labels: list[str] = []
    confidence_scores: list[float] = []

    progress_columns = (
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    progress = Progress(*progress_columns) if show_progress else None
    with progress if progress is not None else nullcontext():
        task_id = (
            progress.add_task("Rotulando sentimento via Hugging Face", total=len(batches))
            if progress is not None
            else None
        )
        for batch in batches:
            for prediction in pipeline(batch):
                sentiment_label, confidence_score = _map_prediction_to_label(
                    prediction, label_mapping=resolved_label_mapping
                )
                sentiment_labels.append(sentiment_label)
                confidence_scores.append(confidence_score)
            if progress is not None and task_id is not None:
                progress.advance(task_id)

    result = pl.DataFrame(
        {
            id_column: row_ids,
            "sentiment_label": sentiment_labels,
            "confidence_score": confidence_scores,
        }
    )
    logger.info(
        "Rotulagem via Hugging Face concluída: %d amostra(s) em %d lote(s).",
        dataframe.height,
        len(batches),
    )
    return result
