"""Embeddings contextuais (sentence embeddings) de textos em português brasileiro.

Implementa a Seção 4.4 do documento mestre: representação densa por
documento via mean pooling sobre os embeddings de subpalavra de um encoder
pré-treinado em português (BERTimbau/Sentence-BERT pt-BR, ver
``configs/model_params.yaml -> embeddings.contextual``), servindo de entrada
tanto para os classificadores clássicos quanto para o autoencoder de
``src/features/reduction.py``.

``transformers``/``torch`` são dependências pesadas e opcionais, ainda não
instaladas no projeto: o import ocorre de forma tardia, dentro de
:func:`load_contextual_encoder`, para que o restante do módulo permaneça
importável sem elas.
"""

import logging
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
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

from exceptions.model import ModelError
from utils.validation import validate_not_empty_collection

logger = logging.getLogger(__name__)


class ContextualEncoder(Protocol):
    """Interface mínima de um encoder de sentence embeddings contextuais.

    Permite injetar um encoder real (via :func:`load_contextual_encoder`) ou
    um dublê de teste em :func:`extract_contextual_embeddings`, sem
    acoplamento à implementação concreta (``transformers``/BERTimbau ou
    ``sentence-transformers``).
    """

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Codifica um lote de textos em vetores densos.

        Parameters
        ----------
        texts : Sequence[str]
            Lote de textos de entrada.

        Returns
        -------
        np.ndarray
            Matriz ``(len(texts), dimensao_do_encoder)`` de embeddings.
        """
        ...


class _TransformersMeanPoolingEncoder:
    """Encoder contextual via ``transformers``, com mean pooling sobre os subtokens.

    Implementa :class:`ContextualEncoder` em torno de um modelo Hugging Face
    (ex.: BERTimbau), aplicando mean pooling ponderado pela máscara de
    atenção — estratégia ``"mean"`` de ``configs/model_params.yaml ->
    embeddings.contextual.pooling_strategy``, mais robusta que usar apenas o
    token ``[CLS]`` para textos curtos como tweets.
    """

    def __init__(self, tokenizer: Any, model: Any, *, max_length: int, device: str) -> None:
        self._tokenizer = tokenizer
        self._model = model
        self._max_length = max_length
        self._device = device

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Codifica um lote de textos com mean pooling sobre a máscara de atenção.

        Parameters
        ----------
        texts : Sequence[str]
            Lote de textos de entrada.

        Returns
        -------
        np.ndarray
            Matriz ``(len(texts), hidden_size)`` de embeddings.
        """
        import torch  # type: ignore[reportMissingImports]

        encoded_input = self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        ).to(self._device)
        with torch.no_grad():
            model_output = self._model(**encoded_input)

        token_embeddings = model_output.last_hidden_state
        attention_mask = encoded_input["attention_mask"].unsqueeze(-1).float()
        summed_embeddings = (token_embeddings * attention_mask).sum(dim=1)
        token_counts = attention_mask.sum(dim=1).clamp(min=1e-9)
        mean_pooled_embeddings = summed_embeddings / token_counts
        return mean_pooled_embeddings.cpu().numpy()


def load_contextual_encoder(
    model_name: str = "neuralmind/bert-base-portuguese-cased",
    *,
    revision: str = "main",
    max_length: int = 128,
    device: str | None = None,
) -> ContextualEncoder:
    """Carrega um encoder contextual pré-treinado do Hugging Face Hub.

    Parameters
    ----------
    model_name : str, optional
        Nome do modelo no Hugging Face Hub, by default
        "neuralmind/bert-base-portuguese-cased" (BERTimbau base, ver
        ``configs/model_params.yaml -> embeddings.contextual.model_name``).
    revision : str, optional
        Revisão (branch, tag ou commit SHA) do modelo no Hugging Face Hub a
        ser baixada, by default "main". Fixar uma revisão específica evita
        que o modelo mude silenciosamente entre execuções (CWE-494).
    max_length : int, optional
        Comprimento máximo de subtokens por texto, truncando o excedente,
        by default 128.
    device : str | None, optional
        Dispositivo PyTorch (``"cpu"``, ``"cuda"``). Se ``None``, usa
        ``"cuda"`` quando disponível e ``"cpu"`` caso contrário, by default
        None.

    Returns
    -------
    ContextualEncoder
        Encoder pronto para uso em :func:`extract_contextual_embeddings`.

    Raises
    ------
    ModelError
        Se as bibliotecas ``transformers``/``torch`` não estiverem
        instaladas.

    Examples
    --------
    >>> load_contextual_encoder()  # doctest: +SKIP
    """
    try:
        import torch  # type: ignore[reportMissingImports]
        from transformers import AutoModel, AutoTokenizer  # type: ignore[reportMissingImports]
    except ImportError as exception:
        raise ModelError(
            "As bibliotecas 'transformers'/'torch' não estão instaladas. Instale com "
            "`uv add transformers torch` para extrair embeddings contextuais."
        ) from exception

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModel.from_pretrained(model_name, revision=revision).to(resolved_device).eval()
    logger.info(
        "Encoder contextual '%s' carregado no dispositivo '%s'.", model_name, resolved_device
    )
    return _TransformersMeanPoolingEncoder(
        tokenizer, model, max_length=max_length, device=resolved_device
    )


def extract_contextual_embeddings(
    dataframe: pl.DataFrame,
    encoder: ContextualEncoder,
    *,
    id_column: str = "id",
    text_column: str = "text",
    batch_size: int = 32,
) -> pl.DataFrame:
    """Extrai embeddings contextuais para cada documento de um corpus, em lotes.

    Parameters
    ----------
    dataframe : pl.DataFrame
        Corpus de entrada, contendo ao menos ``id_column`` e
        ``text_column``. Não vazio.
    encoder : ContextualEncoder
        Encoder contextual, via :func:`load_contextual_encoder`.
    id_column : str, optional
        Nome da coluna identificadora de cada documento, by default "id".
    text_column : str, optional
        Nome da coluna de texto, by default "text".
    batch_size : int, optional
        Quantidade de documentos codificados por chamada a
        ``encoder.encode``, by default 32 (ver ``configs/model_params.yaml
        -> embeddings.contextual.batch_size``).

    Returns
    -------
    pl.DataFrame
        DataFrame largo com ``id_column`` e uma coluna ``embedding_<i>`` por
        dimensão do encoder.

    Raises
    ------
    EmptyDatasetError
        Se ``dataframe`` estiver vazio.

    Examples
    --------
    >>> class _FakeEncoder:
    ...     def encode(self, texts):
    ...         return np.array([[float(len(text)), 0.0] for text in texts])
    >>> df = pl.DataFrame({"id": ["1", "2"], "text": ["oi", "bom dia"]})
    >>> extract_contextual_embeddings(df, _FakeEncoder(), batch_size=1).to_dicts()
    [{'id': '1', 'embedding_0': 2.0, 'embedding_1': 0.0},
    {'id': '2', 'embedding_0': 7.0, 'embedding_1': 0.0}]
    """
    validate_not_empty_collection(dataframe, collection_name="dataframe")

    document_ids = dataframe[id_column].to_list()
    texts = dataframe[text_column].to_list()
    batches = [texts[start : start + batch_size] for start in range(0, len(texts), batch_size)]

    progress_columns = (
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    encoded_batches: list[np.ndarray] = []
    with Progress(*progress_columns) as progress:
        task = progress.add_task("Extraindo embeddings contextuais", total=len(batches))
        for batch in batches:
            encoded_batches.append(encoder.encode(batch))
            progress.advance(task)

    embeddings = np.concatenate(encoded_batches, axis=0)
    dimension = embeddings.shape[1]
    embedding_columns = {
        f"embedding_{dimension_index}": embeddings[:, dimension_index]
        for dimension_index in range(dimension)
    }
    logger.info(
        "Embeddings contextuais extraídos: %d documento(s) em %d lote(s), dimensão %d.",
        len(texts),
        len(batches),
        dimension,
    )
    return pl.DataFrame({id_column: document_ids} | embedding_columns)
