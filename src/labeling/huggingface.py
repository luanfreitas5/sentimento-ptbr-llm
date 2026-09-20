"""Rotulagem de sentimento via modelo do Hugging Face (base ``tweets_data_huggingface``).

Implementa a seção ``huggingface`` de ``configs/labeling.yaml``: um classificador
de sequência do Hugging Face Hub (ex.: ``pysentimento/bertweet-pt-sentiment``,
BERTweet-pt ajustado para sentimento em tweets) é carregado localmente via
``transformers`` e classifica cada tweet em ``negativo``/``neutro``/``positivo``.

Diferente da base OpenAI (LLM gerativo guiado por prompt), este modelo é um
*encoder* já ajustado para a tarefa: não há prompt nem geração de texto. A
confiança é a probabilidade ``softmax`` que o próprio modelo atribui à classe
escolhida (probabilidade de fato, não confiança verbalizada).

Decisões de execução:

* inferência determinística (``model.eval()`` + ``torch.inference_mode``), sem
  amostragem: o mesmo tweet sempre recebe o mesmo rótulo;
* o mapeamento entre os índices do modelo e as classes do projeto é lido de
  ``model.config.id2label`` (``NEG``/``NEU``/``POS``) e validado ao carregar;
* a memória da GPU é liberada a cada lote (``gc.collect`` +
  ``torch.cuda.empty_cache``), em caso de *out of memory* (o lote é dividido
  ao meio) e ao descarregar o modelo (:func:`open_huggingface_classifier`);
* ``transformers``/``torch`` são dependências pesadas: o import ocorre de
  forma tardia, dentro de :func:`load_huggingface_model`.
"""

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from constants.labels import NEGATIVE_LABEL, NEUTRAL_LABEL, POSITIVE_LABEL
from exceptions.model import ModelError
from labeling.incremental import BatchClassifier, LabelPrediction
from utils.memory import release_gpu_memory

logger = logging.getLogger(__name__)

DEFAULT_HUGGINGFACE_MODEL = "pysentimento/bertweet-pt-sentiment"
_DTYPE_CHOICES: tuple[str, ...] = ("auto", "float16", "bfloat16", "float32")

# Nomes de classe usados pelos modelos de sentimento do Hub -> classe do projeto.
_MODEL_LABEL_TO_PROJECT_LABEL: dict[str, str] = {
    "neg": NEGATIVE_LABEL,
    "negative": NEGATIVE_LABEL,
    "negativo": NEGATIVE_LABEL,
    "neu": NEUTRAL_LABEL,
    "neutral": NEUTRAL_LABEL,
    "neutro": NEUTRAL_LABEL,
    "pos": POSITIVE_LABEL,
    "positive": POSITIVE_LABEL,
    "positivo": POSITIVE_LABEL,
}


@dataclass
class HuggingFaceModel:
    """Classificador de sequência do Hugging Face já carregado, pronto para inferência.

    Attributes
    ----------
    model : Any
        ``transformers.AutoModelForSequenceClassification`` em modo de avaliação.
    tokenizer : Any
        Tokenizador correspondente.
    device : str
        Dispositivo do modelo (``"cpu"``, ``"cuda"``, ...).
    model_name : str
        Nome do modelo no Hugging Face Hub.
    class_labels : list[str]
        Classe do projeto (``negativo``/``neutro``/``positivo``) de cada saída
        (logit) do modelo, na ordem dos índices do modelo.
    """

    model: Any
    tokenizer: Any
    device: str
    model_name: str
    class_labels: list[str]


def _resolve_device(device: str | None, torch_module: Any) -> str:
    """Resolve o dispositivo: ``None``/``"auto"`` escolhe CUDA quando disponível.

    Examples
    --------
    >>> class _Cuda:
    ...     @staticmethod
    ...     def is_available():
    ...         return False
    >>> class _Torch:
    ...     cuda = _Cuda()
    >>> _resolve_device("auto", _Torch())
    'cpu'
    >>> _resolve_device("cuda:1", _Torch())
    'cuda:1'
    """
    if device is None or device == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    return device


def _resolve_dtype(dtype: str, device: str, torch_module: Any) -> Any:
    """Converte o nome do tipo numérico em ``torch.dtype``; ``"auto"`` depende do dispositivo.

    Raises
    ------
    ModelError
        Se ``dtype`` não for um de ``auto``, ``float16``, ``bfloat16`` ou ``float32``.
    """
    if dtype not in _DTYPE_CHOICES:
        raise ModelError(f"dtype inválido '{dtype}'; valores aceitos: {list(_DTYPE_CHOICES)}")
    if dtype != "auto":
        return getattr(torch_module, dtype)
    if not device.startswith("cuda"):
        return torch_module.float32
    return torch_module.bfloat16 if torch_module.cuda.is_bf16_supported() else torch_module.float16


def _map_model_labels(id2label: dict[int, str], model_name: str) -> list[str]:
    """Traduz ``model.config.id2label`` para as classes do projeto, na ordem dos índices.

    Parameters
    ----------
    id2label : dict[int, str]
        Mapa índice -> nome de classe do modelo (ex.: ``{0: "NEG", 1: "NEU", 2: "POS"}``).
    model_name : str
        Nome do modelo, usado apenas na mensagem de erro.

    Returns
    -------
    list[str]
        Classe do projeto para cada índice de saída do modelo.

    Raises
    ------
    ModelError
        Se algum nome de classe não for reconhecido ou as três classes do projeto
        não estiverem todas presentes.

    Examples
    --------
    >>> _map_model_labels({0: "NEG", 1: "NEU", 2: "POS"}, "m")
    ['negativo', 'neutro', 'positivo']
    """
    class_labels: list[str] = []
    for index in sorted(id2label):
        project_label = _MODEL_LABEL_TO_PROJECT_LABEL.get(str(id2label[index]).strip().lower())
        if project_label is None:
            raise ModelError(
                f"O modelo '{model_name}' tem a classe '{id2label[index]}', que não corresponde "
                f"a negativo/neutro/positivo; id2label={id2label}"
            )
        class_labels.append(project_label)
    if sorted(class_labels) != sorted([NEGATIVE_LABEL, NEUTRAL_LABEL, POSITIVE_LABEL]):
        raise ModelError(
            f"O modelo '{model_name}' deve ter exatamente as classes negativo/neutro/positivo; "
            f"recebido: {class_labels}"
        )
    return class_labels


def load_huggingface_model(
    model_name: str = DEFAULT_HUGGINGFACE_MODEL,
    *,
    device: str | None = "auto",
    dtype: str = "auto",
    token: str | None = None,
    revision: str = "main",
) -> HuggingFaceModel:
    """Carrega um classificador de sentimento do Hugging Face Hub para rotulagem.

    Parameters
    ----------
    model_name : str, optional
        Modelo no Hugging Face Hub, by default :data:`DEFAULT_HUGGINGFACE_MODEL`.
    device : str | None, optional
        ``"cpu"``, ``"cuda"``, ``"cuda:0"`` ou ``"auto"``/``None`` (CUDA se disponível),
        by default "auto".
    dtype : {"auto", "float16", "bfloat16", "float32"}, optional
        Tipo numérico dos pesos, by default "auto".
    token : str | None, optional
        Token do Hub para modelos com acesso restrito (``SENTIMENTO_HUGGINGFACE_TOKEN``,
        ``.env``); o modelo padrão é público, by default None.
    revision : str, optional
        Branch, tag ou SHA do commit do modelo no Hub; fixe um SHA para garantir
        reprodutibilidade (o ``main`` pode mudar), by default "main".

    Returns
    -------
    HuggingFaceModel
        Modelo e tokenizador prontos para :func:`create_huggingface_batch_classifier`.

    Raises
    ------
    ModelError
        Se ``transformers``/``torch`` não estiverem instalados, ``dtype`` for inválido,
        o download falhar ou as classes do modelo não forem negativo/neutro/positivo.

    Examples
    --------
    >>> load_huggingface_model(device="cpu")  # doctest: +SKIP
    """
    try:
        import torch  # type: ignore[reportMissingImports]
        from transformers import (  # type: ignore[reportMissingImports]
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except ImportError as exception:
        raise ModelError(
            "As bibliotecas 'transformers'/'torch' não estão instaladas. Instale com "
            "`make install-labeling` para rotular via modelo do Hugging Face."
        ) from exception

    resolved_device = _resolve_device(device, torch)
    resolved_dtype = _resolve_dtype(dtype, resolved_device, torch)
    logger.info(
        "Carregando o modelo '%s' (dispositivo=%s, dtype=%s)...",
        model_name,
        resolved_device,
        resolved_dtype,
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, revision=revision)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, revision=revision, token=token, dtype=resolved_dtype
        )
    except (OSError, ValueError, ImportError) as exception:
        raise ModelError(
            f"Não foi possível carregar o modelo '{model_name}': {exception}"
        ) from exception
    class_labels = _map_model_labels(dict(model.config.id2label), model_name)
    model.to(resolved_device)
    model.eval()
    return HuggingFaceModel(
        model=model,
        tokenizer=tokenizer,
        device=resolved_device,
        model_name=model_name,
        class_labels=class_labels,
    )


def unload_huggingface_model(hf_model: HuggingFaceModel) -> None:
    """Descarrega o modelo e devolve a memória da GPU.

    Parameters
    ----------
    hf_model : HuggingFaceModel
        Modelo a descarregar; não deve ser usado depois desta chamada.

    Examples
    --------
    >>> unload_huggingface_model(hf_model)  # doctest: +SKIP
    """
    hf_model.model = None
    hf_model.tokenizer = None
    release_gpu_memory()
    logger.info("Modelo '%s' descarregado e memória da GPU liberada.", hf_model.model_name)


def _predict_probabilities(
    hf_model: HuggingFaceModel, texts: Sequence[str], *, max_input_tokens: int
) -> list[list[float]]:
    """Calcula as probabilidades ``softmax`` de cada texto, dividindo o lote em caso de OOM.

    Parameters
    ----------
    hf_model : HuggingFaceModel
        Modelo carregado.
    texts : Sequence[str]
        Textos (já sanitizados) a classificar.
    max_input_tokens : int
        Máximo de tokens de entrada (o excedente é truncado).

    Returns
    -------
    list[list[float]]
        Para cada texto, a probabilidade de cada saída do modelo (ordem de
        :attr:`HuggingFaceModel.class_labels`).

    Raises
    ------
    ModelError
        Se um único texto ainda estourar a memória da GPU.
    """
    import torch  # type: ignore[reportMissingImports]

    try:
        inputs = hf_model.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        ).to(hf_model.model.device)
        with torch.inference_mode():
            logits = hf_model.model(**inputs).logits
        return torch.softmax(logits.float(), dim=-1).cpu().tolist()
    except torch.cuda.OutOfMemoryError as exception:
        release_gpu_memory()
        if len(texts) == 1:
            raise ModelError(
                "Memória da GPU insuficiente até para um único tweet; reduza max_input_tokens "
                "ou use device='cpu'."
            ) from exception
        middle = len(texts) // 2
        logger.warning(
            "Memória da GPU insuficiente para lote de %d; dividindo em dois.", len(texts)
        )
        return _predict_probabilities(
            hf_model, texts[:middle], max_input_tokens=max_input_tokens
        ) + _predict_probabilities(hf_model, texts[middle:], max_input_tokens=max_input_tokens)


def create_huggingface_batch_classifier(
    hf_model: HuggingFaceModel, *, max_input_tokens: int = 128
) -> BatchClassifier:
    """Cria o classificador de lotes da base ``tweets_data_huggingface``.

    Parameters
    ----------
    hf_model : HuggingFaceModel
        Modelo carregado por :func:`load_huggingface_model`.
    max_input_tokens : int, optional
        Máximo de tokens de entrada; o BERTweet-pt aceita até 128, by default 128.

    Returns
    -------
    BatchClassifier
        Função ``textos -> [(rótulo, probabilidade)]`` para
        :func:`labeling.incremental.run_incremental_labeling`; libera o cache da GPU a cada lote.

    Examples
    --------
    >>> classifier = create_huggingface_batch_classifier(hf_model)  # doctest: +SKIP
    """

    def classify_batch(texts: Sequence[str]) -> list[LabelPrediction | None]:
        probabilities = _predict_probabilities(hf_model, texts, max_input_tokens=max_input_tokens)
        release_gpu_memory()
        results: list[LabelPrediction | None] = []
        for text_probabilities in probabilities:
            best_index = max(range(len(text_probabilities)), key=text_probabilities.__getitem__)
            results.append((hf_model.class_labels[best_index], text_probabilities[best_index]))
        return results

    return classify_batch


@contextmanager
def open_huggingface_classifier(
    *,
    model_name: str = DEFAULT_HUGGINGFACE_MODEL,
    device: str | None = "auto",
    dtype: str = "auto",
    token: str | None = None,
    revision: str = "main",
    max_input_tokens: int = 128,
) -> Iterator[BatchClassifier]:
    """Carrega o modelo, entrega o classificador e, ao sair, descarrega o modelo (limpa a GPU).

    Parameters
    ----------
    model_name, device, dtype, token, revision
        Ver :func:`load_huggingface_model`.
    max_input_tokens
        Ver :func:`create_huggingface_batch_classifier`.

    Yields
    ------
    BatchClassifier
        Classificador de lotes; válido apenas dentro do bloco ``with``.

    Examples
    --------
    >>> with open_huggingface_classifier() as classify:  # doctest: +SKIP
    ...     classify(["adorei"])
    """
    hf_model = load_huggingface_model(
        model_name, device=device, dtype=dtype, token=token, revision=revision
    )
    try:
        yield create_huggingface_batch_classifier(hf_model, max_input_tokens=max_input_tokens)
    finally:
        unload_huggingface_model(hf_model)
