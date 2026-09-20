"""Rotulagem de sentimento via LLM do Hugging Face (base ``tweets_data_huggingface``).

Implementa a seção ``huggingface`` de ``configs/labeling.yaml``: um LLM
instruct do Hugging Face Hub (ex.: ``meta-llama/Meta-Llama-3.1-8B-Instruct``)
é carregado localmente via ``transformers`` e classifica cada tweet com o
mesmo prompt versionado em ``prompts/`` usado pela base OpenAI, o que torna
as duas bases comparáveis. A saída é o JSON do prompt
(``{"label": ..., "probs": {...}}``); a confiança é a probabilidade que o
próprio modelo atribui ao rótulo (confiança verbalizada, não uma
probabilidade calibrada de token).

Decisões de execução:

* geração gulosa (``do_sample=False``) na primeira tentativa, o que a torna
  determinística; tweets cuja resposta não pôde ser interpretada são
  regerados com amostragem (``retry_temperature``, semente global fixada);
* a memória da GPU é liberada a cada lote (``gc.collect`` +
  ``torch.cuda.empty_cache``), em caso de *out of memory* (o lote é dividido
  ao meio) e ao descarregar o modelo (:func:`open_huggingface_classifier`);
* ``transformers``/``torch`` são dependências pesadas: o import ocorre de
  forma tardia, dentro de :func:`load_huggingface_llm`.
"""

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from constants.labels import SENTIMENT_CLASSES
from exceptions.model import ModelError
from labeling.incremental import BatchClassifier, LabelPrediction
from labeling.llm_response import build_labeling_prompt, parse_llm_label_response
from utils.memory import release_gpu_memory

logger = logging.getLogger(__name__)

DEFAULT_HUGGINGFACE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
_DTYPE_CHOICES: tuple[str, ...] = ("auto", "float16", "bfloat16", "float32")


@dataclass
class HuggingFaceLLM:
    """LLM do Hugging Face já carregado, pronto para gerar texto.

    Attributes
    ----------
    model : Any
        ``transformers.AutoModelForCausalLM`` em modo de avaliação.
    tokenizer : Any
        Tokenizador correspondente, com preenchimento à esquerda (necessário
        para geração em lote).
    device : str
        Dispositivo do modelo (``"cpu"``, ``"cuda"``, ...).
    model_name : str
        Nome do modelo no Hugging Face Hub.
    """

    model: Any
    tokenizer: Any
    device: str
    model_name: str


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


def load_huggingface_llm(
    model_name: str = DEFAULT_HUGGINGFACE_MODEL,
    *,
    device: str | None = "auto",
    dtype: str = "auto",
    load_in_4bit: bool = False,
    token: str | None = None,
    revision: str = "main",
) -> HuggingFaceLLM:
    """Carrega um LLM causal do Hugging Face Hub para rotulagem.

    Parameters
    ----------
    model_name : str, optional
        Modelo no Hugging Face Hub, by default :data:`DEFAULT_HUGGINGFACE_MODEL`.
    device : str | None, optional
        ``"cpu"``, ``"cuda"``, ``"cuda:0"`` ou ``"auto"``/``None`` (CUDA se disponível),
        by default "auto".
    dtype : {"auto", "float16", "bfloat16", "float32"}, optional
        Tipo numérico dos pesos, by default "auto".
    load_in_4bit : bool, optional
        Quantiza os pesos em 4 bits (exige ``bitsandbytes`` e GPU), útil para caber
        modelos de 7-9B em GPUs de 8-12 GB, by default False.
    token : str | None, optional
        Token do Hub para modelos com licença restrita (``SENTIMENTO_HUGGINGFACE_TOKEN``,
        ``.env``), by default None.
    revision : str, optional
        Branch, tag ou SHA do commit do modelo no Hub; fixe um SHA para garantir
        reprodutibilidade (o ``main`` pode mudar), by default "main".

    Returns
    -------
    HuggingFaceLLM
        Modelo e tokenizador prontos para :func:`create_huggingface_batch_classifier`.

    Raises
    ------
    ModelError
        Se ``transformers``/``torch`` não estiverem instalados, ``dtype`` for inválido,
        ``load_in_4bit`` for pedido sem ``bitsandbytes``/GPU ou o download falhar.

    Examples
    --------
    >>> load_huggingface_llm("Qwen/Qwen2.5-0.5B-Instruct", device="cpu")  # doctest: +SKIP
    """
    try:
        import torch  # type: ignore[reportMissingImports]
        from transformers import (  # type: ignore[reportMissingImports]
            AutoModelForCausalLM,
            AutoTokenizer,
        )
    except ImportError as exception:
        raise ModelError(
            "As bibliotecas 'transformers'/'torch' não estão instaladas. Instale com "
            "`make install-labeling` para rotular via LLM do Hugging Face."
        ) from exception

    resolved_device = _resolve_device(device, torch)
    resolved_dtype = _resolve_dtype(dtype, resolved_device, torch)
    load_kwargs: dict[str, Any] = {"dtype": resolved_dtype, "token": token}
    if load_in_4bit:
        if not resolved_device.startswith("cuda"):
            raise ModelError("load_in_4bit exige GPU CUDA (device='cuda').")
        try:
            from transformers import BitsAndBytesConfig  # type: ignore[reportMissingImports]

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=resolved_dtype
            )
        except ImportError as exception:
            raise ModelError(
                "load_in_4bit exige 'bitsandbytes'. Instale com `uv add bitsandbytes`."
            ) from exception

    logger.info(
        "Carregando o LLM '%s' (dispositivo=%s, dtype=%s, 4bit=%s)...",
        model_name,
        resolved_device,
        resolved_dtype,
        load_in_4bit,
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=token, revision=revision, padding_side="left"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, device_map={"": resolved_device}, **load_kwargs
        )
    except (OSError, ValueError, ImportError) as exception:
        raise ModelError(
            f"Não foi possível carregar o modelo '{model_name}': {exception}"
        ) from exception
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return HuggingFaceLLM(
        model=model, tokenizer=tokenizer, device=resolved_device, model_name=model_name
    )


def unload_huggingface_llm(llm: HuggingFaceLLM) -> None:
    """Descarrega o modelo e devolve a memória da GPU.

    Parameters
    ----------
    llm : HuggingFaceLLM
        Modelo a descarregar; não deve ser usado depois desta chamada.

    Examples
    --------
    >>> unload_huggingface_llm(llm)  # doctest: +SKIP
    """
    llm.model = None
    llm.tokenizer = None
    release_gpu_memory()
    logger.info("Modelo '%s' descarregado e memória da GPU liberada.", llm.model_name)


def _format_prompt(llm: HuggingFaceLLM, prompt: str) -> str:
    """Aplica o template de chat do modelo (quando existe) ao prompt do usuário."""
    if getattr(llm.tokenizer, "chat_template", None):
        return llm.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
    return prompt


def _generate_texts(
    llm: HuggingFaceLLM,
    prompts: Sequence[str],
    *,
    max_input_tokens: int,
    max_new_tokens: int,
    sampling_temperature: float | None,
) -> list[str]:
    """Gera a resposta de cada prompt em um único ``generate``, dividindo o lote em caso de OOM.

    Parameters
    ----------
    llm : HuggingFaceLLM
        Modelo carregado.
    prompts : Sequence[str]
        Prompts já formatados.
    max_input_tokens : int
        Máximo de tokens de entrada (o excedente é truncado à esquerda).
    max_new_tokens : int
        Máximo de tokens gerados por resposta.
    sampling_temperature : float | None
        ``None`` para geração gulosa; um valor > 0 ativa amostragem.

    Returns
    -------
    list[str]
        Respostas decodificadas (sem o prompt), na ordem de ``prompts``.

    Raises
    ------
    ModelError
        Se um único prompt ainda estourar a memória da GPU.
    """
    import torch  # type: ignore[reportMissingImports]

    try:
        inputs = llm.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        ).to(llm.model.device)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": llm.tokenizer.pad_token_id,
            "do_sample": sampling_temperature is not None,
        }
        if sampling_temperature is not None:
            generation_kwargs["temperature"] = sampling_temperature
        with torch.inference_mode():
            outputs = llm.model.generate(**inputs, **generation_kwargs)
        new_tokens = outputs[:, inputs["input_ids"].shape[1] :]
        return list(llm.tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    except torch.cuda.OutOfMemoryError as exception:
        release_gpu_memory()
        if len(prompts) == 1:
            raise ModelError(
                "Memória da GPU insuficiente até para um único tweet; use um modelo menor, "
                "load_in_4bit ou reduza max_input_tokens."
            ) from exception
        middle = len(prompts) // 2
        logger.warning(
            "Memória da GPU insuficiente para lote de %d; dividindo em dois.", len(prompts)
        )
        common = {
            "max_input_tokens": max_input_tokens,
            "max_new_tokens": max_new_tokens,
            "sampling_temperature": sampling_temperature,
        }
        return _generate_texts(llm, prompts[:middle], **common) + _generate_texts(
            llm, prompts[middle:], **common
        )


def create_huggingface_batch_classifier(
    llm: HuggingFaceLLM,
    prompt_template: str,
    *,
    max_new_tokens: int = 64,
    max_input_tokens: int = 1536,
    max_retries: int = 3,
    retry_temperature: float = 0.3,
    allowed_labels: Sequence[str] = SENTIMENT_CLASSES,
) -> BatchClassifier:
    """Cria o classificador de lotes da base ``tweets_data_huggingface``.

    Parameters
    ----------
    llm : HuggingFaceLLM
        Modelo carregado por :func:`load_huggingface_llm`.
    prompt_template : str
        Template do prompt (marcador ``{{TEXTO}}``).
    max_new_tokens : int, optional
        Máximo de tokens gerados por tweet (o JSON de resposta tem ~40 tokens), by default 64.
    max_input_tokens : int, optional
        Máximo de tokens de entrada (prompt + tweet), by default 1536.
    max_retries : int, optional
        Tentativas por tweet: a 1ª é gulosa; as demais amostram com
        ``retry_temperature``, by default 3.
    retry_temperature : float, optional
        Temperatura das tentativas de reparo, by default 0.3.
    allowed_labels : Sequence[str], optional
        Classes aceitas, by default :data:`constants.labels.SENTIMENT_CLASSES`.

    Returns
    -------
    BatchClassifier
        Função ``textos -> [(rótulo, confiança) | None]`` para
        :func:`labeling.incremental.run_incremental_labeling`; libera o cache da GPU a cada lote.

    Examples
    --------
    >>> classifier = create_huggingface_batch_classifier(llm, prompt_template)  # doctest: +SKIP
    """

    def classify_batch(texts: Sequence[str]) -> list[LabelPrediction | None]:
        prompts = [_format_prompt(llm, build_labeling_prompt(prompt_template, t)) for t in texts]
        results: list[LabelPrediction | None] = [None] * len(prompts)
        pending = list(range(len(prompts)))
        for attempt in range(max_retries):
            if not pending:
                break
            generated = _generate_texts(
                llm,
                [prompts[index] for index in pending],
                max_input_tokens=max_input_tokens,
                max_new_tokens=max_new_tokens,
                sampling_temperature=None if attempt == 0 else retry_temperature,
            )
            still_pending: list[int] = []
            for index, raw_response in zip(pending, generated, strict=True):
                parsed = parse_llm_label_response(raw_response, allowed_labels=allowed_labels)
                if parsed is None:
                    still_pending.append(index)
                else:
                    results[index] = parsed
            pending = still_pending
            release_gpu_memory()
        if pending:
            logger.warning("%d tweet(s) do lote sem resposta interpretável.", len(pending))
        return results

    return classify_batch


@contextmanager
def open_huggingface_classifier(
    prompt_template: str,
    *,
    model_name: str = DEFAULT_HUGGINGFACE_MODEL,
    device: str | None = "auto",
    dtype: str = "auto",
    load_in_4bit: bool = False,
    token: str | None = None,
    revision: str = "main",
    max_new_tokens: int = 64,
    max_input_tokens: int = 1536,
    max_retries: int = 3,
    retry_temperature: float = 0.3,
) -> Iterator[BatchClassifier]:
    """Carrega o LLM, entrega o classificador e, ao sair, descarrega o modelo (limpa a GPU).

    Parameters
    ----------
    prompt_template : str
        Template do prompt (marcador ``{{TEXTO}}``).
    model_name, device, dtype, load_in_4bit, token, revision
        Ver :func:`load_huggingface_llm`.
    max_new_tokens, max_input_tokens, max_retries, retry_temperature
        Ver :func:`create_huggingface_batch_classifier`.

    Yields
    ------
    BatchClassifier
        Classificador de lotes; válido apenas dentro do bloco ``with``.

    Examples
    --------
    >>> with open_huggingface_classifier("...{{TEXTO}}...") as classify:  # doctest: +SKIP
    ...     classify(["adorei"])
    """
    llm = load_huggingface_llm(
        model_name,
        device=device,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        token=token,
        revision=revision,
    )
    try:
        yield create_huggingface_batch_classifier(
            llm,
            prompt_template,
            max_new_tokens=max_new_tokens,
            max_input_tokens=max_input_tokens,
            max_retries=max_retries,
            retry_temperature=retry_temperature,
        )
    finally:
        unload_huggingface_llm(llm)
