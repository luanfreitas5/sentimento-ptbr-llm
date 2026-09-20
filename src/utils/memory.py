"""Gerenciamento de memória de GPU/CPU para modelos pesados.

Centraliza a liberação de memória usada pela rotulagem via LLM local
(``src/labeling/huggingface.py``), que carrega modelos de bilhões de
parâmetros e precisa devolver a VRAM entre lotes e ao final da execução.
"""

import gc
import logging

logger = logging.getLogger(__name__)


def release_gpu_memory() -> bool:
    """Libera memória de GPU: coleta o lixo do Python e esvazia o cache CUDA.

    Seguro em ambientes sem ``torch`` ou sem GPU (nesses casos apenas executa
    ``gc.collect()``). Referências vivas ao modelo/tensores devem ser
    removidas (``del``) pelo chamador antes, senão a VRAM não é devolvida.

    Returns
    -------
    bool
        ``True`` se o cache CUDA foi esvaziado; ``False`` se não havia GPU
        disponível (ou ``torch`` não está instalado).

    Examples
    --------
    >>> isinstance(release_gpu_memory(), bool)
    True
    """
    gc.collect()
    try:
        import torch  # type: ignore[reportMissingImports]
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    logger.debug("Cache de memória da GPU liberado.")
    return True
