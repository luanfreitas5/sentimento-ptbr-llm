"""Fixação de sementes aleatórias para reprodutibilidade.

``random_state`` isolado em uma única chamada de ``scikit-learn`` não é
suficiente para reprodutibilidade fim a fim: é preciso fixar todas as fontes
de aleatoriedade do processo (hash de strings, NumPy e, quando presente,
PyTorch) antes de qualquer operação estocástica.
"""

import logging
import os
import random

import numpy as np

from constants.defaults import DEFAULT_RANDOM_SEED

logger = logging.getLogger(__name__)


def seed_everything(seed: int = DEFAULT_RANDOM_SEED) -> None:
    """Fixa todas as fontes de aleatoriedade conhecidas do processo.

    Define ``PYTHONHASHSEED``, a semente do módulo ``random``, do NumPy e,
    se o PyTorch estiver instalado, também a sua semente e o modo
    determinístico de seus algoritmos.

    Parameters
    ----------
    seed : int, optional
        Valor da semente aleatória, by default :data:`DEFAULT_RANDOM_SEED`.

    Returns
    -------
    None

    Examples
    --------
    >>> seed_everything(42)
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        logger.debug("PyTorch não está instalado; semente aplicada apenas a random e NumPy.")
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    logger.debug("Semente %d aplicada a random, NumPy e PyTorch.", seed)
