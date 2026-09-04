"""Métodos de seleção de neurônios do SAE mais preditivos de uma variável-alvo.

Dado o vetor de ativações de um Sparse Autoencoder (``sae.py``) e um alvo
(sentimento binário ou uma nota contínua), seleciona o subconjunto de
neurônios mais relevante para explicar/prever o alvo — os candidatos que
depois serão interpretados em linguagem natural por
``interpret_neurons.py``. Implementa três estratégias intercambiáveis
(LASSO, correlação, score de separação) e um ponto de extensão para
métricas customizadas.
"""

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _fit_alpha_model(
    x_scaled: np.ndarray, target: np.ndarray, classification: bool, alpha: float, max_iter: int
) -> np.ndarray:
    """Ajusta o modelo linear (Lasso ou logística L1) para um dado ``alpha``
    e retorna os coeficientes."""
    model = (
        LogisticRegression(penalty="l1", solver="liblinear", C=1 / alpha, max_iter=max_iter)
        if classification
        else Lasso(alpha=alpha, max_iter=max_iter)
    )
    model.fit(x_scaled, target)
    return model.coef_.flatten()


def _log_alpha_search_result(
    verbose: bool, n_nonzero: int, n_select: int, alpha: float, total_start_time: float
) -> None:
    """Registra o resultado final da busca binária por ``alpha``."""
    if verbose and n_nonzero == n_select:
        logger.info(
            "Alpha=%.2e encontrado, produzindo exatamente %d feature(s) em %.2fs",
            alpha,
            n_select,
            time.time() - total_start_time,
        )
    if n_nonzero != n_select:
        logger.warning("Busca encerrada com %d feature(s) (alvo: %d)", n_nonzero, n_select)


def _search_alpha_for_n_select(
    x_scaled: np.ndarray,
    target: np.ndarray,
    n_select: int,
    classification: bool,
    max_iter: int,
    verbose: bool,
) -> tuple[float, np.ndarray]:
    """Busca por bisseção o ``alpha`` que produz exatamente ``n_select`` coeficientes não-nulos."""
    alpha_low, alpha_high = 1e-6, 1e4
    alpha = alpha_low
    coefficients = np.zeros(x_scaled.shape[1])
    n_nonzero = -1

    total_start_time = time.time()
    for iteration in range(20):  # número máximo de iterações da busca binária
        iteration_start_time = time.time()
        alpha = float(np.sqrt(alpha_low * alpha_high))

        coefficients = _fit_alpha_model(x_scaled, target, classification, alpha, max_iter)
        n_nonzero = int(np.sum(coefficients != 0))

        if verbose:
            logger.info(
                "Iteração LASSO %2d | alpha=%.2e | # features=%d | %.2fs",
                iteration,
                alpha,
                n_nonzero,
                time.time() - iteration_start_time,
            )

        if n_nonzero == n_select:
            break
        if n_nonzero < n_select:
            alpha_high = alpha
        else:
            alpha_low = alpha

    _log_alpha_search_result(verbose, n_nonzero, n_select, alpha, total_start_time)
    return alpha, coefficients


def select_neurons_lasso(
    activations: np.ndarray,
    target: np.ndarray,
    n_select: int,
    classification: bool = False,
    alpha: float | None = None,
    max_iter: int = 1000,
    verbose: bool = False,
) -> tuple[list[int], list[float]]:
    """Seleciona neurônios via modelo linear com regularização L1 (LASSO).

    A regularização L1 produz vetores de coeficientes esparsos, adequados
    para seleção de features. Quando ``alpha`` não é informado, faz busca
    binária pelo valor que produz exatamente ``n_select`` coeficientes
    não-nulos.

    Parameters
    ----------
    activations : np.ndarray
        Matriz de ativações do SAE, formato ``(n_amostras, n_neuronios)``.
    target : np.ndarray
        Variável-alvo, formato ``(n_amostras,)``.
    n_select : int
        Número de neurônios a selecionar.
    classification : bool, optional
        Se ``True``, usa regressão logística com penalidade L1 em vez de
        LASSO, by default False.
    alpha : float | None, optional
        Força de regularização; se ``None``, busca o valor que produz
        ``n_select`` features não-nulas, by default None.
    max_iter : int, optional
        Número máximo de iterações do otimizador, by default 1000.
    verbose : bool, optional
        Se registra o progresso da busca de ``alpha``, by default False.

    Returns
    -------
    tuple[list[int], list[float]]
        Índices dos neurônios selecionados e seus coeficientes brutos.
    """
    x_scaled = StandardScaler().fit_transform(activations)

    if alpha is not None:
        if verbose:
            logger.info("Usando alpha informado: %.2e", alpha)

        start_time = time.time()
        coefficients = _fit_alpha_model(x_scaled, target, classification, alpha, max_iter)
        if verbose:
            logger.info("Ajuste concluído em %.2fs", time.time() - start_time)

    else:
        _, coefficients = _search_alpha_for_n_select(
            x_scaled, target, n_select, classification, max_iter, verbose
        )

    sorted_indices = np.argsort(-np.abs(coefficients))[:n_select]
    selected_coefficients = coefficients[sorted_indices]

    return sorted_indices.tolist(), selected_coefficients.tolist()


def select_neurons_correlation(
    activations: np.ndarray, target: np.ndarray, n_select: int, **_kwargs: object
) -> tuple[list[int], list[float]]:
    """Seleciona os neurônios com maior correlação de Pearson (em módulo) com o alvo.

    Parameters
    ----------
    activations : np.ndarray
        Matriz de ativações do SAE, formato ``(n_amostras, n_neuronios)``.
    target : np.ndarray
        Variável-alvo, formato ``(n_amostras,)``.
    n_select : int
        Número de neurônios a selecionar.
    **_kwargs : object
        Ignorado; presente apenas para compatibilidade com a assinatura
        comum de :func:`select_neurons`.

    Returns
    -------
    tuple[list[int], list[float]]
        Índices dos neurônios selecionados e suas correlações brutas
        (com sinal). Neurônios "mortos" (ativação constante, comum dado a
        esparsidade do SAE) recebem correlação 0.0 em vez de ``nan``.
    """
    correlations = np.array(
        [
            pearsonr(activations[:, i], target)[0] if np.std(activations[:, i]) > 0 else 0.0
            for i in range(activations.shape[1])
        ]
    )

    sorted_indices = np.argsort(-np.abs(correlations))[:n_select]
    selected_correlations = correlations[sorted_indices]

    return sorted_indices.tolist(), selected_correlations.tolist()


def select_neurons_separation_score(
    activations: np.ndarray,
    target: np.ndarray,
    n_select: int,
    n_top_activating: int = 100,
    n_zero_activating: int | None = None,
    **_kwargs: object,
) -> tuple[list[int], list[float]]:
    """Seleciona neurônios pela separação entre ativações altas e ativações nulas.

    O score de separação é ``E[alvo | top-N ativações] - E[alvo | ativação
    zero]``: quanto maior em módulo, mais o neurônio distingue exemplos com
    alvos diferentes.

    Parameters
    ----------
    activations : np.ndarray
        Matriz de ativações do SAE, formato ``(n_amostras, n_neuronios)``.
    target : np.ndarray
        Variável-alvo, formato ``(n_amostras,)``.
    n_select : int
        Número de neurônios a selecionar.
    n_top_activating : int, optional
        Número de ativações mais altas usadas na média superior, by
        default 100.
    n_zero_activating : int | None, optional
        Se informado, amostra aleatoriamente esta quantidade de exemplos
        com ativação zero (em vez de usar todos), by default None.
    **_kwargs : object
        Ignorado; presente apenas para compatibilidade com a assinatura
        comum de :func:`select_neurons`.

    Returns
    -------
    tuple[list[int], list[float]]
        Índices dos neurônios selecionados e seus scores de separação.
    """
    scores = []
    for neuron_idx in range(activations.shape[1]):
        neuron_activations = activations[:, neuron_idx]
        positive_indices = np.where(neuron_activations >= 0)[0]
        n_positive = len(positive_indices)
        sorted_positive_indices = positive_indices[
            np.argsort(-neuron_activations[positive_indices])
        ]

        top_mean = (
            np.mean(target[sorted_positive_indices[:n_top_activating]]) if n_positive != 0 else 0
        )

        if n_positive < n_top_activating:
            logger.warning(
                "Apenas %d exemplo(s) com ativação positiva para o neurônio %d; "
                "usando todos os disponíveis para o score de separação",
                n_positive,
                neuron_idx,
            )

        zero_mask = neuron_activations == 0
        if n_zero_activating is not None:
            zero_indices = np.where(zero_mask)[0]
            random_zero_indices = np.random.default_rng().choice(
                zero_indices, size=n_zero_activating, replace=False
            )
            zero_mean = np.mean(target[random_zero_indices])
        else:
            zero_mean = np.mean(target[zero_mask])

        scores.append(top_mean - zero_mean)

    scores_array = np.array(scores)
    sorted_indices = np.argsort(-np.abs(scores_array))[:n_select]
    selected_scores = scores_array[sorted_indices]

    return sorted_indices.tolist(), selected_scores.tolist()


def select_neurons_custom(
    activations: np.ndarray,
    target: np.ndarray,
    n_select: int,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
) -> tuple[list[int], list[float]]:
    """Seleciona neurônios usando uma função de métrica customizada.

    Parameters
    ----------
    activations : np.ndarray
        Matriz de ativações do SAE, formato ``(n_amostras, n_neuronios)``.
    target : np.ndarray
        Variável-alvo, formato ``(n_amostras,)``.
    n_select : int
        Número de neurônios a selecionar.
    metric_fn : Callable[[np.ndarray, np.ndarray], float]
        Função que recebe (ativações do neurônio, alvo) e retorna um score
        escalar; neurônios com maior score são selecionados.

    Returns
    -------
    tuple[list[int], list[float]]
        Índices dos neurônios selecionados e seus scores.
    """
    scores = np.array([metric_fn(activations[:, i], target) for i in range(activations.shape[1])])

    sorted_indices = np.argsort(scores)[-n_select:]
    selected_scores = scores[sorted_indices]

    return sorted_indices.tolist(), selected_scores.tolist()


_SELECTION_METHODS: dict[str, Callable[..., tuple[list[int], list[float]]]] = {
    "lasso": select_neurons_lasso,
    "correlation": select_neurons_correlation,
    "separation_score": select_neurons_separation_score,
}


def select_neurons(
    activations: np.ndarray,
    target: np.ndarray,
    n_select: int,
    method: str = "lasso",
    classification: bool = False,
    **kwargs: Any,
) -> tuple[list[int], list[float]]:
    """Seleciona neurônios do SAE mais preditivos do alvo, pelo método escolhido.

    Parameters
    ----------
    activations : np.ndarray
        Matriz de ativações do SAE, formato ``(n_amostras, n_neuronios)``.
    target : np.ndarray
        Variável-alvo, formato ``(n_amostras,)``.
    n_select : int
        Número de neurônios a selecionar.
    method : str, optional
        Um de "lasso", "correlation", "separation_score" ou "custom", by
        default "lasso".
    classification : bool, optional
        Se esta é uma tarefa de classificação binária, by default False.
    **kwargs : Any
        Argumentos adicionais repassados ao método específico; o tipo
        concreto depende de ``method`` e é validado pela função de destino.

    Returns
    -------
    tuple[list[int], list[float]]
        Índices dos neurônios selecionados e seus scores brutos
        (coeficientes para "lasso", correlações para "correlation" etc.).

    Raises
    ------
    ValueError
        Se ``classification=True`` e o alvo tiver mais de duas classes, ou
        se ``method`` for desconhecido.
    """
    if classification and len(np.unique(target)) > 2:
        raise ValueError(
            "classification=True, mas a variável-alvo tem mais de duas classes. "
            "Classificação multiclasse não é suportada; converta para "
            "classificação binária um-contra-todos."
        )

    if method == "custom":
        if "metric_fn" not in kwargs:
            raise ValueError("É necessário informar metric_fn para o método 'custom'")
        return select_neurons_custom(
            activations=activations, target=target, n_select=n_select, **kwargs
        )

    selection_fn = _SELECTION_METHODS.get(method)
    if selection_fn is None:
        raise ValueError(f"Método de seleção desconhecido: {method}")

    return selection_fn(
        activations=activations,
        target=target,
        n_select=n_select,
        classification=classification,
        **kwargs,
    )
