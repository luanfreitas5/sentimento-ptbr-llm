"""Funções puras de agendamento de taxa de aprendizado (learning rate).

Cada função calcula a taxa de aprendizado para um passo de treino
específico, sem depender de nenhum framework de deep learning: podem ser
usadas em um laço de treino customizado em PyTorch (ex.: um laço próprio
para :class:`models.lstm.LSTMSentimentClassifier`/
:class:`models.cnn.CNNSentimentClassifier`, caso venham a expor controle
externo de época) sem acoplar este módulo a ``torch``. O fine-tuning de
Transformer (:class:`models.base.TransformerSentimentClassifier`) já usa o
agendador linear com aquecimento nativo do ``transformers``
(``get_linear_schedule_with_warmup``); as funções aqui evitam reimplementar
essa lógica para os demais modelos que venham a precisar de agendamento.
"""

import math

_MINIMUM_TOTAL_STEPS = 1


def _validate_schedule_arguments(step: int, total_steps: int, warmup_ratio: float) -> int:
    """Valida os argumentos comuns aos agendadores e retorna o número de passos de aquecimento.

    Parameters
    ----------
    step : int
        Passo de treino atual (indexado a partir de 0).
    total_steps : int
        Número total de passos de treino planejados.
    warmup_ratio : float
        Fração de ``total_steps`` usada para o aquecimento linear.

    Returns
    -------
    int
        Número de passos de aquecimento (``warmup_ratio * total_steps``).

    Raises
    ------
    ValueError
        Se ``step`` ou ``total_steps`` forem negativos/zero, ou se
        ``warmup_ratio`` estiver fora de ``[0, 1]``.
    """
    if total_steps < _MINIMUM_TOTAL_STEPS:
        raise ValueError(f"total_steps deve ser >= 1, recebido: {total_steps}")
    if step < 0:
        raise ValueError(f"step não pode ser negativo, recebido: {step}")
    if not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError(f"warmup_ratio deve estar em [0, 1], recebido: {warmup_ratio}")
    return int(warmup_ratio * total_steps)


def linear_warmup_decay(
    step: int,
    total_steps: int,
    *,
    base_lr: float,
    warmup_ratio: float = 0.1,
    end_lr: float = 0.0,
) -> float:
    """Calcula a taxa de aprendizado sob aquecimento linear seguido de decaimento linear.

    Cresce linearmente de 0 até ``base_lr`` durante o aquecimento e decai
    linearmente de ``base_lr`` até ``end_lr`` no restante do treino.

    Parameters
    ----------
    step : int
        Passo de treino atual (indexado a partir de 0).
    total_steps : int
        Número total de passos de treino planejados.
    base_lr : float
        Taxa de aprendizado de pico, atingida ao final do aquecimento.
    warmup_ratio : float, optional
        Fração de ``total_steps`` usada para o aquecimento, by default 0.1.
    end_lr : float, optional
        Taxa de aprendizado ao final do treino, by default 0.0.

    Returns
    -------
    float
        Taxa de aprendizado para ``step``.

    Examples
    --------
    >>> round(linear_warmup_decay(0, 100, base_lr=0.1, warmup_ratio=0.1), 4)
    0.0
    >>> round(linear_warmup_decay(10, 100, base_lr=0.1, warmup_ratio=0.1), 4)
    0.1
    >>> round(linear_warmup_decay(100, 100, base_lr=0.1, warmup_ratio=0.1, end_lr=0.0), 4)
    0.0
    """
    warmup_steps = _validate_schedule_arguments(step, total_steps, warmup_ratio)

    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step / warmup_steps)

    decay_steps = max(total_steps - warmup_steps, 1)
    decay_progress = min((step - warmup_steps) / decay_steps, 1.0)
    return base_lr + (end_lr - base_lr) * decay_progress


def cosine_warmup_decay(
    step: int,
    total_steps: int,
    *,
    base_lr: float,
    warmup_ratio: float = 0.1,
    min_lr: float = 0.0,
) -> float:
    """Calcula a taxa de aprendizado sob aquecimento linear seguido de decaimento em cosseno.

    Cresce linearmente de 0 até ``base_lr`` durante o aquecimento e decai
    seguindo meio ciclo de cosseno de ``base_lr`` até ``min_lr`` no restante
    do treino.

    Parameters
    ----------
    step : int
        Passo de treino atual (indexado a partir de 0).
    total_steps : int
        Número total de passos de treino planejados.
    base_lr : float
        Taxa de aprendizado de pico, atingida ao final do aquecimento.
    warmup_ratio : float, optional
        Fração de ``total_steps`` usada para o aquecimento, by default 0.1.
    min_lr : float, optional
        Taxa de aprendizado mínima, atingida ao final do treino, by default
        0.0.

    Returns
    -------
    float
        Taxa de aprendizado para ``step``.

    Examples
    --------
    >>> round(cosine_warmup_decay(0, 100, base_lr=0.1, warmup_ratio=0.1), 4)
    0.0
    >>> round(cosine_warmup_decay(10, 100, base_lr=0.1, warmup_ratio=0.1), 4)
    0.1
    >>> round(cosine_warmup_decay(100, 100, base_lr=0.1, warmup_ratio=0.1, min_lr=0.0), 4)
    0.0
    """
    warmup_steps = _validate_schedule_arguments(step, total_steps, warmup_ratio)

    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step / warmup_steps)

    decay_steps = max(total_steps - warmup_steps, 1)
    decay_progress = min((step - warmup_steps) / decay_steps, 1.0)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return min_lr + (base_lr - min_lr) * cosine_factor


def constant_with_warmup(
    step: int,
    *,
    base_lr: float,
    warmup_steps: int = 0,
) -> float:
    """Calcula a taxa de aprendizado constante após um aquecimento linear inicial.

    Parameters
    ----------
    step : int
        Passo de treino atual (indexado a partir de 0).
    base_lr : float
        Taxa de aprendizado constante, atingida ao final do aquecimento.
    warmup_steps : int, optional
        Número de passos de aquecimento linear, by default 0 (sem
        aquecimento: ``base_lr`` desde o primeiro passo).

    Returns
    -------
    float
        Taxa de aprendizado para ``step``.

    Raises
    ------
    ValueError
        Se ``step`` ou ``warmup_steps`` forem negativos.

    Examples
    --------
    >>> constant_with_warmup(0, base_lr=0.1, warmup_steps=10)
    0.0
    >>> constant_with_warmup(10, base_lr=0.1, warmup_steps=10)
    0.1
    >>> constant_with_warmup(0, base_lr=0.1)
    0.1
    """
    if step < 0:
        raise ValueError(f"step não pode ser negativo, recebido: {step}")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps não pode ser negativo, recebido: {warmup_steps}")

    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step / warmup_steps)
    return base_lr
