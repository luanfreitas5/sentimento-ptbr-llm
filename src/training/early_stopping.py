"""Critério de parada antecipada, agnóstico ao modelo/framework de treino.

Complementa a parada antecipada já embutida em modelos iterativos como
:class:`models.base.TransformerSentimentClassifier` e
:class:`models.lstm.LSTMSentimentClassifier` (que monitoram a própria perda
de validação internamente): esta classe monitora qualquer sequência externa
de valores de métrica — por exemplo, uma pontuação por dobra em
``src/training/cross_validation.py`` — e não assume nada sobre como esses
valores são produzidos.
"""

import logging
from typing import Literal

logger = logging.getLogger(__name__)

EarlyStoppingMode = Literal["min", "max"]


class EarlyStopping:
    """Monitor de parada antecipada sobre uma sequência externa de valores de métrica.

    A cada chamada de :meth:`step`, compara o novo valor com o melhor valor
    observado até então; se ``patience`` chamadas consecutivas não
    produzirem melhora superior a ``min_delta``, sinaliza que o treino deve
    parar.

    Parameters
    ----------
    patience : int, optional
        Número de chamadas consecutivas sem melhora toleradas antes de
        sinalizar parada, by default 3.
    mode : {"min", "max"}, optional
        Se ``"min"``, uma melhora é uma diminuição do valor monitorado (ex.:
        perda de validação); se ``"max"``, uma melhora é um aumento (ex.:
        F1-macro), by default "min".
    min_delta : float, optional
        Variação mínima, em valor absoluto, para considerar uma mudança
        como melhora, by default 0.0.

    Examples
    --------
    >>> early_stopping = EarlyStopping(patience=2, mode="max")
    >>> early_stopping.step(0.70)
    False
    >>> early_stopping.step(0.65)
    False
    >>> early_stopping.step(0.60)
    True
    """

    def __init__(
        self,
        *,
        patience: int = 3,
        mode: EarlyStoppingMode = "min",
        min_delta: float = 0.0,
    ) -> None:
        if patience < 1:
            raise ValueError(f"patience deve ser >= 1, recebido: {patience}")
        if min_delta < 0:
            raise ValueError(f"min_delta não pode ser negativo, recebido: {min_delta}")

        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self.best_value: float | None = None
        self.best_step: int = -1
        self.wait: int = 0
        self.stopped: bool = False
        self._current_step: int = -1

    def _is_improvement(self, value: float) -> bool:
        """Indica se ``value`` representa uma melhora sobre :attr:`best_value`.

        Parameters
        ----------
        value : float
            Novo valor de métrica observado.

        Returns
        -------
        bool
            ``True`` se ``value`` melhora o melhor valor conhecido em mais
            de :attr:`min_delta`.
        """
        if self.best_value is None:
            return True
        if self.mode == "max":
            return value > self.best_value + self.min_delta
        return value < self.best_value - self.min_delta

    def step(self, value: float) -> bool:
        """Registra um novo valor de métrica e avalia se o treino deve parar.

        Parameters
        ----------
        value : float
            Novo valor observado da métrica monitorada.

        Returns
        -------
        bool
            ``True`` se ``patience`` chamadas consecutivas sem melhora
            foram atingidas (o treino deve parar); ``False`` caso
            contrário.

        Examples
        --------
        >>> early_stopping = EarlyStopping(patience=1, mode="min")
        >>> early_stopping.step(1.0)
        False
        >>> early_stopping.step(1.5)
        True
        """
        self._current_step += 1
        if self._is_improvement(value):
            self.best_value = value
            self.best_step = self._current_step
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped = True
                logger.info(
                    "Parada antecipada acionada no passo %d (melhor valor=%.4f no passo %d).",
                    self._current_step,
                    self.best_value,
                    self.best_step,
                )
        return self.stopped

    def reset(self) -> None:
        """Reinicia o estado interno do monitor, permitindo reutilizá-lo em um novo treino.

        Returns
        -------
        None
        """
        self.best_value = None
        self.best_step = -1
        self.wait = 0
        self.stopped = False
        self._current_step = -1
