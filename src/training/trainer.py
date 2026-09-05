"""Orquestrador genérico de treino, com logging, callbacks e MLflow opcional.

Compõe os demais módulos de ``src/training``: constrói um modelo via
``model_builder`` (tipicamente ``src/models/factory.py``), executa o
``fit`` atômico definido por :class:`models.base.SentimentClassifier`
(nenhum modelo do projeto expõe controle externo de época — modelos
iterativos como :class:`models.base.TransformerSentimentClassifier` e
:class:`models.lstm.LSTMSentimentClassifier` já gerenciam seu próprio laço
de treino e parada antecipada internamente) e, opcionalmente, delega a
:func:`training.cross_validation.run_stratified_cross_validation` para
avaliação robusta entre dobras. Os :class:`training.callbacks.Callback`
são notificados por "passo": uma única chamada de treino em :meth:`Trainer.fit`
ou uma dobra de validação cruzada em :meth:`Trainer.fit_with_cross_validation`.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from constants.defaults import DEFAULT_CROSS_VALIDATION_FOLDS, DEFAULT_RANDOM_SEED
from models.base import SentimentClassifier
from models.persistence import log_classifier_to_mlflow
from training.callbacks import Callback, CallbackList
from training.cross_validation import (
    CrossValidationResult,
    compute_classification_score,
    run_stratified_cross_validation,
)
from utils.seed import seed_everything
from utils.timing import measure_execution_time

logger = logging.getLogger(__name__)

_VALIDATION_METRICS: tuple[str, ...] = ("accuracy", "f1_macro", "mcc")


def _compute_validation_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, float]:
    """Calcula o conjunto padrão de métricas de avaliação de sentimento.

    Parameters
    ----------
    y_true : Sequence[str]
        Rótulos de sentimento verdadeiros.
    y_pred : Sequence[str]
        Rótulos de sentimento preditos, mesmo tamanho de ``y_true``.

    Returns
    -------
    dict[str, float]
        Dicionário com as chaves de :data:`_VALIDATION_METRICS`
        (acurácia, F1-macro e MCC — métricas definidas em ``CLAUDE.md``
        -> "Project-Specific Overrides").
    """
    return {
        metric_name: compute_classification_score(y_true, y_pred, scoring=metric_name)
        for metric_name in _VALIDATION_METRICS
    }


@dataclass
class TrainingResult:
    """Resultado de uma execução de treino via :class:`Trainer`.

    Parameters
    ----------
    model : Any
        Modelo treinado ao final da execução, pronto para
        ``predict``/``predict_proba``.
    metrics : dict[str, float]
        Métricas de avaliação (sobre o conjunto de validação, quando
        informado, ou agregadas da validação cruzada).
    elapsed_seconds : float
        Tempo total de execução, em segundos.
    cross_validation : CrossValidationResult | None
        Resultado detalhado por dobra, presente apenas quando o treino foi
        executado via :meth:`Trainer.fit_with_cross_validation`, by default
        None.
    """

    model: Any
    metrics: dict[str, float] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    cross_validation: CrossValidationResult | None = None


class Trainer:
    """Orquestrador genérico de treino de um classificador de sentimento.

    Parameters
    ----------
    model_builder : Callable[[], SentimentClassifier]
        Função que constrói uma nova instância de modelo não treinada a
        cada chamada (ex.: ``functools.partial(create_classifier,
        "logistic_regression")``, de ``src/models/factory.py``).
    callbacks : Sequence[Callback] | None, optional
        Callbacks notificados a cada passo de treino (ver
        ``src/training/callbacks.py``), by default None (nenhum callback).
    random_state : int, optional
        Semente aleatória de todo o processo, by default
        :data:`constants.defaults.DEFAULT_RANDOM_SEED`.
    track_with_mlflow : bool, optional
        Se ``True``, registra parâmetros, métricas e o modelo final em uma
        execução do MLflow (``configs/config.yaml -> experiment``), by
        default False.
    """

    def __init__(
        self,
        model_builder: Callable[[], SentimentClassifier],
        *,
        callbacks: Sequence[Callback] | None = None,
        random_state: int = DEFAULT_RANDOM_SEED,
        track_with_mlflow: bool = False,
    ) -> None:
        self.model_builder = model_builder
        self.callback_list = CallbackList(callbacks)
        self.random_state = random_state
        self.track_with_mlflow = track_with_mlflow

    def _log_to_mlflow(
        self, metrics: dict[str, float], model: Any, extra_params: dict[str, Any]
    ) -> None:
        """Registra parâmetros, métricas e o modelo final em uma execução do MLflow.

        Parameters
        ----------
        metrics : dict[str, float]
            Métricas a registrar.
        model : Any
            Modelo treinado a registrar como artefato.
        extra_params : dict[str, Any]
            Parâmetros adicionais a registrar (ex.: número de dobras).
        """
        import mlflow

        with mlflow.start_run():
            mlflow.log_param("random_state", self.random_state)
            for name, value in extra_params.items():
                mlflow.log_param(name, value)
            mlflow.log_metrics(metrics)
            log_classifier_to_mlflow(model, "modelo")

    def fit(
        self,
        X_train: Sequence[Any] | np.ndarray,
        y_train: Sequence[str],
        X_val: Sequence[Any] | np.ndarray | None = None,
        y_val: Sequence[str] | None = None,
    ) -> TrainingResult:
        """Treina um único modelo sobre ``X_train``/``y_train``.

        Parameters
        ----------
        X_train : Sequence[Any] | np.ndarray
            Amostras de treino.
        y_train : Sequence[str]
            Rótulos de sentimento de treino, mesmo tamanho de ``X_train``.
        X_val : Sequence[Any] | np.ndarray | None, optional
            Amostras de validação, usadas para calcular métricas ao final
            do treino, by default None (nenhuma métrica calculada).
        y_val : Sequence[str] | None, optional
            Rótulos de sentimento de validação, mesmo tamanho de ``X_val``,
            by default None.

        Returns
        -------
        TrainingResult
            Modelo treinado, métricas de validação (se ``X_val``/``y_val``
            informados) e tempo de execução.

        Examples
        --------
        >>> from functools import partial
        >>> from models.factory import create_classifier
        >>> trainer = Trainer(partial(create_classifier, "naive_bayes"))
        >>> resultado = trainer.fit(["bom", "ruim"], ["positivo", "negativo"])  # doctest: +SKIP
        """
        seed_everything(self.random_state)
        self.callback_list.on_train_begin()

        with measure_execution_time() as timing:
            model = self.model_builder()
            model.fit(X_train, y_train)

            metrics: dict[str, float] = {}
            if X_val is not None and y_val is not None:
                y_pred = model.predict(X_val)
                metrics = _compute_validation_metrics(y_val, y_pred)

            self.callback_list.on_step_end(0, model, metrics)

        self.callback_list.on_train_end()
        logger.info("Treino concluído em %.2fs.", timing.elapsed_seconds)

        if self.track_with_mlflow:
            self._log_to_mlflow(metrics, model, {"n_training_samples": len(X_train)})

        return TrainingResult(model=model, metrics=metrics, elapsed_seconds=timing.elapsed_seconds)

    def fit_with_cross_validation(
        self,
        X: Sequence[Any] | np.ndarray,
        y: Sequence[str],
        *,
        cv: int = DEFAULT_CROSS_VALIDATION_FOLDS,
        scoring: str = "f1_macro",
    ) -> TrainingResult:
        """Treina com validação cruzada estratificada e reajusta o modelo final em todos os dados.

        Cada dobra é tratada como um "passo" para efeito dos callbacks
        (:class:`training.callbacks.Callback`): um
        :class:`training.callbacks.EarlyStoppingCallback` pode interromper
        a validação cruzada antes da última dobra, e um
        :class:`training.callbacks.ModelCheckpointCallback` pode salvar o
        melhor modelo de dobra encontrado. Ao final, um novo modelo é
        treinado sobre o conjunto completo (``X``, ``y``) para uso em
        produção — a validação cruzada mede a estabilidade esperada da
        abordagem, não substitui o treino final (ver CLAUDE.md,
        "Rigorous evaluation").

        Parameters
        ----------
        X : Sequence[Any] | np.ndarray
            Amostras de entrada.
        y : Sequence[str]
            Rótulos de sentimento, mesmo tamanho de ``X``.
        cv : int, optional
            Número de dobras, by default
            :data:`constants.defaults.DEFAULT_CROSS_VALIDATION_FOLDS`.
        scoring : str, optional
            Métrica avaliada por dobra, by default "f1_macro".

        Returns
        -------
        TrainingResult
            Modelo reajustado sobre todos os dados, métricas agregadas da
            validação cruzada (média, desvio padrão e IC 95%) e o
            detalhamento por dobra em :attr:`TrainingResult.cross_validation`.

        Examples
        --------
        >>> from functools import partial
        >>> from models.factory import create_classifier
        >>> trainer = Trainer(partial(create_classifier, "naive_bayes"))
        >>> resultado = trainer.fit_with_cross_validation(
        ...     ["bom", "ruim", "ok", "péssimo"],
        ...     ["positivo", "negativo", "neutro", "negativo"],
        ...     cv=2,
        ... )  # doctest: +SKIP
        """
        seed_everything(self.random_state)
        self.callback_list.on_train_begin()

        with measure_execution_time() as timing:

            def _on_fold_end(
                fold_index: int, fold_score: float, fold_model: SentimentClassifier
            ) -> bool:
                return self.callback_list.on_step_end(fold_index, fold_model, {scoring: fold_score})

            cross_validation_result = run_stratified_cross_validation(
                self.model_builder,
                X,
                y,
                cv=cv,
                scoring=scoring,
                random_state=self.random_state,
                on_fold_end=_on_fold_end,
            )

            final_model = self.model_builder()
            final_model.fit(X, y)

        self.callback_list.on_train_end()
        metrics = {
            f"{scoring}_mean": cross_validation_result.mean,
            f"{scoring}_std": cross_validation_result.std,
            f"{scoring}_ci95": cross_validation_result.confidence_interval_95,
        }
        logger.info("Treino com validação cruzada concluído em %.2fs.", timing.elapsed_seconds)

        if self.track_with_mlflow:
            self._log_to_mlflow(metrics, final_model, {"cv": cv, "scoring": scoring})

        return TrainingResult(
            model=final_model,
            metrics=metrics,
            elapsed_seconds=timing.elapsed_seconds,
            cross_validation=cross_validation_result,
        )
