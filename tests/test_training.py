"""Testes do módulo de treino de classificadores de sentimento (``src/training``)."""

from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from exceptions.data import EmptyDatasetError
from models.factory import create_classifier
from training.callbacks import (
    Callback,
    CallbackList,
    EarlyStoppingCallback,
    LoggingCallback,
    ModelCheckpointCallback,
)
from training.checkpoint import ModelCheckpoint
from training.cross_validation import (
    CrossValidationResult,
    compute_classification_score,
    run_stratified_cross_validation,
)
from training.early_stopping import EarlyStopping
from training.resume import TrainingCheckpointState, resume_training_state, save_training_state
from training.scheduler import constant_with_warmup, cosine_warmup_decay, linear_warmup_decay
from training.trainer import Trainer


def _binary_feature_dataset() -> tuple[np.ndarray, list[str]]:
    """Monta um dataset binário, linearmente separável, para testes de treino/CV."""
    X = np.array(
        [[1, 0], [2, 0], [3, 0], [0, 1], [0, 2], [0, 3]],
        dtype=np.float64,
    )
    y = ["positivo", "positivo", "positivo", "negativo", "negativo", "negativo"]
    return X, y


class _RecordingCallback:
    """Callback de teste: registra cada evento recebido, na ordem de chamada."""

    def __init__(self, *, should_stop: bool = False) -> None:
        self.should_stop = should_stop
        self.events: list[str] = []

    def on_train_begin(self) -> None:
        """Registra o evento de início do treino."""
        self.events.append("train_begin")

    def on_step_end(self, step_index: int, model: Any, metrics: dict[str, float]) -> bool:
        """Registra o evento de fim de passo e retorna ``should_stop``."""
        self.events.append(f"step_end:{step_index}")
        return self.should_stop

    def on_train_end(self) -> None:
        """Registra o evento de fim do treino."""
        self.events.append("train_end")


class TestEarlyStopping:
    """Testes do monitor de parada antecipada agnóstico ao framework."""

    def test_first_value_is_always_an_improvement(self) -> None:
        """O primeiro valor observado nunca sinaliza parada."""
        early_stopping = EarlyStopping(patience=1, mode="min")
        assert early_stopping.step(1.0) is False
        assert early_stopping.best_value == 1.0

    def test_signals_stop_after_patience_exhausted_mode_min(self) -> None:
        """Em modo 'min', valores crescentes esgotam a paciência e sinalizam parada."""
        early_stopping = EarlyStopping(patience=2, mode="min")
        assert early_stopping.step(1.0) is False
        assert early_stopping.step(1.1) is False
        assert early_stopping.step(1.2) is True

    def test_mode_max_treats_increase_as_improvement(self) -> None:
        """Em modo 'max', um valor maior reinicia a contagem de paciência."""
        early_stopping = EarlyStopping(patience=1, mode="max")
        assert early_stopping.step(0.5) is False
        assert early_stopping.step(0.9) is False
        assert early_stopping.best_value == 0.9

    def test_min_delta_requires_meaningful_improvement(self) -> None:
        """Uma melhora menor que ``min_delta`` não reinicia a contagem de paciência."""
        early_stopping = EarlyStopping(patience=1, mode="max", min_delta=0.1)
        assert early_stopping.step(0.5) is False
        assert early_stopping.step(0.55) is True

    def test_reset_clears_internal_state(self) -> None:
        """``reset`` permite reutilizar o mesmo monitor em um novo treino."""
        early_stopping = EarlyStopping(patience=1, mode="min")
        early_stopping.step(1.0)
        early_stopping.step(2.0)
        assert early_stopping.stopped is True
        early_stopping.reset()
        assert early_stopping.stopped is False
        assert early_stopping.best_value is None

    def test_raises_for_invalid_patience(self) -> None:
        """``patience`` menor que 1 deve levantar ``ValueError``."""
        with pytest.raises(ValueError, match="patience"):
            EarlyStopping(patience=0)

    def test_raises_for_negative_min_delta(self) -> None:
        """``min_delta`` negativo deve levantar ``ValueError``."""
        with pytest.raises(ValueError, match="min_delta"):
            EarlyStopping(min_delta=-0.1)


class TestScheduler:
    """Testes das funções puras de agendamento de taxa de aprendizado."""

    def test_linear_warmup_reaches_base_lr_at_end_of_warmup(self) -> None:
        """A taxa de aprendizado deve atingir ``base_lr`` ao final do aquecimento."""
        lr = linear_warmup_decay(10, 100, base_lr=0.1, warmup_ratio=0.1)
        assert lr == pytest.approx(0.1)

    def test_linear_decay_reaches_end_lr_at_final_step(self) -> None:
        """A taxa de aprendizado deve atingir ``end_lr`` no último passo."""
        lr = linear_warmup_decay(100, 100, base_lr=0.1, warmup_ratio=0.1, end_lr=0.0)
        assert lr == pytest.approx(0.0)

    def test_linear_warmup_is_zero_at_first_step(self) -> None:
        """A taxa de aprendizado deve começar em zero no primeiro passo do aquecimento."""
        lr = linear_warmup_decay(0, 100, base_lr=0.1, warmup_ratio=0.1)
        assert lr == pytest.approx(0.0)

    def test_cosine_warmup_decay_reaches_min_lr_at_final_step(self) -> None:
        """A taxa de aprendizado deve decair até ``min_lr`` no último passo."""
        lr = cosine_warmup_decay(100, 100, base_lr=0.1, warmup_ratio=0.1, min_lr=0.0)
        assert lr == pytest.approx(0.0, abs=1e-9)

    def test_constant_with_warmup_ramps_then_stays_constant(self) -> None:
        """A taxa de aprendizado deve crescer linearmente e depois permanecer constante."""
        assert constant_with_warmup(0, base_lr=0.1, warmup_steps=10) == pytest.approx(0.0)
        assert constant_with_warmup(5, base_lr=0.1, warmup_steps=10) == pytest.approx(0.05)
        assert constant_with_warmup(10, base_lr=0.1, warmup_steps=10) == pytest.approx(0.1)
        assert constant_with_warmup(20, base_lr=0.1, warmup_steps=10) == pytest.approx(0.1)

    def test_raises_for_total_steps_less_than_one(self) -> None:
        """``total_steps`` menor que 1 deve levantar ``ValueError``."""
        with pytest.raises(ValueError, match="total_steps"):
            linear_warmup_decay(0, 0, base_lr=0.1)

    def test_raises_for_warmup_ratio_out_of_range(self) -> None:
        """``warmup_ratio`` fora de ``[0, 1]`` deve levantar ``ValueError``."""
        with pytest.raises(ValueError, match="warmup_ratio"):
            cosine_warmup_decay(0, 10, base_lr=0.1, warmup_ratio=1.5)


class TestModelCheckpoint:
    """Testes do checkpointing de modelos a partir de uma métrica monitorada."""

    def test_step_saves_on_first_improvement(self, tmp_path: Path) -> None:
        """O primeiro valor observado deve sempre ser salvo."""
        checkpoint = ModelCheckpoint(tmp_path, monitor="f1_macro", mode="max")
        saved_path = checkpoint.step(create_classifier("naive_bayes"), 0.8, step_index=0)
        assert saved_path is not None
        assert saved_path.exists()
        assert checkpoint.best_value == 0.8

    def test_step_skips_when_no_improvement(self, tmp_path: Path) -> None:
        """Uma métrica pior que a melhor conhecida não deve gerar um novo arquivo."""
        checkpoint = ModelCheckpoint(tmp_path, monitor="f1_macro", mode="max")
        checkpoint.step(create_classifier("naive_bayes"), 0.8, step_index=0)
        saved_path = checkpoint.step(create_classifier("naive_bayes"), 0.5, step_index=1)
        assert saved_path is None
        assert checkpoint.best_value == 0.8

    def test_save_best_only_reuses_the_same_file(self, tmp_path: Path) -> None:
        """Com ``save_best_only=True``, cada melhora deve sobrescrever o mesmo arquivo."""
        checkpoint = ModelCheckpoint(tmp_path, monitor="f1_macro", mode="max", save_best_only=True)
        first_path = checkpoint.step(create_classifier("naive_bayes"), 0.5, step_index=0)
        second_path = checkpoint.step(create_classifier("naive_bayes"), 0.9, step_index=1)
        assert first_path == second_path


class TestCrossValidationResult:
    """Testes das estatísticas agregadas de :class:`CrossValidationResult`."""

    def test_empty_result_has_zeroed_statistics(self) -> None:
        """Um resultado sem dobras deve ter média, desvio padrão e IC 95% iguais a zero."""
        result = CrossValidationResult(scoring="f1_macro")
        assert result.mean == 0.0
        assert result.std == 0.0
        assert result.confidence_interval_95 == 0.0

    def test_confidence_interval_uses_1_96_std_over_sqrt_n(self) -> None:
        """O IC 95% deve seguir ``1.96 * desvio_padrao / sqrt(n_dobras)``."""
        result = CrossValidationResult(scoring="f1_macro", fold_scores=[0.6, 0.8])
        expected_ci = 1.96 * result.std / np.sqrt(2)
        assert result.confidence_interval_95 == pytest.approx(expected_ci)

    def test_single_fold_has_zero_std_and_ci(self) -> None:
        """Uma única dobra não permite estimar desvio padrão nem IC 95%."""
        result = CrossValidationResult(scoring="f1_macro", fold_scores=[0.7])
        assert result.std == 0.0
        assert result.confidence_interval_95 == 0.0


class TestComputeClassificationScore:
    """Testes do cálculo de métricas de classificação nomeadas."""

    def test_perfect_predictions_score_one(self) -> None:
        """Predições perfeitas devem produzir pontuação 1.0 em todas as métricas suportadas."""
        y_true = ["positivo", "negativo", "positivo"]
        assert compute_classification_score(y_true, y_true, scoring="accuracy") == 1.0
        assert compute_classification_score(y_true, y_true, scoring="f1_macro") == 1.0
        assert compute_classification_score(y_true, y_true, scoring="mcc") == 1.0

    def test_raises_for_unsupported_scoring(self) -> None:
        """Uma métrica desconhecida deve levantar ``ValueError``."""
        with pytest.raises(ValueError, match="não suportada"):
            compute_classification_score(["a"], ["a"], scoring="roc_auc")


class TestRunStratifiedCrossValidation:
    """Testes da validação cruzada estratificada com sementes fixas."""

    def test_returns_one_score_per_fold(self) -> None:
        """Sem interrupção antecipada, deve haver exatamente ``cv`` pontuações."""
        X, y = _binary_feature_dataset()
        result = run_stratified_cross_validation(
            partial(create_classifier, "naive_bayes"), X, y, cv=3, scoring="f1_macro"
        )
        assert len(result.fold_scores) == 3
        assert result.mean == pytest.approx(1.0)

    def test_on_fold_end_can_stop_early(self) -> None:
        """``on_fold_end`` retornando ``True`` deve interromper as dobras restantes."""
        X, y = _binary_feature_dataset()
        result = run_stratified_cross_validation(
            partial(create_classifier, "naive_bayes"),
            X,
            y,
            cv=3,
            on_fold_end=lambda fold_index, score, model: fold_index == 0,
        )
        assert len(result.fold_scores) == 1

    def test_raises_for_empty_dataset(self) -> None:
        """Um conjunto de dados vazio deve levantar ``EmptyDatasetError``."""
        with pytest.raises(EmptyDatasetError):
            run_stratified_cross_validation(partial(create_classifier, "naive_bayes"), [], [])


class TestCallbackList:
    """Testes do agregador de callbacks."""

    def test_notifies_all_callbacks_in_order(self) -> None:
        """Todos os callbacks devem ser notificados, na ordem em que foram registrados."""
        first = _RecordingCallback()
        second = _RecordingCallback()
        callback_list = CallbackList([first, second])

        callback_list.on_train_begin()
        callback_list.on_step_end(0, None, {})
        callback_list.on_train_end()

        assert first.events == ["train_begin", "step_end:0", "train_end"]
        assert second.events == ["train_begin", "step_end:0", "train_end"]

    def test_on_step_end_aggregates_stop_requests_with_or(self) -> None:
        """Se qualquer callback solicitar parada, o agregado deve retornar ``True``."""
        callback_list = CallbackList(
            [_RecordingCallback(should_stop=False), _RecordingCallback(should_stop=True)]
        )
        assert callback_list.on_step_end(0, None, {}) is True

    def test_empty_callback_list_never_stops(self) -> None:
        """Uma lista de callbacks vazia nunca deve solicitar parada."""
        callback_list: CallbackList = CallbackList()
        assert callback_list.on_step_end(0, None, {}) is False

    def test_satisfies_callback_protocol(self) -> None:
        """Os callbacks concretos do projeto devem satisfazer o Protocol ``Callback``."""
        assert isinstance(LoggingCallback(), Callback)
        assert isinstance(EarlyStoppingCallback("f1_macro"), Callback)


class TestEarlyStoppingCallback:
    """Testes do adaptador de :class:`EarlyStopping` à interface ``Callback``."""

    def test_signals_stop_after_patience_exhausted(self) -> None:
        """Deve sinalizar parada quando a métrica monitorada piora por ``patience`` passos."""
        callback = EarlyStoppingCallback("f1_macro", patience=2, mode="max")
        callback.on_train_begin()
        assert callback.on_step_end(0, None, {"f1_macro": 0.8}) is False
        assert callback.on_step_end(1, None, {"f1_macro": 0.7}) is False
        assert callback.on_step_end(2, None, {"f1_macro": 0.6}) is True

    def test_ignores_step_when_monitored_metric_is_missing(self) -> None:
        """Se a métrica monitorada estiver ausente, o passo deve ser ignorado sem erro."""
        callback = EarlyStoppingCallback("f1_macro", patience=1)
        assert callback.on_step_end(0, None, {"accuracy": 0.9}) is False


class TestModelCheckpointCallback:
    """Testes do adaptador de :class:`ModelCheckpoint` à interface ``Callback``."""

    def test_saves_model_when_metric_improves(self, tmp_path: Path) -> None:
        """Deve salvar o modelo quando a métrica monitorada melhora."""
        callback = ModelCheckpointCallback(tmp_path, "f1_macro", mode="max")
        should_stop = callback.on_step_end(0, create_classifier("naive_bayes"), {"f1_macro": 0.8})
        assert should_stop is False
        assert callback.checkpoint.best_path is not None
        assert callback.checkpoint.best_path.exists()

    def test_skips_when_model_is_none(self) -> None:
        """Sem uma instância de modelo, nenhum checkpoint deve ser salvo."""
        callback = ModelCheckpointCallback(Path("models/checkpoints/inexistente"), "f1_macro")
        assert callback.on_step_end(0, None, {"f1_macro": 0.9}) is False
        assert callback.checkpoint.best_path is None


class TestResume:
    """Testes de salvamento/retomada do estado completo de um treino."""

    def test_save_and_resume_roundtrip(self, tmp_path: Path) -> None:
        """O estado retomado deve reproduzir o modelo e os metadados salvos."""
        model = create_classifier("naive_bayes")
        X, y = _binary_feature_dataset()
        model.fit(X, y)
        state = TrainingCheckpointState(
            model=model, completed_steps=2, metrics_history=[{"f1_macro": 0.8}, {"f1_macro": 0.9}]
        )

        save_training_state(state, tmp_path)
        restored = resume_training_state(tmp_path)

        assert restored.completed_steps == 2
        assert restored.metrics_history == [{"f1_macro": 0.8}, {"f1_macro": 0.9}]
        assert list(restored.model.predict(X)) == list(model.predict(X))


class TestTrainer:
    """Testes do orquestrador genérico de treino."""

    def test_fit_trains_and_computes_validation_metrics(self) -> None:
        """``fit`` deve treinar o modelo e calcular métricas quando há dados de validação."""
        X, y = _binary_feature_dataset()
        trainer = Trainer(partial(create_classifier, "naive_bayes"))

        result = trainer.fit(X, y, X_val=X, y_val=y)

        assert result.metrics["f1_macro"] == pytest.approx(1.0)
        assert result.metrics["accuracy"] == pytest.approx(1.0)
        assert result.elapsed_seconds >= 0.0

    def test_fit_without_validation_data_returns_empty_metrics(self) -> None:
        """Sem ``X_val``/``y_val``, nenhuma métrica deve ser calculada."""
        X, y = _binary_feature_dataset()
        trainer = Trainer(partial(create_classifier, "naive_bayes"))

        result = trainer.fit(X, y)

        assert result.metrics == {}
        assert result.model is not None

    def test_fit_notifies_callbacks(self) -> None:
        """``fit`` deve notificar os callbacks registrados em cada etapa."""
        X, y = _binary_feature_dataset()
        recorder = _RecordingCallback()
        trainer = Trainer(partial(create_classifier, "naive_bayes"), callbacks=[recorder])

        trainer.fit(X, y)

        assert recorder.events == ["train_begin", "step_end:0", "train_end"]

    def test_fit_with_cross_validation_refits_on_all_data(self) -> None:
        """O modelo final deve ser reajustado sobre todo o conjunto de dados."""
        X, y = _binary_feature_dataset()
        trainer = Trainer(partial(create_classifier, "naive_bayes"))

        result = trainer.fit_with_cross_validation(X, y, cv=3, scoring="f1_macro")

        assert result.cross_validation is not None
        assert len(result.cross_validation.fold_scores) == 3
        assert result.metrics["f1_macro_mean"] == pytest.approx(1.0)
        assert list(result.model.predict(X)) == y

    def test_fit_with_cross_validation_stops_early_via_callback(self) -> None:
        """Um callback de parada antecipada deve interromper as dobras restantes."""
        X, y = _binary_feature_dataset()
        early_stopping_callback = EarlyStoppingCallback("f1_macro", patience=1, mode="min")
        trainer = Trainer(
            partial(create_classifier, "naive_bayes"), callbacks=[early_stopping_callback]
        )

        result = trainer.fit_with_cross_validation(X, y, cv=3, scoring="f1_macro")

        assert result.cross_validation is not None
        assert len(result.cross_validation.fold_scores) < 3
