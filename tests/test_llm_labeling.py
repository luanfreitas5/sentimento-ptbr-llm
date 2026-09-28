"""Testes da rotulagem por LLM: prompt/resposta, checkpoint, execução incremental e fontes."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from evaluation.llm_comparison import build_comparison_frame
from exceptions.base import ProjectError
from exceptions.configuration import MissingEnvironmentVariableError
from exceptions.data import DataValidationError, EmptyDatasetError
from exceptions.model import ModelError
from exceptions.pipeline import IncompleteLabelingError
from labeling import huggingface, openai_labeler
from labeling import rate_limiter as rate_limiter_module
from labeling.checkpoint import (
    append_labeling_checkpoint,
    build_checkpoint_path,
    read_labeling_checkpoint,
)
from labeling.huggingface import (
    HuggingFaceModel,
    _map_model_labels,
    _resolve_device,
    _resolve_dtype,
)
from labeling.incremental import run_incremental_labeling
from labeling.llm_response import (
    _extract_confidence,
    build_labeling_prompt,
    parse_llm_label_response,
)
from labeling.openai_labeler import _classify_single_text, create_openai_batch_classifier
from labeling.rate_limiter import (
    MAX_COOLDOWN_SECONDS,
    RateLimiter,
    get_retry_after_seconds,
    is_rate_limit_error,
)
from schemas.labeling import validate_labeled_source
from utils.memory import release_gpu_memory


class TestBuildLabelingPrompt:
    """Testes de :func:`labeling.llm_response.build_labeling_prompt`."""

    def test_substitutes_text_placeholder(self) -> None:
        """O marcador ``{{TEXTO}}`` deve ser substituído pelo texto do tweet."""
        assert build_labeling_prompt('Tweet: "{{TEXTO}}"', "adorei") == 'Tweet: "adorei"'

    def test_keeps_template_without_placeholder_unchanged(self) -> None:
        """Um template sem marcador não deve ser alterado."""
        assert build_labeling_prompt("sem marcador", "adorei") == "sem marcador"


class TestParseLlmLabelResponse:
    """Testes de :func:`labeling.llm_response.parse_llm_label_response`."""

    def test_parses_label_and_confidence_fields(self) -> None:
        """Uma resposta com ``label``/``confidence`` explícitos deve ser interpretada direto."""
        assert parse_llm_label_response('{"label": "positivo", "confidence": 0.9}') == (
            "positivo",
            0.9,
        )

    def test_parses_confidence_from_probs_when_present(self) -> None:
        """Quando ``probs`` está presente, a confiança deve vir de ``probs[label]``."""
        raw_response = '{"label":"negativo","probs":{"positivo":0.1,"negativo":0.8,"neutro":0.1}}'
        assert parse_llm_label_response(raw_response) == ("negativo", 0.8)

    def test_defaults_confidence_to_one_when_absent(self) -> None:
        """Sem ``confidence`` nem ``probs``, a confiança deve assumir 1.0."""
        assert parse_llm_label_response('{"label": "neutro"}') == ("neutro", 1.0)

    def test_returns_none_for_response_without_json(self) -> None:
        """Uma resposta sem nenhum objeto JSON deve retornar ``None``."""
        assert parse_llm_label_response("resposta sem json") is None

    def test_returns_none_for_malformed_json(self) -> None:
        """Um objeto JSON malformado deve retornar ``None`` em vez de levantar exceção."""
        assert parse_llm_label_response('{"label": "positivo",}') is None

    def test_returns_none_for_label_outside_allowed_classes(self) -> None:
        """Um rótulo fora de ``allowed_labels`` deve ser rejeitado."""
        assert parse_llm_label_response('{"label": "muito_positivo", "confidence": 0.9}') is None

    def test_strips_reasoning_block_before_think_tag(self) -> None:
        """Conteúdo de raciocínio antes de ``</think>`` deve ser descartado."""
        raw_response = '<think>raciocinando...</think>{"label": "positivo", "confidence": 0.7}'
        assert parse_llm_label_response(raw_response) == ("positivo", 0.7)

    def test_strips_markdown_code_fence(self) -> None:
        """Uma resposta envolta em cerca de código ``` json deve ser interpretada normalmente."""
        raw_response = '```json\n{"label": "positivo", "confidence": 0.8}\n```'
        assert parse_llm_label_response(raw_response) == ("positivo", 0.8)

    def test_is_case_and_whitespace_insensitive_for_label(self) -> None:
        """O rótulo deve ser normalizado (minúsculas, sem espaços) antes da validação."""
        assert parse_llm_label_response('{"label": " POSITIVO ", "confidence": 0.5}') == (
            "positivo",
            0.5,
        )


class TestExtractConfidence:
    """Testes de :func:`labeling.llm_response._extract_confidence`."""

    def test_reads_confidence_from_probs_for_label(self) -> None:
        """A confiança deve ser lida de ``probs[label]`` quando disponível."""
        parsed = {"label": "positivo", "probs": {"positivo": 0.7, "negativo": 0.3}}
        assert _extract_confidence(parsed, "positivo") == pytest.approx(0.7)

    def test_falls_back_to_confidence_when_label_missing_from_probs(self) -> None:
        """Se ``probs`` não contiver o rótulo, deve recorrer a ``confidence``."""
        parsed = {"label": "neutro", "probs": {"positivo": 0.7}, "confidence": 0.4}
        assert _extract_confidence(parsed, "neutro") == pytest.approx(0.4)

    def test_clamps_value_above_one(self) -> None:
        """Valores de confiança acima de 1.0 devem ser limitados (clamp) a 1.0."""
        assert _extract_confidence({"confidence": 1.5}, "positivo") == pytest.approx(1.0)

    def test_clamps_negative_value_to_zero(self) -> None:
        """Valores de confiança negativos devem ser limitados (clamp) a 0.0."""
        assert _extract_confidence({"confidence": -0.5}, "positivo") == pytest.approx(0.0)

    def test_defaults_to_one_for_non_numeric_confidence(self) -> None:
        """Um valor de confiança não numérico deve resultar no padrão 1.0."""
        assert _extract_confidence({"confidence": "alta"}, "positivo") == pytest.approx(1.0)


class TestLabelingCheckpoint:
    """Testes de :mod:`labeling.checkpoint`."""

    def test_missing_checkpoint_is_empty(self, tmp_path: Path) -> None:
        """Um checkpoint inexistente equivale a nenhum tweet rotulado."""
        assert read_labeling_checkpoint(tmp_path / "nao_existe.jsonl") == {}

    def test_append_then_read_roundtrip(self, tmp_path: Path) -> None:
        """O que foi anexado deve ser lido de volta, inclusive em várias gravações."""
        path = tmp_path / "sub" / "ckpt.jsonl"
        append_labeling_checkpoint(path, {"1": ("positivo", 0.9)})
        append_labeling_checkpoint(path, {"2": ("negativo", 0.7)})

        assert read_labeling_checkpoint(path) == {"1": ("positivo", 0.9), "2": ("negativo", 0.7)}

    def test_ignores_truncated_last_line(self, tmp_path: Path) -> None:
        """Uma linha truncada por interrupção é ignorada; o tweet será reprocessado."""
        path = tmp_path / "ckpt.jsonl"
        append_labeling_checkpoint(path, {"1": ("positivo", 0.9)})
        with path.open("a", encoding="utf-8") as file:
            file.write('{"id": "2", "sentiment_la')

        assert read_labeling_checkpoint(path) == {"1": ("positivo", 0.9)}

    def test_last_occurrence_wins_for_duplicated_id(self, tmp_path: Path) -> None:
        """Em ids duplicados, vale a última gravação."""
        path = tmp_path / "ckpt.jsonl"
        append_labeling_checkpoint(path, {"1": ("positivo", 0.9)})
        append_labeling_checkpoint(path, {"1": ("negativo", 0.6)})

        assert read_labeling_checkpoint(path) == {"1": ("negativo", 0.6)}

    def test_empty_append_does_not_create_file(self, tmp_path: Path) -> None:
        """Anexar um lote vazio não deve criar arquivo."""
        path = tmp_path / "ckpt.jsonl"
        append_labeling_checkpoint(path, {})
        assert not path.exists()

    def test_path_changes_with_model_prompt_or_temperature(self, tmp_path: Path) -> None:
        """Mudar modelo, prompt ou temperatura inicia um checkpoint novo (nunca mistura rótulos)."""
        base = build_checkpoint_path(
            tmp_path, "openai", model_name="m", prompt_template="p", temperature=0.0
        )
        assert base == build_checkpoint_path(
            tmp_path, "openai", model_name="m", prompt_template="p", temperature=0.0
        )
        for changed in (
            {"model_name": "outro"},
            {"prompt_template": "outro"},
            {"temperature": 0.5},
        ):
            arguments: dict[str, Any] = {
                "model_name": "m",
                "prompt_template": "p",
                "temperature": 0.0,
            } | changed
            assert build_checkpoint_path(tmp_path, "openai", **arguments) != base

    def test_checkpoint_file_is_json_lines(self, tmp_path: Path) -> None:
        """Cada linha do checkpoint é um JSON independente (formato JSON Lines)."""
        path = tmp_path / "ckpt.jsonl"
        append_labeling_checkpoint(path, {"1": ("positivo", 0.9), "2": ("neutro", 0.5)})

        lines = path.read_text(encoding="utf-8").splitlines()
        assert [json.loads(line)["id"] for line in lines] == ["1", "2"]


def _corpus(n_tweets: int = 5) -> pl.DataFrame:
    """Corpus sintético com ``n_tweets`` tweets ``t0``, ``t1``..."""
    return pl.DataFrame(
        {
            "id": [str(index) for index in range(n_tweets)],
            "text_normalized": [f"t{index}" for index in range(n_tweets)],
        },
        schema={"id": pl.String, "text_normalized": pl.String},
    )


class TestRunIncrementalLabeling:
    """Testes de :func:`labeling.incremental.run_incremental_labeling`."""

    def test_labels_every_tweet_in_corpus_order(self, tmp_path: Path) -> None:
        """Devolve uma linha por tweet, na ordem do corpus, gravando cada lote no checkpoint."""
        batches: list[list[str]] = []

        def _classify(texts: Sequence[str]) -> list[tuple[str, float] | None]:
            batches.append(list(texts))
            return [("positivo", 0.91234567) for _ in texts]

        result = run_incremental_labeling(
            _corpus(),
            _classify,
            source_name="teste",
            checkpoint_path=tmp_path / "c.jsonl",
            batch_size=2,
            show_progress=False,
        )

        assert batches == [["t0", "t1"], ["t2", "t3"], ["t4"]]
        assert result["id"].to_list() == ["0", "1", "2", "3", "4"]
        assert result["confidence_score"].to_list() == [0.9123] * 5

    def test_skips_tweets_already_in_checkpoint(self, tmp_path: Path) -> None:
        """Tweets já rotulados no checkpoint não são enviados novamente ao classificador."""
        checkpoint = tmp_path / "c.jsonl"
        append_labeling_checkpoint(checkpoint, {"0": ("negativo", 0.5), "3": ("neutro", 0.6)})
        received: list[str] = []

        def _classify(texts: Sequence[str]) -> list[tuple[str, float] | None]:
            received.extend(texts)
            return [("positivo", 0.9) for _ in texts]

        result = run_incremental_labeling(
            _corpus(),
            _classify,
            source_name="teste",
            checkpoint_path=checkpoint,
            batch_size=10,
            show_progress=False,
        )

        assert received == ["t1", "t2", "t4"]
        assert result["sentiment_label"].to_list() == [
            "negativo",
            "positivo",
            "positivo",
            "neutro",
            "positivo",
        ]

    def test_raises_incomplete_when_a_tweet_has_no_valid_label_and_keeps_progress(
        self, tmp_path: Path
    ) -> None:
        """Falhas isoladas levantam ``IncompleteLabelingError`` sem perder o progresso."""
        checkpoint = tmp_path / "c.jsonl"

        def _classify(texts: Sequence[str]) -> list[tuple[str, float] | None]:
            return [None if text == "t2" else ("positivo", 0.9) for text in texts]

        with pytest.raises(IncompleteLabelingError, match="1 tweet"):
            run_incremental_labeling(
                _corpus(),
                _classify,
                source_name="teste",
                checkpoint_path=checkpoint,
                batch_size=10,
                show_progress=False,
            )

        assert set(read_labeling_checkpoint(checkpoint)) == {"0", "1", "3", "4"}

    def test_raises_for_empty_corpus(self, tmp_path: Path) -> None:
        """Um corpus vazio deve levantar ``EmptyDatasetError``."""
        with pytest.raises(EmptyDatasetError):
            run_incremental_labeling(
                _corpus(0),
                lambda texts: [],
                source_name="teste",
                checkpoint_path=tmp_path / "c.jsonl",
                batch_size=2,
                show_progress=False,
            )

    def test_raises_for_duplicated_ids(self, tmp_path: Path) -> None:
        """Ids duplicados tornariam o checkpoint ambíguo: erro explícito."""
        corpus = pl.DataFrame({"id": ["1", "1"], "text_normalized": ["a", "b"]})
        with pytest.raises(DataValidationError, match="duplicados"):
            run_incremental_labeling(
                corpus,
                lambda texts: [("positivo", 0.9)] * len(texts),
                source_name="teste",
                checkpoint_path=tmp_path / "c.jsonl",
                batch_size=2,
                show_progress=False,
            )

    def test_raises_when_classifier_returns_wrong_number_of_results(self, tmp_path: Path) -> None:
        """Um classificador que devolve menos resultados que textos viola o contrato do lote."""
        with pytest.raises(DataValidationError, match="resultado"):
            run_incremental_labeling(
                _corpus(),
                lambda texts: [("positivo", 0.9)],
                source_name="teste",
                checkpoint_path=tmp_path / "c.jsonl",
                batch_size=3,
                show_progress=False,
            )

    def test_raises_for_invalid_batch_size(self, tmp_path: Path) -> None:
        """``batch_size`` menor que 1 é inválido."""
        with pytest.raises(DataValidationError, match="batch_size"):
            run_incremental_labeling(
                _corpus(),
                lambda texts: [],
                source_name="teste",
                checkpoint_path=tmp_path / "c.jsonl",
                batch_size=0,
                show_progress=False,
            )


class _FakeRateLimitError(Exception):
    """Dublê de ``openai.RateLimitError`` (HTTP 429), com cabeçalho ``Retry-After``."""

    status_code = 429

    def __init__(self, retry_after: str | None = None) -> None:
        super().__init__("429 simulado")
        headers = {} if retry_after is None else {"retry-after": retry_after}
        self.response = type("Response", (), {"headers": headers})()


class _StubRateLimiter:
    """Limitador de mentira: não espera e registra as pausas globais pedidas por 429."""

    def __init__(self) -> None:
        self.cooldowns: list[float] = []

    def acquire(self) -> None:
        """Não bloqueia."""

    def report_success(self) -> None:
        """Ignora sucessos."""

    def report_rate_limited(self, wait_seconds: float) -> None:
        """Registra a pausa pedida."""
        self.cooldowns.append(wait_seconds)


class TestOpenAILabeler:
    """Testes de :mod:`labeling.openai_labeler` (a API é substituída por dublês)."""

    @staticmethod
    def _classify(text: str = "adorei", **overrides: Any) -> tuple[str, float] | None:
        arguments: dict[str, Any] = {
            "model": "m",
            "temperature": 0.0,
            "max_tokens": 100,
            "max_retries": 3,
            "max_rate_limit_retries": 3,
            "rate_limiter": _StubRateLimiter(),
            "request_timeout_seconds": 30.0,
            "allowed_labels": ("negativo", "neutro", "positivo"),
        }
        return _classify_single_text(text, 'Tweet: "{{TEXTO}}"', **(arguments | overrides))

    @pytest.fixture(autouse=True)
    def _no_sleep(self, monkeypatch: pytest.MonkeyPatch) -> list[float]:
        """Evita esperas reais (backoff/limitador) e registra os tempos pedidos."""
        sleeps: list[float] = []
        monkeypatch.setattr(openai_labeler.time, "sleep", sleeps.append)
        monkeypatch.setattr(rate_limiter_module.time, "sleep", sleeps.append)
        return sleeps

    def test_passes_messages_model_timeout_and_max_tokens_to_the_api(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """O prompt com o texto, o modelo, o timeout e ``max_tokens`` devem chegar à chamada."""
        captured: dict[str, Any] = {}

        def _fake_completion(messages: list[dict[str, Any]], **kwargs: Any) -> str:
            captured.update(kwargs, messages=messages)
            return '{"label":"neutro","confidence":0.5}'

        monkeypatch.setattr(openai_labeler, "generate_chat_completion", _fake_completion)

        assert self._classify() == ("neutro", 0.5)

        assert captured["messages"] == [{"role": "user", "content": 'Tweet: "adorei"'}]
        assert captured["model"] == "m"
        assert captured["timeout"] == 30.0
        assert captured["max_tokens"] == 100

    def test_reads_confidence_from_probs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """O formato do prompt padrão (``probs``) deve render a confiança da classe escolhida."""
        monkeypatch.setattr(
            openai_labeler,
            "generate_chat_completion",
            lambda messages, **kwargs: '{"label":"positivo","probs":{"positivo":0.9}}',
        )

        assert self._classify() == ("positivo", 0.9)

    def test_retries_after_api_error_with_exponential_backoff(
        self, monkeypatch: pytest.MonkeyPatch, _no_sleep: list[float]
    ) -> None:
        """Erros da API (ex.: timeout) são retentados, com espera crescente entre tentativas."""
        calls = {"n": 0}

        def _flaky(messages: list[dict[str, Any]], **kwargs: Any) -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise TimeoutError("simulado")
            return '{"label":"negativo","confidence":0.7}'

        monkeypatch.setattr(openai_labeler, "generate_chat_completion", _flaky)

        assert self._classify() == ("negativo", 0.7)
        assert calls["n"] == 3
        assert len(_no_sleep) == 2
        assert 2.0 <= _no_sleep[0] < 3.0
        assert 4.0 <= _no_sleep[1] < 5.0

    def test_returns_none_after_exhausting_api_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Falha persistente da API devolve ``None`` (não vira ``indefinido``: será retentado)."""
        calls = {"n": 0}

        def _always_fails(messages: list[dict[str, Any]], **kwargs: Any) -> str:
            calls["n"] += 1
            raise TimeoutError("simulado")

        monkeypatch.setattr(openai_labeler, "generate_chat_completion", _always_fails)

        assert self._classify(max_retries=2) is None
        assert calls["n"] == 2

    def test_unparseable_response_becomes_undefined_with_zero_score(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Resposta fora do formato vira ``indefinido`` com score 0.0, sem nova chamada."""
        calls = {"n": 0}

        def _no_json(messages: list[dict[str, Any]], **kwargs: Any) -> str:
            calls["n"] += 1
            return "sem json"

        monkeypatch.setattr(openai_labeler, "generate_chat_completion", _no_json)

        assert self._classify() == ("indefinido", 0.0)
        assert calls["n"] == 1

    def test_unknown_label_becomes_undefined(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Rótulo fora das classes aceitas vira ``indefinido`` com score 0.0."""
        monkeypatch.setattr(
            openai_labeler,
            "generate_chat_completion",
            lambda messages, **kwargs: '{"label":"raiva","confidence":0.9}',
        )

        assert self._classify() == ("indefinido", 0.0)

    def test_empty_text_is_undefined_without_calling_the_api(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Texto vazio não gasta chamada: vira ``indefinido`` com score 0.0."""

        def _must_not_be_called(messages: list[dict[str, Any]], **kwargs: Any) -> str:
            raise AssertionError("a API não deveria ser chamada")

        monkeypatch.setattr(openai_labeler, "generate_chat_completion", _must_not_be_called)

        assert self._classify(text="   ") == ("indefinido", 0.0)

    def test_rate_limit_waits_for_retry_after_and_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No HTTP 429, o limitador pausa e a chamada é repetida sem consumir ``max_retries``."""
        calls = {"n": 0}
        limiter = _StubRateLimiter()

        def _rate_limited_once(messages: list[dict[str, Any]], **kwargs: Any) -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                raise _FakeRateLimitError(retry_after="2")
            return '{"label":"positivo","confidence":0.8}'

        monkeypatch.setattr(openai_labeler, "generate_chat_completion", _rate_limited_once)

        assert self._classify(rate_limiter=limiter, max_retries=1) == ("positivo", 0.8)
        assert calls["n"] == 2
        assert limiter.cooldowns == [2.0]

    def test_rate_limit_returns_none_after_exhausting_rate_limit_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """429 persistente esgota ``max_rate_limit_retries`` e devolve ``None``."""
        calls = {"n": 0}

        def _always_429(messages: list[dict[str, Any]], **kwargs: Any) -> str:
            calls["n"] += 1
            raise _FakeRateLimitError(retry_after="0")

        monkeypatch.setattr(openai_labeler, "generate_chat_completion", _always_429)

        limiter = _StubRateLimiter()

        assert self._classify(rate_limiter=limiter, max_rate_limit_retries=2) is None
        assert calls["n"] == 3
        assert limiter.cooldowns == [0.0, 0.0]

    def test_configuration_errors_are_not_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Falha de configuração (chave ausente) interrompe a execução em vez de ser retentada."""
        calls = {"n": 0}

        def _missing_key(messages: list[dict[str, Any]], **kwargs: Any) -> str:
            calls["n"] += 1
            raise MissingEnvironmentVariableError("OPENAI_KEY")

        monkeypatch.setattr(openai_labeler, "generate_chat_completion", _missing_key)

        with pytest.raises(ProjectError):
            self._classify()
        assert calls["n"] == 1

    def test_batch_classifier_preserves_order_and_isolates_failures(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No lote, a ordem é preservada e a falha de API de um tweet não afeta os demais."""

        def _fake_completion(messages: list[dict[str, Any]], **kwargs: Any) -> str:
            content = messages[0]["content"]
            if "falha" in content:
                raise TimeoutError("simulado")
            if "ruim" in content:
                return "sem json"
            return '{"label":"positivo","confidence":0.8}'

        monkeypatch.setattr(openai_labeler, "generate_chat_completion", _fake_completion)
        classify = create_openai_batch_classifier(
            "{{TEXTO}}", model="m", max_retries=1, n_workers=3, requests_per_minute=None
        )

        assert classify(["bom", "falha", "ruim", "otimo"]) == [
            ("positivo", 0.8),
            None,
            ("indefinido", 0.0),
            ("positivo", 0.8),
        ]


class TestRateLimiter:
    """Testes de :mod:`labeling.rate_limiter`."""

    def test_rejects_non_positive_rate(self) -> None:
        """Taxa menor ou igual a zero é configuração inválida."""
        with pytest.raises(ValueError, match="requests_per_minute"):
            RateLimiter(0)

    def test_acquire_without_limit_does_not_wait(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Sem taxa base, ``acquire`` não pausa."""
        sleeps: list[float] = []
        monkeypatch.setattr(rate_limiter_module.time, "sleep", sleeps.append)

        limiter = RateLimiter(None)
        limiter.acquire()
        limiter.acquire()

        assert sleeps == []

    def test_second_acquire_waits_for_the_base_interval(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Com 60 req/min, a segunda chamada imediata espera cerca de 1 s."""
        sleeps: list[float] = []
        monkeypatch.setattr(rate_limiter_module.time, "monotonic", lambda: 100.0)
        monkeypatch.setattr(
            rate_limiter_module.time,
            "sleep",
            lambda seconds: (
                sleeps.append(seconds),
                monkeypatch.setattr(rate_limiter_module.time, "monotonic", lambda: 102.0),
            ),
        )

        limiter = RateLimiter(60)
        limiter.acquire()
        limiter.acquire()

        assert sleeps == [pytest.approx(1.0)]

    def test_rate_limited_report_imposes_cooldown_capped_at_maximum(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Um 429 pausa todas as threads, no máximo por ``MAX_COOLDOWN_SECONDS``."""
        sleeps: list[float] = []
        clock = {"now": 100.0}
        monkeypatch.setattr(rate_limiter_module.time, "monotonic", lambda: clock["now"])

        def _advance(seconds: float) -> None:
            sleeps.append(seconds)
            clock["now"] += seconds

        monkeypatch.setattr(rate_limiter_module.time, "sleep", _advance)

        limiter = RateLimiter(None)
        limiter.report_rate_limited(10_000.0)
        limiter.acquire()

        assert sleeps == [pytest.approx(MAX_COOLDOWN_SECONDS)]

    def test_retry_after_header_is_read_in_seconds_and_milliseconds(self) -> None:
        """``retry-after-ms`` tem prioridade sobre ``retry-after``; ausentes devolvem ``None``."""
        error_ms = _FakeRateLimitError()
        error_ms.response.headers.update({"retry-after-ms": "1500", "retry-after": "9"})
        error_s = _FakeRateLimitError(retry_after="7")

        assert get_retry_after_seconds(error_ms) == pytest.approx(1.5)
        assert get_retry_after_seconds(error_s) == pytest.approx(7.0)
        assert get_retry_after_seconds(_FakeRateLimitError()) is None
        assert get_retry_after_seconds(TimeoutError()) is None

    def test_is_rate_limit_error_checks_status_code(self) -> None:
        """Só o status HTTP 429 caracteriza limite de taxa."""
        assert is_rate_limit_error(_FakeRateLimitError())
        assert not is_rate_limit_error(TimeoutError())


class TestUndefinedLabelInBases:
    """``indefinido`` é aceito na base por fonte e excluído dos consumidores de 3 classes."""

    @staticmethod
    def _source(labels: list[str], scores: list[float]) -> pl.DataFrame:
        n = len(labels)
        return pl.DataFrame(
            {
                "id": [str(i) for i in range(n)],
                "text": ["t"] * n,
                "text_normalized": ["t"] * n,
                "sentiment_label": labels,
                "confidence_score": scores,
            }
        )

    def test_labeled_source_schema_accepts_undefined_label(self) -> None:
        """A base por fonte aceita ``indefinido`` com score 0.0."""
        frame = self._source(["positivo", "indefinido"], [0.9, 0.0])

        assert validate_labeled_source(frame).height == 2

    def test_labeled_source_schema_still_rejects_unknown_labels(self) -> None:
        """Rótulos fora das classes e de ``indefinido`` continuam inválidos."""
        with pytest.raises(DataValidationError):
            validate_labeled_source(self._source(["raiva"], [0.9]))

    def test_comparison_frame_drops_tweets_undefined_in_either_base(self) -> None:
        """A comparação entre modelos ignora tweets ``indefinido`` em qualquer base."""
        huggingface_base = self._source(["positivo", "negativo", "neutro"], [0.9, 0.8, 0.7])
        openai_base = self._source(["positivo", "indefinido", "neutro"], [0.9, 0.0, 0.6])

        frame = build_comparison_frame(huggingface_base, openai_base)

        assert frame["id"].to_list() == ["0", "2"]
        assert frame["agree"].to_list() == [True, True]


def _build_fake_model() -> HuggingFaceModel:
    """Classificador de mentira (sem modelo real), com saídas na ordem NEG/NEU/POS."""
    return HuggingFaceModel(
        model=object(),
        tokenizer=object(),
        device="cpu",
        model_name="m",
        class_labels=["negativo", "neutro", "positivo"],
    )


class TestHuggingFaceLabeler:
    """Testes de :mod:`labeling.huggingface` que não exigem GPU nem download de modelo."""

    def test_resolve_device_prefers_cuda_when_available(self) -> None:
        """``auto`` escolhe CUDA quando disponível e CPU caso contrário."""

        class _Cuda:
            available = True

            def is_available(self) -> bool:
                return self.available

        class _Torch:
            cuda = _Cuda()

        assert _resolve_device("auto", _Torch()) == "cuda"
        assert _resolve_device(None, _Torch()) == "cuda"
        _Torch.cuda.available = False
        assert _resolve_device("auto", _Torch()) == "cpu"
        assert _resolve_device("cuda:1", _Torch()) == "cuda:1"

    def test_resolve_dtype_rejects_unknown_name(self) -> None:
        """Um ``dtype`` desconhecido deve levantar ``ModelError``."""
        with pytest.raises(ModelError, match="dtype"):
            _resolve_dtype("float8", "cpu", object())

    def test_resolve_dtype_uses_float32_on_cpu(self) -> None:
        """No CPU, ``auto`` usa float32."""

        class _Torch:
            float32 = "fp32"

        assert _resolve_dtype("auto", "cpu", _Torch()) == "fp32"

    def test_unload_releases_gpu_memory_and_drops_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ao descarregar, o modelo é solto e a memória da GPU é liberada."""
        released: list[bool] = []
        monkeypatch.setattr(huggingface, "release_gpu_memory", lambda: released.append(True))
        hf_model = _build_fake_model()

        huggingface.unload_huggingface_model(hf_model)

        assert released == [True]
        assert hf_model.model is None
        assert hf_model.tokenizer is None

    def test_map_model_labels_translates_pysentimento_classes(self) -> None:
        """``NEG``/``NEU``/``POS`` viram as classes do projeto, na ordem dos índices."""
        assert _map_model_labels({0: "NEG", 1: "NEU", 2: "POS"}, "m") == [
            "negativo",
            "neutro",
            "positivo",
        ]
        assert _map_model_labels({0: "positive", 1: "negative", 2: "neutral"}, "m") == [
            "positivo",
            "negativo",
            "neutro",
        ]

    def test_map_model_labels_rejects_unknown_class(self) -> None:
        """Uma classe fora de negativo/neutro/positivo levanta ``ModelError``."""
        with pytest.raises(ModelError, match="não corresponde"):
            _map_model_labels({0: "NEG", 1: "NEU", 2: "RAIVA"}, "m")

    def test_map_model_labels_rejects_missing_class(self) -> None:
        """Um modelo binário (sem neutro) é recusado: o projeto exige as três classes."""
        with pytest.raises(ModelError, match="exatamente"):
            _map_model_labels({0: "NEG", 1: "POS"}, "m")

    def test_classifier_picks_argmax_class_with_its_probability(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """O rótulo é a classe de maior probabilidade e a confiança é essa probabilidade."""
        released: list[bool] = []
        probabilities = {"bom": [0.1, 0.2, 0.7], "ruim": [0.8, 0.15, 0.05]}

        def _fake_predict(
            hf_model: Any, texts: Sequence[str], *, max_input_tokens: int
        ) -> list[list[float]]:
            assert max_input_tokens == 128
            return [probabilities[text] for text in texts]

        monkeypatch.setattr(huggingface, "_predict_probabilities", _fake_predict)
        monkeypatch.setattr(huggingface, "release_gpu_memory", lambda: released.append(True))
        classify = huggingface.create_huggingface_batch_classifier(_build_fake_model())

        assert classify(["bom", "ruim"]) == [("positivo", 0.7), ("negativo", 0.8)]
        assert released == [True]

    def test_classifier_respects_model_output_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A ordem das saídas do modelo vem de ``class_labels``, não de uma ordem fixa."""
        hf_model = _build_fake_model()
        hf_model.class_labels = ["positivo", "negativo", "neutro"]
        monkeypatch.setattr(
            huggingface, "_predict_probabilities", lambda *args, **kwargs: [[0.6, 0.3, 0.1]]
        )
        monkeypatch.setattr(huggingface, "release_gpu_memory", lambda: True)

        classify = huggingface.create_huggingface_batch_classifier(hf_model)

        assert classify(["x"]) == [("positivo", 0.6)]


class TestReleaseGpuMemory:
    """Testes de :func:`utils.memory.release_gpu_memory`."""

    def test_is_safe_without_gpu(self) -> None:
        """Sem GPU (ou sem torch), devolve um booleano e nunca levanta exceção."""
        assert isinstance(release_gpu_memory(), bool)


class TestLabeledSourceSchema:
    """Testes de :func:`schemas.labeling.validate_labeled_source`."""

    @staticmethod
    def _valid() -> pl.DataFrame:
        return pl.DataFrame(
            {
                "id": ["1", "2"],
                "text": ["Adorei @a", "ruim"],
                "text_normalized": ["adorei [MENCAO]", "ruim"],
                "sentiment_label": ["positivo", "negativo"],
                "confidence_score": [0.9, 0.7],
            }
        )

    def test_accepts_valid_base(self) -> None:
        """Uma base conforme o contrato passa sem alterações."""
        assert validate_labeled_source(self._valid()).height == 2

    @pytest.mark.parametrize(
        ("column", "value"),
        [
            ("sentiment_label", "muito_positivo"),
            ("confidence_score", 1.5),
            ("confidence_score", -0.1),
        ],
    )
    def test_rejects_out_of_range_values(self, column: str, value: Any) -> None:
        """Classes desconhecidas e confianças fora de [0, 1] violam o contrato."""
        invalid = self._valid().with_columns(pl.lit(value).alias(column))
        with pytest.raises(DataValidationError):
            validate_labeled_source(invalid)

    def test_rejects_duplicated_ids(self) -> None:
        """O ``id`` é a chave de comparação entre as bases: deve ser único."""
        with pytest.raises(DataValidationError):
            validate_labeled_source(self._valid().with_columns(pl.lit("1").alias("id")))

    def test_rejects_extra_columns(self) -> None:
        """O contrato é estrito: colunas extras (ex.: modelo) ficam nos metadados, não na base."""
        with pytest.raises(DataValidationError):
            validate_labeled_source(self._valid().with_columns(pl.lit("m").alias("model")))
