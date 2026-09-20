"""Testes da rotulagem por LLM: prompt/resposta, checkpoint, execução incremental e fontes."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from exceptions.base import ProjectError
from exceptions.configuration import MissingEnvironmentVariableError
from exceptions.data import DataValidationError, EmptyDatasetError
from exceptions.model import ModelError
from exceptions.pipeline import IncompleteLabelingError
from labeling import huggingface, openai_labeler
from labeling.checkpoint import (
    append_labeling_checkpoint,
    build_checkpoint_path,
    read_labeling_checkpoint,
)
from labeling.huggingface import HuggingFaceLLM, _resolve_device, _resolve_dtype
from labeling.incremental import run_incremental_labeling
from labeling.llm_response import (
    _extract_confidence,
    build_labeling_prompt,
    parse_llm_label_response,
)
from labeling.openai_labeler import _classify_single_text, create_openai_batch_classifier
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


class TestOpenAILabeler:
    """Testes de :mod:`labeling.openai_labeler` (a API é substituída por dublês)."""

    @staticmethod
    def _classify(**overrides: Any) -> tuple[str, float] | None:
        arguments: dict[str, Any] = {
            "model": "m",
            "temperature": 0.0,
            "max_retries": 3,
            "request_interval_seconds": 0.5,
            "request_timeout_seconds": 30.0,
            "allowed_labels": ("negativo", "neutro", "positivo"),
        }
        return _classify_single_text("adorei", 'Tweet: "{{TEXTO}}"', **(arguments | overrides))

    def test_sleeps_before_every_call_to_avoid_rate_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deve pausar ``request_interval_seconds`` antes de cada chamada (proteção contra 429)."""
        sleeps: list[float] = []
        monkeypatch.setattr(openai_labeler.time, "sleep", sleeps.append)
        monkeypatch.setattr(
            openai_labeler,
            "generate_completion",
            lambda **kwargs: '{"label":"positivo","confidence":0.9}',
        )

        assert self._classify() == ("positivo", 0.9)
        assert sleeps == [0.5]

    def test_passes_timeout_and_prompt_with_text_to_the_api(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """O timeout configurado e o prompt com o texto devem chegar à chamada."""
        monkeypatch.setattr(openai_labeler.time, "sleep", lambda seconds: None)
        captured: dict[str, Any] = {}

        def _fake_completion(**kwargs: Any) -> str:
            captured.update(kwargs)
            return '{"label":"neutro","confidence":0.5}'

        monkeypatch.setattr(openai_labeler, "generate_completion", _fake_completion)

        self._classify()

        assert captured["timeout"] == 30.0
        assert captured["provider"] == "openai"
        assert captured["prompt"] == 'Tweet: "adorei"'

    def test_retries_after_api_error_with_exponential_backoff(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Erros da API (ex.: 429/timeout) são retentados, com espera crescente entre tentativas."""
        sleeps: list[float] = []
        monkeypatch.setattr(openai_labeler.time, "sleep", sleeps.append)
        calls = {"n": 0}

        def _flaky(**kwargs: Any) -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise TimeoutError("simulado")
            return '{"label":"negativo","confidence":0.7}'

        monkeypatch.setattr(openai_labeler, "generate_completion", _flaky)

        assert self._classify() == ("negativo", 0.7)
        assert calls["n"] == 3
        backoffs = [seconds for seconds in sleeps if seconds >= 1.0]
        assert backoffs == [1.0, 2.0]

    def test_returns_none_after_exhausting_retries_on_unparseable_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Respostas fora do formato esgotam as tentativas e devolvem ``None`` (sem exceção)."""
        monkeypatch.setattr(openai_labeler.time, "sleep", lambda seconds: None)
        monkeypatch.setattr(openai_labeler, "generate_completion", lambda **kwargs: "sem json")

        assert self._classify(max_retries=2) is None

    def test_configuration_errors_are_not_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Falha de configuração (chave ausente) interrompe a execução em vez de ser retentada."""
        monkeypatch.setattr(openai_labeler.time, "sleep", lambda seconds: None)
        calls = {"n": 0}

        def _missing_key(**kwargs: Any) -> str:
            calls["n"] += 1
            raise MissingEnvironmentVariableError("OPENAI_KEY")

        monkeypatch.setattr(openai_labeler, "generate_completion", _missing_key)

        with pytest.raises(ProjectError):
            self._classify()
        assert calls["n"] == 1

    def test_batch_classifier_preserves_order_and_isolates_failures(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No lote, a ordem é preservada e a falha de um tweet não afeta os demais."""
        monkeypatch.setattr(openai_labeler.time, "sleep", lambda seconds: None)

        def _fake_completion(*, prompt: str, **kwargs: Any) -> str:
            if "ruim" in prompt:
                return "sem json"
            return '{"label":"positivo","confidence":0.8}'

        monkeypatch.setattr(openai_labeler, "generate_completion", _fake_completion)
        classify = create_openai_batch_classifier(
            "{{TEXTO}}", model="m", max_retries=1, n_workers=3, request_interval_seconds=0.0
        )

        assert classify(["bom", "ruim", "otimo"]) == [
            ("positivo", 0.8),
            None,
            ("positivo", 0.8),
        ]


def _build_fake_llm() -> HuggingFaceLLM:
    """LLM de mentira (sem modelo real) com um tokenizador sem template de chat."""
    tokenizer = type("_Tokenizer", (), {"chat_template": None})()
    return HuggingFaceLLM(model=object(), tokenizer=tokenizer, device="cpu", model_name="m")


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
        llm = _build_fake_llm()

        huggingface.unload_huggingface_llm(llm)

        assert released == [True]
        assert llm.model is None
        assert llm.tokenizer is None

    def test_classifier_retries_only_unparsed_tweets_and_releases_gpu_each_round(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 1ª tentativa é gulosa; só os tweets sem resposta válida são regerados."""
        calls: list[tuple[int, float | None]] = []
        released: list[bool] = []
        valid = '{"label":"positivo","probs":{"positivo":0.9}}'

        def _fake_generate(
            llm: Any, prompts: Sequence[str], *, sampling_temperature: float | None, **kwargs: Any
        ) -> list[str]:
            calls.append((len(prompts), sampling_temperature))
            if sampling_temperature is None:
                return ["lixo" if "ruim" in prompt else valid for prompt in prompts]
            return ['{"label":"negativo","confidence":0.6}' for _ in prompts]

        monkeypatch.setattr(huggingface, "_generate_texts", _fake_generate)
        monkeypatch.setattr(huggingface, "release_gpu_memory", lambda: released.append(True))
        classify = huggingface.create_huggingface_batch_classifier(
            _build_fake_llm(), "{{TEXTO}}", max_retries=3, retry_temperature=0.3
        )

        results = classify(["bom", "ruim"])

        assert results == [("positivo", 0.9), ("negativo", 0.6)]
        assert calls == [(2, None), (1, 0.3)]
        assert len(released) == 2

    def test_classifier_returns_none_when_retries_are_exhausted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sem resposta interpretável após todas as tentativas, o tweet fica como ``None``."""
        monkeypatch.setattr(
            huggingface, "_generate_texts", lambda llm, prompts, **kwargs: ["lixo"] * len(prompts)
        )
        monkeypatch.setattr(huggingface, "release_gpu_memory", lambda: True)
        classify = huggingface.create_huggingface_batch_classifier(
            _build_fake_llm(), "{{TEXTO}}", max_retries=2
        )

        assert classify(["a", "b"]) == [None, None]


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
