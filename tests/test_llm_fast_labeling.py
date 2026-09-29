"""Testes dos modos rápidos de rotulagem por LLM (sem rede: a API é substituída por dublês)."""

import math
import threading
from typing import Any

import polars as pl
import pytest

from exceptions.data import DataValidationError
from labeling import openai_labeler
from labeling import rate_limiter as rate_limiter_module
from labeling.incremental import run_incremental_labeling
from labeling.llm_batching import (
    build_label_only_prompt,
    build_multi_tweet_prompt,
    parse_first_token_label,
    parse_multi_tweet_response,
)
from labeling.openai_labeler import create_openai_batch_classifier

LABELS = ("negativo", "neutro", "positivo")


class TestParseFirstTokenLabel:
    """Testes de :func:`labeling.llm_batching.parse_first_token_label`."""

    def test_confidence_is_normalized_softmax_over_class_prefixes(self) -> None:
        logprobs = {"pos": math.log(0.6), "neg": math.log(0.3), "neu": math.log(0.1)}
        label, confidence = parse_first_token_label("positivo", logprobs, allowed_labels=LABELS)  # type: ignore[misc]
        assert label == "positivo"
        assert confidence == pytest.approx(0.6)

    def test_ignores_tokens_that_match_no_class(self) -> None:
        logprobs = {"pos": math.log(0.5), "Olá": math.log(0.5)}
        _, confidence = parse_first_token_label("positivo", logprobs, allowed_labels=LABELS)  # type: ignore[misc]
        assert confidence == pytest.approx(1.0)

    def test_ambiguous_prefix_is_ignored(self) -> None:
        # "ne" é prefixo de negativo e de neutro: não pode ser atribuído a nenhum
        assert parse_first_token_label("", {"ne": -0.1}, allowed_labels=LABELS) is None

    def test_falls_back_to_text_without_logprobs(self) -> None:
        assert parse_first_token_label("Negativo.", {}, allowed_labels=LABELS) == ("negativo", 1.0)

    def test_returns_none_when_nothing_matches(self) -> None:
        assert parse_first_token_label("talvez", {}, allowed_labels=LABELS) is None


class TestMultiTweetFormat:
    """Testes do prompt e do parser multi-tweet."""

    def test_prompt_numbers_tweets_and_drops_placeholder_line(self) -> None:
        prompt = build_multi_tweet_prompt('Instrução\nTweet: "{{TEXTO}}"', ["a\nb", "c"])
        assert "1. a b" in prompt
        assert "2. c" in prompt
        assert "{{TEXTO}}" not in prompt

    def test_label_only_prompt_embeds_text_and_labels(self) -> None:
        prompt = build_label_only_prompt('Tweet: "{{TEXTO}}"', "adorei", allowed_labels=LABELS)
        assert "adorei" in prompt
        assert "positivo" in prompt

    def test_parser_maps_ids_and_marks_missing_or_invalid_as_none(self) -> None:
        raw = (
            '[{"id":1,"label":"positivo","confidence":0.9},'
            '{"id":3,"label":"inexistente"},{"id":9,"label":"neutro"}]'
        )
        assert parse_multi_tweet_response(raw, 3, allowed_labels=LABELS) == [
            ("positivo", 0.9),
            None,
            None,
        ]

    def test_parser_marks_duplicated_ids_as_none(self) -> None:
        raw = '[{"id":1,"label":"positivo"},{"id":1,"label":"negativo"}]'
        assert parse_multi_tweet_response(raw, 1, allowed_labels=LABELS) == [None]

    @pytest.mark.parametrize("raw", ["", "sem json", "[não é json]", '{"id":1}'])
    def test_parser_tolerates_garbage(self, raw: str) -> None:
        assert parse_multi_tweet_response(raw, 2, allowed_labels=LABELS) == [None, None]


class TestFastClassifier:
    """Testes de dedup, multi-tweet e logprobs em :func:`create_openai_batch_classifier`."""

    @pytest.fixture(autouse=True)
    def _no_sleep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(openai_labeler.time, "sleep", lambda _: None)
        monkeypatch.setattr(rate_limiter_module.time, "sleep", lambda _: None)

    def test_duplicated_texts_are_sent_once_across_batches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prompts: list[str] = []

        def fake_completion(messages: list[dict[str, Any]], **_: Any) -> str:
            prompts.append(messages[0]["content"])
            return '{"label":"positivo","confidence":0.8}'

        monkeypatch.setattr(openai_labeler, "generate_chat_completion", fake_completion)
        classify = create_openai_batch_classifier("{{TEXTO}}", model="m", requests_per_minute=None)

        assert classify(["bom", "bom", "  "]) == [
            ("positivo", 0.8),
            ("positivo", 0.8),
            ("indefinido", 0.0),
        ]
        assert classify(["bom"]) == [("positivo", 0.8)]
        assert prompts == ["bom"]

    def test_failed_tweets_are_not_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = {"n": 0}

        def flaky(messages: list[dict[str, Any]], **_: Any) -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionError("queda")
            return '{"label":"neutro","confidence":0.7}'

        monkeypatch.setattr(openai_labeler, "generate_chat_completion", flaky)
        classify = create_openai_batch_classifier(
            "{{TEXTO}}", model="m", max_retries=1, requests_per_minute=None
        )

        assert classify(["x"]) == [None]
        assert classify(["x"]) == [("neutro", 0.7)]

    def test_label_logprobs_mode_uses_first_token_distribution(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_logprobs(messages: list[dict[str, Any]], **kwargs: Any) -> tuple[str, dict]:
            assert kwargs["max_tokens"] <= 4
            return "positivo", {"pos": math.log(0.75), "neg": math.log(0.25)}

        monkeypatch.setattr(
            openai_labeler, "generate_chat_completion_first_token_logprobs", fake_logprobs
        )
        classify = create_openai_batch_classifier(
            "{{TEXTO}}", model="m", requests_per_minute=None, response_mode="label_logprobs"
        )
        assert classify(["ótimo"]) == [("positivo", pytest.approx(0.75))]

    def test_multi_tweet_sends_one_request_per_group_and_retries_missing_items(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prompts: list[str] = []

        def fake_completion(messages: list[dict[str, Any]], **_: Any) -> str:
            prompt = messages[0]["content"]
            prompts.append(prompt)
            if "1. a" in prompt:  # grupo multi: devolve só o item 1
                return '[{"id":1,"label":"positivo","confidence":0.9}]'
            return '{"label":"negativo","confidence":0.6}'  # reclassificação individual

        monkeypatch.setattr(openai_labeler, "generate_chat_completion", fake_completion)
        classify = create_openai_batch_classifier(
            "Instr {{TEXTO}}", model="m", requests_per_minute=None, tweets_per_request=2
        )

        assert classify(["a", "b"]) == [("positivo", 0.9), ("negativo", 0.6)]
        assert len(prompts) == 2  # 1 requisição de grupo + 1 reclassificação do item ausente

    def test_multi_tweet_group_failure_returns_none_for_all(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def always_fail(*_: Any, **__: Any) -> str:
            raise ConnectionError("queda")

        monkeypatch.setattr(openai_labeler, "generate_chat_completion", always_fail)
        classify = create_openai_batch_classifier(
            "{{TEXTO}}", model="m", max_retries=1, requests_per_minute=None, tweets_per_request=2
        )
        assert classify(["a", "b"]) == [None, None]

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"response_mode": "xml"}, "response_mode"),
            ({"tweets_per_request": 0}, "tweets_per_request"),
        ],
    )
    def test_invalid_options_are_rejected(self, kwargs: dict[str, Any], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            create_openai_batch_classifier("{{TEXTO}}", model="m", **kwargs)


class TestParallelBatches:
    """Testes do pool de lotes de :func:`labeling.incremental.run_incremental_labeling`."""

    @staticmethod
    def _corpus(n: int) -> pl.DataFrame:
        return pl.DataFrame(
            {"id": [str(i) for i in range(n)], "text_normalized": [f"t{i}" for i in range(n)]}
        )

    def test_parallel_batches_give_same_result_as_sequential(self, tmp_path: Any) -> None:
        thread_names: set[str] = set()

        def classify(texts: Any) -> list[tuple[str, float]]:
            thread_names.add(threading.current_thread().name)
            return [("positivo", 0.5)] * len(texts)

        result = run_incremental_labeling(
            self._corpus(10),
            classify,
            source_name="t",
            checkpoint_path=tmp_path / "ckpt.jsonl",
            batch_size=2,
            n_parallel_batches=3,
            show_progress=False,
        )
        assert result["id"].to_list() == [str(i) for i in range(10)]
        assert set(result["sentiment_label"]) == {"positivo"}

    def test_failed_batch_items_raise_incomplete_and_keep_checkpoint(self, tmp_path: Any) -> None:
        from exceptions.pipeline import IncompleteLabelingError

        def classify(texts: Any) -> list[tuple[str, float] | None]:
            return [None if text == "t3" else ("neutro", 0.9) for text in texts]

        with pytest.raises(IncompleteLabelingError):
            run_incremental_labeling(
                self._corpus(6),
                classify,
                source_name="t",
                checkpoint_path=tmp_path / "ckpt.jsonl",
                batch_size=2,
                n_parallel_batches=2,
                show_progress=False,
            )
        assert len((tmp_path / "ckpt.jsonl").read_text(encoding="utf-8").splitlines()) == 5

    def test_invalid_parallel_batches_is_rejected(self, tmp_path: Any) -> None:
        with pytest.raises(DataValidationError, match="n_parallel_batches"):
            run_incremental_labeling(
                self._corpus(2),
                lambda texts: [("neutro", 1.0)] * len(texts),
                source_name="t",
                checkpoint_path=tmp_path / "ckpt.jsonl",
                batch_size=1,
                n_parallel_batches=0,
                show_progress=False,
            )
