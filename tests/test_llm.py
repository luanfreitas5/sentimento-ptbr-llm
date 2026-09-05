"""Testes do módulo de orquestração LangChain de LLMs (``src/llm``)."""

import sys
from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from exceptions.model import UnsupportedModelError
from llm.backends import (
    LLM_BACKEND_NAMES,
    HuggingFaceLLMBackend,
    LLMBackend,
    OllamaLLMBackend,
    create_llm_backend,
)
from llm.chains import build_sentiment_classification_chain, run_chain_with_retry
from llm.classifier import LangChainSentimentClassifier
from llm.parsers import (
    SentimentLLMOutput,
    extract_json_object,
    generate_and_parse_with_retry,
    parse_structured_llm_output,
)
from llm.prompts import build_sentiment_prompt
from models.base import SentimentClassifier


class _FakeLLMBackend:
    """Backend LLM de teste: retorna respostas pré-programadas, na ordem das chamadas."""

    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        """Registra a chamada e retorna a próxima resposta pré-programada (repete a última)."""
        self.calls.append(prompt)
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[index]


class _FakeRunnable:
    """Objeto de teste com método ``invoke``, imitando um LLM Runnable do LangChain."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.received_prompts: list[str] = []

    def invoke(self, prompt: str) -> str:
        """Registra o prompt recebido e retorna a resposta pré-programada."""
        self.received_prompts.append(prompt)
        return self.response


class TestPromptBuilders:
    """Testes dos templates de prompt versionados."""

    def test_zero_shot_prompt_ignores_examples(self) -> None:
        """A estratégia zero-shot não deve incluir nenhum exemplo no prompt."""
        prompt = build_sentiment_prompt(
            "ótimo produto", strategy="zero_shot", few_shot_examples=[("exemplo", "positivo")]
        )
        assert "exemplo" not in prompt
        assert prompt.endswith("Resposta:")

    def test_few_shot_prompt_includes_examples(self) -> None:
        """A estratégia few-shot deve incluir os exemplos informados no prompt."""
        prompt = build_sentiment_prompt(
            "produto razoável", strategy="few_shot", few_shot_examples=[("péssimo", "negativo")]
        )
        assert "péssimo" in prompt
        assert '"negativo"' in prompt

    def test_chain_of_thought_prompt_ends_with_reasoning_cue(self) -> None:
        """A estratégia de cadeia de pensamento deve terminar solicitando o raciocínio."""
        prompt = build_sentiment_prompt("produto ok", strategy="chain_of_thought")
        assert prompt.endswith("Raciocínio:")

    def test_raises_for_unknown_strategy(self) -> None:
        """Uma estratégia desconhecida deve levantar ``ValueError``."""
        with pytest.raises(ValueError, match="Estratégia"):
            build_sentiment_prompt("texto", strategy="inexistente")  # type: ignore[arg-type]

    def test_raises_for_unknown_version(self) -> None:
        """Uma versão de template desconhecida deve levantar ``ValueError``."""
        with pytest.raises(ValueError, match="Versão"):
            build_sentiment_prompt("texto", version="v99")


class TestSentimentLLMOutput:
    """Testes do contrato de saída estruturada de um LLM."""

    def test_normalizes_label_casing_and_whitespace(self) -> None:
        """O rótulo deve ser normalizado para minúsculas, sem espaços nas bordas."""
        output = SentimentLLMOutput(sentimento=" Positivo ", confianca=0.5)
        assert output.sentimento == "positivo"

    def test_rejects_confidence_out_of_range(self) -> None:
        """Uma confiança fora de ``[0.0, 1.0]`` deve levantar ``ValidationError``."""
        with pytest.raises(ValidationError):
            SentimentLLMOutput(sentimento="positivo", confianca=1.5)


class TestExtractJsonObject:
    """Testes da extração de objetos JSON de um texto bruto."""

    def test_extracts_json_substring(self) -> None:
        """Deve extrair apenas o trecho JSON, ignorando texto ao redor."""
        raw_output = 'Raciocínio: parece positivo\nResposta: {"sentimento": "positivo"}'
        assert extract_json_object(raw_output) == '{"sentimento": "positivo"}'

    def test_returns_none_without_json(self) -> None:
        """Deve retornar ``None`` quando não há nenhum objeto JSON no texto."""
        assert extract_json_object("resposta sem json") is None


class TestParseStructuredLLMOutput:
    """Testes do parser estruturado (Pydantic) de respostas de LLM."""

    def test_parses_valid_json(self) -> None:
        """Uma resposta JSON válida deve ser interpretada corretamente."""
        result = parse_structured_llm_output(
            '{"sentimento": "positivo", "confianca": 0.9, "justificativa": "elogio"}'
        )
        assert result is not None
        assert result.sentimento == "positivo"
        assert result.confianca == 0.9
        assert result.justificativa == "elogio"

    def test_returns_none_for_invalid_json(self) -> None:
        """Um JSON malformado deve resultar em ``None``, sem levantar exceção."""
        assert parse_structured_llm_output('{"sentimento": "positivo",}') is None

    def test_returns_none_for_missing_sentiment_key(self) -> None:
        """A ausência da chave obrigatória ``sentimento`` deve resultar em ``None``."""
        assert parse_structured_llm_output('{"confianca": 0.5}') is None

    def test_returns_none_for_label_outside_allowed_labels(self) -> None:
        """Um rótulo fora das classes conhecidas deve resultar em ``None``."""
        result = parse_structured_llm_output(
            '{"sentimento": "irritado"}', allowed_labels=("positivo", "negativo")
        )
        assert result is None


class TestGenerateAndParseWithRetry:
    """Testes da geração com retentativas e fallback."""

    def test_returns_parsed_output_on_first_success(self) -> None:
        """Uma resposta interpretável na primeira tentativa não deve gerar retentativas."""
        backend = _FakeLLMBackend(['{"sentimento": "positivo", "confianca": 0.9}'])
        result = generate_and_parse_with_retry(backend, "prompt")
        assert result.sentimento == "positivo"
        assert len(backend.calls) == 1

    def test_retries_until_success(self) -> None:
        """Deve retentar até obter uma resposta interpretável."""
        backend = _FakeLLMBackend(["resposta inválida", '{"sentimento": "negativo"}'])
        result = generate_and_parse_with_retry(backend, "prompt", max_retries=3)
        assert result.sentimento == "negativo"
        assert len(backend.calls) == 2

    def test_returns_fallback_after_exhausting_retries(self) -> None:
        """Após esgotar as tentativas, deve retornar o rótulo de fallback com confiança zero."""
        backend = _FakeLLMBackend(["sempre inválido"])
        result = generate_and_parse_with_retry(
            backend, "prompt", max_retries=2, fallback_label="neutro"
        )
        assert result.sentimento == "neutro"
        assert result.confianca == 0.0
        assert len(backend.calls) == 2

    def test_raises_for_invalid_max_retries(self) -> None:
        """``max_retries`` menor que 1 deve levantar ``ValueError``."""
        with pytest.raises(ValueError, match="max_retries"):
            generate_and_parse_with_retry(_FakeLLMBackend(["{}"]), "prompt", max_retries=0)


class TestLLMBackendsProtocol:
    """Testes da interface comum de backend LLM e da fábrica única."""

    def test_fake_backend_satisfies_protocol(self) -> None:
        """Um dublê de teste com ``generate`` deve satisfazer o Protocol ``LLMBackend``."""
        assert isinstance(_FakeLLMBackend(["{}"]), LLMBackend)

    def test_ollama_backend_delegates_to_runnable_invoke(self) -> None:
        """``OllamaLLMBackend.generate`` deve delegar para ``invoke`` do Runnable subjacente."""
        runnable = _FakeRunnable("resposta gerada")
        backend = OllamaLLMBackend(runnable)
        assert backend.generate("prompt de teste") == "resposta gerada"
        assert runnable.received_prompts == ["prompt de teste"]

    def test_huggingface_backend_delegates_to_runnable_invoke(self) -> None:
        """``HuggingFaceLLMBackend.generate`` deve delegar para ``invoke`` do Runnable subjacente"""
        runnable = _FakeRunnable("outra resposta")
        backend = HuggingFaceLLMBackend(runnable)
        assert backend.generate("prompt") == "outra resposta"

    def test_create_llm_backend_raises_for_unknown_name(self) -> None:
        """Um nome de backend desconhecido deve levantar ``UnsupportedModelError``."""
        with pytest.raises(UnsupportedModelError):
            create_llm_backend("inexistente")  # type: ignore[arg-type]

    def test_backend_names_are_sorted_and_known(self) -> None:
        """As constantes de nomes de backend devem incluir Ollama e Hugging Face."""
        assert set(LLM_BACKEND_NAMES) == {"ollama", "huggingface"}

    def test_load_ollama_backend_raises_when_langchain_ollama_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deve levantar ``ModelError`` quando ``langchain-ollama`` não está instalado."""
        from exceptions.model import ModelError

        monkeypatch.setitem(sys.modules, "langchain_ollama", None)
        with pytest.raises(ModelError, match="langchain-ollama"):
            create_llm_backend("ollama")

    def test_load_huggingface_backend_raises_when_langchain_huggingface_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deve levantar ``ModelError`` quando ``langchain-huggingface`` não está instalado."""
        from exceptions.model import ModelError

        monkeypatch.setitem(sys.modules, "langchain_huggingface", None)
        with pytest.raises(ModelError, match="langchain-huggingface"):
            create_llm_backend("huggingface")


class TestSentimentClassificationChain:
    """Testes do encadeamento LangChain (LCEL) de classificação de sentimento."""

    def test_chain_classifies_text(self) -> None:
        """A cadeia deve compor prompt, geração e parsing, retornando a saída estruturada."""
        pytest.importorskip("langchain_core")
        backend = _FakeLLMBackend(['{"sentimento": "positivo", "confianca": 0.8}'])
        chain = build_sentiment_classification_chain(backend, strategy="zero_shot")
        result = chain.invoke("ótimo produto")
        assert result is not None
        assert result.sentimento == "positivo"

    def test_run_chain_with_retry_returns_fallback_when_unparseable(self) -> None:
        """Deve retornar o fallback quando a cadeia nunca produz uma saída interpretável."""
        pytest.importorskip("langchain_core")
        backend = _FakeLLMBackend(["resposta sem json"])
        chain = build_sentiment_classification_chain(backend, strategy="zero_shot")
        result = run_chain_with_retry(chain, "texto", max_retries=2, fallback_label="neutro")
        assert result.sentimento == "neutro"

    def test_raises_for_invalid_max_retries(self) -> None:
        """``max_retries`` menor que 1 deve levantar ``ValueError``, mesmo sem invocar a cadeia."""
        pytest.importorskip("langchain_core")
        backend = _FakeLLMBackend(["{}"])
        chain = build_sentiment_classification_chain(backend)
        with pytest.raises(ValueError, match="max_retries"):
            run_chain_with_retry(chain, "texto", max_retries=0)


class TestLangChainSentimentClassifier:
    """Testes do classificador LLM completo (backend + prompt + parser)."""

    def test_satisfies_sentiment_classifier_protocol(self) -> None:
        """O classificador deve satisfazer o Protocol ``SentimentClassifier`` por duck typing."""
        classifier = LangChainSentimentClassifier(_FakeLLMBackend(["{}"]))
        assert isinstance(classifier, SentimentClassifier)

    def test_predict_returns_labels(self) -> None:
        """``predict`` deve retornar um rótulo de sentimento por texto de entrada."""
        backend = _FakeLLMBackend(['{"sentimento": "positivo", "confianca": 0.9}'])
        classifier = LangChainSentimentClassifier(backend, strategy="zero_shot")
        predictions = classifier.predict(["ótimo produto", "outro texto"])
        assert list(predictions) == ["positivo", "positivo"]

    def test_predict_proba_assigns_confidence_to_predicted_class(self) -> None:
        """A probabilidade da classe predita deve corresponder à confiança relatada pelo LLM."""
        backend = _FakeLLMBackend(['{"sentimento": "positivo", "confianca": 0.9}'])
        classifier = LangChainSentimentClassifier(
            backend, strategy="zero_shot", allowed_labels=("positivo", "negativo")
        )
        probabilities = classifier.predict_proba(["ótimo produto"])
        predicted_index = list(classifier.allowed_labels).index("positivo")
        assert probabilities.shape == (1, 2)
        assert probabilities[0, predicted_index] == pytest.approx(0.9)

    def test_fit_selects_balanced_few_shot_examples(self) -> None:
        """``fit`` deve selecionar exemplos balanceados quando a estratégia usa exemplos."""
        classifier = LangChainSentimentClassifier(
            _FakeLLMBackend(["{}"]), strategy="few_shot", n_examples_per_class=1
        )
        classifier.fit(["bom", "ruim", "ok"], ["positivo", "negativo", "neutro"])
        assert len(classifier._few_shot_examples) == 3

    def test_zero_shot_strategy_skips_example_selection(self) -> None:
        """``fit`` não deve selecionar exemplos quando a estratégia é zero-shot."""
        classifier = LangChainSentimentClassifier(_FakeLLMBackend(["{}"]), strategy="zero_shot")
        classifier.fit(["bom", "ruim"], ["positivo", "negativo"])
        assert classifier._few_shot_examples == []

    def test_predict_with_justification_preserves_justification(self) -> None:
        """A justificativa textual do LLM deve ser preservada na saída completa."""
        backend = _FakeLLMBackend(
            ['{"sentimento": "positivo", "confianca": 0.9, "justificativa": "elogio direto"}']
        )
        classifier = LangChainSentimentClassifier(backend, strategy="zero_shot")
        outputs = classifier.predict_with_justification(["ótimo produto"])
        assert outputs[0].justificativa == "elogio direto"

    def test_uses_fallback_label_after_unparseable_responses(self) -> None:
        """Quando o LLM nunca responde de forma interpretável, deve usar o rótulo de fallback."""
        backend = _FakeLLMBackend(["resposta sem json"])
        classifier = LangChainSentimentClassifier(
            backend, strategy="zero_shot", max_retries=1, fallback_label="neutro"
        )
        predictions = classifier.predict(["texto qualquer"])
        assert list(predictions) == ["neutro"]
