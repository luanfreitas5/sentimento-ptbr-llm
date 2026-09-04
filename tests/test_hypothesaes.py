"""Testes do módulo HypotheSAEs (``src/hypothesaes``).

Requer ``torch`` instalado (dependência pesada e opcional do subpacote,
ver ``src/hypothesaes/__init__.py``): o módulo inteiro é pulado via
``pytest.importorskip`` quando ausente. Chamadas reais a LLMs (OpenAI) são
sempre substituídas por *fakes* via ``monkeypatch`` — nenhum teste faz
requisições de rede.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from exceptions.configuration import MissingEnvironmentVariableError
from exceptions.data import DataNotFoundError
from hypothesaes import (
    annotate,
    embedding,
    evaluation,
    interpret_neurons,
    llm_api,
    quickstart,
    sae,
    utils,
)
from hypothesaes.interpret_neurons import (
    InterpretConfig,
    LLMConfig,
    NeuronInterpreter,
    SamplingConfig,
    ScoringConfig,
)
from hypothesaes.sae import SparseAutoencoder

# `select_neurons` (o módulo) é reexportado em `hypothesaes/__init__.py` sob o
# mesmo nome da função `select_neurons()` que ele contém; importar o módulo
# via `from hypothesaes import select_neurons` retornaria a função (o último
# nome vence no namespace do pacote), não o módulo. Importar diretamente do
# submódulo evita a colisão.
from hypothesaes.select_neurons import (
    select_neurons,
    select_neurons_correlation,
    select_neurons_custom,
    select_neurons_separation_score,
)


# =============================================================================
# utils.py
# =============================================================================
class TestTruncateText:
    """Testes do truncamento de texto por palavras/caracteres/tokens."""

    def test_returns_original_text_when_no_limit_given(self) -> None:
        """Sem nenhum limite informado, o texto não deve ser alterado."""
        assert utils.truncate_text("texto qualquer") == "texto qualquer"

    def test_truncates_by_words_and_appends_message(self) -> None:
        """Deve cortar no número de palavras e anexar a mensagem de truncamento."""
        result = utils.truncate_text("um dois tres quatro", max_words=2)
        assert result.startswith("um dois")
        assert result.endswith("[... restante do texto foi truncado]")

    def test_does_not_truncate_when_under_word_limit(self) -> None:
        """Um texto mais curto que o limite não deve ser truncado."""
        assert utils.truncate_text("texto curto", max_words=10) == "texto curto"

    def test_truncates_by_characters(self) -> None:
        """Deve cortar no número de caracteres informado."""
        result = utils.truncate_text("abcdefghij", max_chars=4)
        assert result.startswith("abcd")

    def test_idempotent_on_already_truncated_text(self) -> None:
        """Aplicar novamente sobre um texto já truncado não deve duplicar a mensagem."""
        once = utils.truncate_text("um dois tres", max_words=1)
        twice = utils.truncate_text(once, max_words=1)
        assert once == twice


class TestFormatTextForDisplay:
    """Testes da formatação compacta de texto para exibição em logs."""

    def test_replaces_newlines_with_spaces(self) -> None:
        """Quebras de linha devem virar espaços simples."""
        assert utils.format_text_for_display("linha um\nlinha dois") == "linha um linha dois"

    def test_truncates_to_max_chars(self) -> None:
        """O texto deve ser truncado ao limite de caracteres informado."""
        result = utils.format_text_for_display("a" * 200, max_chars=10)
        assert result.startswith("aaaaaaaaaa")


class TestFilterInvalidTexts:
    """Testes da filtragem de textos ``None``/vazios."""

    def test_removes_none_and_blank_strings(self) -> None:
        """Deve remover ``None`` e strings vazias/apenas-espaço, preservando a ordem."""
        assert utils.filter_invalid_texts(["ok", None, "  ", "outro", ""]) == ["ok", "outro"]

    def test_returns_empty_list_for_all_invalid(self) -> None:
        """Uma lista inteiramente inválida deve retornar lista vazia."""
        assert utils.filter_invalid_texts([None, ""]) == []


class TestSaveLoadJson:
    """Testes de round-trip de leitura/escrita de JSON."""

    def test_round_trip_preserves_data(self, tmp_path: Path) -> None:
        """Dados salvos devem ser lidos de volta identicamente."""
        destination = tmp_path / "cache" / "dados.json"
        utils.save_json({"a": 1, "b": [1, 2, 3]}, destination)
        assert utils.load_json(destination) == {"a": 1, "b": [1, 2, 3]}

    def test_load_json_returns_empty_dict_for_missing_file(self, tmp_path: Path) -> None:
        """Um arquivo inexistente deve retornar um dicionário vazio."""
        assert utils.load_json(tmp_path / "inexistente.json") == {}


class TestLoadPromptTemplate:
    """Testes do carregamento de templates de prompt."""

    def test_loads_annotate_template(self) -> None:
        """O template 'annotate' deve conter os placeholders esperados."""
        content = utils.load_prompt_template("annotate")
        assert "{hypothesis}" in content
        assert "{text}" in content

    def test_raises_for_unknown_template(self) -> None:
        """Um nome de template inexistente deve levantar ``DataNotFoundError``."""
        with pytest.raises(DataNotFoundError):
            utils.load_prompt_template("template-que-nao-existe")


# =============================================================================
# llm_api.py
# =============================================================================
class TestNormalizeLlmKwargs:
    """Testes da normalização de argumentos de requisição do LLM."""

    def test_applies_defaults_when_absent(self) -> None:
        """Valores padrão devem ser aplicados quando ausentes em ``llm_kwargs``."""
        result = llm_api.normalize_llm_kwargs(
            {}, default_timeout=30.0, default_reasoning_effort="low"
        )
        assert result == {"timeout": 30.0, "reasoning_effort": "low"}

    def test_does_not_override_explicit_values(self) -> None:
        """Valores já informados pelo usuário não devem ser sobrescritos."""
        result = llm_api.normalize_llm_kwargs({"timeout": 5.0}, default_timeout=30.0)
        assert result == {"timeout": 5.0}

    def test_verbosity_skipped_when_text_already_present(self) -> None:
        """Se 'text' já foi informado, o padrão de verbosidade não deve ser aplicado."""
        result = llm_api.normalize_llm_kwargs({"text": {"foo": "bar"}}, default_verbosity="low")
        assert "verbosity" not in result


class TestResolveApiKey:
    """Testes da resolução da chave de API da OpenAI."""

    def test_raises_when_missing_for_hosted_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Sem ``OPENAI_KEY_SAE`` e apontando para a OpenAI, deve levantar erro de configuração."""
        monkeypatch.delenv("OPENAI_KEY_SAE", raising=False)
        with pytest.raises(MissingEnvironmentVariableError):
            llm_api._resolve_api_key(base_url=None)

    def test_returns_placeholder_for_local_server_without_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Para um servidor local sem chave configurada, deve usar o placeholder local."""
        monkeypatch.delenv("OPENAI_KEY_SAE", raising=False)
        result = llm_api._resolve_api_key(base_url="http://127.0.0.1:8000/v1")
        assert result == llm_api.LOCAL_OPENAI_API_KEY_PLACEHOLDER

    def test_returns_configured_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Uma chave válida configurada deve ser retornada diretamente."""
        monkeypatch.setenv("OPENAI_KEY_SAE", "sk-teste-123")
        assert llm_api._resolve_api_key(base_url=None) == "sk-teste-123"


class TestModelAbbreviationMap:
    """Testes do mapa de abreviações de modelo."""

    def test_known_abbreviations_resolve(self) -> None:
        """Abreviações conhecidas devem mapear para o ID de modelo completo."""
        assert llm_api.MODEL_ABBREVIATION_TO_ID["gpt5-mini"] == "gpt-5-mini"


# =============================================================================
# embedding.py
# =============================================================================
class TestEmbeddingCache:
    """Testes do cache fragmentado de embeddings em disco."""

    def test_round_trip_save_and_load(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Embeddings salvos em fragmentos devem ser recuperados integralmente."""
        monkeypatch.setattr(embedding, "EMBEDDING_CACHE_DIR", tmp_path)
        chunk_embeddings = {"texto um": np.array([1.0, 2.0]), "texto dois": np.array([3.0, 4.0])}

        next_index = embedding._save_embedding_chunk("meu_cache", chunk_embeddings, 0)
        assert next_index == 1

        loaded = embedding.load_embedding_cache("meu_cache")
        assert set(loaded.keys()) == {"texto um", "texto dois"}
        np.testing.assert_array_equal(loaded["texto um"], [1.0, 2.0])

    def test_load_returns_empty_for_missing_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Um cache inexistente deve retornar um dicionário vazio."""
        monkeypatch.setattr(embedding, "EMBEDDING_CACHE_DIR", tmp_path)
        assert embedding.load_embedding_cache("nao_existe") == {}

    def test_load_returns_empty_when_cache_name_is_none(self) -> None:
        """``cache_name=None`` deve sempre retornar vazio, sem tocar o disco."""
        assert embedding.load_embedding_cache(None) == {}

    def test_next_chunk_index_increments(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """O próximo índice de fragmento deve ser um a mais que o maior existente."""
        monkeypatch.setattr(embedding, "EMBEDDING_CACHE_DIR", tmp_path)
        embedding._save_embedding_chunk("cache", {"a": np.array([1.0])}, 0)
        embedding._save_embedding_chunk("cache", {"b": np.array([2.0])}, 1)
        assert embedding._find_next_chunk_index("cache") == 2


# =============================================================================
# annotate.py
# =============================================================================
class TestParseCompletion:
    """Testes da interpretação de respostas em texto livre (Sim/Não)."""

    def test_parses_yes(self) -> None:
        """Uma resposta iniciando com 'yes' deve retornar 1."""
        assert annotate.parse_completion("yes, o texto menciona o atendimento.") == 1

    def test_parses_no(self) -> None:
        """Uma resposta iniciando com 'no' deve retornar 0."""
        assert annotate.parse_completion("no, não menciona.") == 0

    def test_returns_none_for_ambiguous_response(self) -> None:
        """Uma resposta ambígua deve retornar ``None``."""
        assert annotate.parse_completion("talvez") is None

    def test_strips_thinking_block(self) -> None:
        """Conteúdo dentro de ``<think>...</think>`` deve ser ignorado na interpretação."""
        assert annotate.parse_completion("<think>analisando...</think>yes") == 1


class TestParseCompletionJson:
    """Testes da interpretação de respostas em formato JSON."""

    def test_parses_yes_answer(self) -> None:
        """Um JSON com ``"answer": "yes"`` deve retornar 1."""
        completion = '{"answer": "yes", "reasoning": "evidência clara"}'
        assert annotate.parse_completion_json(completion) == 1

    def test_parses_no_answer(self) -> None:
        """Um JSON com ``"answer": "no"`` deve retornar 0."""
        completion = '{"answer": "no", "reasoning": "sem evidência"}'
        assert annotate.parse_completion_json(completion) == 0

    def test_returns_none_for_malformed_json(self) -> None:
        """Uma resposta sem JSON válido deve retornar ``None``."""
        assert annotate.parse_completion_json("isto não é um JSON") is None


class TestGenerateCacheKey:
    """Testes da geração de chave de cache de anotação."""

    def test_key_is_deterministic(self) -> None:
        """A mesma entrada deve sempre produzir a mesma chave."""
        key1 = annotate.generate_cache_key("conceito", "um texto qualquer")
        key2 = annotate.generate_cache_key("conceito", "um texto qualquer")
        assert key1 == key2

    def test_key_differs_by_concept(self) -> None:
        """Conceitos diferentes devem produzir chaves diferentes para o mesmo texto."""
        assert annotate.generate_cache_key("a", "texto") != annotate.generate_cache_key(
            "b", "texto"
        )


class TestAnnotationCache:
    """Testes de round-trip do cache de anotações em disco."""

    def test_round_trip(self, tmp_path: Path) -> None:
        """Anotações salvas devem ser lidas de volta identicamente."""
        cache_path = tmp_path / "cache.json"
        annotate.save_annotation_cache(cache_path, {"chave": 1})
        assert annotate.load_annotation_cache(cache_path) == {"chave": 1}

    def test_returns_empty_for_missing_file(self, tmp_path: Path) -> None:
        """Um arquivo inexistente deve retornar cache vazio."""
        assert annotate.load_annotation_cache(tmp_path / "nao_existe.json") == {}

    def test_returns_empty_for_none_path(self) -> None:
        """``cache_path=None`` deve retornar cache vazio."""
        assert annotate.load_annotation_cache(None) == {}

    def test_recovers_from_corrupted_cache_file(self, tmp_path: Path) -> None:
        """Um arquivo de cache corrompido deve ser descartado, retornando cache vazio."""
        cache_path = tmp_path / "corrompido.json"
        cache_path.write_text("{ isto nao eh json valido", encoding="utf-8")
        assert annotate.load_annotation_cache(cache_path) == {}
        assert not cache_path.exists()


class TestAnnotateSingleText:
    """Testes da anotação de um único par (texto, conceito) via LLM (mockado)."""

    def test_returns_annotation_from_mocked_completion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deve retornar 1 quando o LLM mockado responde afirmativamente."""
        monkeypatch.setattr(
            annotate, "generate_completion", lambda **kwargs: "yes, evidência clara."
        )
        result, api_time = annotate.annotate_single_text(
            "ótimo atendimento", "elogia o atendimento"
        )
        assert result == 1
        assert api_time >= 0

    def test_returns_none_after_exhausting_retries_on_ambiguous_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Se o parser nunca interpretar a resposta, retorna ``None`` após esgotar tentativas."""
        monkeypatch.setattr(annotate, "generate_completion", lambda **kwargs: "resposta ambígua")
        result, _ = annotate.annotate_single_text("texto", "conceito", max_retries=1)
        assert result is None


class TestAnnotateTasks:
    """Testes da anotação paralela de múltiplas tarefas (texto, conceito)."""

    def test_annotates_all_tasks_without_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Sem cache, todas as tarefas devem ser anotadas via o LLM mockado."""
        monkeypatch.setattr(annotate, "generate_completion", lambda **kwargs: "yes")
        tasks = [("texto um", "conceito a"), ("texto dois", "conceito a")]
        results = annotate.annotate_tasks(tasks, n_workers=2, show_progress=False)
        assert results["conceito a"]["texto um"] == 1
        assert results["conceito a"]["texto dois"] == 1

    def test_use_cache_only_maps_uncached_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Com ``use_cache_only=True``, itens fora do cache devem receber ``uncached_value``."""

        def _fail_if_called(**kwargs: object) -> str:
            raise AssertionError(
                "generate_completion não deveria ser chamado com use_cache_only=True"
            )

        monkeypatch.setattr(annotate, "generate_completion", _fail_if_called)
        tasks = [("texto novo", "conceito a")]
        results = annotate.annotate_tasks(
            tasks, use_cache_only=True, uncached_value=0, show_progress=False
        )
        assert results["conceito a"]["texto novo"] == 0


class TestAnnotateTextsWithConcepts:
    """Testes da anotação em produto cartesiano (textos x conceitos)."""

    def test_returns_array_per_concept_in_text_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cada conceito deve mapear para um vetor de anotações na ordem dos textos."""

        def _fake_completion(**kwargs: object) -> str:
            prompt = str(kwargs.get("prompt", ""))
            return "yes" if "positivo" in prompt else "no"

        monkeypatch.setattr(annotate, "generate_completion", _fake_completion)
        texts = ["review positivo", "review negativo"]
        result = annotate.annotate_texts_with_concepts(
            texts, ["elogia o produto"], cache_name=None, n_workers=2, show_progress=False
        )
        np.testing.assert_array_equal(result["elogia o produto"], [1, 0])


# =============================================================================
# select_neurons.py
# =============================================================================
class TestSelectNeuronsCorrelation:
    """Testes da seleção de neurônios por correlação com o alvo."""

    def test_selects_most_correlated_neurons(self) -> None:
        """Neurônios fortemente correlacionados (positiva ou negativa) devem ser selecionados."""
        rng = np.random.default_rng(42)
        target = np.linspace(-1, 1, 50)
        activations = np.column_stack(
            [
                target + rng.normal(scale=0.01, size=50),  # forte correlação positiva
                -target + rng.normal(scale=0.01, size=50),  # forte correlação negativa
                rng.normal(size=50),  # ruído, sem correlação
            ]
        )
        indices, scores = select_neurons_correlation(activations, target, n_select=2)
        assert set(indices) == {0, 1}
        assert len(scores) == 2


class TestSelectNeuronsSeparationScore:
    """Testes da seleção de neurônios por score de separação."""

    def test_identifies_neuron_that_separates_target(self) -> None:
        """O neurônio cuja ativação separa claramente o alvo deve ter o maior score em módulo."""
        target = np.array([10.0] * 5 + [0.0] * 5)
        informative_neuron = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        dead_neuron = np.zeros(10)
        activations = np.column_stack([informative_neuron, dead_neuron])

        indices, scores = select_neurons_separation_score(
            activations, target, n_select=1, n_top_activating=5
        )
        assert indices == [0]
        assert scores[0] == pytest.approx(10.0)


class TestSelectNeuronsCustom:
    """Testes da seleção de neurônios por métrica customizada."""

    def test_selects_by_custom_metric(self) -> None:
        """A métrica customizada deve determinar diretamente quais neurônios são selecionados."""
        activations = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 100.0]])
        target = np.array([0.0, 0.0, 0.0])  # ignorado pela métrica customizada

        indices, scores = select_neurons_custom(
            activations, target, n_select=1, metric_fn=lambda acts, _target: float(np.sum(acts))
        )
        assert indices == [1]
        assert scores[0] == pytest.approx(102.0)


class TestSelectNeuronsDispatcher:
    """Testes da função de despacho ``select_neurons``."""

    def test_raises_for_multiclass_classification(self) -> None:
        """Alvo com mais de duas classes e ``classification=True`` deve levantar ``ValueError``."""
        activations = np.random.default_rng(0).normal(size=(10, 2))
        target = np.array([0, 1, 2] * 3 + [0])
        with pytest.raises(ValueError, match="multiclasse"):
            select_neurons(activations, target, n_select=1, method="lasso", classification=True)

    def test_raises_for_unknown_method(self) -> None:
        """Um método desconhecido deve levantar ``ValueError``."""
        activations = np.random.default_rng(0).normal(size=(10, 2))
        target = np.random.default_rng(0).normal(size=10)
        with pytest.raises(ValueError, match="desconhecido"):
            select_neurons(activations, target, n_select=1, method="inexistente")

    def test_dispatches_to_correlation_method(self) -> None:
        """``method='correlation'`` deve delegar para :func:`select_neurons_correlation`."""
        target = np.linspace(-1, 1, 30)
        activations = np.column_stack([target, np.random.default_rng(1).normal(size=30)])
        indices, _ = select_neurons(activations, target, n_select=1, method="correlation")
        assert indices == [0]


# =============================================================================
# sae.py
# =============================================================================
class TestSparseAutoencoderForward:
    """Testes do forward pass do Sparse Autoencoder."""

    def test_reconstruction_matches_input_shape(self) -> None:
        """A reconstrução deve ter o mesmo formato da entrada."""
        torch.manual_seed(0)
        model = SparseAutoencoder(input_dim=8, m_total_neurons=16, k_active_neurons=4, device="cpu")
        x = torch.randn(5, 8)
        reconstruction, info = model(x)
        assert reconstruction.shape == x.shape
        assert info["activations"].shape == (5, 16)

    def test_activations_are_at_most_k_sparse(self) -> None:
        """Cada exemplo deve ter no máximo ``k_active_neurons`` ativações não-nulas."""
        torch.manual_seed(0)
        model = SparseAutoencoder(input_dim=8, m_total_neurons=16, k_active_neurons=4, device="cpu")
        x = torch.randn(10, 8)
        _, info = model(x)
        nonzero_per_row = (info["activations"] != 0).sum(dim=1)
        assert (nonzero_per_row <= 4).all()

    def test_rejects_prefix_lengths_not_ending_in_total_neurons(self) -> None:
        """``prefix_lengths`` deve terminar em ``m_total_neurons``, senão levanta ``ValueError``."""
        with pytest.raises(ValueError, match="prefix_length"):
            SparseAutoencoder(
                input_dim=8,
                m_total_neurons=16,
                k_active_neurons=4,
                prefix_lengths=[8, 12],
                device="cpu",
            )


class TestSparseAutoencoderFit:
    """Testes do treinamento (``fit``) do Sparse Autoencoder."""

    def test_fit_runs_and_returns_history(self) -> None:
        """O treinamento por poucas épocas deve retornar um histórico consistente."""
        torch.manual_seed(0)
        model = SparseAutoencoder(input_dim=6, m_total_neurons=12, k_active_neurons=3, device="cpu")
        x_train = torch.randn(32, 6)
        history = model.fit(x_train, n_epochs=3, batch_size=8, show_progress=False)
        assert len(history["train_loss"]) == 3
        assert all(np.isfinite(loss) for loss in history["train_loss"])

    def test_fit_with_matryoshka_prefixes(self) -> None:
        """O treinamento com perda Matryoshka deve rodar sem erros, com múltiplos prefixos."""
        torch.manual_seed(0)
        model = SparseAutoencoder(
            input_dim=6,
            m_total_neurons=12,
            k_active_neurons=3,
            prefix_lengths=[4, 12],
            device="cpu",
        )
        x_train = torch.randn(16, 6)
        history = model.fit(x_train, n_epochs=2, batch_size=8, show_progress=False)
        assert len(history["train_loss"]) == 2

    def test_fit_saves_checkpoint(self, tmp_path: Path) -> None:
        """Ao informar ``save_dir``, um checkpoint deve ser salvo com o nome esperado."""
        torch.manual_seed(0)
        model = SparseAutoencoder(input_dim=6, m_total_neurons=12, k_active_neurons=3, device="cpu")
        x_train = torch.randn(16, 6)
        model.fit(x_train, n_epochs=1, batch_size=8, save_dir=tmp_path, show_progress=False)
        expected_path = tmp_path / sae.build_sae_checkpoint_name(12, 3)
        assert expected_path.is_file()


class TestSparseAutoencoderSaveLoad:
    """Testes de persistência (save/load) do Sparse Autoencoder."""

    def test_load_reproduces_forward_pass(self, tmp_path: Path) -> None:
        """Um modelo carregado deve produzir a mesma reconstrução que o modelo original."""
        torch.manual_seed(0)
        model = SparseAutoencoder(input_dim=6, m_total_neurons=12, k_active_neurons=3, device="cpu")
        model.eval()
        x = torch.randn(4, 6)
        original_reconstruction, _ = model(x)

        save_path = tmp_path / "modelo.pt"
        model.save(save_path)
        loaded_model = sae.load_model(save_path, device="cpu")
        loaded_model.eval()
        loaded_reconstruction, _ = loaded_model(x)

        assert torch.allclose(original_reconstruction, loaded_reconstruction)
        assert loaded_model.m_total_neurons == 12
        assert loaded_model.k_active_neurons == 3

    def test_raises_for_missing_checkpoint(self, tmp_path: Path) -> None:
        """Carregar um checkpoint inexistente deve levantar ``DataNotFoundError``."""
        with pytest.raises(DataNotFoundError):
            sae.load_model(tmp_path / "inexistente.pt", device="cpu")


class TestBuildSaeCheckpointName:
    """Testes da construção do nome de arquivo de checkpoint."""

    def test_name_without_matryoshka(self) -> None:
        """Sem prefixos Matryoshka, o nome deve seguir o padrão simples."""
        assert sae.build_sae_checkpoint_name(256, 8) == "SAE_M=256_K=8.pt"

    def test_name_with_matryoshka_prefixes(self) -> None:
        """Com prefixos Matryoshka, o nome deve incluir a lista de prefixos."""
        assert (
            sae.build_sae_checkpoint_name(256, 8, [32, 256])
            == "SAE_matryoshka_M=256_K=8_prefixes=32-256.pt"
        )


class TestComputeActivations:
    """Testes do cálculo de ativações em lote."""

    def test_returns_expected_shape(self) -> None:
        """As ativações calculadas devem ter o formato ``(n_amostras, m_total_neurons)``."""
        torch.manual_seed(0)
        model = SparseAutoencoder(input_dim=6, m_total_neurons=12, k_active_neurons=3, device="cpu")
        activations = model.compute_activations(
            np.random.default_rng(0).normal(size=(20, 6)).astype(np.float32), show_progress=False
        )
        assert activations.shape == (20, 12)

    def test_raises_for_unsupported_input_type(self) -> None:
        """Um tipo de entrada não suportado deve levantar ``TypeError``."""
        model = SparseAutoencoder(input_dim=6, m_total_neurons=12, k_active_neurons=3, device="cpu")
        with pytest.raises(TypeError):
            model.compute_activations("nao eh um array")


# =============================================================================
# interpret_neurons.py
# =============================================================================
class TestSampleTopZero:
    """Testes da amostragem de exemplos de topo/zero para interpretação."""

    def test_selects_highest_activating_and_zero_examples(self) -> None:
        """Deve retornar os exemplos de maior ativação e exemplos de ativação zero."""
        texts = [f"texto {i}" for i in range(6)]
        activations = np.zeros((6, 1))
        activations[:, 0] = [5.0, 4.0, 0.0, 0.0, 0.0, 0.0]  # 2 positivos, 4 zeros

        result = interpret_neurons.sample_top_zero(
            texts, activations, neuron_idx=0, n_examples=4, random_seed=0
        )

        assert set(result["positive_texts"]) == {"texto 0", "texto 1"}
        assert len(result["negative_texts"]) == 2
        assert all(
            text in {"texto 2", "texto 3", "texto 4", "texto 5"}
            for text in result["negative_texts"]
        )

    def test_truncates_examples_by_word_count(self) -> None:
        """Exemplos devem ser truncados quando ``max_words_per_example`` é informado."""
        texts = [
            "uma frase bem longa com varias palavras",
            "outra frase bem longa com varias palavras",
        ]
        activations = np.array([[5.0], [0.0]])
        result = interpret_neurons.sample_top_zero(
            texts, activations, neuron_idx=0, n_examples=2, max_words_per_example=2, random_seed=0
        )
        positive_text = result["positive_texts"][0]
        assert isinstance(positive_text, str)
        assert "[..." in positive_text


class TestSamplePercentileBins:
    """Testes da amostragem por faixas de percentil de ativação."""

    def test_selects_high_and_low_percentile_examples(self) -> None:
        """Deve retornar exemplos das faixas alta e baixa configuradas."""
        texts = [f"texto {i}" for i in range(10)]
        activations = np.arange(10, dtype=float).reshape(-1, 1)  # 0..9

        result = interpret_neurons.sample_percentile_bins(
            texts,
            activations,
            neuron_idx=0,
            n_examples=4,
            high_percentile=(90, 100),
            low_percentile=(0, 20),
            random_seed=0,
        )
        assert len(result["positive_texts"]) <= 2
        assert len(result["negative_texts"]) <= 2


class TestSamplingAndLlmConfigDefaults:
    """Testes dos valores padrão das dataclasses de configuração."""

    def test_sampling_config_defaults(self) -> None:
        """``SamplingConfig`` deve ter os valores padrão documentados."""
        config = SamplingConfig()
        assert config.function is interpret_neurons.sample_top_zero
        assert config.n_examples == 20

    def test_interpret_config_wraps_sub_configs(self) -> None:
        """``InterpretConfig`` deve compor ``SamplingConfig`` e ``LLMConfig`` por padrão."""
        config = InterpretConfig()
        assert isinstance(config.sampling, SamplingConfig)
        assert isinstance(config.llm, LLMConfig)
        assert config.n_candidates == 1


class TestNeuronInterpreterParseInterpretation:
    """Testes do parsing da resposta bruta do LLM intérprete."""

    def test_strips_dash_and_quotes(self) -> None:
        """Deve remover o prefixo '- \"' e as aspas ao redor do texto."""
        interpreter = NeuronInterpreter()
        assert (
            interpreter._parse_interpretation('- "menciona atendimento rápido"')
            == "menciona atendimento rápido"
        )

    def test_returns_none_for_incomplete_thinking_block(self) -> None:
        """Um bloco ``<think>`` sem fechamento deve retornar ``None`` (resposta incompleta)."""
        interpreter = NeuronInterpreter()
        assert interpreter._parse_interpretation("<think>ainda pensando") is None

    def test_strips_completed_thinking_block(self) -> None:
        """O conteúdo antes de ``</think>`` deve ser descartado."""
        interpreter = NeuronInterpreter()
        result = interpreter._parse_interpretation('<think>racionínio</think>\n- "resultado final"')
        assert result == "resultado final"


class TestNeuronInterpreterComputeMetrics:
    """Testes do cálculo de métricas de fidelidade de uma interpretação."""

    def test_computes_expected_recall_precision_f1(self) -> None:
        """Recall, precisão e F1 devem corresponder ao cálculo manual esperado."""
        interpreter = NeuronInterpreter()
        annotations = np.array([1, 1, 0, 1])
        labels = np.array([1, 1, 0, 0])
        activations = np.array([3.0, 2.0, 0.0, 1.0])

        metrics = interpreter._compute_metrics(annotations, labels, activations)

        assert metrics["recall"] == pytest.approx(1.0)
        assert metrics["precision"] == pytest.approx(2 / 3)
        assert metrics["f1"] == pytest.approx(2 * 1.0 * (2 / 3) / (1.0 + 2 / 3))

    def test_returns_zeroed_metrics_when_labels_have_single_class(self) -> None:
        """Se os rótulos não tiverem as duas classes, todas as métricas devem ser 0."""
        interpreter = NeuronInterpreter()
        metrics = interpreter._compute_metrics(
            annotations=np.array([1, 0]), labels=np.array([1, 1]), activations=np.array([1.0, 2.0])
        )
        assert metrics == {"recall": 0.0, "precision": 0.0, "f1": 0.0, "correlation": 0.0}


class TestNeuronInterpreterInterpretNeurons:
    """Testes de integração da geração de interpretações (LLM mockado)."""

    def test_returns_parsed_interpretation_for_each_neuron(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cada neurônio deve receber a interpretação (parseada) retornada pelo LLM mockado."""
        monkeypatch.setattr(
            interpret_neurons, "generate_completion", lambda **kwargs: '- "menciona atendimento"'
        )

        texts = [f"texto {i}" for i in range(8)]
        activations = np.zeros((8, 1))
        activations[:, 0] = [5.0, 4.0, 3.0, 2.0, 0.0, 0.0, 0.0, 0.0]

        interpreter = NeuronInterpreter(n_workers_interpretation=2)
        config = InterpretConfig(sampling=SamplingConfig(n_examples=4))
        result = interpreter.interpret_neurons(
            texts, activations, neuron_indices=[0], config=config
        )

        assert result[0] == ["menciona atendimento"]

    def test_returns_none_for_dead_neuron(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Um neurônio sem nenhuma ativação positiva ('morto') não deve gerar interpretação."""
        monkeypatch.setattr(
            interpret_neurons, "generate_completion", lambda **kwargs: '- "não deveria ser chamado"'
        )

        texts = [f"texto {i}" for i in range(4)]
        activations = np.zeros((4, 1))

        interpreter = NeuronInterpreter()
        result = interpreter.interpret_neurons(texts, activations, neuron_indices=[0])
        assert result[0] == [None]


class TestNeuronInterpreterScoreInterpretations:
    """Testes de integração da pontuação de fidelidade (anotação mockada)."""

    def test_scores_are_within_valid_range(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """As métricas retornadas devem estar em faixas válidas (0 a 1 para recall/precision/f1)."""
        monkeypatch.setattr(annotate, "generate_completion", lambda **kwargs: "yes")

        texts = [f"texto {i}" for i in range(8)]
        activations = np.zeros((8, 1))
        activations[:, 0] = [5.0, 4.0, 3.0, 2.0, 0.0, 0.0, 0.0, 0.0]

        interpreter = NeuronInterpreter(n_workers_annotation=2)
        interpretations = {0: ["menciona atendimento"]}
        scoring_config = ScoringConfig(n_examples=4)

        metrics = interpreter.score_interpretations(
            texts, activations, interpretations, config=scoring_config, show_progress=False
        )

        neuron_metrics = metrics[0]["menciona atendimento"]
        assert 0.0 <= neuron_metrics["recall"] <= 1.0
        assert 0.0 <= neuron_metrics["precision"] <= 1.0
        assert 0.0 <= neuron_metrics["f1"] <= 1.0


# =============================================================================
# evaluation.py
# =============================================================================
class TestComputePairwiseCorrelationMatrix:
    """Testes do cálculo de correlações entre pares de hipóteses."""

    def test_identical_vectors_have_correlation_one(self) -> None:
        """Duas hipóteses com o mesmo padrão de anotação devem ter correlação 1.0."""
        vector = np.array([1, 0, 1, 0, 1])
        result = evaluation.compute_pairwise_correlation_matrix({"ref": vector}, {"pred": vector})
        assert result[("ref", "pred")] == pytest.approx(1.0)


class TestMatchHypothesisPairs:
    """Testes do pareamento ótimo de hipóteses (algoritmo húngaro)."""

    def test_matches_maximize_total_similarity(self) -> None:
        """O pareamento deve maximizar a similaridade total (não necessariamente por posição)."""
        list_1 = ["a", "b"]
        list_2 = ["x", "y"]
        similarity = {("a", "x"): 0.1, ("a", "y"): 0.9, ("b", "x"): 0.9, ("b", "y"): 0.1}

        matches = evaluation.match_hypothesis_pairs(list_1, list_2, similarity)
        matched_pairs = {(hyp1, hyp2) for hyp1, hyp2, _ in matches}
        assert matched_pairs == {("a", "y"), ("b", "x")}

    def test_raises_for_mismatched_list_lengths(self) -> None:
        """Listas de tamanhos diferentes devem levantar ``ValueError``."""
        with pytest.raises(ValueError):
            evaluation.match_hypothesis_pairs(["a"], ["x", "y"], {})


class TestEvaluatePredicateSurfaceSimilarity:
    """Testes da similaridade de superfície entre predicados (LLM mockado)."""

    def test_returns_one_when_all_responses_are_yes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Se todas as amostras do LLM responderem 'yes', o score deve ser 1.0."""
        monkeypatch.setattr(evaluation, "generate_completion", lambda **kwargs: "yes")
        score = evaluation.evaluate_predicate_surface_similarity("a", "b", n_samples=3)
        assert score == pytest.approx(1.0)

    def test_returns_zero_when_all_responses_are_no(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Se todas as amostras do LLM responderem 'no', o score deve ser 0.0."""
        monkeypatch.setattr(evaluation, "generate_completion", lambda **kwargs: "no")
        score = evaluation.evaluate_predicate_surface_similarity("a", "b", n_samples=3)
        assert score == pytest.approx(0.0)


class TestComputeHypothesisSeparationScores:
    """Testes do cálculo do score de separação por hipótese."""

    def test_positive_effect_when_concept_correlates_with_high_target(self) -> None:
        """Um conceito presente apenas em exemplos de alvo alto deve ter efeito positivo."""
        y_true = np.array([10.0, 10.0, 0.0, 0.0])
        annotations = {"hipotese": np.array([1, 1, 0, 0])}

        scores = evaluation.compute_hypothesis_separation_scores(annotations, y_true)

        effect_size, _p_value = scores["hipotese"]
        assert effect_size == pytest.approx(10.0)


class TestComputeOlsMetrics:
    """Testes do ajuste de regressão (OLS/logística) sobre as anotações."""

    def test_ols_returns_r2_for_regression(self) -> None:
        """Para alvo contínuo, deve retornar a métrica ``r2``."""
        rng = np.random.default_rng(0)
        y_true = rng.normal(size=40)
        annotations = {"h1": (y_true > 0).astype(int), "h2": rng.integers(0, 2, size=40)}

        metrics, coefficients = evaluation.compute_ols_metrics(
            annotations, y_true, classification=False
        )

        assert "r2" in metrics
        assert set(coefficients.keys()) == {"h1", "h2"}


class TestScoreHypotheses:
    """Testes da avaliação completa de um conjunto de hipóteses."""

    def test_returns_metrics_and_sorted_dataframe(self) -> None:
        """Deve retornar métricas com contagem de hipóteses significativas e
        um DataFrame ordenado."""
        rng = np.random.default_rng(0)
        y_true = rng.normal(size=60)
        annotations = {
            "hipotese forte": (y_true > np.median(y_true)).astype(int),
            "hipotese fraca": rng.integers(0, 2, size=60),
        }

        metrics, hypothesis_df = evaluation.score_hypotheses(annotations, y_true)

        assert "Significant" in metrics
        assert isinstance(hypothesis_df, pd.DataFrame)
        assert list(hypothesis_df.columns) == [
            "hypothesis",
            "separation_score",
            "separation_pval",
            "regression_coef",
            "regression_pval",
            "feature_prevalence",
        ]
        assert hypothesis_df["separation_score"].is_monotonic_decreasing


# =============================================================================
# quickstart.py
# =============================================================================
class TestQuickstartTrainSae:
    """Testes do fluxo de alto nível de treinamento do SAE."""

    def test_trains_a_new_model(self) -> None:
        """Sem checkpoint existente, deve treinar um novo modelo do zero."""
        torch.manual_seed(0)
        embeddings = np.random.default_rng(0).normal(size=(24, 6)).astype(np.float32)
        model = quickstart.train_sae(
            embeddings, 8, 2, n_epochs=2, batch_size=8, show_progress=False
        )
        assert isinstance(model, SparseAutoencoder)
        assert model.m_total_neurons == 8
        assert model.k_active_neurons == 2

    def test_loads_existing_checkpoint_instead_of_retraining(self, tmp_path: Path) -> None:
        """Com um checkpoint existente e ``overwrite_checkpoint=False``,
        deve carregá-lo em vez de retreinar."""
        torch.manual_seed(0)
        embeddings = np.random.default_rng(0).normal(size=(24, 6)).astype(np.float32)
        quickstart.train_sae(
            embeddings, 8, 2, n_epochs=1, batch_size=8, checkpoint_dir=tmp_path, show_progress=False
        )

        loaded = quickstart.train_sae(
            embeddings, 8, 2, n_epochs=1, batch_size=8, checkpoint_dir=tmp_path, show_progress=False
        )
        assert loaded.m_total_neurons == 8
        assert loaded.k_active_neurons == 2


class TestQuickstartInterpretSae:
    """Testes do fluxo de alto nível de interpretação de neurônios."""

    def test_raises_when_selection_params_are_ambiguous(self) -> None:
        """Informar zero ou mais de um critério de seleção deve levantar ``ValueError``."""
        torch.manual_seed(0)
        model = SparseAutoencoder(input_dim=6, m_total_neurons=8, k_active_neurons=2, device="cpu")
        with pytest.raises(ValueError):
            quickstart.interpret_sae(["texto"], np.zeros((1, 6), dtype=np.float32), model)

    def test_returns_dataframe_with_interpretations(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Deve retornar um DataFrame com uma linha por neurônio interpretado."""
        monkeypatch.setattr(
            interpret_neurons, "generate_completion", lambda **kwargs: '- "hipótese de teste"'
        )

        torch.manual_seed(0)
        model = SparseAutoencoder(input_dim=6, m_total_neurons=8, k_active_neurons=3, device="cpu")
        embeddings = np.random.default_rng(0).normal(size=(20, 6)).astype(np.float32)
        texts = [f"texto {i}" for i in range(20)]

        result = quickstart.interpret_sae(
            texts,
            embeddings,
            model,
            n_top_neurons=2,
            print_examples_n=0,
            n_examples_for_interpretation=4,
        )

        assert list(result.columns[:2]) == ["neuron_idx", "interpretation"]
        assert len(result) == 2


class TestQuickstartGenerateHypotheses:
    """Testes do fluxo de alto nível de geração de hipóteses."""

    def test_returns_dataframe_without_fidelity_scoring(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Com ``n_scoring_examples=0``, deve pular a pontuação de fidelidade."""
        monkeypatch.setattr(
            interpret_neurons, "generate_completion", lambda **kwargs: '- "hipótese de teste"'
        )

        torch.manual_seed(0)
        model = SparseAutoencoder(input_dim=6, m_total_neurons=8, k_active_neurons=3, device="cpu")
        embeddings = np.random.default_rng(0).normal(size=(30, 6)).astype(np.float32)
        texts = [f"texto {i}" for i in range(30)]
        labels = np.random.default_rng(0).normal(size=30)

        result = quickstart.generate_hypotheses(
            texts,
            labels,
            embeddings,
            model,
            selection_method="correlation",
            n_selected_neurons=2,
            n_scoring_examples=0,
            n_examples_for_interpretation=4,
        )

        assert "neuron_idx" in result.columns
        assert "target_correlation" in result.columns
        assert "interpretation" in result.columns
        assert len(result) == 2

    def test_raises_when_n_selected_exceeds_total_neurons(self) -> None:
        """Selecionar mais neurônios do que o total do SAE deve levantar ``ValueError``."""
        torch.manual_seed(0)
        model = SparseAutoencoder(input_dim=6, m_total_neurons=4, k_active_neurons=2, device="cpu")
        embeddings = np.random.default_rng(0).normal(size=(10, 6)).astype(np.float32)
        with pytest.raises(ValueError):
            quickstart.generate_hypotheses(
                [f"texto {i}" for i in range(10)],
                np.random.default_rng(0).normal(size=10),
                embeddings,
                model,
                n_selected_neurons=99,
                n_scoring_examples=0,
            )


class TestQuickstartEvaluateHypotheses:
    """Testes do fluxo de alto nível de avaliação de hipóteses em holdout."""

    def test_returns_metrics_and_evaluation_dataframe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deve retornar métricas agregadas e um DataFrame de avaliação por hipótese."""

        def _fake_completion(**kwargs: object) -> str:
            prompt = str(kwargs.get("prompt", ""))
            return "yes" if "positivo" in prompt else "no"

        monkeypatch.setattr(annotate, "generate_completion", _fake_completion)

        texts = ["texto positivo"] * 10 + ["texto negativo"] * 10
        labels = np.array([1] * 10 + [0] * 10)
        hypotheses_df = pd.DataFrame(
            {"neuron_idx": [0], "interpretation": ["menciona algo positivo"]}
        )

        metrics, evaluation_df = quickstart.evaluate_hypotheses(
            hypotheses_df, texts, labels, classification=True, n_workers_annotation=2
        )

        assert "Significant" in metrics
        assert "hypothesis" in evaluation_df.columns
