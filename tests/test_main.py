"""Testes de fixação (pinning) da integração da CLI (``src/main.py``).

A revisão final da branch apontou que nada no test suite exercitava
``src/main.py`` diretamente: nenhum teste garantia que ``--max-workers``
realmente chegava aos kwargs da etapa ``preprocessing`` (ver Tasks 7/10/11
do plano de paralelização). Este arquivo também cobre a composição da etapa
``labeling`` (``_build_labeling_stage_kwargs``), que carrega o pipeline de
sentimento do Hugging Face — não é um teste de integração completo da CLI.
"""

import pytest

import main
from config.paths import load_project_paths
from config.settings import create_settings, load_general_config
from main import _build_labeling_stage_kwargs, _build_preprocessing_stage_kwargs, parse_arguments


class _FakeSentimentPipeline:
    """Dublê de pipeline Hugging Face, sem baixar o modelo real da Hub."""

    def __call__(self, texts):
        """Ignora os textos e nunca é efetivamente chamado nestes testes."""
        raise AssertionError("o pipeline dublê não deveria ser chamado nestes testes")


class TestBuildPreprocessingStageKwargs:
    """Testes de :func:`main._build_preprocessing_stage_kwargs`."""

    def test_includes_max_workers_from_cli_argument(self) -> None:
        """``--max-workers`` informado na CLI deve chegar aos kwargs da etapa."""
        args = parse_arguments(["--stage", "preprocessing", "--max-workers", "4"])
        paths = load_project_paths()
        general_config = load_general_config()
        settings = create_settings()

        kwargs = _build_preprocessing_stage_kwargs(paths, general_config, settings, args)

        assert kwargs["max_workers"] == 4

    def test_defaults_max_workers_to_none_when_not_informed(self) -> None:
        """Sem ``--max-workers``, o valor repassado deve ser ``None`` (o executor decide)."""
        args = parse_arguments(["--stage", "preprocessing"])
        paths = load_project_paths()
        general_config = load_general_config()
        settings = create_settings()

        kwargs = _build_preprocessing_stage_kwargs(paths, general_config, settings, args)

        assert kwargs["max_workers"] is None


class TestBuildLabelingStageKwargs:
    """Testes de :func:`main._build_labeling_stage_kwargs`.

    O carregamento real do pipeline Hugging Face (rede/modelo pesado) é
    substituído por um dublê via ``monkeypatch`` — estes testes cobrem
    apenas a composição dos kwargs a partir de ``configs/labeling.yaml``.
    """

    def test_includes_huggingface_kwargs_from_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Os parâmetros de ``configs/labeling.yaml -> huggingface`` devem chegar aos kwargs."""
        fake_pipeline = _FakeSentimentPipeline()
        monkeypatch.setattr(
            main, "load_huggingface_sentiment_pipeline", lambda **kwargs: fake_pipeline
        )
        args = parse_arguments(["--stage", "labeling"])
        paths = load_project_paths()
        general_config = load_general_config()
        settings = create_settings()

        kwargs = _build_labeling_stage_kwargs(paths, general_config, settings, args)

        assert kwargs["pipeline"] is fake_pipeline
        assert kwargs["label_mapping"] == {"POS": "positivo", "NEG": "negativo", "NEU": "neutro"}
        assert kwargs["huggingface_batch_size"] == 32
        assert kwargs["low_confidence_threshold"] == 0.5

    def test_loads_pipeline_with_model_name_from_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """O modelo carregado deve ser o configurado em ``huggingface.model``."""
        captured_kwargs: dict[str, object] = {}

        def _fake_loader(**kwargs: object) -> _FakeSentimentPipeline:
            captured_kwargs.update(kwargs)
            return _FakeSentimentPipeline()

        monkeypatch.setattr(main, "load_huggingface_sentiment_pipeline", _fake_loader)
        args = parse_arguments(["--stage", "labeling"])
        paths = load_project_paths()
        general_config = load_general_config()
        settings = create_settings()

        _build_labeling_stage_kwargs(paths, general_config, settings, args)

        assert captured_kwargs["model_name"] == "pysentimiento/bertweet-pt-sentiment"
        assert captured_kwargs["device"] == "auto"
        assert captured_kwargs["batch_size"] == 32
        assert captured_kwargs["max_length"] == 128

    def test_includes_llm_relabeling_kwargs_from_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Os parâmetros de ``configs/labeling.yaml -> llm_relabeling`` devem chegar aos kwargs."""
        monkeypatch.setattr(
            main,
            "load_huggingface_sentiment_pipeline",
            lambda **kwargs: _FakeSentimentPipeline(),
        )
        args = parse_arguments(["--stage", "labeling"])
        paths = load_project_paths()
        general_config = load_general_config()
        settings = create_settings()

        kwargs = _build_labeling_stage_kwargs(paths, general_config, settings, args)

        assert kwargs["llm_relabeling_enabled"] is True
        assert kwargs["llm_relabeling_score_threshold"] == 0.5
        assert (
            kwargs["llm_relabeling_prompt_name"]
            == "labeling_1_rubrica_few-shot_distribuicao_probabilidade"
        )
        assert kwargs["llm_relabeling_model"] == "UnB-Llama-3.3-70B-Instruct"
        assert kwargs["llm_relabeling_temperature"] == 0.0
        assert kwargs["llm_relabeling_max_retries"] == 3
        assert kwargs["llm_relabeling_n_workers"] == 8
        assert kwargs["llm_relabeling_provider"] == "openai"
        assert kwargs["llm_relabeling_ollama_base_url"] == "http://localhost:11434"

    def test_resolves_ollama_model_when_active_provider_is_ollama(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Com ``configs/llm.yaml -> active_provider: "ollama"``, deve usar ``model_ollama``."""
        monkeypatch.setattr(
            main,
            "load_huggingface_sentiment_pipeline",
            lambda **kwargs: _FakeSentimentPipeline(),
        )
        original_read_yaml = main.read_yaml

        def _fake_read_yaml(path):
            config = original_read_yaml(path)
            if path.name == "llm.yaml":
                config["active_provider"] = "ollama"
            return config

        monkeypatch.setattr(main, "read_yaml", _fake_read_yaml)
        args = parse_arguments(["--stage", "labeling"])
        paths = load_project_paths()
        general_config = load_general_config()
        settings = create_settings()

        kwargs = _build_labeling_stage_kwargs(paths, general_config, settings, args)

        assert kwargs["llm_relabeling_provider"] == "ollama"
        assert kwargs["llm_relabeling_model"] == "llama3.2:1b"
