"""Testes de fixação (pinning) da integração da CLI (``src/main.py``).

Garantem que os argumentos da CLI e os YAMLs de ``configs/`` chegam aos kwargs
de cada etapa: ``--max-workers`` em ``preprocessing``/``labeling``, as duas
fontes de rotulagem (Hugging Face e OpenAI) em ``labeling`` e os limiares da
``comparative_evaluation``. Não é um teste de integração completo da CLI: o
modelo do Hugging Face e a API OpenAI nunca são acionados aqui.
"""

import pytest

from config.paths import load_project_paths
from config.settings import create_settings, load_general_config
from main import (
    _build_comparative_evaluation_stage_kwargs,
    _build_labeling_stage_kwargs,
    _build_preprocessing_stage_kwargs,
    parse_arguments,
)


def _build_kwargs(builder, argv: list[str]):
    """Executa um construtor de kwargs com os argumentos de CLI e a configuração reais."""
    return builder(
        load_project_paths(), load_general_config(), create_settings(), parse_arguments(argv)
    )


class TestBuildPreprocessingStageKwargs:
    """Testes de :func:`main._build_preprocessing_stage_kwargs`."""

    def test_includes_max_workers_from_cli_argument(self) -> None:
        """``--max-workers`` informado na CLI deve chegar aos kwargs da etapa."""
        kwargs = _build_kwargs(
            _build_preprocessing_stage_kwargs, ["--stage", "preprocessing", "--max-workers", "4"]
        )
        assert kwargs["max_workers"] == 4

    def test_defaults_max_workers_to_none_when_not_informed(self) -> None:
        """Sem ``--max-workers``, o valor repassado deve ser ``None`` (o executor decide)."""
        kwargs = _build_kwargs(_build_preprocessing_stage_kwargs, ["--stage", "preprocessing"])
        assert kwargs["max_workers"] is None


class TestBuildLabelingStageKwargs:
    """Testes de :func:`main._build_labeling_stage_kwargs`."""

    def test_builds_both_sources_by_default_in_order(self) -> None:
        """Sem ``--label-source``, as duas fontes são montadas: Hugging Face antes da OpenAI."""
        kwargs = _build_kwargs(_build_labeling_stage_kwargs, ["--stage", "labeling"])

        assert [source.name for source in kwargs["sources"]] == ["huggingface", "openai"]
        assert kwargs["downstream_source"] == "huggingface"
        assert kwargs["low_confidence_threshold"] == 0.5

    @pytest.mark.parametrize("source_name", ["huggingface", "openai"])
    def test_label_source_selects_a_single_source(self, source_name: str) -> None:
        """``--label-source`` restringe a execução a uma única fonte."""
        kwargs = _build_kwargs(
            _build_labeling_stage_kwargs, ["--stage", "labeling", "--label-source", source_name]
        )
        assert [source.name for source in kwargs["sources"]] == [source_name]

    def test_sources_use_models_and_shared_prompt_from_config(self) -> None:
        """Modelos e prompt vêm de ``configs/labeling.yaml``; o prompt é o mesmo nas duas fontes."""
        sources = {
            source.name: source
            for source in _build_kwargs(_build_labeling_stage_kwargs, ["--stage", "labeling"])[
                "sources"
            ]
        }

        assert sources["huggingface"].model_name.startswith(
            "meta-llama/Meta-Llama-3.1-8B-Instruct@"
        )
        assert sources["openai"].model_name == "UnB-Llama-3.3-70B-Instruct"
        assert sources["huggingface"].prompt_template == sources["openai"].prompt_template
        assert "{{TEXTO}}" in sources["openai"].prompt_template

    def test_does_not_load_the_huggingface_model_while_building(self) -> None:
        """Montar a fonte não carrega o modelo: isso só ocorre ao abrir o classificador."""
        kwargs = _build_kwargs(
            _build_labeling_stage_kwargs, ["--stage", "labeling", "--label-source", "huggingface"]
        )
        assert callable(kwargs["sources"][0].open_classifier)


class TestBuildComparativeEvaluationStageKwargs:
    """Testes de :func:`main._build_comparative_evaluation_stage_kwargs`."""

    def test_reads_thresholds_from_evaluation_config(self) -> None:
        """Os limiares e o bootstrap vêm de ``configs/evaluation.yaml -> llm_comparison``."""
        kwargs = _build_kwargs(
            _build_comparative_evaluation_stage_kwargs, ["--stage", "comparative_evaluation"]
        )

        assert kwargs["output_subdir"] == "comparativo_hf_openai"
        assert kwargs["high_confidence_threshold"] == 0.8
        assert kwargs["low_confidence_threshold"] == 0.5
        assert kwargs["n_bootstrap"] == 1000
        assert kwargs["run_hypotheses"] is True
        assert kwargs["hypotheses_targets"] == ("disagreement", "uncertainty")

    def test_skip_hypotheses_flag_disables_hypothesaes(self) -> None:
        """``--skip-hypotheses`` desliga o HypotheSAEs sem alterar o restante da avaliação."""
        kwargs = _build_kwargs(
            _build_comparative_evaluation_stage_kwargs,
            ["--stage", "comparative_evaluation", "--skip-hypotheses"],
        )
        assert kwargs["run_hypotheses"] is False

    def test_random_seed_flag_overrides_config(self) -> None:
        """``--random-seed`` sobrescreve a semente do YAML."""
        kwargs = _build_kwargs(
            _build_comparative_evaluation_stage_kwargs,
            ["--stage", "comparative_evaluation", "--random-seed", "7"],
        )
        assert kwargs["random_seed"] == 7

    def test_needs_no_manual_predictions_function(self) -> None:
        """A etapa é autoexecutável: não existe mais ``--predictions-func``."""
        with pytest.raises(SystemExit):
            parse_arguments(["--stage", "comparative_evaluation", "--predictions-func", "a:b"])


class TestRemovedStages:
    """A etapa ``llm_evaluation`` e o relabeling não fazem mais parte da CLI."""

    def test_llm_evaluation_is_not_a_valid_stage(self) -> None:
        """``--stage llm_evaluation`` deve ser rejeitado pelo parser."""
        with pytest.raises(SystemExit):
            parse_arguments(["--stage", "llm_evaluation"])

    def test_llm_evaluation_flags_are_gone(self) -> None:
        """``--llm-backend`` e ``--llm-strategy`` foram removidos junto com a etapa."""
        with pytest.raises(SystemExit):
            parse_arguments(["--stage", "labeling", "--llm-backend", "ollama"])
