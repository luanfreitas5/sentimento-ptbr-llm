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
    _STAGE_KWARGS_BUILDERS,
    _build_comparative_evaluation_stage_kwargs,
    _build_evaluate_stage_kwargs,
    _build_hypotheses_stage_kwargs,
    _build_labeling_stage_kwargs,
    _build_preprocessing_stage_kwargs,
    _build_report_stage_kwargs,
    _build_training_deep_learning_stage_kwargs,
    _build_training_llm_stage_kwargs,
    _build_training_transformer_stage_kwargs,
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

    def test_sources_use_models_and_prompt_from_config(self) -> None:
        """Modelos e prompt vêm de ``configs/labeling.yaml``; só a fonte OpenAI usa o prompt."""
        sources = {
            source.name: source
            for source in _build_kwargs(_build_labeling_stage_kwargs, ["--stage", "labeling"])[
                "sources"
            ]
        }

        assert sources["huggingface"].model_name.startswith("pysentimiento/bertweet-pt-sentiment@")
        assert sources["huggingface"].prompt_template == ""
        assert sources["openai"].model_name == "UnB-Llama-3.3-70B-Instruct"
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
        assert "run_hypotheses" not in kwargs  # hipóteses têm estágio próprio (`hypotheses`)

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


class TestCategoryTrainingStages:
    """Cada categoria de modelo tem estágio próprio, com os modelos e hiperparâmetros corretos."""

    def test_stages_are_registered_in_the_canonical_order(self) -> None:
        """``--stage all`` roda treino por categoria, depois ``evaluate`` e ``report``."""
        stages = load_general_config().stages
        assert stages[stages.index("features") :] == [
            "features",
            "training_classical",
            "training_deep_learning",
            "training_transformer",
            "training_llm",
            "evaluate",
            "report",
        ]
        assert set(stages) <= set(_STAGE_KWARGS_BUILDERS)

    @pytest.mark.parametrize("stage", ["training_deep_learning", "training_transformer"])
    def test_neural_stages_use_only_their_own_models(self, stage: str, monkeypatch) -> None:
        """Deep Learning treina lstm/cnn; Transformer treina os três Transformers."""
        import main

        monkeypatch.setattr(main, "_load_split_texts", lambda path: (["a"], ["positivo"]))
        builder = {
            "training_deep_learning": _build_training_deep_learning_stage_kwargs,
            "training_transformer": _build_training_transformer_stage_kwargs,
        }[stage]

        kwargs = _build_kwargs(builder, ["--stage", stage])

        expected = {
            "training_deep_learning": {"lstm", "cnn"},
            "training_transformer": {"bertimbau", "roberta", "distilbert"},
        }[stage]
        assert set(kwargs["model_names"]) == expected
        assert set(kwargs["model_params"]) == expected
        assert kwargs["X_val"] == ["a"]  # a validação alimenta a parada antecipada

    def test_model_names_flag_restricts_the_stage(self, monkeypatch) -> None:
        """``--model-names`` restringe os modelos do estágio."""
        import main

        monkeypatch.setattr(main, "_load_split_texts", lambda path: (["a"], ["positivo"]))
        kwargs = _build_kwargs(
            _build_training_transformer_stage_kwargs,
            ["--stage", "training_transformer", "--model-names", "bertimbau"],
        )
        assert kwargs["model_names"] == ("bertimbau",)

    def test_llm_stage_reads_ollama_models_and_sample_limit(self, monkeypatch) -> None:
        """O LLM usa ``model_params.yaml -> llm`` e o limite de amostras de ``evaluation.yaml``."""
        import main

        monkeypatch.setattr(main, "_load_split_texts", lambda path: (["a"], ["positivo"]))
        kwargs = _build_kwargs(_build_training_llm_stage_kwargs, ["--stage", "training_llm"])

        assert kwargs["model_names"] == ("llama3_2",)
        assert kwargs["model_params"]["llama3_2"]["model_name"] == "llama3.2"
        assert kwargs["max_validation_samples"] == 500
        assert kwargs["random_seed"] == load_general_config().reproducibility.random_seed


class TestEvaluateAndReportStages:
    """``evaluate`` e ``report`` recebem a configuração de ``configs/``."""

    def test_evaluate_reads_uncertainty_significance_and_ablation(self) -> None:
        """Bootstrap, alfa e componentes da ablação vêm de ``configs/evaluation.yaml``."""
        kwargs = _build_kwargs(_build_evaluate_stage_kwargs, ["--stage", "evaluate"])

        assert kwargs["n_bootstrap"] == 1000
        assert kwargs["confidence_level"] == 0.95
        assert kwargs["alpha"] == 0.05
        assert kwargs["skip_ablation"] is False
        ablation = kwargs["ablation_config"]
        assert "sem_bigramas" in ablation["components"]
        # a ablação parte dos hiperparâmetros da regressão logística de model_params.yaml
        assert ablation["model_params"]["class_weight"] == "balanced"

    def test_skip_ablation_and_model_names_flags(self) -> None:
        """``--skip-ablation`` e ``--model-names`` chegam aos kwargs."""
        kwargs = _build_kwargs(
            _build_evaluate_stage_kwargs,
            ["--stage", "evaluate", "--skip-ablation", "--model-names", "svm,dummy"],
        )
        assert kwargs["skip_ablation"] is True
        assert kwargs["model_names"] == ("svm", "dummy")

    def test_report_only_needs_paths(self) -> None:
        """``report`` só lê o disco: o único argumento é ``paths``."""
        assert set(_build_kwargs(_build_report_stage_kwargs, ["--stage", "report"])) == {"paths"}


class TestHypothesesStageKwargs:
    """O estágio ``hypotheses`` monta apenas os argumentos do modo escolhido."""

    def test_default_mode_is_disagreement_with_targets_from_config(self) -> None:
        """Sem ``--hypotheses-mode``, usa a discordância e os alvos de ``evaluation.yaml``."""
        kwargs = _build_kwargs(_build_hypotheses_stage_kwargs, ["--stage", "hypotheses"])

        assert kwargs["mode"] == "disagreement"
        assert kwargs["skip"] is False
        assert kwargs["targets"] == ("disagreement", "uncertainty")
        assert kwargs["output_subdir"] == "comparativo_hf_openai"
        assert "patterns_kwargs" not in kwargs
        assert "diagnostics_kwargs" not in kwargs

    def test_skip_hypotheses_builds_nothing_heavy(self) -> None:
        """``--skip-hypotheses`` dispensa a etapa sem montar corpus, LLM nem diagnóstico."""
        kwargs = _build_kwargs(
            _build_hypotheses_stage_kwargs,
            ["--stage", "hypotheses", "--hypotheses-mode", "patterns", "--skip-hypotheses"],
        )
        assert kwargs["skip"] is True
        assert "patterns_kwargs" not in kwargs

    def test_diagnostics_mode_forwards_the_diagnostics_flags(self) -> None:
        """No modo ``diagnostics``, as flags ``--diagnostics-*`` viram ``diagnostics_kwargs``."""
        kwargs = _build_kwargs(
            _build_hypotheses_stage_kwargs,
            [
                "--stage", "hypotheses", "--hypotheses-mode", "diagnostics",
                "--diagnostics-step", "validation", "--dry-run",
            ],
        )  # fmt: skip
        assert kwargs["diagnostics_kwargs"]["step"] == "validation"
        assert kwargs["diagnostics_kwargs"]["dry_run"] is True
        assert "paths" not in kwargs["diagnostics_kwargs"]

    def test_removed_stage_names_are_rejected(self) -> None:
        """``hypothesaes_analysis`` e ``diagnostics`` deixaram de ser estágios."""
        for stage in ("hypothesaes_analysis", "diagnostics"):
            with pytest.raises(SystemExit):
                parse_arguments(["--stage", stage])
