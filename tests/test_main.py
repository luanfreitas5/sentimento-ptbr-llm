"""Testes de fixação (pinning) da integração da CLI (``src/main.py``) com ``--max-workers``.

A revisão final da branch apontou que nada no test suite exercitava
``src/main.py`` diretamente: nenhum teste garantia que ``--max-workers``
realmente chegava aos kwargs das etapas ``preprocessing``/``labeling`` (ver
Tasks 7/10/11 do plano de paralelização). Este arquivo cobre apenas esse
ponto específico — não é um teste de integração completo da CLI.
"""

from config.paths import load_project_paths
from config.settings import create_settings, load_general_config
from main import _build_labeling_stage_kwargs, _build_preprocessing_stage_kwargs, parse_arguments


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
    """Testes de :func:`main._build_labeling_stage_kwargs`."""

    def test_includes_max_workers_from_cli_argument(self) -> None:
        """``--max-workers`` informado na CLI deve chegar aos kwargs da etapa."""
        args = parse_arguments(["--stage", "labeling", "--max-workers", "3"])
        paths = load_project_paths()
        general_config = load_general_config()
        settings = create_settings()

        kwargs = _build_labeling_stage_kwargs(paths, general_config, settings, args)

        assert kwargs["max_workers"] == 3

    def test_defaults_max_workers_to_none_when_not_informed(self) -> None:
        """Sem ``--max-workers``, o valor repassado deve ser ``None`` (o executor decide)."""
        args = parse_arguments(["--stage", "labeling"])
        paths = load_project_paths()
        general_config = load_general_config()
        settings = create_settings()

        kwargs = _build_labeling_stage_kwargs(paths, general_config, settings, args)

        assert kwargs["max_workers"] is None
