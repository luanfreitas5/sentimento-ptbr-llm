"""Testes do consenso de rotulagem (módulo ``evaluation.label_consensus`` e a etapa)."""

from pathlib import Path

import polars as pl
import pytest

from config.paths import ProjectPaths
from data.writer import write_dataset
from evaluation.label_consensus import (
    CONSENSUS_LEVELS,
    NO_CONSENSUS_LABEL,
    build_consensus_frame,
    calculate_multi_rater_agreement,
    calculate_pairwise_agreement,
    find_disputed_tweets,
    summarize_consensus_by_class,
    summarize_consensus_levels,
)
from exceptions.data import DataNotFoundError, DataValidationError, EmptyDatasetError
from pipelines.label_consensus import run_label_consensus_stage
from pipelines.workflow import STAGE_REGISTRY
from tests.test_pipelines import pipeline_paths  # noqa: F401  (fixture reutilizada)

_NEG, _NEU, _POS = "negativo", "neutro", "positivo"

# Votos de 4 rotuladores (a, b, c, d) em 6 tweets, cobrindo todos os níveis de consenso.
_VOTES: dict[str, list[str]] = {
    "a": [_POS, _POS, _POS, _POS, _NEG, _NEU],
    "b": [_POS, _POS, _POS, _NEG, _POS, _NEU],
    "c": [_POS, _POS, _NEU, _NEG, _NEU, _POS],
    "d": [_POS, _NEG, _NEU, _NEG, _POS, _NEG],
}
# tweet 0: 4x pos (unanimidade); 1: 3 pos + 1 neg (maioria); 2: 2 pos + 2 neu (empate);
# 3: 1 pos + 3 neg (maioria); 4: 2 pos + 1 neg + 1 neu (pluralidade); 5: 2 neu + 1 pos + 1 neg
# (pluralidade).


def _build_source(labels: list[str], *, confidence: float = 0.8) -> pl.DataFrame:
    """Base rotulada sintética no contrato ``LabeledSourceSchema``."""
    ids = [str(index) for index in range(len(labels))]
    return pl.DataFrame(
        {
            "id": ids,
            "text": [f"texto {i}" for i in ids],
            "text_normalized": [f"texto {i}" for i in ids],
            "sentiment_label": labels,
            "confidence_score": [confidence] * len(labels),
        }
    )


@pytest.fixture
def sources() -> dict[str, pl.DataFrame]:
    """Quatro rotuladores sintéticos sobre os mesmos seis tweets."""
    return {name: _build_source(labels) for name, labels in _VOTES.items()}


class TestBuildConsensusFrame:
    """Testes de :func:`evaluation.label_consensus.build_consensus_frame`."""

    def test_assigns_consensus_level_and_label(self, sources: dict[str, pl.DataFrame]) -> None:
        """Cada tweet recebe o nível e a classe de consenso esperados."""
        frame = build_consensus_frame(sources)

        assert frame["consensus_level"].to_list() == [
            "unanimidade",
            "maioria",
            "empate",
            "maioria",
            "pluralidade",
            "pluralidade",
        ]
        assert frame["consensus_label"].to_list() == [
            _POS,
            _POS,
            NO_CONSENSUS_LABEL,
            _NEG,
            _POS,
            _NEU,
        ]
        assert frame["n_raters"].unique().to_list() == [4]

    def test_votes_sum_to_number_of_raters(self, sources: dict[str, pl.DataFrame]) -> None:
        """Invariante: os votos por classe de cada tweet somam o número de rotuladores."""
        frame = build_consensus_frame(sources)

        total_votes = frame.select(
            (pl.col("votes_negativo") + pl.col("votes_neutro") + pl.col("votes_positivo")).alias(
                "total"
            )
        )["total"]
        assert total_votes.to_list() == [4] * frame.height

    def test_flags_polarity_conflict(self, sources: dict[str, pl.DataFrame]) -> None:
        """Voto ``negativo`` e ``positivo`` no mesmo tweet marca conflito de polaridade."""
        frame = build_consensus_frame(sources)

        assert frame["polarity_conflict"].to_list() == [False, True, False, True, True, True]

    def test_drops_tweets_undefined_in_any_source(self, sources: dict[str, pl.DataFrame]) -> None:
        """Tweet ``indefinido`` em algum rotulador fica fora (a matriz de votos é completa)."""
        undefined = _VOTES["a"].copy()
        undefined[0] = "indefinido"
        sources["a"] = _build_source(undefined)

        frame = build_consensus_frame(sources)

        assert frame.height == 5
        assert "0" not in frame["id"].to_list()

    def test_requires_at_least_two_raters(self, sources: dict[str, pl.DataFrame]) -> None:
        """Um único rotulador não forma consenso."""
        with pytest.raises(DataValidationError):
            build_consensus_frame({"a": sources["a"]})

    def test_rejects_different_tweet_sets(self, sources: dict[str, pl.DataFrame]) -> None:
        """As bases devem conter exatamente os mesmos ``id``."""
        sources["b"] = sources["b"].head(5)

        with pytest.raises(DataValidationError, match="mesmos tweets"):
            build_consensus_frame(sources)

    def test_rejects_empty_source(self, sources: dict[str, pl.DataFrame]) -> None:
        """Base vazia levanta ``EmptyDatasetError``."""
        sources["b"] = sources["b"].clear()

        with pytest.raises(EmptyDatasetError):
            build_consensus_frame(sources)

    def test_rejects_labels_outside_the_contract(self, sources: dict[str, pl.DataFrame]) -> None:
        """Rótulo fora das classes viola o contrato de dados."""
        sources["a"] = _build_source(["feliz", *_VOTES["a"][1:]])

        with pytest.raises(DataValidationError):
            build_consensus_frame(sources)


class TestSummaries:
    """Testes dos resumos de nível, classe, disputa e concordância."""

    def test_levels_cover_every_level_and_sum_to_total(
        self, sources: dict[str, pl.DataFrame]
    ) -> None:
        """Há uma linha por nível (inclusive vazio) e as contagens somam o total."""
        frame = build_consensus_frame(sources)

        levels = summarize_consensus_levels(frame)

        assert levels["consensus_level"].to_list() == list(CONSENSUS_LEVELS)
        assert levels["n_tweets"].to_list() == [1, 2, 2, 1]
        assert levels["n_tweets"].sum() == frame.height
        assert all(
            lower <= proportion <= upper
            for lower, proportion, upper in zip(
                levels["ci_lower"], levels["proportion"], levels["ci_upper"], strict=True
            )
        )

    def test_by_class_counts_levels_within_each_class(
        self, sources: dict[str, pl.DataFrame]
    ) -> None:
        """A tabela cruza classe de consenso e nível, com taxa de unanimidade."""
        frame = build_consensus_frame(sources)

        by_class = summarize_consensus_by_class(frame).to_dicts()
        rows = {row["consensus_label"]: row for row in by_class}

        assert rows[_POS]["n_tweets"] == 3
        assert rows[_POS]["n_unanimidade"] == 1
        assert rows[_POS]["unanimity_rate"] == pytest.approx(1 / 3)
        assert rows[NO_CONSENSUS_LABEL]["n_empate"] == 1
        assert sum(row["n_tweets"] for row in by_class) == frame.height

    def test_pairwise_agreement_is_perfect_for_identical_raters(
        self, sources: dict[str, pl.DataFrame]
    ) -> None:
        """Dois rotuladores idênticos têm concordância 1 e Kappa 1."""
        sources["b"] = sources["a"]
        frame = build_consensus_frame(sources)

        pairwise = calculate_pairwise_agreement(frame, list(sources), n_bootstrap=30)

        row = pairwise.filter((pl.col("source_a") == "a") & (pl.col("source_b") == "b")).row(
            0, named=True
        )
        assert row["agreement_rate"] == 1.0
        assert row["cohen_kappa"] == pytest.approx(1.0)
        assert pairwise.height == 6  # C(4, 2)

    def test_multi_rater_agreement_reports_fleiss_and_krippendorff(
        self, sources: dict[str, pl.DataFrame]
    ) -> None:
        """Kappa de Fleiss e Alpha de Krippendorff saem com IC e taxa de unanimidade."""
        frame = build_consensus_frame(sources)

        summary = calculate_multi_rater_agreement(frame, list(sources), n_bootstrap=30)

        assert summary["n_raters"] == 4
        assert summary["unanimity_rate"] == pytest.approx(1 / 6)
        assert -1.0 <= summary["fleiss_kappa"] <= 1.0
        assert -1.0 <= summary["krippendorff_alpha"] <= 1.0

    def test_unanimous_raters_give_perfect_fleiss_kappa(self) -> None:
        """Rotuladores idênticos (com variação entre tweets) têm Fleiss = 1."""
        base = _build_source([_POS, _NEG, _NEU, _POS])
        frame = build_consensus_frame({"a": base, "b": base, "c": base})

        summary = calculate_multi_rater_agreement(frame, ["a", "b", "c"], n_bootstrap=20)

        assert summary["fleiss_kappa"] == pytest.approx(1.0)
        assert summary["unanimity_rate"] == 1.0

    def test_disputed_tweets_exclude_unanimous_and_start_with_polarity_conflict(
        self, sources: dict[str, pl.DataFrame]
    ) -> None:
        """Tweets em disputa não incluem unânimes e priorizam polaridade oposta."""
        frame = build_consensus_frame(sources)

        disputed = find_disputed_tweets(frame, top_n=3)

        assert disputed.height == 3
        assert "0" not in disputed["id"].to_list()  # único unânime
        assert disputed["polarity_conflict"].to_list()[0] is True
        assert "text" not in disputed.columns  # só o texto normalizado é exposto


def _write_consensus_bases(paths: ProjectPaths, sources: dict[str, pl.DataFrame]) -> None:
    """Grava as quatro bases; o Hugging Face recebe tweets extras (corpus inteiro)."""
    extra = _build_source([_POS] * 4).with_columns(
        (pl.col("id").cast(pl.Int64) + 100).cast(pl.String).alias("id")
    )
    write_dataset(pl.concat([sources["a"], extra]), paths.huggingface_labeled_file)
    write_dataset(sources["b"], paths.huggingface_cardiff_file)
    write_dataset(sources["c"], paths.gold_openai_file)
    write_dataset(sources["d"], paths.gold_unb_file)


class TestRunLabelConsensusStage:
    """Testes de :func:`pipelines.label_consensus.run_label_consensus_stage`."""

    def test_writes_tables_figures_and_metrics(
        self,
        pipeline_paths: ProjectPaths,  # noqa: F811
        sources: dict[str, pl.DataFrame],
    ) -> None:
        """A etapa restringe o HF à amostra e grava tabelas, figuras e o JSON de métricas."""
        _write_consensus_bases(pipeline_paths, sources)

        result = run_label_consensus_stage(pipeline_paths, n_bootstrap=30)

        assert result.consensus_frame.height == 6  # tweets extras do HF ficaram de fora
        assert result.summary["sources"] == [
            "huggingface",
            "huggingface_cardiff",
            "gold_openai",
            "gold_unb",
        ]
        assert set(result.tables) == {
            "consenso_por_tweet",
            "niveis_de_consenso",
            "consenso_por_classe",
            "concordancia_par_a_par",
            "tweets_em_disputa",
        }
        assert all(path.is_file() for path in result.tables.values())
        assert all(path.is_file() for path in result.figures)
        assert len(result.figures) == 4  # 2 gráficos x (PNG + SVG)
        assert result.metrics_path.is_file()
        summary_md = result.tables["consenso_por_tweet"].parent / "resumo_consenso.md"
        assert "Kappa de Fleiss" in summary_md.read_text(encoding="utf-8")

    def test_runs_with_the_available_raters_only(
        self,
        pipeline_paths: ProjectPaths,  # noqa: F811
        sources: dict[str, pl.DataFrame],
    ) -> None:
        """Rotuladores ainda não gerados são dispensados (mínimo de dois)."""
        _write_consensus_bases(pipeline_paths, sources)
        pipeline_paths.gold_unb_file.unlink()
        pipeline_paths.huggingface_cardiff_file.unlink()

        result = run_label_consensus_stage(pipeline_paths, n_bootstrap=30)

        assert result.summary["sources"] == ["huggingface", "gold_openai"]

    def test_raises_when_fewer_than_two_raters(
        self,
        pipeline_paths: ProjectPaths,  # noqa: F811
        sources: dict[str, pl.DataFrame],
    ) -> None:
        """Com menos de dois rotuladores, orienta a gerar as bases faltantes."""
        write_dataset(sources["c"], pipeline_paths.gold_openai_file)

        with pytest.raises(DataNotFoundError, match="make label-hf-cardiff"):
            run_label_consensus_stage(pipeline_paths)

    def test_raises_when_sample_has_tweets_missing_from_huggingface(
        self,
        pipeline_paths: ProjectPaths,  # noqa: F811
        sources: dict[str, pl.DataFrame],
    ) -> None:
        """Tweet da amostra ausente no HF é erro de dados, não descarte silencioso."""
        _write_consensus_bases(pipeline_paths, sources)
        write_dataset(sources["a"].head(4), pipeline_paths.huggingface_labeled_file)

        with pytest.raises(DataValidationError, match="ausente"):
            run_label_consensus_stage(pipeline_paths)

    def test_stage_is_registered(self) -> None:
        """A etapa está disponível na CLI/orquestração."""
        assert "label_consensus" in STAGE_REGISTRY

    def test_default_output_lives_under_reports(
        self,
        pipeline_paths: ProjectPaths,  # noqa: F811
        sources: dict[str, pl.DataFrame],
    ) -> None:
        """Os arquivos ficam em ``reports/{tables,metrics}/consenso_rotulagem``."""
        _write_consensus_bases(pipeline_paths, sources)

        result = run_label_consensus_stage(pipeline_paths, n_bootstrap=20)

        assert result.metrics_path == (
            pipeline_paths.reports_metrics_dir / "consenso_rotulagem.json"
        )
        assert isinstance(next(iter(result.tables.values())), Path)
