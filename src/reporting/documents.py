"""Geração dos Model Cards e do Datasheet a partir dos resultados reais.

Complementa os modelos de documento escritos à mão em ``reports/model_cards/`` e
``reports/datasheets/`` (desenho, hiperparâmetros, considerações éticas) com um
documento gerado com os números efetivamente medidos, para que a seção de
resultados nunca fique desatualizada (Mitchell et al., 2019; Gebru et al., 2018).
Nada aqui contém texto bruto de tweets: apenas contagens e estatísticas agregadas.
"""

from collections.abc import Mapping
from typing import Any

import polars as pl

from reporting.tables import build_results_table, dataframe_to_markdown

# Modelo de documento manual que cada Model Card gerado complementa.
_HAND_WRITTEN_CARDS: dict[str, str] = {
    "classical": "model_card_ml_classico.md",
    "deep_learning": "model_card_deep_learning.md",
    "transformer": "model_card_transformer.md",
    "llm": "model_card_llm_local.md",
}
_CARD_INTENDED_USE: dict[str, str] = {
    "classical": (
        "Baseline interpretável e de baixo custo (dummy, Naive Bayes, Regressão Logística, SVM, "
        "Random Forest, Gradient Boosting sobre TF-IDF) para classificação de sentimento em "
        "tweets pt-BR."
    ),
    "deep_learning": "Redes LSTM e CNN treinadas do zero sobre o corpus de tweets pt-BR.",
    "transformer": (
        "Fine-tuning de Transformers pré-treinados em português (BERTimbau, RoBERTa, DistilBERT)."
    ),
    "llm": (
        "LLMs open-source via Ollama em modo few-shot, sem ajuste de pesos; o LLM descreve, "
        "toda agregação é feita em código determinístico."
    ),
}
_LIMITATIONS = (
    "- Os rótulos de treino e teste vêm de rotuladores automáticos (LLMs); as métricas medem "
    "a concordância com esse pseudo-gold, não o acerto sobre a verdade humana. Só um gold set "
    "(TweetSentBR/RePro) ou amostra rotulada por humanos mede acerto.",
    "- Intervalos de confiança por bootstrap sobre o conjunto de teste; diferenças entre modelos "
    "só devem ser afirmadas quando o McNemar (com correção de Holm) for significativo.",
    "- Nenhum atributo sensível é definido neste domínio; a auditoria de fairness não se aplica.",
    "- Sem monitoramento de drift: o desempenho pode degradar com mudanças de vocabulário.",
)


def _best_row(models: pl.DataFrame) -> dict[str, Any]:
    """Linha do melhor modelo (maior F1-macro) da tabela de métricas."""
    return models.sort("f1_macro", descending=True).row(0, named=True)


def render_model_card(
    category: str,
    models: pl.DataFrame,
    per_class: pl.DataFrame,
    slices: pl.DataFrame,
    *,
    mcnemar: pl.DataFrame | None = None,
    ablation: pl.DataFrame | None = None,
    summary: Mapping[str, Any] | None = None,
) -> str:
    """Gera o Model Card (resultados) de uma categoria de modelo.

    Parameters
    ----------
    category : str
        Categoria (``classical``, ``deep_learning``, ``transformer`` ou ``llm``).
    models : pl.DataFrame
        Linhas de ``metricas_por_modelo`` dos modelos da categoria. Não vazio.
    per_class : pl.DataFrame
        Relatório por classe (``model``, ``class``, ``precision``, ``recall``, ``f1``, ``support``).
    slices : pl.DataFrame
        Métricas por faixa de comprimento do texto (com a coluna ``model``).
    mcnemar : pl.DataFrame | None, optional
        Comparações par a par de todos os modelos, by default None.
    ablation : pl.DataFrame | None, optional
        Tabela de ablação (relevante só para ``classical``), by default None.
    summary : Mapping[str, Any] | None, optional
        ``reports/metrics/avaliacao.json``, by default None.

    Returns
    -------
    str
        Documento em Markdown.

    Examples
    --------
    >>> render_model_card("classical", modelos, por_classe, fatias)  # doctest: +SKIP
    """
    names = models["model"].to_list()
    best = _best_row(models)
    category_label = models["category_label"][0]
    lines = [
        f"# Model Card (resultados) — {category_label}",
        "",
        f"> Gerado automaticamente pela etapa `report`. Complementa o documento escrito à mão "
        f"[`{_HAND_WRITTEN_CARDS[category]}`](./{_HAND_WRITTEN_CARDS[category]}) com os números "
        "medidos no conjunto de teste.",
        "",
        "## Modelos avaliados",
        "",
        ", ".join(f"`{name}`" for name in names),
        "",
        "## Uso pretendido",
        "",
        _CARD_INTENDED_USE[category],
        "",
        "## Métricas no conjunto de teste (IC 95% bootstrap)",
        "",
        dataframe_to_markdown(build_results_table(models)),
        "",
        f"Melhor modelo da categoria: **{best['model']}** (F1-macro = {best['f1_macro']:.3f}, "
        f"IC 95% [{best['f1_macro_ci_low']:.3f}, {best['f1_macro_ci_high']:.3f}]).",
        "",
        "## Desempenho por classe",
        "",
        dataframe_to_markdown(
            per_class.filter(pl.col("model").is_in(names))
            .select("model", "class", "precision", "recall", "f1", "support")
            .with_columns(pl.col("precision", "recall", "f1").round(3))
        ),
        "",
        "## Desempenho por comprimento do texto",
        "",
        dataframe_to_markdown(
            slices.filter(pl.col("model").is_in(names))
            .select("model", "slice", "n_samples", "f1_macro", "mcc")
            .with_columns(pl.col("f1_macro", "mcc").round(3))
        ),
        "",
    ]

    if mcnemar is not None and not mcnemar.is_empty():
        within = mcnemar.filter(pl.col("model_a").is_in(names) & pl.col("model_b").is_in(names))
        if not within.is_empty():
            lines += [
                "## Comparação estatística (McNemar, correção de Holm)",
                "",
                dataframe_to_markdown(
                    within.select(
                        "model_a", "model_b", "n_common", "p_value_holm", "significant"
                    ).with_columns(pl.col("p_value_holm").round(4))
                ),
                "",
            ]

    if category == "classical" and ablation is not None and not ablation.is_empty():
        lines += [
            "## Ablação do pipeline (conjunto de validação)",
            "",
            dataframe_to_markdown(
                ablation.select(
                    "component",
                    "baseline_value",
                    "ablated_value",
                    "impact",
                    "impact_ci_low",
                    "impact_ci_high",
                ).with_columns(pl.col(pl.Float64).round(4))
            ),
            "",
        ]

    if summary is not None:
        lines += [
            "## Contexto da avaliação",
            "",
            f"- Tweets de teste: {summary['n_test']}; modelos avaliados no total: "
            f"{summary['n_models']}.",
            f"- Melhor modelo geral: `{summary['best_model']['model']}`.",
            "",
        ]
    lines += ["## Limitações e considerações éticas", "", *_LIMITATIONS, ""]
    return "\n".join(lines)


def render_datasheet(
    split_summary: pl.DataFrame,
    class_distribution: pl.DataFrame,
    length_summary: pl.DataFrame,
) -> str:
    """Gera o Datasheet (estatísticas reais) do corpus particionado.

    Parameters
    ----------
    split_summary : pl.DataFrame
        Uma linha por partição (``split``, ``n_tweets``).
    class_distribution : pl.DataFrame
        Contagem e proporção por partição e classe (``split``, ``sentiment_label``, ``count``,
        ``proportion``).
    length_summary : pl.DataFrame
        Estatísticas de palavras por tweet por partição (``split``, ``mean_words``,
        ``median_words``, ``p95_words``).

    Returns
    -------
    str
        Documento em Markdown, sem nenhum texto bruto de tweet.

    Examples
    --------
    >>> render_datasheet(particoes, distribuicao, comprimentos)  # doctest: +SKIP
    """
    return "\n".join(
        [
            "# Datasheet (estatísticas) — Corpus de Tweets pt-BR",
            "",
            "> Gerado automaticamente pela etapa `report`. Complementa o documento escrito à mão "
            "[`datasheet_corpus_tweets.md`](./datasheet_corpus_tweets.md) com as contagens reais "
            "das partições. Contém apenas estatísticas agregadas: nenhum texto de tweet, "
            "@menção, URL ou identificador (LGPD).",
            "",
            "## Partições",
            "",
            dataframe_to_markdown(split_summary),
            "",
            "## Distribuição das classes de sentimento",
            "",
            dataframe_to_markdown(class_distribution.with_columns(pl.col("proportion").round(4))),
            "",
            "## Comprimento dos tweets (palavras, texto normalizado)",
            "",
            dataframe_to_markdown(length_summary.with_columns(pl.col(pl.Float64).round(2))),
            "",
            "## Observações",
            "",
            "- Split estratificado pela classe, com semente fixa (`configs/config.yaml -> "
            "data_split`).",
            "- Rótulos atribuídos por rotuladores automáticos (consenso/LLMs); ver o datasheet "
            "manual para a base legal, a coleta e as limitações.",
            "",
        ]
    )
