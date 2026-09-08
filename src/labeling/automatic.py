"""Rotuladores automáticos/heurísticos usados na rotulagem em cascata.

Implementa a etapa ``cascade.labelers`` de ``configs/labeling.yaml``: cada
rotulador (heurística, LLM ou modelo de referência) implementa a interface
:class:`SentimentLabeler` e produz um rótulo candidato com confiança para
cada texto. Este módulo fornece a interface comum, um rotulador heurístico
baseado em léxico de sentimento pt-BR (sem dependência de LLM/modelo
treinado, ainda não implementados em ``src/llm`` e ``src/models``) e a
função que executa a cascata completa sobre um corpus, produzindo o
formato longo validado por ``src/schemas/labeling.py``
(:class:`schemas.labeling.LabelingResultSchema`), posteriormente agregado
em ``src/labeling/consensus.py``.
"""

import functools
import logging
import operator
import re
from collections.abc import Mapping
from typing import Protocol

import polars as pl

from constants.labels import NEGATIVE_LABEL, NEUTRAL_LABEL, POSITIVE_LABEL
from parallel.labeling import run_parallel_sentiment_labeling
from preprocessing.emojis import calculate_emoji_sentiment_counts
from schemas.labeling import validate_labeling_result
from utils.validation import validate_not_empty_collection

logger = logging.getLogger(__name__)

_WORD_PATTERN = re.compile(r"\w+")

# Léxico curado de palavras de sentimento em português brasileiro, comuns em
# tweets. Não pretende ser exaustivo (nenhuma biblioteca de léxico de
# sentimento está entre as dependências do projeto — ver CLAUDE.md, "What to
# Avoid" -> dependências sem justificativa): serve como sinal heurístico
# leve e determinístico, um dos rotuladores da cascata definida em
# ``configs/labeling.yaml``.
POSITIVE_WORDS: frozenset[str] = frozenset(
    {
        "bom",
        "boa",
        "otimo",
        "ótimo",
        "otima",
        "ótima",
        "excelente",
        "adorei",
        "amei",
        "maravilhoso",
        "maravilhosa",
        "incrivel",
        "incrível",
        "perfeito",
        "perfeita",
        "recomendo",
        "sensacional",
        "show",
        "top",
        "gostei",
        "feliz",
        "melhor",
        "lindo",
        "linda",
        "sucesso",
        "eficiente",
        "satisfeito",
        "satisfeita",
        "confiavel",
        "confiável",
        "rapido",
        "rápido",
        "agradavel",
        "agradável",
    }
)

NEGATIVE_WORDS: frozenset[str] = frozenset(
    {
        "pessimo",
        "péssimo",
        "pessima",
        "péssima",
        "ruim",
        "horrivel",
        "horrível",
        "odeio",
        "detestei",
        "decepcao",
        "decepção",
        "lixo",
        "pior",
        "triste",
        "raiva",
        "problema",
        "reclamacao",
        "reclamação",
        "cancelei",
        "chateado",
        "chateada",
        "revoltante",
        "decepcionante",
        "lento",
        "lenta",
        "arrependimento",
        "insatisfeito",
        "insatisfeita",
        "nojento",
        "nojenta",
    }
)


class SentimentLabeler(Protocol):
    """Interface que todo rotulador da cascata (heurística, LLM ou modelo) deve implementar.

    Permite que :func:`run_cascade_labeling` orquestre rotuladores de
    naturezas distintas (heurística determinística, classificador LLM via
    ``src/llm``, modelo de referência via ``src/models``) sem acoplamento à
    implementação concreta de cada um.
    """

    def label(self, text: str) -> tuple[str, float]:
        """Classifica o sentimento de um texto.

        Parameters
        ----------
        text : str
            Texto de entrada a ser classificado.

        Returns
        -------
        tuple[str, float]
            Par ``(rótulo_de_sentimento, confiança)``, com o rótulo
            pertencente a :data:`constants.labels.SENTIMENT_CLASSES` e a
            confiança em ``[0.0, 1.0]``.
        """
        ...


_IndexedText = tuple[int, str]
_IndexedLabel = tuple[int, str, float]


def _label_indexed_item(item: _IndexedText, *, labeler: SentimentLabeler) -> _IndexedLabel:
    """Classifica um texto indexado, preservando sua posição original na lista de entrada.

    Necessário porque a execução paralela (``ProcessPoolExecutor``) coleta
    resultados na ordem de conclusão, não na ordem de submissão.

    Parameters
    ----------
    item : tuple[int, str]
        Par ``(índice_original, texto)``.
    labeler : SentimentLabeler
        Rotulador aplicado ao texto.

    Returns
    -------
    tuple[int, str, float]
        Tripla ``(índice_original, rótulo_de_sentimento, confiança)``.
    """
    original_index, text = item
    sentiment_label, confidence_score = labeler.label(text)
    return original_index, sentiment_label, confidence_score


def calculate_lexicon_sentiment_counts(text: str) -> dict[str, int]:
    """Conta palavras do léxico de sentimento positivo/negativo presentes no texto.

    Parameters
    ----------
    text : str
        Texto de entrada.

    Returns
    -------
    dict[str, int]
        Dicionário com as chaves ``"positivo"`` e ``"negativo"``, cada uma
        contendo a contagem de palavras do léxico correspondente
        encontradas no texto (comparação sem distinção de caixa).

    Examples
    --------
    >>> calculate_lexicon_sentiment_counts("o produto é ótimo, adorei")
    {'positivo': 2, 'negativo': 0}
    """
    words = [word.lower() for word in _WORD_PATTERN.findall(text)]
    return {
        POSITIVE_LABEL: sum(1 for word in words if word in POSITIVE_WORDS),
        NEGATIVE_LABEL: sum(1 for word in words if word in NEGATIVE_WORDS),
    }


def classify_by_lexical_heuristic(text: str) -> tuple[str, float]:
    """Classifica o sentimento de um texto combinando léxico de palavras e emojis.

    Combina :func:`calculate_lexicon_sentiment_counts` com
    :func:`preprocessing.emojis.calculate_emoji_sentiment_counts`. O rótulo
    é a polaridade com maior contagem combinada; a confiança é a proporção
    que essa contagem representa do total de sinais encontrados. Sem
    nenhum sinal, retorna rótulo neutro com confiança zero (ausência de
    evidência, não neutralidade observada).

    Parameters
    ----------
    text : str
        Texto de entrada, idealmente já normalizado por
        ``src/preprocessing/pipeline.py``.

    Returns
    -------
    tuple[str, float]
        Par ``(rótulo_de_sentimento, confiança)``.

    Examples
    --------
    >>> classify_by_lexical_heuristic("o produto é ótimo, adorei")
    ('positivo', 1.0)
    >>> classify_by_lexical_heuristic("texto neutro sem sinal")
    ('neutro', 0.0)
    """
    lexicon_counts = calculate_lexicon_sentiment_counts(text)
    emoji_counts = calculate_emoji_sentiment_counts(text)

    positive_score = lexicon_counts[POSITIVE_LABEL] + emoji_counts[POSITIVE_LABEL]
    negative_score = lexicon_counts[NEGATIVE_LABEL] + emoji_counts[NEGATIVE_LABEL]
    total_score = positive_score + negative_score

    if total_score == 0:
        return NEUTRAL_LABEL, 0.0
    if positive_score > negative_score:
        return POSITIVE_LABEL, positive_score / total_score
    if negative_score > positive_score:
        return NEGATIVE_LABEL, negative_score / total_score
    return NEUTRAL_LABEL, 0.5


class LexicalHeuristicLabeler:
    """Rotulador heurístico determinístico baseado em léxico de sentimento e emojis.

    Implementa :class:`SentimentLabeler` delegando para
    :func:`classify_by_lexical_heuristic`. Corresponde ao rotulador
    ``"heuristica_lexica"`` de ``configs/labeling.yaml``.
    """

    def label(self, text: str) -> tuple[str, float]:
        """Classifica o sentimento de um texto pela heurística léxica.

        Parameters
        ----------
        text : str
            Texto de entrada a ser classificado.

        Returns
        -------
        tuple[str, float]
            Par ``(rótulo_de_sentimento, confiança)``.
        """
        return classify_by_lexical_heuristic(text)


def run_cascade_labeling(
    dataframe: pl.DataFrame,
    labelers: Mapping[str, SentimentLabeler],
    *,
    id_column: str = "id",
    text_column: str = "text",
    weights: Mapping[str, float] | None = None,
    max_workers: int | None = None,
    show_progress: bool = True,
) -> pl.DataFrame:
    """Executa a cascata de rotuladores sobre um corpus, produzindo candidatos por amostra.

    Cada combinação (amostra, rotulador) gera uma linha no formato longo
    exigido por :class:`schemas.labeling.LabelingResultSchema`, insumo de
    ``src/labeling/consensus.py`` e ``src/labeling/confidence.py``. Para
    cada rotulador, todos os textos são classificados em paralelo (ver
    :func:`parallel.labeling.run_parallel_sentiment_labeling`), com os
    resultados reordenados pelo índice original antes de montar as
    colunas — a coleta paralela retorna na ordem de conclusão, não na
    ordem de submissão.

    Parameters
    ----------
    dataframe : pl.DataFrame
        Corpus de entrada, contendo ao menos ``id_column`` e
        ``text_column``. Não vazio.
    labelers : Mapping[str, SentimentLabeler]
        Rotuladores a executar, nomeados pela chave (ex.:
        ``"heuristica_lexica"``, correspondendo a ``cascade.labelers`` de
        ``configs/labeling.yaml``). Não vazio.
    id_column : str, optional
        Nome da coluna identificadora de cada amostra, by default "id".
    text_column : str, optional
        Nome da coluna de texto a ser classificada, by default "text".
    weights : Mapping[str, float] | None, optional
        Peso de cada rotulador pelo nome usado em ``labelers``, repassado a
        ``src/labeling/consensus.py`` na agregação ponderada. Rotuladores
        ausentes do mapeamento recebem peso 1.0, by default None.
    max_workers : int | None, optional
        Número máximo de processos usados por rotulador, by default None
        (o executor escolhe automaticamente).
    show_progress : bool, optional
        Se ``True``, exibe uma barra de progresso no console por
        rotulador, by default True.

    Returns
    -------
    pl.DataFrame
        DataFrame no formato longo (``id``, ``tagger``, ``sentiment_label``,
        ``confidence_score``, ``weight``), validado contra
        :class:`schemas.labeling.LabelingResultSchema`.

    Raises
    ------
    EmptyDatasetError
        Se ``dataframe`` ou ``labelers`` estiverem vazios.
    DataValidationError
        Se algum resultado produzido violar o contrato de dados (ex.:
        rótulo fora de :data:`constants.labels.SENTIMENT_CLASSES`).

    Examples
    --------
    >>> df = pl.DataFrame({"id": ["1"], "text": ["adorei o produto"]})
    >>> labelers = {"heuristica_lexica": LexicalHeuristicLabeler()}
    >>> resultado = run_cascade_labeling(
    ...     df, labelers, weights={"heuristica_lexica": 1.0}, show_progress=False
    ... )
    >>> resultado["sentiment_label"].to_list()
    ['positivo']
    """
    validate_not_empty_collection(dataframe, collection_name="dataframe")
    validate_not_empty_collection(labelers, collection_name="labelers")
    resolved_weights = weights or {}

    row_ids = dataframe[id_column].to_list()
    indexed_texts = list(enumerate(dataframe[text_column].to_list()))

    ids: list[str] = []
    taggers: list[str] = []
    sentiment_labels: list[str] = []
    confidences: list[float] = []
    label_weights: list[float] = []

    for tagger_name, labeler in labelers.items():
        labeling_result = run_parallel_sentiment_labeling(
            functools.partial(_label_indexed_item, labeler=labeler),
            indexed_texts,
            max_workers=max_workers,
            show_progress=show_progress,
        )
        if labeling_result.failures:
            raise labeling_result.failures[0].error
        for original_index, sentiment_label, confidence_score in sorted(
            labeling_result.successes, key=operator.itemgetter(0)
        ):
            ids.append(row_ids[original_index])
            taggers.append(tagger_name)
            sentiment_labels.append(sentiment_label)
            confidences.append(confidence_score)
            label_weights.append(resolved_weights.get(tagger_name, 1.0))

    result = pl.DataFrame(
        {
            "id": ids,
            "tagger": taggers,
            "sentiment_label": sentiment_labels,
            "confidence_score": confidences,
            "weight": label_weights,
        }
    )
    logger.info(
        "Rotulagem em cascata concluída: %d amostra(s) x %d rotulador(es) = %d resultado(s).",
        dataframe.height,
        len(labelers),
        result.height,
    )
    return validate_labeling_result(result)
