"""Composição das etapas de pré-processamento em um pipeline reprodutível.

Encadeia a limpeza (``src/preprocessing/cleaning.py``), a normalização de
elementos estruturais (``src/preprocessing/text.py``) e de emojis
(``src/preprocessing/emojis.py``) em uma única função por texto e aplica o
resultado a um corpus inteiro, junto aos critérios de inclusão/exclusão
(``src/preprocessing/filtering.py``) e, opcionalmente, à tokenização
(``src/preprocessing/tokenization.py``) e à lematização via spaCy
(``src/preprocessing/lemmatization.py``), produzindo o dataset intermediário
usado pelas etapas seguintes (rotulagem e extração de features).
"""

import functools
import logging
import operator
import os

import polars as pl

from exceptions.pipeline import PipelineStageError
from parallel.preprocessing import run_parallel_text_cleaning
from preprocessing.cleaning import clean_tweet_text
from preprocessing.emojis import normalize_emojis
from preprocessing.filtering import filter_by_inclusion_criteria
from preprocessing.lemmatization import lemmatize_text
from preprocessing.text import (
    normalize_hashtags,
    normalize_mentions,
    normalize_repeated_characters,
    normalize_urls,
)
from preprocessing.tokenization import tokenize_and_normalize
from utils.text import normalize_whitespace
from utils.validation import validate_not_empty_collection

logger = logging.getLogger(__name__)

_IndexedText = tuple[int, str]

# Abaixo deste número de linhas, a normalização roda em um laço serial puro
# (sem executor): o custo fixo de inicializar um ProcessPoolExecutor supera
# qualquer ganho de paralelismo para lotes pequenos (ver docs/superpowers/
# specs/2026-09-07-batch-raw-tweet-pipeline-design.md, revisão final).
_SERIAL_EXECUTION_THRESHOLD = 2000


def _calculate_chunk_size(n_items: int, max_workers: int | None) -> int:
    """Calcula o tamanho de lote (chunk) usado na normalização paralela em larga escala.

    Um chunk pequeno demais reintroduz o overhead de round-trip por item
    que a paralelização em lotes existe para eliminar; um chunk grande
    demais reduz o número de lotes abaixo do número de workers disponíveis,
    deixando-os ociosos. ``max(50, ...)`` evita lotes minúsculos quando
    ``n_items`` é apenas um pouco maior que o limiar serial; dividir por
    ``workers * 4`` produz lotes suficientes para balancear o trabalho
    entre os workers mesmo com alguma variação de duração por item.

    Parameters
    ----------
    n_items : int
        Número total de itens a processar.
    max_workers : int | None
        Número máximo de processos configurado pelo chamador; ``None``
        usa ``os.cpu_count()`` (ou 4, se indisponível) como estimativa.

    Returns
    -------
    int
        Tamanho de lote a repassar a ``run_parallel_text_cleaning``.
    """
    workers = max_workers or (os.cpu_count() or 4)
    return max(50, n_items // (workers * 4))


def normalize_tweet_text(text: str, *, keep_hashtag_word: bool = True) -> str:
    """Aplica a sequência completa de limpeza e normalização a um único tweet.

    Ordem das etapas: remoção do marcador de retweet, substituição de URLs
    e menções por tokens, normalização de hashtags, mapeamento de emojis
    para tokens semânticos, redução de repetições ortográficas e
    normalização final de espaçamento. A remoção do marcador de retweet
    precisa ocorrer antes da normalização de menções, pois o padrão de
    retweet depende do ``@usuario`` original no início do texto.

    Parameters
    ----------
    text : str
        Texto bruto de um tweet.
    keep_hashtag_word : bool, optional
        Repassado a :func:`preprocessing.text.normalize_hashtags`, by
        default True.

    Returns
    -------
    str
        Texto normalizado, pronto para tokenização ou extração de features.

    Examples
    --------
    >>> normalize_tweet_text("RT @exemplo: amei o produto!! 😍 #recomendo https://exemplo.com")
    'amei o produto!! [EMOJI_POSITIVO] recomendo [URL]'
    """
    normalized = clean_tweet_text(text)
    normalized = normalize_urls(normalized)
    normalized = normalize_mentions(normalized)
    normalized = normalize_hashtags(normalized, keep_word=keep_hashtag_word)
    normalized = normalize_emojis(normalized)
    normalized = normalize_repeated_characters(normalized)
    return normalize_whitespace(normalized)


def _normalize_row_text(text: str, *, keep_hashtag_word: bool) -> str:
    """Normaliza um texto individual, convertendo falhas em ``PipelineStageError``.

    Parameters
    ----------
    text : str
        Texto bruto de um tweet.
    keep_hashtag_word : bool
        Repassado a :func:`normalize_tweet_text`.

    Returns
    -------
    str
        Texto normalizado.

    Raises
    ------
    PipelineStageError
        Se a normalização falhar para o texto informado.
    """
    try:
        return normalize_tweet_text(text, keep_hashtag_word=keep_hashtag_word)
    except Exception as exception:  # captura ampla e proposital: isola a falha de uma linha
        logger.exception("Falha ao normalizar o texto de um tweet")
        raise PipelineStageError(
            stage_name="normalizacao_texto", detail=str(exception)
        ) from exception


def _normalize_indexed_row_text(item: _IndexedText, *, keep_hashtag_word: bool) -> _IndexedText:
    """Normaliza um texto indexado, preservando sua posição original na lista de entrada.

    Necessário porque a execução paralela (``ProcessPoolExecutor``) coleta
    resultados na ordem de conclusão, não na ordem de submissão — sem o
    índice original, a coluna resultante ficaria desalinhada em relação às
    demais linhas do DataFrame.

    Parameters
    ----------
    item : tuple[int, str]
        Par ``(índice_original, texto_bruto)``.
    keep_hashtag_word : bool
        Repassado a :func:`_normalize_row_text`.

    Returns
    -------
    tuple[int, str]
        Par ``(índice_original, texto_normalizado)``.
    """
    original_index, text = item
    return original_index, _normalize_row_text(text, keep_hashtag_word=keep_hashtag_word)


def run_preprocessing_pipeline(
    dataframe: pl.DataFrame,
    *,
    text_column: str = "text",
    normalized_text_column: str = "text_normalized",
    tokens_column: str | None = None,
    lemmatized_text_column: str | None = None,
    keep_hashtag_word: bool = True,
    expand_slang: bool = True,
    apply_negation_marking: bool = True,
    apply_inclusion_filters: bool = True,
    minimum_characters: int = 5,
    minimum_words: int = 2,
    minimum_portuguese_ratio: float = 0.15,
    max_repeated_word_ratio: float = 0.5,
    drop_duplicate_text: bool = True,
    max_workers: int | None = None,
    show_progress: bool = True,
) -> pl.DataFrame:
    """Executa o pipeline reprodutível de pré-processamento sobre um corpus de tweets.

    Adiciona ``normalized_text_column`` com o resultado de
    :func:`normalize_tweet_text` aplicado a cada linha e, opcionalmente,
    filtra o corpus pelos critérios de inclusão de
    ``src/preprocessing/filtering.py`` (conteúdo mínimo, idioma provável,
    ausência de spam e duplicatas) e adiciona uma coluna de tokens via
    :func:`preprocessing.tokenization.tokenize_and_normalize`.

    Parameters
    ----------
    dataframe : pl.DataFrame
        Corpus de entrada, contendo ao menos ``text_column``. Não vazio.
    text_column : str, optional
        Nome da coluna com o texto bruto, by default "text".
    normalized_text_column : str, optional
        Nome da coluna a ser criada com o texto normalizado, by default
        "text_normalized".
    tokens_column : str | None, optional
        Nome da coluna de tokens a ser criada, aplicada após os filtros de
        inclusão (quando habilitados). Se ``None``, a tokenização não é
        executada, by default None.
    lemmatized_text_column : str | None, optional
        Nome da coluna de texto lematizado a ser criada, via
        :func:`preprocessing.lemmatization.lemmatize_text` (spaCy),
        aplicada após os filtros de inclusão (quando habilitados). Se
        ``None``, a lematização não é executada, by default None. Sem o
        spaCy/modelo ``pt_core_news_sm`` instalados (``make
        install-nlp``), a coluna é criada com o texto normalizado
        inalterado (fallback com aviso no log).
    keep_hashtag_word : bool, optional
        Repassado a :func:`normalize_tweet_text`, by default True.
    max_workers : int | None, optional
        Número máximo de processos usados na normalização paralela, by
        default None (o executor escolhe automaticamente). Abaixo de
        ``_SERIAL_EXECUTION_THRESHOLD`` linhas, a normalização roda em um
        laço serial (sem executor) independentemente deste valor — o custo
        fixo de inicializar um ``ProcessPoolExecutor`` não compensa para
        lotes pequenos. Acima do limiar, os itens são agrupados em lotes
        (``chunk_size`` calculado automaticamente) e uma ``Future`` é
        submetida por lote, não por item, para amortizar o overhead de IPC.
    show_progress : bool, optional
        Se ``True``, exibe uma barra de progresso no console, by default
        True.
    expand_slang : bool, optional
        Repassado a :func:`preprocessing.tokenization.tokenize_and_normalize`,
        usado apenas quando ``tokens_column`` é informado, by default True.
    apply_negation_marking : bool, optional
        Repassado a :func:`preprocessing.tokenization.tokenize_and_normalize`,
        usado apenas quando ``tokens_column`` é informado, by default True.
    apply_inclusion_filters : bool, optional
        Se ``True``, aplica
        :func:`preprocessing.filtering.filter_by_inclusion_criteria` sobre
        ``normalized_text_column`` após a normalização, by default True.
    minimum_characters : int, optional
        Repassado ao filtro de conteúdo mínimo, by default 5.
    minimum_words : int, optional
        Repassado ao filtro de conteúdo mínimo, by default 2.
    minimum_portuguese_ratio : float, optional
        Repassado ao filtro de idioma, by default 0.15.
    max_repeated_word_ratio : float, optional
        Repassado ao filtro de spam, by default 0.5.
    drop_duplicate_text : bool, optional
        Repassado ao filtro de duplicatas, by default True.

    Returns
    -------
    pl.DataFrame
        Corpus com a coluna de texto normalizado (e, quando solicitadas, as
        colunas de tokens e/ou texto lematizado), filtrado pelos critérios
        de inclusão quando ``apply_inclusion_filters`` for ``True``.

    Raises
    ------
    EmptyDatasetError
        Se ``dataframe`` estiver vazio.
    PipelineStageError
        Se a normalização falhar para alguma linha do corpus.

    Examples
    --------
    >>> df = pl.DataFrame({"id": ["1", "2"], "text": ["RT @a: muito bom!! 😍", "RT @b: oi"]})
    >>> resultado = run_preprocessing_pipeline(df)
    >>> resultado["text_normalized"].to_list()
    ['muito bom!! [EMOJI_POSITIVO]']
    """
    validate_not_empty_collection(dataframe, collection_name="dataframe")

    raw_texts = dataframe[text_column].to_list()
    if len(raw_texts) < _SERIAL_EXECUTION_THRESHOLD:
        # Lote pequeno: laço serial puro, sem executor (mesmo comportamento
        # de antes das Tasks 6/9 introduzirem paralelismo) — o overhead fixo
        # de inicializar um ProcessPoolExecutor não compensa para poucos itens.
        normalized_texts = [
            _normalize_row_text(text, keep_hashtag_word=keep_hashtag_word) for text in raw_texts
        ]
    else:
        indexed_texts = list(enumerate(raw_texts))
        normalization_result = run_parallel_text_cleaning(
            functools.partial(_normalize_indexed_row_text, keep_hashtag_word=keep_hashtag_word),
            indexed_texts,
            max_workers=max_workers,
            show_progress=show_progress,
            chunk_size=_calculate_chunk_size(len(indexed_texts), max_workers),
        )
        if normalization_result.failures:
            raise normalization_result.failures[0].error
        normalized_texts = [
            text for _, text in sorted(normalization_result.successes, key=operator.itemgetter(0))
        ]
    result = dataframe.with_columns(pl.Series(normalized_text_column, normalized_texts))

    if apply_inclusion_filters:
        result = filter_by_inclusion_criteria(
            result,
            text_column=normalized_text_column,
            minimum_characters=minimum_characters,
            minimum_words=minimum_words,
            minimum_portuguese_ratio=minimum_portuguese_ratio,
            max_repeated_word_ratio=max_repeated_word_ratio,
            drop_duplicate_text=drop_duplicate_text,
        )

    if tokens_column is not None:
        token_lists = [
            tokenize_and_normalize(
                text, expand_slang=expand_slang, apply_negation_marking=apply_negation_marking
            )
            for text in result[normalized_text_column].to_list()
        ]
        result = result.with_columns(pl.Series(tokens_column, token_lists, dtype=pl.List(pl.Utf8)))

    if lemmatized_text_column is not None:
        lemmatized_texts = [
            lemmatize_text(text) for text in result[normalized_text_column].to_list()
        ]
        result = result.with_columns(pl.Series(lemmatized_text_column, lemmatized_texts))

    logger.info(
        "Pipeline de pré-processamento concluído: %d/%d linha(s) mantida(s)",
        result.height,
        dataframe.height,
    )
    return result
