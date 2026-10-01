"""Representações lexicais (TF-IDF / bag-of-words) de textos em português brasileiro.

Implementa a Seção 4.4 do documento mestre (``projeto-mestrado-analise-sentimentos-ptbr.md``):
baseline lexical para os classificadores clássicos (Regressão Logística, SVM
linear, Naive Bayes multinomial). O cálculo de TF-IDF é implementado
diretamente sobre ``polars``/``numpy`` (sem dependência de ``scikit-learn``,
ainda não presente no projeto — ver CLAUDE.md, "What to Avoid" -> dependências
sem justificativa), consumindo o texto já tokenizado e normalizado por
``src/preprocessing/tokenization.py`` (inclusive a marcação ``_NEG`` de
escopo de negação).

O resultado é produzido no formato longo (``id``, ``term``, ``tfidf_weight``),
mais econômico em memória que uma matriz densa para o vocabulário tipicamente
esparso de TF-IDF (ver ``configs/model_params.yaml -> classical.tfidf``);
:func:`pivot_tfidf_features_to_wide` converte para a matriz densa quando
exigida por um classificador específico.
"""

import logging
import math
from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np
import polars as pl

from utils.validation import validate_not_empty_collection

logger = logging.getLogger(__name__)


def extract_ngrams(tokens: Sequence[str], ngram_range: tuple[int, int]) -> list[str]:
    """Gera n-gramas contíguos de uma sequência de tokens.

    Cada n-grama é representado como uma única string com os tokens unidos
    por ``"_"``, no mesmo formato de termo esperado por
    :func:`build_vocabulary`.

    Parameters
    ----------
    tokens : Sequence[str]
        Tokens de entrada, tipicamente produzidos por
        ``src/preprocessing/tokenization.py``.
    ngram_range : tuple[int, int]
        Par ``(n_minimo, n_maximo)`` (inclusivo) de tamanhos de n-grama a
        gerar. Reflete ``configs/model_params.yaml ->
        classical.tfidf.ngram_range``.

    Returns
    -------
    list[str]
        N-gramas gerados, agrupados por tamanho crescente, na ordem em que
        aparecem no texto.

    Examples
    --------
    >>> extract_ngrams(["não", "gostei_NEG", "nada_NEG"], (1, 2))
    ['não', 'gostei_NEG', 'nada_NEG', 'não_gostei_NEG', 'gostei_NEG_nada_NEG']
    """
    minimum_n, maximum_n = ngram_range
    ngrams: list[str] = []
    for n in range(minimum_n, maximum_n + 1):
        if n > len(tokens):
            continue
        ngrams.extend("_".join(tokens[start : start + n]) for start in range(len(tokens) - n + 1))
    return ngrams


def build_document_frequencies(tokenized_documents: Sequence[Sequence[str]]) -> Counter[str]:
    """Conta em quantos documentos cada termo aparece ao menos uma vez.

    Parameters
    ----------
    tokenized_documents : Sequence[Sequence[str]]
        Um documento por elemento, já expandido em termos (tokens e/ou
        n-gramas) via :func:`extract_ngrams`.

    Returns
    -------
    Counter[str]
        Frequência de documento (``document frequency``) de cada termo.

    Examples
    --------
    >>> sorted(build_document_frequencies([["bom", "dia"], ["bom"]]).items())
    [('bom', 2), ('dia', 1)]
    """
    document_frequencies: Counter[str] = Counter()
    for document_terms in tokenized_documents:
        document_frequencies.update(set(document_terms))
    return document_frequencies


def build_vocabulary(
    tokenized_documents: Sequence[Sequence[str]],
    *,
    max_features: int = 20000,
    min_document_frequency: int = 3,
    max_document_frequency_ratio: float = 0.95,
) -> dict[str, int]:
    """Constrói o vocabulário TF-IDF a partir de um corpus tokenizado.

    Descarta termos raros demais (abaixo de ``min_document_frequency``) e
    frequentes demais (acima de ``max_document_frequency_ratio`` dos
    documentos, tipicamente stopwords residuais), mantendo os
    ``max_features`` termos mais frequentes entre os elegíveis. O índice
    final de cada termo é atribuído em ordem alfabética, para que a ordem
    das colunas produzidas por :func:`pivot_tfidf_features_to_wide` seja
    determinística e independente da ordem de iteração do corpus.

    Parameters
    ----------
    tokenized_documents : Sequence[Sequence[str]]
        Um documento por elemento, já expandido em termos via
        :func:`extract_ngrams`. Não vazio.
    max_features : int, optional
        Tamanho máximo do vocabulário, by default 20000 (ver
        ``configs/model_params.yaml -> classical.tfidf.max_features``).
    min_document_frequency : int, optional
        Frequência de documento mínima (inclusive) para um termo ser
        elegível, by default 3.
    max_document_frequency_ratio : float, optional
        Fração máxima (inclusive) de documentos em que um termo pode
        aparecer para ser elegível, by default 0.95.

    Returns
    -------
    dict[str, int]
        Mapeamento de cada termo selecionado para seu índice de coluna.

    Raises
    ------
    EmptyDatasetError
        Se ``tokenized_documents`` estiver vazio.

    Examples
    --------
    >>> build_vocabulary(
    ...     [["bom", "dia"], ["bom", "produto"], ["bom", "ótimo"]],
    ...     min_document_frequency=1,
    ...     max_document_frequency_ratio=1.0,
    ...     max_features=2,
    ... )
    {'bom': 0, 'dia': 1}
    """
    validate_not_empty_collection(tokenized_documents, collection_name="tokenized_documents")
    document_frequencies = build_document_frequencies(tokenized_documents)
    maximum_document_frequency = max_document_frequency_ratio * len(tokenized_documents)

    eligible_terms = [
        term
        for term, frequency in document_frequencies.items()
        if min_document_frequency <= frequency <= maximum_document_frequency
    ]
    selected_terms = sorted(eligible_terms, key=lambda term: (-document_frequencies[term], term))[
        :max_features
    ]
    return {term: index for index, term in enumerate(sorted(selected_terms))}


def calculate_term_frequency(
    document_terms: Sequence[str],
    vocabulary: Mapping[str, int],
    *,
    sublinear_term_frequency: bool = True,
) -> dict[str, float]:
    """Calcula a frequência de termo (TF) de um documento, restrita ao vocabulário.

    Parameters
    ----------
    document_terms : Sequence[str]
        Termos (tokens e/ou n-gramas) de um único documento, via
        :func:`extract_ngrams`.
    vocabulary : Mapping[str, int]
        Vocabulário produzido por :func:`build_vocabulary`. Termos ausentes
        do vocabulário são ignorados.
    sublinear_term_frequency : bool, optional
        Se ``True``, aplica a escala sublinear ``1 + log(contagem)`` em vez
        da contagem bruta, reduzindo o peso de termos muito repetidos em um
        mesmo documento, by default True (ver ``configs/model_params.yaml
        -> classical.tfidf.sublinear_tf``).

    Returns
    -------
    dict[str, float]
        Frequência de termo por termo do vocabulário presente no documento.

    Examples
    --------
    >>> calculate_term_frequency(
    ...     ["bom", "bom", "dia"], {"bom": 0, "dia": 1}, sublinear_term_frequency=False
    ... )
    {'bom': 2.0, 'dia': 1.0}
    """
    term_counts = Counter(term for term in document_terms if term in vocabulary)
    if sublinear_term_frequency:
        return {term: 1.0 + math.log(count) for term, count in term_counts.items()}
    return {term: float(count) for term, count in term_counts.items()}


def compute_tfidf_features(
    dataframe: pl.DataFrame,
    *,
    id_column: str = "id",
    text_column: str = "text",
    ngram_range: tuple[int, int] = (1, 2),
    max_features: int = 20000,
    min_document_frequency: int = 3,
    max_document_frequency_ratio: float = 0.95,
    sublinear_term_frequency: bool = True,
) -> pl.DataFrame:
    """Calcula os pesos TF-IDF de um corpus, no formato longo (esparso).

    O texto de cada linha é tokenizado por espaço (o corpus de entrada já
    deve estar normalizado e tokenizado por
    ``src/preprocessing/pipeline.py``/``tokenization.py``, inclusive a
    marcação ``_NEG`` de negação), expandido em n-gramas via
    :func:`extract_ngrams`, restrito ao vocabulário de
    :func:`build_vocabulary` e ponderado pelo IDF suavizado
    ``log((n_documentos + 1) / (frequencia_documento + 1)) + 1`` (mesma
    formulação do ``scikit-learn``, evitando IDF nulo/negativo).

    Parameters
    ----------
    dataframe : pl.DataFrame
        Corpus de entrada, contendo ao menos ``id_column`` e
        ``text_column``. Não vazio.
    id_column : str, optional
        Nome da coluna identificadora de cada documento, by default "id".
    text_column : str, optional
        Nome da coluna de texto (tokens separados por espaço), by default
        "text".
    ngram_range : tuple[int, int], optional
        Repassado a :func:`extract_ngrams`, by default (1, 2).
    max_features : int, optional
        Repassado a :func:`build_vocabulary`, by default 20000.
    min_document_frequency : int, optional
        Repassado a :func:`build_vocabulary`, by default 3.
    max_document_frequency_ratio : float, optional
        Repassado a :func:`build_vocabulary`, by default 0.95.
    sublinear_term_frequency : bool, optional
        Repassado a :func:`calculate_term_frequency`, by default True.

    Returns
    -------
    pl.DataFrame
        DataFrame no formato longo (``id``, ``term``, ``tfidf_weight``),
        contendo apenas pares com peso não nulo. Use
        :func:`pivot_tfidf_features_to_wide` para obter a matriz densa.

    Raises
    ------
    EmptyDatasetError
        Se ``dataframe`` estiver vazio.

    Examples
    --------
    >>> df = pl.DataFrame({"id": ["1", "2"], "text": ["bom dia", "bom produto"]})
    >>> resultado = compute_tfidf_features(
    ...     df, min_document_frequency=1, max_document_frequency_ratio=1.0
    ... )
    >>> sorted(resultado["term"].to_list())
    ['bom', 'bom', 'bom_dia', 'bom_produto', 'dia', 'produto']
    """
    validate_not_empty_collection(dataframe, collection_name="dataframe")

    document_ids = dataframe[id_column].to_list()
    tokenized_documents = [text.split() for text in dataframe[text_column].to_list()]
    document_terms = [extract_ngrams(tokens, ngram_range) for tokens in tokenized_documents]

    vocabulary = build_vocabulary(
        document_terms,
        max_features=max_features,
        min_document_frequency=min_document_frequency,
        max_document_frequency_ratio=max_document_frequency_ratio,
    )
    document_frequencies = build_document_frequencies(document_terms)
    n_documents = len(document_terms)
    inverse_document_frequencies = {
        term: math.log((n_documents + 1) / (document_frequencies[term] + 1)) + 1.0
        for term in vocabulary
    }

    ids: list[str] = []
    terms: list[str] = []
    weights: list[float] = []
    for document_id, terms_in_document in zip(document_ids, document_terms, strict=True):
        term_frequencies = calculate_term_frequency(
            terms_in_document, vocabulary, sublinear_term_frequency=sublinear_term_frequency
        )
        for term, term_frequency in term_frequencies.items():
            ids.append(document_id)
            terms.append(term)
            weights.append(term_frequency * inverse_document_frequencies[term])

    logger.info(
        "TF-IDF calculado: %d documento(s), vocabulário de %d termo(s), %d peso(s) não nulo(s).",
        n_documents,
        len(vocabulary),
        len(weights),
    )
    return pl.DataFrame({id_column: ids, "term": terms, "tfidf_weight": weights})


def pivot_tfidf_features_to_wide(
    long_features: pl.DataFrame,
    *,
    id_column: str = "id",
    term_column: str = "term",
    weight_column: str = "tfidf_weight",
) -> pl.DataFrame:
    """Converte pesos TF-IDF do formato longo para uma matriz densa (larga).

    Parameters
    ----------
    long_features : pl.DataFrame
        Resultado de :func:`compute_tfidf_features`, no formato longo. Não
        vazio.
    id_column : str, optional
        Nome da coluna identificadora de cada documento, by default "id".
    term_column : str, optional
        Nome da coluna com o termo, by default "term".
    weight_column : str, optional
        Nome da coluna com o peso TF-IDF, by default "tfidf_weight".

    Returns
    -------
    pl.DataFrame
        Matriz densa com uma linha por documento e uma coluna por termo do
        vocabulário; pares ausentes no formato longo (peso zero) são
        preenchidos com 0.0.

    Raises
    ------
    EmptyDatasetError
        Se ``long_features`` estiver vazio.

    Examples
    --------
    >>> long_features = pl.DataFrame(
    ...     {"id": ["1", "2"], "term": ["bom", "dia"], "tfidf_weight": [1.5, 0.8]}
    ... )
    >>> wide = pivot_tfidf_features_to_wide(long_features).sort("id")
    >>> wide["bom"].to_list()
    [1.5, 0.0]
    >>> wide["dia"].to_list()
    [0.0, 0.8]
    """
    validate_not_empty_collection(long_features, collection_name="long_features")
    return (
        long_features.pivot(
            on=term_column, index=id_column, values=weight_column, aggregate_function="first"
        )
        .fill_null(0.0)
        .sort(id_column)
    )


class LexicalTfidfVectorizer:
    """Vetorizador TF-IDF ajustável: aprende o vocabulário no treino e o aplica a qualquer texto.

    Usa as mesmas funções de :func:`compute_tfidf_features` (n-gramas, vocabulário,
    TF sublinear e IDF suavizado), mas guarda o vocabulário e o IDF para transformar também
    validação e teste sem vazamento de informação. Ao contrário do formato longo, devolve uma
    linha por texto, inclusive os sem nenhum termo do vocabulário (linha de zeros), e a ordem
    das colunas é a do vocabulário (alfabética), logo estável entre execuções.

    Parameters
    ----------
    ngram_range : tuple[int, int], optional
        Repassado a :func:`extract_ngrams`, by default (1, 2).
    max_features : int, optional
        Repassado a :func:`build_vocabulary`, by default 20000.
    min_document_frequency : int, optional
        Repassado a :func:`build_vocabulary`, by default 3.
    max_document_frequency_ratio : float, optional
        Repassado a :func:`build_vocabulary`, by default 0.95.
    sublinear_term_frequency : bool, optional
        Repassado a :func:`calculate_term_frequency`, by default True.

    Examples
    --------
    >>> vectorizer = LexicalTfidfVectorizer(min_document_frequency=1, ngram_range=(1, 1))
    >>> vectorizer.fit(["bom dia", "bom produto"]).transform(["bom"]).shape
    (1, 3)
    """

    def __init__(
        self,
        *,
        ngram_range: tuple[int, int] = (1, 2),
        max_features: int = 20000,
        min_document_frequency: int = 3,
        max_document_frequency_ratio: float = 0.95,
        sublinear_term_frequency: bool = True,
    ) -> None:
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_document_frequency = min_document_frequency
        self.max_document_frequency_ratio = max_document_frequency_ratio
        self.sublinear_term_frequency = sublinear_term_frequency
        self.vocabulary_: dict[str, int] = {}
        self.idf_: np.ndarray = np.empty(0, dtype=np.float32)

    def _extract_terms(self, texts: Sequence[str]) -> list[list[str]]:
        """Tokeniza por espaço e expande cada texto em n-gramas."""
        return [extract_ngrams(text.split(), self.ngram_range) for text in texts]

    def fit(self, texts: Sequence[str]) -> "LexicalTfidfVectorizer":
        """Aprende o vocabulário e o IDF a partir dos textos de treino.

        Parameters
        ----------
        texts : Sequence[str]
            Textos de treino já normalizados (termos separados por espaço). Não vazio.

        Returns
        -------
        LexicalTfidfVectorizer
            A própria instância ajustada.

        Raises
        ------
        EmptyDatasetError
            Se ``texts`` estiver vazio.
        """
        document_terms = self._extract_terms(texts)
        self.vocabulary_ = build_vocabulary(
            document_terms,
            max_features=self.max_features,
            min_document_frequency=self.min_document_frequency,
            max_document_frequency_ratio=self.max_document_frequency_ratio,
        )
        document_frequencies = build_document_frequencies(document_terms)
        n_documents = len(document_terms)
        self.idf_ = np.zeros(len(self.vocabulary_), dtype=np.float32)
        for term, index in self.vocabulary_.items():
            self.idf_[index] = math.log((n_documents + 1) / (document_frequencies[term] + 1)) + 1.0
        logger.info(
            "Vetorizador TF-IDF ajustado: %d documento(s), vocabulário de %d termo(s).",
            n_documents,
            len(self.vocabulary_),
        )
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        """Converte textos em uma matriz TF-IDF densa, no vocabulário aprendido em :meth:`fit`.

        Parameters
        ----------
        texts : Sequence[str]
            Textos normalizados (termos separados por espaço).

        Returns
        -------
        np.ndarray
            Matriz ``(n_textos, n_termos)`` de ``float32``.

        Raises
        ------
        RuntimeError
            Se o vetorizador ainda não foi ajustado.
        """
        if not self.vocabulary_:
            raise RuntimeError("Vetorizador TF-IDF não ajustado: chame fit() antes de transform().")
        matrix = np.zeros((len(texts), len(self.vocabulary_)), dtype=np.float32)
        for row, terms in enumerate(self._extract_terms(texts)):
            term_frequencies = calculate_term_frequency(
                terms, self.vocabulary_, sublinear_term_frequency=self.sublinear_term_frequency
            )
            for term, term_frequency in term_frequencies.items():
                index = self.vocabulary_[term]
                matrix[row, index] = term_frequency * self.idf_[index]
        return matrix
