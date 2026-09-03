"""Testes do módulo de engenharia de features (``src/features``)."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from exceptions.data import DataNotFoundError, EmptyDatasetError
from features.contextual_embeddings import extract_contextual_embeddings
from features.lexical import (
    build_document_frequencies,
    build_vocabulary,
    calculate_term_frequency,
    compute_tfidf_features,
    extract_ngrams,
    pivot_tfidf_features_to_wide,
)
from features.reduction import (
    AutoencoderArtifacts,
    compute_reconstruction_error,
    encode_with_autoencoder,
    train_autoencoder,
)
from features.selection import (
    build_feature_group_mask,
    calculate_feature_correlation_matrix,
    calculate_feature_variance,
    select_features_by_redundancy,
    select_features_by_variance_threshold,
    select_k_best_features_by_target_correlation,
)
from features.static_embeddings import (
    compute_document_embedding,
    extract_static_embeddings,
    load_fasttext_model,
)
from features.statistics import (
    calculate_descriptive_statistics,
    calculate_embedding_norms,
    calculate_feature_sparsity_ratio,
    summarize_feature_matrix,
)


class _FakeStaticEmbeddingModel:
    """Modelo de embeddings estáticos de teste, com vocabulário fixo de 2 dimensões."""

    def get_word_vector(self, word: str) -> np.ndarray:
        """Retorna ``[1, 0]`` para "bom", ``[0, 1]`` para qualquer outra palavra."""
        return np.array([1.0, 0.0]) if word == "bom" else np.array([0.0, 1.0])

    def get_dimension(self) -> int:
        """Retorna a dimensão fixa (2) do modelo de teste."""
        return 2


class _FakeContextualEncoder:
    """Encoder contextual de teste: cada texto vira ``[len(texto), 0.0]``."""

    def encode(self, texts: list[str]) -> np.ndarray:
        """Codifica cada texto pelo seu comprimento em caracteres."""
        return np.array([[float(len(text)), 0.0] for text in texts])


class TestExtractNgrams:
    """Testes da geração de n-gramas contíguos."""

    def test_generates_unigrams_only_for_range_one_one(self) -> None:
        """Com ``ngram_range=(1, 1)``, deve retornar apenas os tokens originais."""
        assert extract_ngrams(["a", "b", "c"], (1, 1)) == ["a", "b", "c"]

    def test_generates_unigrams_and_bigrams(self) -> None:
        """Com ``ngram_range=(1, 2)``, deve retornar unigramas seguidos de bigramas."""
        assert extract_ngrams(["não", "gostei"], (1, 2)) == ["não", "gostei", "não_gostei"]

    def test_returns_empty_list_for_empty_tokens(self) -> None:
        """Uma lista de tokens vazia deve retornar uma lista vazia."""
        assert extract_ngrams([], (1, 2)) == []

    def test_skips_ngram_sizes_larger_than_token_count(self) -> None:
        """Tamanhos de n-grama maiores que a quantidade de tokens são ignorados."""
        assert extract_ngrams(["a"], (1, 3)) == ["a"]


class TestBuildDocumentFrequencies:
    """Testes da contagem de frequência de documento por termo."""

    def test_counts_each_document_once_per_term(self) -> None:
        """Um termo repetido no mesmo documento deve ser contado uma única vez."""
        assert build_document_frequencies([["a", "a", "b"], ["a"]]) == {"a": 2, "b": 1}


class TestBuildVocabulary:
    """Testes da construção do vocabulário TF-IDF."""

    def test_assigns_alphabetical_indices(self) -> None:
        """Os índices do vocabulário devem seguir a ordem alfabética dos termos."""
        vocabulary = build_vocabulary(
            [["b"], ["a"], ["a", "b"]], min_document_frequency=1, max_document_frequency_ratio=1.0
        )
        assert vocabulary == {"a": 0, "b": 1}

    def test_excludes_terms_below_minimum_document_frequency(self) -> None:
        """Termos com frequência de documento abaixo do mínimo devem ser excluídos."""
        vocabulary = build_vocabulary(
            [["raro"], ["comum"], ["comum"]],
            min_document_frequency=2,
            max_document_frequency_ratio=1.0,
        )
        assert vocabulary == {"comum": 0}

    def test_excludes_terms_above_maximum_document_frequency_ratio(self) -> None:
        """Termos presentes em quase todos os documentos devem ser excluídos."""
        vocabulary = build_vocabulary(
            [["muito_comum"], ["muito_comum"], ["raro"]],
            min_document_frequency=1,
            max_document_frequency_ratio=0.5,
        )
        assert vocabulary == {"raro": 0}

    def test_limits_vocabulary_to_max_features(self) -> None:
        """O vocabulário não deve exceder ``max_features`` termos."""
        vocabulary = build_vocabulary(
            [["a", "b", "c"]],
            min_document_frequency=1,
            max_document_frequency_ratio=1.0,
            max_features=2,
        )
        assert len(vocabulary) == 2

    def test_raises_for_empty_documents(self) -> None:
        """Um corpus vazio deve levantar ``EmptyDatasetError``."""
        with pytest.raises(EmptyDatasetError):
            build_vocabulary([])


class TestCalculateTermFrequency:
    """Testes do cálculo de frequência de termo (TF)."""

    def test_ignores_terms_outside_vocabulary(self) -> None:
        """Termos ausentes do vocabulário não devem aparecer no resultado."""
        result = calculate_term_frequency(["a", "fora"], {"a": 0}, sublinear_term_frequency=False)
        assert result == {"a": 1.0}

    def test_applies_sublinear_scaling(self) -> None:
        """A escala sublinear deve produzir ``1 + log(contagem)`` para termos repetidos."""
        result = calculate_term_frequency(["a", "a"], {"a": 0}, sublinear_term_frequency=True)
        assert result["a"] == pytest.approx(1.0 + np.log(2))


class TestComputeTfidfFeatures:
    """Testes do cálculo de pesos TF-IDF em formato longo."""

    def test_produces_higher_weight_for_term_unique_to_one_document(self) -> None:
        """Um termo exclusivo de um documento deve ter peso TF-IDF maior que um termo comum."""
        df = pl.DataFrame({"id": ["1", "2"], "text": ["bom dia raro", "bom dia"]})
        result = compute_tfidf_features(
            df, ngram_range=(1, 1), min_document_frequency=1, max_document_frequency_ratio=1.0
        )
        rows_for_document_one = result.filter(pl.col("id") == "1")
        weights_by_term = dict(
            zip(rows_for_document_one["term"], rows_for_document_one["tfidf_weight"], strict=True)
        )
        assert weights_by_term["raro"] > weights_by_term["bom"]

    def test_raises_for_empty_dataframe(self) -> None:
        """Um DataFrame vazio deve levantar ``EmptyDatasetError``."""
        df = pl.DataFrame({"id": [], "text": []}, schema={"id": pl.Utf8, "text": pl.Utf8})
        with pytest.raises(EmptyDatasetError):
            compute_tfidf_features(df)


class TestPivotTfidfFeaturesToWide:
    """Testes da conversão do formato longo para a matriz densa."""

    def test_fills_missing_pairs_with_zero(self) -> None:
        """Pares (id, termo) ausentes no formato longo devem virar zero na matriz densa."""
        long_features = pl.DataFrame(
            {"id": ["1", "2"], "term": ["a", "b"], "tfidf_weight": [1.5, 0.8]}
        )
        wide = pivot_tfidf_features_to_wide(long_features).sort("id")
        assert wide["a"].to_list() == [1.5, 0.0]
        assert wide["b"].to_list() == [0.0, 0.8]

    def test_raises_for_empty_long_features(self) -> None:
        """Um DataFrame longo vazio deve levantar ``EmptyDatasetError``."""
        df = pl.DataFrame(
            {"id": [], "term": [], "tfidf_weight": []},
            schema={"id": pl.Utf8, "term": pl.Utf8, "tfidf_weight": pl.Float64},
        )
        with pytest.raises(EmptyDatasetError):
            pivot_tfidf_features_to_wide(df)


class TestComputeDocumentEmbedding:
    """Testes do cálculo do embedding médio de um documento."""

    def test_averages_word_vectors(self) -> None:
        """O embedding do documento deve ser a média dos vetores das palavras."""
        result = compute_document_embedding(["bom", "outra"], _FakeStaticEmbeddingModel())
        assert result.tolist() == pytest.approx([0.5, 0.5])

    def test_returns_zero_vector_for_empty_tokens(self) -> None:
        """Um documento sem tokens deve retornar um vetor de zeros na dimensão do modelo."""
        result = compute_document_embedding([], _FakeStaticEmbeddingModel())
        assert result.tolist() == [0.0, 0.0]


class TestExtractStaticEmbeddings:
    """Testes da extração de embeddings estáticos para um corpus."""

    def test_produces_one_embedding_column_per_dimension(self) -> None:
        """Deve haver uma coluna ``embedding_<i>`` por dimensão do modelo."""
        df = pl.DataFrame({"id": ["1"], "text": ["bom dia"]})
        result = extract_static_embeddings(df, _FakeStaticEmbeddingModel())
        assert set(result.columns) == {"id", "embedding_0", "embedding_1"}

    def test_raises_for_empty_dataframe(self) -> None:
        """Um DataFrame vazio deve levantar ``EmptyDatasetError``."""
        df = pl.DataFrame({"id": [], "text": []}, schema={"id": pl.Utf8, "text": pl.Utf8})
        with pytest.raises(EmptyDatasetError):
            extract_static_embeddings(df, _FakeStaticEmbeddingModel())


class TestLoadFasttextModel:
    """Testes do carregamento de um modelo FastText."""

    def test_raises_for_missing_model_file(self, tmp_path: Path) -> None:
        """Um caminho de modelo inexistente deve levantar ``DataNotFoundError``."""
        with pytest.raises(DataNotFoundError):
            load_fasttext_model(tmp_path / "modelo_inexistente.bin")


class TestExtractContextualEmbeddings:
    """Testes da extração de embeddings contextuais em lotes."""

    def test_produces_embeddings_in_batches(self) -> None:
        """A extração em lotes deve preservar a ordem e o conteúdo dos documentos."""
        df = pl.DataFrame({"id": ["1", "2"], "text": ["oi", "bom dia"]})
        result = extract_contextual_embeddings(df, _FakeContextualEncoder(), batch_size=1)
        assert result["embedding_0"].to_list() == [2.0, 7.0]

    def test_raises_for_empty_dataframe(self) -> None:
        """Um DataFrame vazio deve levantar ``EmptyDatasetError``."""
        df = pl.DataFrame({"id": [], "text": []}, schema={"id": pl.Utf8, "text": pl.Utf8})
        with pytest.raises(EmptyDatasetError):
            extract_contextual_embeddings(df, _FakeContextualEncoder())


class TestTrainAutoencoder:
    """Testes do treinamento do autoencoder (requerem ``torch`` instalado)."""

    def test_reduces_dimensionality_of_encoded_output(self) -> None:
        """O espaço latente codificado deve ter a dimensão configurada em ``latent_dim``."""
        pytest.importorskip("torch")
        embeddings = np.random.default_rng(42).normal(size=(20, 16)).astype(np.float32)
        artifacts = train_autoencoder(
            embeddings,
            input_dim=16,
            latent_dim=4,
            hidden_layers=(8,),
            batch_size=4,
            epochs=2,
            early_stopping_patience=2,
        )
        encoded = encode_with_autoencoder(embeddings, artifacts)
        assert encoded.shape == (20, 4)

    def test_raises_for_empty_training_embeddings(self) -> None:
        """Uma matriz de embeddings de treino vazia deve levantar ``EmptyDatasetError``."""
        with pytest.raises(EmptyDatasetError):
            train_autoencoder(np.empty((0, 16)))


class TestEncodeWithAutoencoder:
    """Testes da projeção de embeddings no espaço latente do autoencoder."""

    def test_raises_for_empty_embeddings(self) -> None:
        """Uma matriz de embeddings vazia deve levantar ``EmptyDatasetError``."""
        artifacts = AutoencoderArtifacts(
            module=None, input_dim=16, latent_dim=4, training_loss_history=[]
        )
        with pytest.raises(EmptyDatasetError):
            encode_with_autoencoder(np.empty((0, 16)), artifacts)


class TestComputeReconstructionError:
    """Testes do cálculo do erro de reconstrução por amostra."""

    def test_returns_one_error_value_per_sample(self) -> None:
        """Deve haver um valor de erro de reconstrução por amostra de entrada."""
        pytest.importorskip("torch")
        embeddings = np.random.default_rng(42).normal(size=(20, 16)).astype(np.float32)
        artifacts = train_autoencoder(
            embeddings,
            input_dim=16,
            latent_dim=4,
            hidden_layers=(8,),
            batch_size=4,
            epochs=2,
            early_stopping_patience=2,
        )
        errors = compute_reconstruction_error(embeddings, artifacts)
        assert errors.shape == (20,)
        assert (errors >= 0).all()

    def test_raises_for_empty_embeddings(self) -> None:
        """Uma matriz de embeddings vazia deve levantar ``EmptyDatasetError``."""
        artifacts = AutoencoderArtifacts(
            module=None, input_dim=16, latent_dim=4, training_loss_history=[]
        )
        with pytest.raises(EmptyDatasetError):
            compute_reconstruction_error(np.empty((0, 16)), artifacts)


class TestCalculateFeatureVariance:
    """Testes do cálculo de variância por coluna de feature."""

    def test_computes_population_variance_per_feature(self) -> None:
        """A variância populacional deve ser calculada corretamente por feature."""
        df = pl.DataFrame({"id": ["1", "2", "3"], "a": [1.0, 1.0, 1.0], "b": [1.0, 2.0, 3.0]})
        result = calculate_feature_variance(df).sort("feature")
        assert result["feature"].to_list() == ["a", "b"]
        assert [round(value, 4) for value in result["variance"].to_list()] == [0.0, 0.6667]

    def test_raises_for_empty_feature_matrix(self) -> None:
        """Uma matriz de features vazia deve levantar ``EmptyDatasetError``."""
        df = pl.DataFrame({"id": [], "a": []}, schema={"id": pl.Utf8, "a": pl.Float64})
        with pytest.raises(EmptyDatasetError):
            calculate_feature_variance(df)


class TestSelectFeaturesByVarianceThreshold:
    """Testes da remoção de features com variância abaixo de um limiar."""

    def test_removes_constant_feature(self) -> None:
        """Uma feature constante deve ser removida com o limiar padrão."""
        df = pl.DataFrame({"id": ["1", "2", "3"], "a": [1.0, 1.0, 1.0], "b": [1.0, 2.0, 3.0]})
        assert select_features_by_variance_threshold(df).columns == ["id", "b"]


class TestCalculateFeatureCorrelationMatrix:
    """Testes do cálculo da matriz de correlação entre features."""

    def test_perfectly_correlated_features_have_correlation_one(self) -> None:
        """Duas features perfeitamente correlacionadas devem ter correlação 1.0."""
        df = pl.DataFrame({"id": ["1", "2", "3"], "a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
        feature_names, correlation = calculate_feature_correlation_matrix(df)
        assert feature_names == ["a", "b"]
        assert round(float(correlation[0, 1]), 4) == 1.0


class TestSelectFeaturesByRedundancy:
    """Testes da remoção de features redundantes por correlação."""

    def test_removes_redundant_feature_keeping_the_first(self) -> None:
        """Entre duas features altamente correlacionadas, a primeira deve ser mantida."""
        df = pl.DataFrame(
            {
                "id": ["1", "2", "3"],
                "a": [1.0, 2.0, 3.0],
                "b": [2.0, 4.0, 6.0],
                "c": [3.0, 1.0, 2.0],
            }
        )
        assert select_features_by_redundancy(df).columns == ["id", "a", "c"]


class TestSelectKBestFeaturesByTargetCorrelation:
    """Testes da seleção supervisionada por correlação com o alvo."""

    def test_selects_feature_most_correlated_with_target(self) -> None:
        """A feature mais correlacionada (em módulo) com o alvo deve ser mantida."""
        df = pl.DataFrame({"id": ["1", "2", "3"], "a": [1.0, 2.0, 3.0], "b": [3.0, 1.0, 2.0]})
        result = select_k_best_features_by_target_correlation(df, [1.0, 2.0, 3.0], k=1)
        assert result.columns == ["id", "a"]


class TestBuildFeatureGroupMask:
    """Testes da máscara de grupos de features para o ablation study."""

    def test_keeps_only_columns_from_enabled_groups(self) -> None:
        """Apenas as colunas dos grupos habilitados devem ser mantidas."""
        df = pl.DataFrame({"id": ["1"], "tfidf_a": [0.5], "emb_0": [0.1], "emb_1": [0.2]})
        groups = {"tfidf": ["tfidf_a"], "embeddings": ["emb_0", "emb_1"]}
        result = build_feature_group_mask(df, groups, enabled_groups=["embeddings"])
        assert result.columns == ["id", "emb_0", "emb_1"]

    def test_raises_for_empty_feature_matrix(self) -> None:
        """Uma matriz de features vazia deve levantar ``EmptyDatasetError``."""
        df = pl.DataFrame({"id": [], "a": []}, schema={"id": pl.Utf8, "a": pl.Float64})
        with pytest.raises(EmptyDatasetError):
            build_feature_group_mask(df, {"grupo": ["a"]}, enabled_groups=["grupo"])


class TestCalculateDescriptiveStatistics:
    """Testes das estatísticas descritivas por feature."""

    def test_computes_zero_ratio(self) -> None:
        """A fração de valores zero deve ser calculada corretamente."""
        df = pl.DataFrame({"id": ["1", "2", "3"], "a": [0.0, 0.0, 3.0]})
        result = calculate_descriptive_statistics(df)
        assert result["feature"].to_list() == ["a"]
        assert round(result["zero_ratio"].to_list()[0], 4) == 0.6667

    def test_raises_for_empty_feature_matrix(self) -> None:
        """Uma matriz de features vazia deve levantar ``EmptyDatasetError``."""
        df = pl.DataFrame({"id": [], "a": []}, schema={"id": pl.Utf8, "a": pl.Float64})
        with pytest.raises(EmptyDatasetError):
            calculate_descriptive_statistics(df)


class TestCalculateEmbeddingNorms:
    """Testes do cálculo da norma L2 por documento."""

    def test_computes_l2_norm_per_row(self) -> None:
        """A norma L2 de cada linha deve ser calculada corretamente."""
        df = pl.DataFrame({"id": ["1", "2"], "a": [3.0, 0.0], "b": [4.0, 0.0]})
        assert calculate_embedding_norms(df)["l2_norm"].to_list() == [5.0, 0.0]


class TestCalculateFeatureSparsityRatio:
    """Testes do cálculo da fração global de valores zero."""

    def test_computes_global_sparsity_ratio(self) -> None:
        """A fração global de zeros deve ser calculada sobre todas as células."""
        df = pl.DataFrame({"id": ["1", "2"], "a": [0.0, 1.0], "b": [0.0, 0.0]})
        assert calculate_feature_sparsity_ratio(df) == pytest.approx(0.75)


class TestSummarizeFeatureMatrix:
    """Testes do resumo agregado de uma matriz de features."""

    def test_summary_contains_expected_keys(self) -> None:
        """O resumo deve conter a contagem de documentos e de features."""
        df = pl.DataFrame({"id": ["1", "2"], "a": [3.0, 0.0], "b": [4.0, 0.0]})
        summary = summarize_feature_matrix(df, matrix_name="exemplo")
        assert summary["n_documents"] == 2
        assert summary["n_features"] == 2


class TestFeaturesProperties:
    """Testes baseados em propriedade (hypothesis) para invariantes do módulo."""

    @given(st.lists(st.text(alphabet="ab", min_size=1, max_size=5), min_size=0, max_size=10))
    def test_extract_ngrams_unigram_count_matches_token_count(self, tokens: list[str]) -> None:
        """Com ``ngram_range=(1, 1)``, a quantidade de n-gramas deve igualar a de tokens."""
        assert len(extract_ngrams(tokens, (1, 1))) == len(tokens)

    @given(
        st.lists(
            st.floats(min_value=-1000, max_value=1000, allow_nan=False), min_size=1, max_size=10
        )
    )
    def test_calculate_feature_variance_is_never_negative(self, values: list[float]) -> None:
        """A variância calculada nunca deve ser negativa."""
        df = pl.DataFrame({"id": [str(i) for i in range(len(values))], "a": values})
        result = calculate_feature_variance(df)
        assert (result["variance"] >= 0).all()

    @given(st.lists(st.floats(min_value=0, max_value=1, allow_nan=False), min_size=1, max_size=10))
    def test_calculate_feature_sparsity_ratio_is_in_valid_range(self, values: list[float]) -> None:
        """A taxa de esparsidade deve sempre estar entre 0.0 e 1.0."""
        df = pl.DataFrame({"id": [str(i) for i in range(len(values))], "a": values})
        assert 0.0 <= calculate_feature_sparsity_ratio(df) <= 1.0
