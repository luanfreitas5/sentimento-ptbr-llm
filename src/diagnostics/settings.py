"""Configuração validada da camada de diagnóstico (``configs/diagnostics.yaml``).

Carrega o YAML e o valida com Pydantic (``extra="forbid"``), de modo que uma
configuração inválida falhe na inicialização com um erro tipado, e não no meio
de uma execução que já gastou chamadas de LLM.
"""

import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from config.paths import CONFIGS_DIR
from exceptions.configuration import ConfigurationFileNotFoundError, InvalidConfigurationError
from io_utils.yaml import read_yaml

logger = logging.getLogger(__name__)

DEFAULT_DIAGNOSTICS_CONFIG_FILE: Path = CONFIGS_DIR / "diagnostics.yaml"


class _Strict(BaseModel):
    """Base dos modelos de configuração: rejeita chaves desconhecidas."""

    model_config = ConfigDict(extra="forbid")


class DataSettings(_Strict):
    """Mapeamento das colunas do corpus rotulado para o contrato de diagnóstico."""

    text_column: str = "text_normalized"
    agreement_column: str = "confidence_score"
    model_columns: dict[str, str]
    output_dir: str = "diagnostics"


class SplitSettings(_Strict):
    """Proporções das partições (descoberta/validação/holdout)."""

    holdout_size: float = Field(gt=0, lt=1, default=0.2)
    validation_size: float = Field(gt=0, lt=1, default=0.1)


class EmbeddingSettings(_Strict):
    """Modelo e lote dos embeddings locais."""

    model_name: str
    batch_size: int = Field(gt=0, default=128)


class SaeSettings(_Strict):
    """Dimensões do Sparse Autoencoder (M neurônios, K ativos)."""

    m_total_neurons: int = Field(gt=0, default=256)
    k_active_neurons: int = Field(gt=0, default=8)
    matryoshka_prefix_lengths: list[int] | None = None
    checkpoint_subdir: str = "diagnostics"


class SanitySettings(_Strict):
    """Parâmetros do gate de sanidade (Ridge nos embeddings)."""

    ridge_alpha: float = Field(gt=0, default=1.0)
    n_bootstrap: int = Field(gt=0, default=1000)
    n_permutations: int = Field(gt=0, default=1000)
    confidence_level: float = Field(gt=0, lt=1, default=0.95)
    significance_alpha: float = Field(gt=0, lt=1, default=0.05)
    min_effect: float = Field(ge=0, default=0.0)


class HypothesesSettings(_Strict):
    """Parâmetros de ``generate_hypotheses``."""

    selection_method: Literal["separation_score", "correlation", "lasso"] = "lasso"
    n_selected_neurons: int = Field(gt=0, default=20)
    n_candidate_interpretations: int = Field(gt=0, default=3)
    n_examples_for_interpretation: int = Field(gt=0, default=20)
    max_words_per_example: int = Field(gt=0, default=60)
    max_interpretation_tokens: int | None = 200
    n_scoring_examples: int = Field(ge=0, default=100)
    scoring_metric: str = "f1"
    n_workers: int = Field(gt=0, default=8)
    task_specific_instructions: str | None = None


class LLMSettings(_Strict):
    """Cliente LLM único (OpenAI-compatível: Ollama ou OpenAI)."""

    provider: Literal["ollama", "openai"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    openai_base_url: str | None = None
    interpreter_model: str
    annotator_model: str
    max_concurrency: int = Field(gt=0, default=8)
    request_timeout_seconds: float = Field(gt=0, default=120.0)
    max_retries: int = Field(ge=0, default=3)
    temperature: float = Field(ge=0, default=0.0)
    cache_dir: str = "data/interim/diagnostics/llm_cache"
    max_annotation_failure_rate: float = Field(ge=0, le=1, default=0.2)


class PriceSettings(_Strict):
    """Preço (USD por 1M de tokens) de entrada e saída."""

    input: float = Field(ge=0, default=0.0)
    output: float = Field(ge=0, default=0.0)


class PricingSettings(_Strict):
    """Tabela de preços por modelo, com um padrão."""

    default: PriceSettings = PriceSettings()
    models: dict[str, PriceSettings] = Field(default_factory=dict)


class ValidationSettings(_Strict):
    """Validação das hipóteses no holdout (Bonferroni)."""

    corrected_pval_threshold: float = Field(gt=0, lt=1, default=0.1)
    top_k_concepts: int = Field(gt=0, default=10)
    max_tweets_per_run: int = Field(gt=0, default=500)
    max_annotation_calls: int = Field(gt=0, default=10_000)


class SamplingSettings(_Strict):
    """Amostragem estratificada por conceito para rotulagem humana."""

    tweets_per_concept: int = Field(gt=0, default=20)
    salt_env_var: str = "DIAGNOSTICS_SAMPLE_SALT"


class PromptSettings(_Strict):
    """Versões de prompt de rotulagem (v1 intacto, v2 derivado das hipóteses)."""

    directory: str = "prompts"
    v1: str
    v2: str = "v2.md"
    labeling_model_temperature: float = Field(ge=0, default=0.0)


class ComparisonSettings(_Strict):
    """Comparação pareada v1 vs v2 no gold set."""

    gold_eval_fraction: float = Field(gt=0, lt=1, default=0.5)
    gold_max_tweets: int = Field(gt=0, default=2000)
    n_folds: int = Field(ge=6, default=10)
    n_bootstrap: int = Field(gt=0, default=1000)
    confidence_level: float = Field(gt=0, lt=1, default=0.95)
    significance_alpha: float = Field(gt=0, lt=1, default=0.05)


class DiagnosticsSettings(_Strict):
    """Configuração completa de ``configs/diagnostics.yaml``."""

    random_seed: int = 42
    data: DataSettings
    splits: SplitSettings = SplitSettings()
    embedding: EmbeddingSettings
    sae: SaeSettings = SaeSettings()
    sanity: SanitySettings = SanitySettings()
    hypotheses: HypothesesSettings = HypothesesSettings()
    llm: LLMSettings
    pricing: PricingSettings = PricingSettings()
    targets: dict[str, dict[str, str]] = Field(default_factory=dict)
    validation: ValidationSettings = ValidationSettings()
    sampling: SamplingSettings = SamplingSettings()
    prompts: PromptSettings
    comparison: ComparisonSettings = ComparisonSettings()


def load_diagnostics_settings(
    config_file: Path = DEFAULT_DIAGNOSTICS_CONFIG_FILE,
) -> DiagnosticsSettings:
    """Lê e valida ``configs/diagnostics.yaml``.

    Parameters
    ----------
    config_file : Path, optional
        Caminho do YAML, by default :data:`DEFAULT_DIAGNOSTICS_CONFIG_FILE`.

    Returns
    -------
    DiagnosticsSettings
        Configuração validada.

    Raises
    ------
    ConfigurationFileNotFoundError
        Se o arquivo não existir.
    InvalidConfigurationError
        Se o conteúdo violar o schema (chave desconhecida, valor fora da faixa etc.).

    Examples
    --------
    >>> load_diagnostics_settings().llm.provider  # doctest: +SKIP
    'ollama'
    """
    if not config_file.is_file():
        raise ConfigurationFileNotFoundError(str(config_file))
    try:
        settings = DiagnosticsSettings.model_validate(read_yaml(config_file))
    except ValidationError as exception:
        raise InvalidConfigurationError(f"{config_file.name}: {exception}") from exception
    logger.info("Configuração de diagnóstico carregada de '%s'.", config_file)
    return settings
