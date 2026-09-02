"""Valores padrão de hiperparâmetros e limiares usados em todo o projeto.

Espelham os valores definidos em ``configs/config.yaml``,
``configs/model_params.yaml`` e ``configs/evaluation.yaml``, servindo como
``fallback`` quando um parâmetro não é explicitamente informado.
"""

# Reprodutibilidade (ver src/utils/seed.py e src/config/environment.py)
DEFAULT_RANDOM_SEED = 42

# Particionamento de dados (ver configs/config.yaml -> data_split)
DEFAULT_TEST_SIZE = 0.2
DEFAULT_VALIDATION_SIZE = 0.1

# Validação cruzada (ver configs/evaluation.yaml -> cross_validation)
DEFAULT_CROSS_VALIDATION_FOLDS = 5

# Estimativa de incerteza (ver configs/evaluation.yaml -> uncertainty)
DEFAULT_BOOTSTRAP_ITERATIONS = 1000
DEFAULT_CONFIDENCE_LEVEL = 0.95

# Testes de significância estatística (ver configs/evaluation.yaml)
DEFAULT_SIGNIFICANCE_ALPHA = 0.05

# Metas de qualidade de código (ver pyproject.toml)
DEFAULT_MINIMUM_TEST_COVERAGE = 0.80

# Limiares de regressão de métrica (ver configs/evaluation.yaml -> regression_thresholds)
DEFAULT_F1_MACRO_MINIMUM = 0.65
DEFAULT_MCC_MINIMUM = 0.45
