# Referência da API

Gerada automaticamente a partir das docstrings NumPy (em pt-BR) via [mkdocstrings](https://mkdocstrings.github.io/). `paths: [src]` em `mkdocs.yml` faz `src/` a raiz de importação — os identificadores abaixo são relativos a ela (ver CLAUDE.md, "Import style").

## Orquestração

::: main

## `config` — configuração validada, ambiente, logging, caminhos

::: config.settings
::: config.environment
::: config.paths
::: config.constants
::: config.logging
::: config.version

## `constants` — constantes, enums e valores padrão

::: constants.columns
::: constants.labels
::: constants.metrics
::: constants.regex
::: constants.tokens
::: constants.defaults

## `schemas` — contratos de dados (Pandera)

::: schemas.dataset
::: schemas.labeling
::: schemas.prediction
::: schemas.training
::: schemas.experiment

## `data` — ingestão e carregamento

::: data.downloader
::: data.loader
::: data.splitter
::: data.sampler
::: data.writer
::: data.catalog

## `preprocessing` — limpeza e normalização de texto pt-BR

::: preprocessing.text
::: preprocessing.emojis
::: preprocessing.tokenization
::: preprocessing.cleaning
::: preprocessing.filtering
::: preprocessing.pipeline

## `labeling` — rotulagem em cascata

::: labeling.automatic
::: labeling.consensus
::: labeling.confidence
::: labeling.manual
::: labeling.validation

## `hypothesaes` — diagnóstico via Autoencoder Esparso

::: hypothesaes.sae
::: hypothesaes.embedding
::: hypothesaes.select_neurons
::: hypothesaes.interpret_neurons
::: hypothesaes.annotate
::: hypothesaes.evaluation
::: hypothesaes.llm_api
::: hypothesaes.utils

## `features` — representações e engenharia de features

::: features.lexical
::: features.static_embeddings
::: features.contextual_embeddings
::: features.reduction
::: features.selection
::: features.statistics

## `models` — modelos por paradigma

::: models.base
::: models.factory
::: models.naive_bayes
::: models.logistic_regression
::: models.svm
::: models.random_forest
::: models.gradient_boosting
::: models.autoencoder
::: models.lstm
::: models.cnn
::: models.bertimbau
::: models.roberta
::: models.distilbert
::: models.llm
::: models.persistence

## `training` — treino, callbacks e checkpointing

::: training.trainer
::: training.callbacks
::: training.checkpoint
::: training.scheduler
::: training.early_stopping
::: training.cross_validation
::: training.resume

## `llm` — LLMs locais e orquestração LangChain

::: llm.backends
::: llm.prompts
::: llm.parsers
::: llm.chains
::: llm.classifier

## `inference` — predição

::: inference.predictor
::: inference.batch
::: inference.online
::: inference.llm_batch
::: inference.postprocessing

## `metrics` e `evaluation` — métricas e análises estatísticas

::: metrics.classification
::: metrics.ranking
::: metrics.confidence
::: metrics.operational
::: evaluation.evaluator
::: evaluation.calibration
::: evaluation.significance
::: evaluation.ablation
::: evaluation.slice_evaluation
::: evaluation.hypothesaes_report
::: evaluation.reports

## `visualization` — gráficos e diagnósticos

::: visualization.theme
::: visualization.distributions
::: visualization.wordcloud
::: visualization.ngrams
::: visualization.confusion_matrix
::: visualization.roc_pr_curves
::: visualization.embeddings
::: visualization.interpretability
::: visualization.diagnostics
::: visualization.hypothesaes

## `experiment` — rastreamento de experimentos (MLflow)

::: experiment.tracker
::: experiment.registry
::: experiment.reproducibility

## `pipelines` — estágios executáveis

::: pipelines.workflow
::: pipelines.ingestion
::: pipelines.preprocessing
::: pipelines.labeling
::: pipelines.features
::: pipelines.training_classical
::: pipelines.training_deep_learning
::: pipelines.llm_evaluation
::: pipelines.comparative_evaluation
::: pipelines.hypothesaes_analysis

## `parallel` — programação paralela transversal

::: parallel.core
::: parallel.scraping
::: parallel.preprocessing
::: parallel.inference
::: parallel.experiments

## `io_utils`, `logging_utils` e `utils` — infraestrutura transversal

::: io_utils.yaml
::: io_utils.json
::: io_utils.csv
::: io_utils.parquet
::: io_utils.model
::: logging_utils.logger
::: logging_utils.formatter
::: logging_utils.handlers
::: logging_utils.timer
::: utils.seed
::: utils.hashing
::: utils.text
::: utils.decorators
::: utils.validation
::: utils.timing
