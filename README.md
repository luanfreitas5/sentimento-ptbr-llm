# 🇧🇷 Análise de Sentimentos em Textos do Twitter/X em Português Brasileiro

### Um Estudo Comparativo entre Algoritmos Clássicos de Machine Learning, Deep Learning e LLMs Locais

**Repositório GitHub**: [`luanfreitas5/sentimento-ptbr-llm`](https://github.com/luanfreitas5/sentimento-ptbr-llm)

[![CI](https://github.com/luanfreitas5/sentimento-ptbr-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/luanfreitas5/sentimento-ptbr-llm/actions/workflows/ci.yml)
[![Tests](https://github.com/luanfreitas5/sentimento-ptbr-llm/actions/workflows/tests.yml/badge.svg)](https://github.com/luanfreitas5/sentimento-ptbr-llm/actions/workflows/tests.yml)
[![Docs](https://github.com/luanfreitas5/sentimento-ptbr-llm/actions/workflows/docs.yml/badge.svg)](https://luanfreitas5.github.io/sentimento-ptbr-llm)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![uv](https://img.shields.io/badge/managed%20by-uv-de5fe9.svg)](https://docs.astral.sh/uv/)

---

## 🎯 Sobre o projeto

Redes sociais como o Twitter/X constituem uma das principais fontes de dados textuais em português brasileiro para estudos de opinião pública, monitoramento de marcas, análise de crises e pesquisa sociolinguística computacional. Apesar do volume expressivo de dados disponíveis, o português brasileiro permanece uma língua de recursos limitados no que diz respeito a corpora rotulados de sentimento provenientes de mídias sociais — domínio caracterizado por ruído ortográfico, gírias, ironia, emojis e construções sintáticas informais que desafiam tanto abordagens clássicas de NLP quanto modelos de linguagem de propósito geral.

Este projeto desenvolve e avalia comparativamente um pipeline de análise de sentimentos (positivo/negativo/neutro) para tweets em português brasileiro, contrastando **quatro paradigmas** de classificação sob o mesmo protocolo experimental — mesmo split, mesmas métricas, mesmos testes estatísticos:

| Paradigma | Exemplos de modelo | Representação |
|---|---|---|
| ML clássico | Naive Bayes, Regressão Logística, SVM, Random Forest, Gradient Boosting | TF-IDF, FastText, embeddings contextuais + autoencoder |
| Deep Learning | BiLSTM, CNN para texto | Embeddings estáticos (FastText) |
| Transformers (fine-tuning) | BERTimbau, RoBERTa pt-BR, DistilBERT pt-BR | Embeddings contextuais |
| LLMs locais | Llama 3.1, Gemma 2 via Ollama/Hugging Face | Prompting zero-shot / few-shot / chain-of-thought |

A execução local dos LLMs (Ollama/Hugging Face) é particularmente relevante para dados de redes sociais, que envolvem restrições de privacidade e custo que desaconselham o uso de APIs proprietárias na nuvem. A métrica principal é o **F1-macro** (robusto ao desbalanceamento entre classes), complementado pelo **MCC**, testes de significância (McNemar, Wilcoxon, Friedman + post-hoc de Nemenyi), calibração de probabilidades e avaliação por slice — com ênfase em rigor metodológico, escalabilidade computacional e reprodutibilidade.

---

## 📦 Estrutura do projeto

```text
sentimento-ptbr-llm/
├── .devcontainer/
│   └── devcontainer.json
├── .dvc/
│   └── config
├── .github/
│   ├── workflows/
│   │   ├── bump-version.yml
│   │   ├── ci.yml
│   │   ├── dependency-review.yml
│   │   ├── docs.yml
│   │   ├── pre-commit.yml
│   │   ├── release.yml
│   │   ├── reproducibility.yml
│   │   └── tests.yml
│   └── dependabot.yml
├── .vscode/
│   ├── extensions.json
│   ├── launch.json
│   └── settings.json
├── app/                        # API FastAPI e/ou dashboard Streamlit (opcional)
│   ├── __init__.py
│   ├── api.py
│   └── dashboard.py
├── configs/                    # Configurações versionadas (YAML), validadas em runtime com Pydantic
│   ├── config.yaml
│   ├── deploy.yaml
│   ├── evaluation.yaml
│   ├── hypothesaes.yaml
│   ├── labeling.yaml
│   ├── llm.yaml
│   ├── logging.yaml
│   ├── model_params.yaml
│   └── paths.yaml
├── data/                       # raw/ (nunca modificar) → interim/ → processed/ (fora do Git; ver dvc.yaml)
│   ├── external/
│   ├── interim/
│   ├── processed/
│   └── raw/
├── docs/                       # Documentação técnica (MkDocs Material + mkdocstrings)
│   ├── assets/
│   ├── guides/
│   │   ├── metodologia.md
│   │   ├── pipeline.md
│   │   ├── reprodutibilidade.md
│   │   └── setup.md
│   ├── index.md
│   └── reference.md
├── logs/                       # Logs diários (rich + logging, fora do Git)
├── models/                     # Checkpoints, artefatos e registry (fora do Git; ver dvc.yaml)
│   ├── artifacts/
│   ├── checkpoints/
│   └── registry/
├── notebooks/                  # Exploração e relatório final (lógica de produção fica em src/)
│   ├── 01_eda_corpus.ipynb
│   ├── 02_diagnostico_rotulagem.ipynb
│   ├── 03_engenharia_features.ipynb
│   ├── 04_ml_classico.ipynb
│   ├── 05_deep_learning_transformers.ipynb
│   ├── 06_llm_prompting.ipynb
│   └── 07_avaliacao_comparativa.ipynb
├── reports/                    # Figuras, métricas, tabelas, model cards e datasheets
│   ├── ablation/
│   ├── datasheets/
│   │   └── datasheet_corpus_tweets.md
│   ├── figures/
│   ├── interpretability/
│   ├── metrics/
│   ├── model_cards/
│   │   ├── model_card_deep_learning.md
│   │   ├── model_card_llm_local.md
│   │   ├── model_card_ml_classico.md
│   │   └── model_card_transformer.md
│   ├── statistics/
│   └── tables/
├── scripts/                    # Utilitários de manutenção do repositório
├── src/                        # Código de produção — ver src/__init__.py para a lista de packages
│   ├── config/                 # Configuração validada, ambiente, logging, caminhos
│   ├── constants/               # Constantes, enums e valores padrão
│   ├── data/                     # Ingestão e carregamento
│   ├── evaluation/                # Métricas, significância, calibração, slices, ablation
│   ├── exceptions/                 # Hierarquia de exceções customizadas
│   ├── experiment/                  # Rastreamento de experimentos (MLflow)
│   ├── features/                     # Representações e engenharia de features
│   ├── hypothesaes/                   # Diagnóstico via Autoencoder Esparso (HypotheSAEs)
│   │   └── prompts/
│   ├── inference/                      # Camada comum de inferência
│   ├── io_utils/                        # Leitura/escrita (YAML, JSON, CSV, Parquet, modelo)
│   ├── labeling/                         # Rotulagem em cascata
│   ├── llm/                               # LLMs locais e orquestração LangChain
│   ├── logging_utils/                      # Handlers, formatadores e utilitários de log
│   ├── metrics/                             # Métricas de classificação/ranking/confiança
│   ├── models/                               # Modelos por paradigma (clássico, DL, Transformer, LLM)
│   ├── parallel/                              # Programação paralela transversal
│   ├── pipelines/                              # Estágios executáveis (`--stage`)
│   ├── preprocessing/                           # Limpeza e normalização de texto pt-BR
│   ├── schemas/                                  # Contratos de dados (Pandera)
│   ├── training/                                  # Treino, callbacks e checkpointing
│   ├── utils/                                      # Utilitários genéricos
│   ├── visualization/                               # Gráficos e diagnósticos
│   ├── __init__.py
│   └── main.py                                       # Orquestração via `--stage`
├── tests/                       # Testes unitários, integração, propriedade e comportamentais
│   ├── conftest.py
│   └── test_*.py                # um módulo de teste por package de src/
├── .env.example
├── .gitattributes
├── .gitignore
├── .pre-commit-config.yaml
├── CHANGELOG.md
├── CLAUDE.md
├── Dockerfile
├── LICENSE
├── Makefile
├── README.md
├── docker-compose.yml
├── dvc.yaml
├── mkdocs.yml
├── params.yaml
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

---

## 🚀 Setup rápido

Pré-requisitos: [uv](https://docs.astral.sh/uv/) e Python 3.10–3.13.

```bash
git clone https://github.com/luanfreitas5/sentimento-ptbr-llm.git
cd sentimento-ptbr-llm

make init          # uv sync --dev + hooks do pre-commit
cp .env.example .env  # preencha os segredos localmente (nunca commitado)
```

Extras opcionais, instalados sob demanda conforme a etapa do pipeline usada:

```bash
make install-collect   # twscrape (coleta de tweets)
make install-nlp       # spaCy + modelo pt-BR (lematização) e nltk + corpus de stopwords
make install-llm       # PyTorch + Transformers + Accelerate + Ollama
uv sync --extra viz    # wordcloud, networkx, umap-learn (figuras opcionais)
uv sync --extra dvc    # versionamento de dados/modelos
uv sync --extra app    # Streamlit + Plotly (dashboard)
```

Guia detalhado: [`docs/guides/setup.md`](docs/guides/setup.md).

---

## ▶️ Uso

O pipeline é orquestrado por estágios via `src/main.py --stage <nome>` (ver `configs/config.yaml -> stages`), com atalhos no `Makefile`:

```bash
make pipeline-ingestion SCRAPE_FUNC=parallel.scraping:collect_tweets QUERIES="palavra1 palavra2"
make pipeline-preprocessing
make pipeline-labeling
make pipeline-features
make pipeline-training-classical
make pipeline-training-deep-learning
make pipeline-llm-evaluation
make pipeline-comparative-evaluation PREDICTIONS_FUNC=modulo:funcao
make pipeline-all          # executa todos os estágios, na ordem de configs/config.yaml

make mlflow    # UI do MLflow (mlruns/)
make app       # dashboard Streamlit (app/dashboard.py)
```

Detalhes de cada estágio: [`docs/guides/pipeline.md`](docs/guides/pipeline.md).

---

## ✅ Qualidade e testes

```bash
make quality   # ruff + basedPyright + bandit/pip-audit + vulture + xenon + interrogate + refurb
make test      # pytest -m "not slow", cobertura mínima de 80%
make test-all  # inclui os testes lentos
make coverage  # relatório HTML em htmlcov/
```

CI no GitHub Actions roda a mesma suíte a cada push/PR (`.github/workflows/ci.yml`, `tests.yml`).

---

## 🔁 Reprodutibilidade

- Seed única (`42`) propagada por `src/config/environment.py::seed_everything` (`random`, `numpy`, `PYTHONHASHSEED`, e frameworks de DL/Transformers quando presentes).
- `uv.lock` commitado — ambiente resolvido de forma determinística.
- Cada execução é rastreada no MLflow (`mlruns/`) junto com o Git SHA e o hash SHA-256 dos dados de entrada.

Detalhes: [`docs/guides/reprodutibilidade.md`](docs/guides/reprodutibilidade.md).

---

## 📚 Documentação

Documentação completa (metodologia, guias, referência de API gerada via `mkdocstrings`) publicada em **[luanfreitas5.github.io/sentimento-ptbr-llm](https://luanfreitas5.github.io/sentimento-ptbr-llm)**.

```bash
make docs-serve   # servidor local em http://127.0.0.1:8000
```

---

## 📄 Licença

Distribuído sob a licença [MIT](LICENSE). Os dados coletados/rotulados **não são redistribuídos** neste repositório — ver [`reports/datasheets/datasheet_corpus_tweets.md`](reports/datasheets/datasheet_corpus_tweets.md).

## ✍️ Autor

**Luan Freitas** — [luan.mgf@gmail.com](mailto:luan.mgf@gmail.com) — [GitHub](https://github.com/luanfreitas5)
