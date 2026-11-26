# 🇧🇷 Interpretabilidade de Modelos BERT com Autoencoders Esparsos na Análise de Sentimentos de Publicações do X em Português Brasileiro

### Qualidade de rótulos automáticos, consenso entre rotuladores e hipóteses interpretáveis (HypotheSAEs)

> Dissertação de mestrado (PPCA/UnB) — texto de qualificação em [`Monografia/`](Monografia/). Autor: Luan Mendes Gonçalves Freitas. Orientador: Prof. Dr. Marcos Fagundes Caetano; coorientador: Prof. Dr. Thiago Paulo Faleiros.

**Repositório GitHub**: [`luanfreitas5/sentimento-ptbr-llm`](https://github.com/luanfreitas5/sentimento-ptbr-llm)

[![CI](https://github.com/luanfreitas5/sentimento-ptbr-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/luanfreitas5/sentimento-ptbr-llm/actions/workflows/ci.yml)
[![Tests](https://github.com/luanfreitas5/sentimento-ptbr-llm/actions/workflows/tests.yml/badge.svg)](https://github.com/luanfreitas5/sentimento-ptbr-llm/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/luanfreitas5/sentimento-ptbr-llm/branch/main/graph/badge.svg)](https://codecov.io/gh/luanfreitas5/sentimento-ptbr-llm)
[![Docs](https://github.com/luanfreitas5/sentimento-ptbr-llm/actions/workflows/docs.yml/badge.svg)](https://luanfreitas5.github.io/sentimento-ptbr-llm)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![uv](https://img.shields.io/badge/managed%20by-uv-de5fe9.svg)](https://docs.astral.sh/uv/)

---

## 🎯 Sobre o projeto

Redes sociais como o X (antigo Twitter) concentram grande volume de opinião em português brasileiro, mas o português é uma língua de recursos limitados quando se trata de corpora de sentimento de mídias sociais — domínio com ruído ortográfico, gírias, ironia, emojis e construções informais. Modelos da família BERT e LLMs são as principais ferramentas para classificar sentimento, porém pouco transparentes, e o corpus disponível (**64.086 posts** de 2020–2021) **não possui rótulos humanos**.

O projeto investiga a relação entre a **interpretabilidade das representações internas de modelos BERT** e a **concordância de suas predições de sentimento** (positivo/negativo/neutro) com rótulos de referência, usando **autoencoders esparsos (SAEs)** para identificar padrões latentes associados às decisões dos modelos. O problema tem duas partes que se alimentam:

1. **Rótulos:** como obter rótulos em larga escala, a baixo custo e com qualidade quantificada, sem anotação humana em escala?
2. **Interpretabilidade:** como decompor as representações de modelos BERT em padrões latentes interpretáveis e relacioná-los às divergências e à incerteza dos rotuladores?

### Desenho experimental

| Fonte | Papel | Cobertura |
|---|---|---|
| 8 classificadores BERT do Hugging Face (primário: `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual`; secundário: `pysentimiento/bertweet-pt-sentiment`; + 6 adicionais), revisões fixadas, executados localmente | Rotuladores do corpus | Todo o corpus |
| `gold_openai` (`gpt-6-luna`, *reasoning effort low*) | Referência primária (*gold set* de LLM) | Amostra comum de 2.000 posts |
| `gold_unb` (`UnB-Llama-3.3-70B-Instruct`) | Referência secundária | Mesma amostra |
| Validação humana estratificada por confiança | Estimativa do acerto real | n = 500 |

O texto enviado a APIs externas é sanitizado (sem @menções e URLs, conforme a LGPD). Os *gold sets* são rótulos de LLM, não de humanos: medem concordância, não acerto absoluto — por isso a validação humana. A qualidade dos rótulos é medida por **Kappa de Cohen** (simples e ponderado), **Kappa de Fleiss**, **Alpha de Krippendorff**, Macro F1, MCC e McNemar, sempre com **IC de 95% por *bootstrap*** (1.000 reamostragens). As hipóteses interpretáveis são geradas pelo **HypotheSAEs** (SAE com 256 neurônios, 8 ativos, sobre embeddings `paraphrase-multilingual-mpnet-base-v2`) para dois alvos — *divergência* e *incerteza* — em 16 pares modelo × *gold* (32 alvos), com *gate* de sanidade estatístico, validação em *holdout* e correção para comparações múltiplas (FDR de Benjamini–Hochberg).

### Perguntas de pesquisa

| # | Pergunta | Hipótese |
|---|---|---|
| PQ1 | Os rótulos dos BERTs concordam entre si e com os *gold sets* a ponto de serem indício de qualidade? | **H1.** κ ≥ 0,6 com os *gold sets*, com discordância concentrada no neutro |
| PQ2 | Onde e por que os rotuladores divergem? | **H2.** Divergência e baixa confiança associam-se a padrões linguísticos recuperáveis como hipóteses |
| PQ3 | Os padrões latentes se relacionam com as decisões e a concordância dos modelos? | **H3.** Atributos interpretáveis associam-se à divergência e à incerteza |
| PQ4 | As hipóteses são estáveis entre *gold sets* diferentes? | **H4.** As de incerteza são consistentes; as de divergência dependem do rotulador de referência |

### Resultados preliminares (qualificação)

Todos os números vêm de artefatos gerados pelo pipeline em `reports/` (IC de 95% entre colchetes):

| Achado | Resultado |
|---|---|
| Corpus | 71.995 posts na bronze → **64.086** na silver (89,0%); neutro 40,1%, negativo 32,3%, positivo 27,6% |
| Rotulador primário × *gold sets* | κ = 0,46–0,53 (ponderado quadrático 0,64–0,66); divergência concentrada na fronteira com o neutro |
| Acerto contra o humano (n = 500) | primário **71,6%** [68,0; 75,2]; `gold_unb` 74,6%; `gold_openai` **79,8%** (McNemar: significativamente melhor que o primário) |
| Consenso entre 10 rotuladores | κ de Fleiss = **0,28**; unanimidade em apenas 6,4% dos posts; o neutro é a classe de menor acordo |
| HypotheSAEs | 32/32 alvos aprovados no *gate* de sanidade; no *holdout*, 0/10 hipóteses de divergência e 5/10 de incerteza sobreviveram ao FDR (2 ao Bonferroni), com efeitos pequenos |

**Leitura:** o rotulador primário é um sinal razoável, porém ruidoso (H1 parcial: κ abaixo do limiar, neutro confirmado); sua confiança é informativa (concordância com o humano sobe de 0,52 na faixa baixa a 0,89 na alta); as hipóteses são **exploratórias**. Ficam como próximos passos um segundo anotador humano, a repetição da interpretação com um LLM maior e a análise da relação entre interpretabilidade e concordância.

> **Nota de escopo:** a versão inicial do projeto previa comparar famílias de classificadores (ML clássico, *deep learning*, *transformers* e LLMs locais) e usar TweetSentBR/RePro como *gold sets*. Isso saiu do escopo da dissertação — sem rótulos humanos em escala não há treino/avaliação supervisionados confiáveis — e o foco passou a ser a qualidade dos rótulos e a interpretabilidade. Os estágios supervisionados (`features`, `training_*`, `evaluate`) continuam no código como trabalho exploratório e não fazem parte dos resultados da monografia.

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
│   ├── diagnostics.yaml
│   ├── hypothesaes.yaml
│   ├── labeling.yaml
│   ├── llm.yaml
│   ├── logging.yaml
│   ├── model_params.yaml
│   └── paths.yaml
├── data/                       # medallion: bronze/ (imutável, inclui external/) → silver/ → gold/ (fora do Git; ver dvc.yaml)
│   ├── bronze/                 # dados como recebidos (tweets por usuário, catalog.json)
│   ├── silver/                 # limpos, tipados e validados (corpus normalizado, checkpoints)
│   └── gold/                   # prontos para modelagem (corpus rotulado, treino/validação/teste, features)
├── docs/                       # Documentação técnica (MkDocs Material + mkdocstrings)
│   ├── assets/
│   ├── guides/
│   │   ├── diagnostico-hypothesaes.md
│   │   ├── metodologia.md
│   │   ├── pipeline.md
│   │   ├── reprodutibilidade.md
│   │   ├── setup.md
│   │   └── validade-e-privacidade.md
│   ├── index.md
│   └── reference.md
├── Monografia/                 # Texto de qualificação (LaTeX, modelo UnB-CIC), bibliografia e apresentação
│   ├── tex/                    # Capítulos: introdução, fundamentação, metodologia, resultados, ameaças, cronograma
│   └── monografia.tex
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
│   ├── 07_avaliacao_comparativa.ipynb
│   └── 08–10_cloud_gpu_*.ipynb   # execução em GPU na nuvem
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
├── src/                        # Código de produção — ver src/__init__.py para a lista de packages
│   ├── config/                 # Configuração validada, ambiente, logging, caminhos
│   ├── constants/               # Constantes, enums e valores padrão
│   ├── data/                     # Ingestão e carregamento
│   ├── diagnostics/                # Diagnóstico: comparação, validação humana, hipóteses, reconstrução do SAE
│   ├── evaluation/                # Métricas, significância, calibração, slices, ablation
│   ├── exceptions/                 # Hierarquia de exceções customizadas
│   ├── experiment/                  # Rastreamento de experimentos (MLflow)
│   ├── features/                     # Representações e engenharia de features
│   ├── hypothesaes/                   # Diagnóstico via Autoencoder Esparso (HypotheSAEs)
│   │   └── prompts/
│   ├── inference/                      # Camada comum de inferência
│   ├── io_utils/                        # Leitura/escrita (YAML, JSON, CSV, Parquet, modelo)
│   ├── labeling/                         # Rotulagem por LLM (Hugging Face e OpenAI), com checkpoint
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
make install-labeling  # PyTorch + Transformers + Accelerate + cliente OpenAI (rotulagem)
make install-hypothesaes  # + Ollama, sentence-transformers (HypotheSAEs)
uv sync --extra viz    # wordcloud, networkx, umap-learn (figuras opcionais)
uv sync --extra dvc    # versionamento de dados/modelos
uv sync --extra app    # Streamlit + Plotly (dashboard)
```

Guia detalhado: [`docs/guides/setup.md`](docs/guides/setup.md).

---

## ▶️ Uso

O pipeline é orquestrado por estágios via `src/main.py --stage <nome>` (ver `configs/config.yaml -> stages`; os estágios da monografia vão de `ingestion` a `hypotheses`), com atalhos no `Makefile`:

```bash
make ingest SCRAPE_FUNC=parallel.scraping:collect_tweets QUERIES="palavra1 palavra2"
make preprocess
make label               # HF (todo o corpus) + gold sets (gpt-6-luna e UnB-Llama)
make label-hf-extra      # modelos adicionais do Hugging Face (todo o corpus)
make label-hf            # só o modelo do Hugging Face (todos os tweets)
make label-gold-openai   # só o gold set gpt-6-luna (API; OPENAI_KEY no .env)
make label-gold-unb      # só o gold set UnB-Llama-3.3-70B-Instruct (API)
make consensus           # consenso entre rotuladores: votos, Kappa de Fleiss, Alpha de Krippendorff
make compare             # compara o Hugging Face com cada gold set: concordância, Kappa, divergências
make hypotheses          # hipóteses do HypotheSAEs (MODE=disagreement|patterns|diagnostics)
make diagnostics         # camada de diagnóstico (STEP=hypotheses|validation|comparison)

# Estágios supervisionados exploratórios (fora do escopo da monografia):
make features            # split + TF-IDF
make classical           # Baseline (dummy) + ML tradicional
make deep                # Deep Learning (LSTM, CNN)
make transformer         # Transformers (BERTimbau, RoBERTa, DistilBERT)
make llm                 # LLM open-source via Ollama (few-shot)
make evaluate            # teste + estatística (McNemar) + ablação
make report              # figuras + tabelas + Model Cards + Datasheet
make all                 # todos os estágios, na ordem de configs/config.yaml

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
make install-ci && make test-ci   # ambiente idêntico ao do CI (PyTorch CPU + clientes LLM): único que garante o fail_under de 80%
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
