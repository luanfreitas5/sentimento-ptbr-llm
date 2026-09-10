# Setup

## Pré-requisitos

- [uv](https://docs.astral.sh/uv/) ≥ 0.9 (gerenciador de pacotes e ambientes)
- Python 3.10 a 3.13 (`requires-python` em `pyproject.toml`)
- Git
- Opcional: [Ollama](https://ollama.com/) instalado localmente para os estágios de LLM (`configs/llm.yaml -> backends.ollama`)

## Instalação

```bash
git clone https://github.com/luanfreitas5/sentimento-ptbr-llm.git
cd sentimento-ptbr-llm

make init          # equivalente a: uv sync --dev && make hooks
cp .env.example .env
```

Edite `.env` com os valores reais (nunca commitado — ver `.gitignore`). As variáveis aceitas estão documentadas em `.env.example` e validadas em `src/config/settings.py::Settings`.

## Extras opcionais

O projeto instala apenas as dependências de runtime "leves" por padrão. Extras pesados ou específicos de uma etapa ficam isolados em `[project.optional-dependencies]` (`pyproject.toml`) e são instalados sob demanda:

| Extra | Instala | Quando precisa |
|---|---|---|
| `llm` | PyTorch, Transformers, Accelerate, Ollama, LangChain | Estágios `llm_evaluation` e análise HypotheSAEs |
| `collect` | twscrape | Estágio `ingestion` (coleta de tweets) |
| `nlp` | spaCy + modelo `pt_core_news_sm`, nltk + corpus `stopwords` | Lematização e stopwords em pt-BR no pré-processamento (com fallback por regex/lista curada se ausentes) |
| `viz` | wordcloud, networkx, umap-learn | Figuras opcionais (nuvem de palavras, rede de similaridade, projeção UMAP) |
| `dvc` | DVC | Versionamento de dados/modelos (ver `dvc.yaml`) |
| `app` | Streamlit, Plotly | Dashboard comparativo (`app/dashboard.py`) |

```bash
make install-collect
make install-nlp        # já baixa o modelo pt-BR do spaCy e o corpus de stopwords do nltk
make install-llm
uv sync --extra viz
uv sync --extra dvc
uv sync --extra app

make install-all        # todos os extras + dev, de uma vez
```

## Verificando a instalação

```bash
make quality   # ruff + basedPyright + bandit/pip-audit + vulture + xenon + interrogate + refurb
make test      # pytest -m "not slow", cobertura mínima de 80%
```

Se `make quality` e `make test` passarem, o ambiente está pronto para rodar o pipeline — ver o [guia de pipeline](pipeline.md).

## Ambiente containerizado (opcional)

```bash
docker build -t sentimento-ptbr-llm .
docker run --rm -it --env-file .env sentimento-ptbr-llm --stage preprocessing

docker compose up mlflow                # UI do MLflow em http://localhost:5000
docker compose --profile llm up ollama  # servidor Ollama
```

Ver `.devcontainer/devcontainer.json` para um ambiente de desenvolvimento reprodutível no VS Code (Dev Containers).
