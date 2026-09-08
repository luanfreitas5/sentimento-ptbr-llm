# Pipeline

O pipeline é orquestrado por **estágios**, executáveis isoladamente via:

```bash
uv run python src/main.py --stage <nome_do_estagio>
uv run python src/main.py --stage all   # executa todos, na ordem de configs/config.yaml -> stages
```

O acoplamento entre estágios é o sistema de arquivos (Parquet em `data/`, checkpoints em `models/`) — qualquer estágio pode ser reexecutado sem repetir os anteriores, desde que suas entradas já existam em disco. A lista canônica e a ordem dos estágios vivem em `configs/config.yaml -> stages` e no registro `src/pipelines/workflow.py::STAGE_REGISTRY`.

## Estágios

| Estágio | Atalho `make` | Módulo | Entrada | Saída |
|---|---|---|---|---|
| `ingestion` | `pipeline-ingestion` | `src/pipelines/ingestion.py` | função de coleta definida pelo usuário (`--scrape-func`) | `data/interim/tweets_coletados.parquet` |
| `preprocessing` | `pipeline-preprocessing` | `src/pipelines/preprocessing.py` | lote de tweets brutos coletados por usuário, `data/raw/*.parquet` (ver `src/data/loader.py::load_raw_tweet_batch`) | `data/interim/corpus_normalizado.parquet` |
| `labeling` | `pipeline-labeling` | `src/pipelines/labeling.py` | corpus normalizado + gold sets (`data/external/`) | `data/processed/corpus_rotulado.parquet` |
| `features` | `pipeline-features` | `src/pipelines/features.py` | corpus rotulado | splits treino/validação/teste + features TF-IDF |
| `training_classical` | `pipeline-training-classical` | `src/pipelines/training_classical.py` | splits + features | checkpoints em `models/checkpoints/` |
| `training_deep_learning` | `pipeline-training-deep-learning` | `src/pipelines/training_deep_learning.py` | splits + embeddings | checkpoints (DL/Transformers/autoencoder) |
| `llm_evaluation` | `pipeline-llm-evaluation` | `src/pipelines/llm_evaluation.py` | conjunto de teste | classificação + métricas via LLM local |
| `comparative_evaluation` | `pipeline-comparative-evaluation` | `src/pipelines/comparative_evaluation.py` | função de predições definida pelo usuário (`--predictions-func`) | `reports/metrics/comparativo.csv`, `reports/tables/`, `reports/statistics/` |
| `hypothesaes_analysis` | — | `src/pipelines/hypothesaes_analysis.py` | corpus rotulado | diagnóstico de rotulagem em `reports/interpretability/` |

## Estágios que exigem argumentos explícitos

Dois estágios não têm uma implementação fixa por design — o projeto não assume uma fonte de scraping nem um formato de predições:

```bash
# ingestion: --scrape-func aponta para uma função "modulo:funcao" implementada
# por você (ver src/data/downloader.py para o contrato esperado).
make pipeline-ingestion SCRAPE_FUNC=meu_modulo:minha_funcao_de_coleta QUERIES="termo1 termo2"

# comparative_evaluation: --predictions-func deve retornar (model_predictions, y_true)
# a partir dos modelos já treinados/registrados no MLflow.
make pipeline-comparative-evaluation PREDICTIONS_FUNC=meu_modulo:minha_funcao_de_predicoes
```

Ambos os caminhos são resolvidos dinamicamente por `src/main.py::_import_callable_from_dotted_path`.

## Versionamento do DAG com DVC (opcional)

`dvc.yaml` espelha os mesmos estágios para quem instalou o extra `dvc` (`uv sync --extra dvc`):

```bash
dvc repro                    # roda todo o DAG, pulando estágios sem mudança em deps/params
dvc repro training_classical # roda só esse estágio (e dependências desatualizadas)
dvc dag                      # visualiza o grafo de dependências
```

Parâmetros de linha de comando (`--scrape-func`, `--queries`, `--predictions-func`) ficam em `params.yaml`, na raiz do repositório — edite antes de rodar `dvc repro`.

## Serviços auxiliares

```bash
make mlflow   # UI do MLflow (mlruns/) — inspeciona métricas/params/artefatos por execução
make app      # dashboard Streamlit (app/dashboard.py), se o extra `app` estiver instalado
```
