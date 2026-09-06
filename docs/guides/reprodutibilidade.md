# Reprodutibilidade

`random_state` isolado não é reprodutibilidade. Este projeto fixa ambiente, sementes e rastreia a proveniência completa de cada execução (código + dados + parâmetros).

## Seeds

Uma única seed (`42`) é propagada por todo o pipeline via `src/config/environment.py::seed_everything`:

- `PYTHONHASHSEED` (também exportado no `Makefile`, antes do interpretador iniciar)
- `random.seed`, `numpy.random.seed`
- sementes de frameworks de DL/Transformers (PyTorch), quando presentes
- `random_state` explícito em toda operação estocástica: splits (`configs/config.yaml -> data_split`), modelos clássicos e validação cruzada (`configs/model_params.yaml`, `configs/evaluation.yaml -> cross_validation`)

## Ambiente fixado

- **`uv.lock`** commitado — resolução de dependências determinística, nunca "unpinned".
- **`requires-python = ">=3.10,<3.14"`** (`pyproject.toml`) — evita variação de comportamento entre versões do interpretador.
- **`.pre-commit-config.yaml`** e CI usam as mesmas ferramentas/versões declaradas em `pyproject.toml`, sem duplicação de configuração.

## Hash de dados

Cada dataset intermediário pode ser verificado com `src/utils/hashing.py` (SHA-256), permitindo detectar mudanças silenciosas de dados entre execuções. O catálogo de datasets (`src/data/catalog.py`) registra os hashes junto aos caminhos em `data/raw/catalog.json`.

## Rastreamento com MLflow

Todo treino/avaliação com `track_with_mlflow=True` grava params, métricas e artefatos em `mlruns/` (`configs/config.yaml -> experiment.tracking_uri`, sobrescrito por `SENTIMENTO_MLFLOW_TRACKING_URI` no `.env`). Cada execução é identificável por:

- **Git SHA** do commit em que rodou
- **Hash SHA-256** dos dados de entrada
- **Parâmetros** e **métricas** completos (não apenas o valor final)

```bash
make mlflow   # UI local em http://127.0.0.1:5000
```

## Versionamento de dados e modelos com DVC

Dados (`data/`) e modelos (`models/`) **não são versionados no Git** — apenas o ponteiro (`.dvc`) é commitado; o conteúdo fica no remote configurado em `.dvc/config` (ver `dvc.yaml` para o DAG completo de estágios).

```bash
uv sync --extra dvc
dvc repro          # reexecuta apenas os estágios cujas deps/params mudaram
dvc push           # envia dados/modelos para o remote
```

## Notebooks

Notebooks (`notebooks/`) contêm apenas exploração e relatório final — nenhuma lógica de produção. Antes de commitar, o kernel deve ser reiniciado e todas as células executadas na ordem; `nbstripout` (hook do pre-commit) remove outputs/metadados automaticamente.

## Checklist antes de reportar um resultado

1. O experimento foi rastreado no MLflow, com Git SHA e hash de dados associados?
2. A métrica principal veio acompanhada de intervalo de confiança (bootstrap, `configs/evaluation.yaml -> uncertainty`)?
3. A comparação entre modelos passou por um teste de significância (McNemar/Wilcoxon/Friedman), não apenas por diferença bruta de médias?
4. A métrica foi reportada por slice relevante, além do agregado (`configs/evaluation.yaml -> slice_evaluation`)?
