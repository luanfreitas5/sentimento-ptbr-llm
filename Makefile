# --- Configuração ----------------------------------------------------------
PYTHON := python
RUN := uv run python src/main.py    # 'src' vira raiz do path ao rodar o script

# PYTHONHASHSEED precisa ser exportado ANTES do interpretador iniciar: definido
# dentro do processo, não afeta a ordem de iteração de conjuntos já criada.
export PYTHONHASHSEED := 42

.DEFAULT_GOAL := help
.PHONY: help init venv install install-all  update lock export \
	lint typecheck security deadcode complexity docstrings modernize quality \
	test smoke test-all coverage hooks pre-commit update-hooks release docs docs-serve docs-deploy profile clean cache jupyter notebook add remove tree \
	clean-processed clean-reports clean-outputs clean-notebooks \

help:  ## Lista os alvos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

init:  ## Inicializa o projeto (instala dependências + hooks)
	$(MAKE) install
	$(MAKE) hooks

venv:  ## Cria o ambiente virtual (requer: uv)
	uv venv

install:  ## Instala dependências (runtime + dev)
	uv sync --dev

install-all:  ## Instala tudo (todos os extras + dev)
	uv sync --all-extras --dev



update:  ## Atualiza todas as dependências e sincroniza
	uv lock --upgrade
	uv sync --all-groups

lock:
	uv lock

export:
	uv export --no-hashes -o requirements.txt

# --- Qualidade -------------------------------------------------------------
lint: ## Lint com ruff (Format + Check)
	uv run ruff format .
	uv run ruff check --fix .

typecheck:  ## Type checking estático (basedPyright)
	uv run basedpyright

security:  ## Análise de segurança (bandit + pip-audit)
	uv run bandit -r src -c pyproject.toml
	uv run pip-audit --ignore-vuln PYSEC-2026-2447 --ignore-vuln PYSEC-2026-3552

deadcode:  ## Detecta código morto (vulture)
	uv run vulture src

complexity:  ## Limites de complexidade (xenon)
	uv run xenon --max-absolute B --max-modules A --max-average A src

docstrings:  ## Cobertura de docstrings (interrogate)
	uv run interrogate -v src

modernize:  ## Detecta código redundante (refurb)
	uv run refurb src

quality: lint typecheck security deadcode complexity docstrings modernize   ## Roda toda a suíte de qualidade (espelha o CI)

# --- Testes ----------------------------------------------------------------
test:  ## Roda os testes com cobertura
	uv run pytest -m "not slow"

smoke:  ## Roda apenas os smoke tests
	uv run pytest -m smoke -q

test-all:  ## Roda a suíte completa, inclusive os testes lentos
	uv run pytest

coverage:  ## Gera o relatório de cobertura em HTML
	uv run pytest -m "not slow" --cov-report=html
	@echo "Relatório disponível em htmlcov/index.html"

hooks:  ## Instala os hooks do pre-commit
	uv run pre-commit install
	uv run pre-commit install --hook-type commit-msg
	uv run detect-secrets scan > .secrets.baseline

pre-commit:  ## Roda todos os hooks do pre-commit em todos os arquivos
	uv run pre-commit run --all-files

update-hooks:  ## Atualiza os hooks do pre-commit
	uv run pre-commit autoupdate

release:  ## Cria uma nova release (versão + changelog + tag)
	uv run cz bump --changelog

# --- Limpeza de saídas do pipeline ------------------------------------------
clean-processed:  ## Remove os artefatos de dados processados
	rm -rf data/processed/*.parquet

clean-reports:  ## Remove os relatórios gerados (pastas por modelo + comparação)
	find reports -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +

clean-outputs: clean-processed clean-reports  ## Remove todas as saídas do pipeline

clean-notebooks:  ## Remove os notebooks com células vazias
	uv run nbstripout notebooks

# --- Documentação ----------------------------------------------------------
docs:  ## Constrói a documentação (modo estrito)
	uv run mkdocs build --strict

docs-serve:  ## Servidor local da documentação
	uv run mkdocs serve

docs-deploy:  ## Publica a documentação no GitHub Pages
	uv run mkdocs gh-deploy --force

# --- Utilitários -----------------------------------------------------------
profile:  ## Exemplo de profiling com scalene (ajuste o alvo)
	uv run scalene src/main.py

clean:  ## Remove caches e artefatos temporários
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov coverage.xml site
	find . -type d -name __pycache__ -exec rm -rf {} +

cache:
	uv cache clean

# --- Jupyter ----------------------------------------------------------------
jupyter:
	uv run jupyter lab

notebook:
	uv run jupyter notebook

# --- Gerenciamento de pacotes -----------------------------------------------
add:
	uv add $(PKG)

remove:
	uv remove $(PKG)

tree:
	uv tree

# --- Pipeline ---------------------------------------------------------------
# Cada alvo executa uma etapa isolada; o acoplamento entre elas é o sistema de
# arquivos, então qualquer etapa pode ser reexecutada sem repetir as anteriores.


# --- Serviços auxiliares ----------------------------------------------------
mlflow:  ## Sobe a interface do MLflow para inspecionar os experimentos
	uv run mlflow ui --backend-store-uri mlruns

app:  ## Sobe o dashboard Streamlit de resultados
	uv run streamlit run app/dashboard.py
