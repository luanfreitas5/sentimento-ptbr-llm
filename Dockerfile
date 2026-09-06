# =============================================================================
# Dockerfile — imagem de execução do pipeline sentimento-ptbr-llm.
#
# Build multi-stage: os artefatos do `uv sync` são resolvidos em um estágio
# com toolchain completa e copiados para uma imagem final enxuta, rodando
# como usuário não-root (ver configs/deploy.yaml -> docker.non_root_user).
#
# Uso:
#   docker build -t sentimento-ptbr-llm .
#   docker run --rm -it --env-file .env sentimento-ptbr-llm --stage preprocessing
# =============================================================================

# --- Estágio 1: build -------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

# Ferramenta `uv` fixada por dígito de versão, copiada da imagem oficial.
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /build

# Camada de dependências isolada do código-fonte para maximizar o cache do
# Docker: `uv.lock`/`pyproject.toml` mudam com menos frequência que `src/`.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY src/ ./src/
COPY configs/ ./configs/

# --- Estágio 2: runtime ------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

ARG APP_USER=appuser
ARG APP_UID=1000

RUN groupadd --gid "${APP_UID}" "${APP_USER}" \
    && useradd --uid "${APP_UID}" --gid "${APP_USER}" --create-home --shell /usr/sbin/nologin "${APP_USER}"

WORKDIR /app

COPY --from=builder --chown=${APP_USER}:${APP_USER} /build/.venv /app/.venv
COPY --from=builder --chown=${APP_USER}:${APP_USER} /build/src /app/src
COPY --from=builder --chown=${APP_USER}:${APP_USER} /build/configs /app/configs

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONHASHSEED=42 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# data/, models/, reports/, logs/, mlruns/ são montados como volumes em
# tempo de execução (ver docker-compose.yml) — nunca embutidos na imagem.
RUN mkdir -p /app/data /app/models /app/reports /app/logs /app/mlruns \
    && chown -R ${APP_USER}:${APP_USER} /app/data /app/models /app/reports /app/logs /app/mlruns

USER ${APP_USER}

ENTRYPOINT ["python", "src/main.py"]
CMD ["--help"]
