FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

EXPOSE 8080

WORKDIR /app

ENV UV_LINK_MODE=copy

COPY uv.lock pyproject.toml ./

RUN uv sync --frozen --no-install-project

COPY README.md README.md
COPY src src/
COPY configs configs/
COPY artifacts artifacts/

RUN uv sync --frozen --no-install-project

CMD uv run uvicorn src.artsy.api:app --host 0.0.0.0 --port ${PORT}
