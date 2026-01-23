FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

EXPOSE 8080

WORKDIR /

COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
COPY README.md README.md
COPY src src/
COPY data/api data/api/
COPY configs configs/
COPY outputs/2026-01-21/14-51-58/models outputs/2026-01-21/14-51-58/models/
COPY data/processed/styles.txt data/processed/styles.txt

RUN uv sync --frozen --no-dev && rm -rf /root/.cache/pypoetry /root/.cache/pip

CMD uv run uvicorn src.artsy.api:app --host 0.0.0.0 --port ${PORT}
