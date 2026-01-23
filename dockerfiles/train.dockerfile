# Base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV UV_LINK_MODE=copy

COPY uv.lock pyproject.toml ./

RUN uv sync --frozen --no-install-project

COPY src src/
COPY configs configs/
COPY README.md LICENSE ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

ENTRYPOINT ["uv", "run", "src/artsy/train.py"]
