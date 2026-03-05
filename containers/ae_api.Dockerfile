# Base image and global ARGs
ARG PYTHON_VERSION=3.10
ARG POETRY_VERSION=2.1.2
ARG VIRTUAL_ENV=/app/.venv

# Builder stage
FROM python:${PYTHON_VERSION}-slim AS builder

# Use global ARGs
ARG POETRY_VERSION
ARG VIRTUAL_ENV

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Europe/Warsaw \
    POETRY_HOME=/opt/poetry \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/opt/.cache \
    POETRY_VIRTUALENVS_PATH=${VIRTUAL_ENV}

# Install Poetry
RUN pip install "poetry==${POETRY_VERSION}"

# Set working directory
WORKDIR /app

# Copy project dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN --mount=type=cache,target=${POETRY_CACHE_DIR} \
    poetry install --only main

# Runtime stage
FROM python:${PYTHON_VERSION}-slim AS runtime

# Use global ARGs
ARG VIRTUAL_ENV

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Europe/Warsaw \
    PYTHONPATH="/app" \
    VIRTUAL_ENV=${VIRTUAL_ENV} \
    PATH="$VIRTUAL_ENV/bin:$PATH"

# Set working directory
WORKDIR /app

# Create non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Install dumb-init and any other runtime dependencies
RUN --mount=type=cache,target=/var/cache/apt\
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends dumb-init && \
    rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# Copy application code
COPY --chown=appuser:appgroup pyproject.toml /app/pyproject.toml
COPY --chown=appuser:appgroup src/v2 /app/src/v2

# Switch to non-root user
USER appuser

# Set entrypoint to use dumb-init
ENTRYPOINT ["/usr/bin/dumb-init", "--"]

# Set default command
CMD ["python", "src/v2/api/main.py"]
