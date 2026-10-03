# syntax=docker/dockerfile:1.7

############################################
# Stage 1: build Python dependencies
############################################
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip setuptools wheel \
    && /opt/venv/bin/pip install -r requirements.txt

############################################
# Stage 2: production runtime image
############################################
FROM python:3.12-slim AS runtime

LABEL maintainer="ai-operating-system" \
      org.opencontainers.image.title="ai-os-backend" \
      org.opencontainers.image.description="FastAPI backend for the AI Operating System"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV="/opt/venv" \
    APP_HOME=/app \
    PLAYWRIGHT_BROWSERS_PATH=/opt/playwright \
    HOST=0.0.0.0 \
    PORT=8000 \
    LOG_LEVEL=info \
    UVICORN_WORKERS=4

WORKDIR ${APP_HOME}

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
        ffmpeg \
        tesseract-ocr \
        libpq5 \
        libnss3 \
        libnspr4 \
        libatk1.0-0 \
        libatk-bridge2.0-0 \
        libcups2 \
        libdrm2 \
        libdbus-1-3 \
        libxkbcommon0 \
        libxcomposite1 \
        libxdamage1 \
        libxfixes3 \
        libxrandr2 \
        libgbm1 \
        libasound2 \
        libpango-1.0-0 \
        libcairo2 \
        libxshmfence1 \
        fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser \
    && mkdir -p ${PLAYWRIGHT_BROWSERS_PATH} ${APP_HOME}/data \
    && chown -R appuser:appuser ${APP_HOME} ${PLAYWRIGHT_BROWSERS_PATH}

COPY --chown=appuser:appuser backend/app/ ${APP_HOME}/app/
COPY --chown=appuser:appuser requirements.txt ${APP_HOME}/requirements.txt
COPY --chown=appuser:appuser entrypoint.sh ${APP_HOME}/entrypoint.sh
COPY --chown=appuser:appuser healthcheck.sh ${APP_HOME}/healthcheck.sh

RUN chmod +x ${APP_HOME}/entrypoint.sh ${APP_HOME}/healthcheck.sh

USER appuser

RUN python -m playwright install chromium

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD ["/app/healthcheck.sh"]

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "backend.app.integration.app:app", "--host", "0.0.0.0", "--port", "8000"]
