#!/usr/bin/env bash
set -euo pipefail

log() {
    printf '[entrypoint] %s\n' "$1"
}

wait_for_postgres() {
    local host="${POSTGRES_HOST:-postgres}"
    local port="${POSTGRES_PORT:-5432}"
    local retries=0
    local max_retries="${WAIT_MAX_RETRIES:-60}"

    log "waiting for postgres at ${host}:${port}"
    until pg_isready -h "${host}" -p "${port}" -q >/dev/null 2>&1; do
        retries=$((retries + 1))
        if [ "${retries}" -ge "${max_retries}" ]; then
            log "postgres did not become ready in time"
            exit 1
        fi
        sleep 1
    done
    log "postgres is ready"
}

wait_for_redis() {
    local host="${REDIS_HOST:-redis}"
    local port="${REDIS_PORT:-6379}"
    local retries=0
    local max_retries="${WAIT_MAX_RETRIES:-60}"

    log "waiting for redis at ${host}:${port}"
    until redis-cli -h "${host}" -p "${port}" ping >/dev/null 2>&1; do
        retries=$((retries + 1))
        if [ "${retries}" -ge "${max_retries}" ]; then
            log "redis did not become ready in time"
            exit 1
        fi
        sleep 1
    done
    log "redis is ready"
}

run_migrations() {
    log "running database migrations"
    python -m alembic upgrade head
}

initialize_vector_db() {
    log "initializing vector database"
    python -m backend.app.scripts.init_vector_db
}

warmup_prompts() {
    log "warming up prompt registry"
    python -m backend.app.scripts.warmup_prompts
}

warmup_capabilities() {
    log "warming up capability registry"
    python -m backend.app.scripts.warmup_capabilities
}

start_server() {
    log "starting uvicorn"
    uvicorn backend.app.integration.app:app \
        --host "${HOST:-0.0.0.0}" \
        --port "${PORT:-8000}" \
        --workers "${UVICORN_WORKERS:-4}" \
        --log-level "${LOG_LEVEL:-info}" \
        --proxy-headers \
        --forwarded-allow-ips "*" &
    UVICORN_PID=$!
}

wait_for_app_health() {
    local port="${PORT:-8000}"
    local retries=0
    local max_retries="${WAIT_MAX_RETRIES:-30}"

    log "waiting for application health check on port ${port}"
    until curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; do
        if ! kill -0 "${UVICORN_PID}" 2>/dev/null; then
            log "uvicorn process exited before becoming healthy"
            exit 1
        fi
        retries=$((retries + 1))
        if [ "${retries}" -ge "${max_retries}" ]; then
            log "application did not become healthy in time"
            exit 1
        fi
        sleep 1
    done
    log "application is healthy"
}

shutdown() {
    log "received termination signal, shutting down uvicorn"
    if [ -n "${UVICORN_PID:-}" ]; then
        kill -TERM "${UVICORN_PID}" 2>/dev/null || true
        wait "${UVICORN_PID}" 2>/dev/null || true
    fi
    exit 0
}

trap shutdown TERM INT

main() {
    wait_for_postgres
    wait_for_redis
    run_migrations
    initialize_vector_db
    warmup_prompts
    warmup_capabilities
    start_server
    wait_for_app_health
    wait "${UVICORN_PID}"
}

main "$@"
