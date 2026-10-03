#!/usr/bin/env bash
set -uo pipefail

readonly EXIT_OK=0
readonly EXIT_FASTAPI_DOWN=1
readonly EXIT_REDIS_DOWN=2
readonly EXIT_POSTGRES_DOWN=3
readonly EXIT_CHROMA_DOWN=4
readonly EXIT_CAPABILITIES_DOWN=5

PORT="${PORT:-8000}"
REDIS_HOST="${REDIS_HOST:-redis}"
REDIS_PORT="${REDIS_PORT:-6379}"
POSTGRES_HOST="${POSTGRES_HOST:-postgres}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_USER="${POSTGRES_USER:-ai_os}"
POSTGRES_DB="${POSTGRES_DB:-ai_os}"
CHROMA_HOST="${CHROMA_HOST:-chroma}"
CHROMA_PORT="${CHROMA_PORT:-8000}"

check_fastapi() {
    curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1
}

check_redis() {
    redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" ping 2>/dev/null | grep -q PONG
}

check_postgres() {
    pg_isready -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -q
}

check_chroma() {
    curl -fsS --max-time 5 "http://${CHROMA_HOST}:${CHROMA_PORT}/api/v1/heartbeat" >/dev/null 2>&1
}

check_capabilities() {
    curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1
}

if ! check_fastapi; then
    echo "unhealthy: fastapi is not responding"
    exit "${EXIT_FASTAPI_DOWN}"
fi

if ! check_redis; then
    echo "unhealthy: redis is not responding"
    exit "${EXIT_REDIS_DOWN}"
fi

if ! check_postgres; then
    echo "unhealthy: postgresql is not responding"
    exit "${EXIT_POSTGRES_DOWN}"
fi

if ! check_chroma; then
    echo "unhealthy: chroma is not responding"
    exit "${EXIT_CHROMA_DOWN}"
fi

if ! check_capabilities; then
    echo "unhealthy: capability registry health check failed"
    exit "${EXIT_CAPABILITIES_DOWN}"
fi

echo "healthy"
exit "${EXIT_OK}"
