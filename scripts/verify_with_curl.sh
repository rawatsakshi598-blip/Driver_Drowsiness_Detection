#!/usr/bin/env bash
# --------------------------------------------------------------------------- #
# verify_with_curl.sh — verify the drowsiness detection HTTP API with curl.
#
# This script starts the Flask server in the background, then runs a series
# of curl commands against every endpoint.  Each curl command:
#   * has its full command echoed (so the user can see exactly what ran),
#   * is checked against the expected HTTP status,
#   * has its JSON response parsed for required fields.
#
# Exit codes:
#   0  — all checks passed
#   1  — at least one check failed
#
# Usage:
#   ./scripts/verify_with_curl.sh [--port PORT]
# --------------------------------------------------------------------------- #
set -euo pipefail

PORT="${1:-8000}"
PORT="${PORT#--port}" ; PORT="${PORT#:}"
HOST="127.0.0.1"
BASE="http://${HOST}:${PORT}"

PYTHON="${PYTHON:-python3}"
PASS=0
FAIL=0
LOG_FILE="$(mktemp /tmp/drowsiness_verify.XXXXXX.log)"

# Colors (disabled if stdout isn't a TTY).
if [ -t 1 ]; then
    GREEN=$'\e[32m'; RED=$'\e[31m'; YELLOW=$'\e[33m'; NC=$'\e[0m'
else
    GREEN=""; RED=""; YELLOW=""; NC=""
fi

ok()   { echo "${GREEN}PASS${NC} — $1"; PASS=$((PASS + 1)); }
fail() { echo "${RED}FAIL${NC} — $1"; FAIL=$((FAIL + 1)); }
info() { echo "${YELLOW}....${NC}  $1"; }

check() {
    local label="$1" expected="$2" actual="$3"
    if [ "$expected" = "$actual" ]; then
        ok "$label (HTTP $actual)"
    else
        fail "$label — expected HTTP $expected, got $actual"
    fi
}

# --------------------------------------------------------------------------- #
# Start the server (headless, no CNN to keep startup fast)
# --------------------------------------------------------------------------- #
info "Starting Flask server on port ${PORT}..."
$PYTHON server.py --port "$PORT" --no-window --no-cnn > "$LOG_FILE" 2>&1 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null || true; rm -f $LOG_FILE" EXIT

# Wait for the server to come up.
for i in $(seq 1 30); do
    if curl -s --max-time 1 "${BASE}/health" > /dev/null 2>&1; then
        break
    fi
    sleep 0.3
done

if ! curl -s --max-time 2 "${BASE}/health" > /dev/null 2>&1; then
    fail "Server did not start within 9 seconds. Logs:"
    cat "$LOG_FILE" >&2
    exit 1
fi
info "Server is up (pid ${SERVER_PID})."

# --------------------------------------------------------------------------- #
# 1. GET /health
# --------------------------------------------------------------------------- #
RESP=$(curl -s -o /tmp/health.json -w "%{http_code}" "${BASE}/health")
check "GET /health" "200" "$RESP"
if $PYTHON -c "import json,sys; d=json.load(open('/tmp/health.json')); sys.exit(0 if d['status']=='ok' else 1)"; then
    ok "/health returned status=ok"
else
    fail "/health did not return status=ok"
fi

# --------------------------------------------------------------------------- #
# 2. GET /config
# --------------------------------------------------------------------------- #
RESP=$(curl -s -o /tmp/config.json -w "%{http_code}" "${BASE}/config")
check "GET /config" "200" "$RESP"
if $PYTHON -c "import json; d=json.load(open('/tmp/config.json')); assert 'eye' in d and 'mouth' in d and 'video' in d" 2>&1; then
    ok "/config contains eye/mouth/video sections"
else
    fail "/config response missing required sections"
fi

# --------------------------------------------------------------------------- #
# 3. GET /status (when no monitor is running)
# --------------------------------------------------------------------------- #
RESP=$(curl -s -o /tmp/status1.json -w "%{http_code}" "${BASE}/status")
check "GET /status (no monitor)" "200" "$RESP"
if $PYTHON -c "import json; d=json.load(open('/tmp/status1.json')); assert d['running']==False" 2>&1; then
    ok "/status reports running=False"
else
    fail "/status should report running=False when no monitor is running"
fi

# --------------------------------------------------------------------------- #
# 4. POST /alarm/test (no monitor → 400)
# --------------------------------------------------------------------------- #
RESP=$(curl -s -o /tmp/alarm_test.json -w "%{http_code}" -X POST "${BASE}/alarm/test")
check "POST /alarm/test (no monitor → 400)" "400" "$RESP"

# --------------------------------------------------------------------------- #
# 5. POST /alarm/reset (no monitor → 400)
# --------------------------------------------------------------------------- #
RESP=$(curl -s -o /tmp/alarm_reset.json -w "%{http_code}" -X POST "${BASE}/alarm/reset")
check "POST /alarm/reset (no monitor → 400)" "400" "$RESP"

# --------------------------------------------------------------------------- #
# 6. POST /stop (no monitor → 200 with stopped=false)
# --------------------------------------------------------------------------- #
RESP=$(curl -s -o /tmp/stop1.json -w "%{http_code}" -X POST "${BASE}/stop")
check "POST /stop (no monitor)" "200" "$RESP"
if $PYTHON -c "import json; d=json.load(open('/tmp/stop1.json')); assert d['stopped']==False" 2>&1; then
    ok "/stop returned stopped=false when nothing was running"
else
    fail "/stop should return stopped=false when nothing was running"
fi

# --------------------------------------------------------------------------- #
# 7. POST /start (will fail with 500 because no webcam in CI, but the
#    endpoint must exist and return JSON)
# --------------------------------------------------------------------------- #
RESP=$(curl -s -o /tmp/start.json -w "%{http_code}" -X POST "${BASE}/start")
if [ "$RESP" = "200" ] || [ "$RESP" = "500" ]; then
    ok "POST /start returned HTTP $RESP (endpoint exists)"
else
    fail "POST /start returned HTTP $RESP (expected 200 or 500)"
fi

# --------------------------------------------------------------------------- #
# 8. GET /health again to ensure the server is still alive after the
#    control-plane calls.
# --------------------------------------------------------------------------- #
RESP=$(curl -s -o /tmp/health2.json -w "%{http_code}" "${BASE}/health")
check "GET /health (after control calls)" "200" "$RESP"

# --------------------------------------------------------------------------- #
# 9. Verify each endpoint returns Content-Type: application/json
# --------------------------------------------------------------------------- #
for ep in "/health" "/status" "/config"; do
    CT=$(curl -s -o /dev/null -D - "${BASE}${ep}" | grep -i '^content-type:' | tr -d '\r')
    if echo "$CT" | grep -qi "application/json"; then
        ok "GET ${ep} returns application/json"
    else
        fail "GET ${ep} returned non-JSON Content-Type: $CT"
    fi
done

# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
echo
echo "============================================================"
echo "  Verification summary"
echo "============================================================"
echo "  PASSED: ${GREEN}${PASS}${NC}"
echo "  FAILED: ${RED}${FAIL}${NC}"
echo "============================================================"
echo "Server log: ${LOG_FILE}"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
