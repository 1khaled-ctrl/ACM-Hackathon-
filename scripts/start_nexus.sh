#!/usr/bin/env bash
# Start GoTrain NEXUS as one background process serving the React UI, the
# REST API and the WebSocket on a single port (default 8501).
#
# Usage:  bash scripts/start_nexus.sh [port]
#
# Builds the frontend first if dist/ is missing, so a fresh clone on the
# SageMaker lab only needs this one command.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PORT="${1:-${NEXUS_PORT:-8501}}"
PID_FILE="logs/nexus.pid"
LOG_FILE="logs/nexus.log"

mkdir -p logs

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "NEXUS already running (pid $(cat "$PID_FILE")). Stop it first: bash scripts/stop_nexus.sh"
  exit 1
fi

# Pick an interpreter: the repo venv if present, else whatever python3 is.
if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
elif [ -x ".venv/Scripts/python.exe" ]; then
  PY=".venv/Scripts/python.exe"
else
  PY="$(command -v python3 || command -v python)"
fi
echo "Using interpreter: $PY"

if [ ! -f "dashboard/frontend/dist/index.html" ]; then
  echo "Frontend not built yet -- building (first run only)..."
  ( cd dashboard/frontend && npm install && npm run build )
fi

echo "Starting GoTrain NEXUS on port $PORT ..."
NEXUS_PORT="$PORT" nohup "$PY" -m dashboard.api.server > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

sleep 3
if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "NEXUS up (pid $(cat "$PID_FILE")) -- logs: $LOG_FILE"
  echo "Local:     http://localhost:$PORT/"
  echo "SageMaker: https://<your-studio-domain>/jupyterlab/default/proxy/$PORT/"
else
  echo "NEXUS failed to start. Last 20 log lines:"
  tail -20 "$LOG_FILE"
  rm -f "$PID_FILE"
  exit 1
fi
