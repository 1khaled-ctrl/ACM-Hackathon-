#!/usr/bin/env bash
# Stop the GoTrain NEXUS server started by scripts/start_nexus.sh.
#
# This stops the DASHBOARD only. It deliberately does not touch a running
# GoTrain agent -- pause that from the UI (or POST /api/agent/stop) so the
# control plane records it, rather than killing training from a shell.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PID_FILE="logs/nexus.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "No $PID_FILE -- NEXUS does not appear to be running."
  exit 0
fi

PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  sleep 2
  if kill -0 "$PID" 2>/dev/null; then
    echo "Process $PID ignored SIGTERM; sending SIGKILL."
    kill -9 "$PID" 2>/dev/null || true
  fi
  echo "NEXUS stopped (pid $PID)."
else
  echo "Stale pid file (process $PID not running)."
fi

rm -f "$PID_FILE"
