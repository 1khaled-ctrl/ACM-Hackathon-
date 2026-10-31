#!/usr/bin/env bash
# Start GoTrain NEXUS as one background process serving the React UI, the
# REST API and the WebSocket on a single port (default 8501).
#
# Usage:  bash scripts/start_nexus.sh [port]
#
# Rebuilds the frontend on every start whenever the committed frontend
# source has moved past what dist/ was last built from (see the git-hash
# check below) -- not just on a fresh clone. Without this, `git pull`
# alone silently leaves the OLD compiled dist/ running: dist/ is
# gitignored (it's a build artifact, never committed), so pulling updates
# the .jsx source but does nothing to the bundle actually being served
# until something rebuilds it. A stale dist/ looks identical to a
# successful pull -- same URL, same port, server starts fine -- so this
# check exists specifically so "I pulled but nothing changed" can't
# happen silently again.
#
# This is a GIT-HASH check, not a file-mtime check, on purpose: mtimes
# are not a reliable signal here. If dist/ was built from a commit the
# lab already had, then `git pull` bringing in newer commits does NOT
# make the .jsx files' mtimes newer than dist/ -- their content is
# unchanged relative to what was already checked out when that build
# ran. A mtime comparison would (and, in practice, did) conclude "no
# rebuild needed" right after a pull that actually changed the UI.
# Comparing against the commit dist/ was built from is the only signal
# that's correct independent of wall-clock time.

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

# BUILD_MARKER records which commit dist/ was built from, written right
# after a successful build below. Compared against the commit that last
# touched frontend source -- not HEAD overall, so an unrelated backend
# commit landing after a build doesn't trigger a pointless rebuild.
BUILD_MARKER="dashboard/frontend/dist/.build_commit"
SOURCE_COMMIT="$(git log -1 --format=%H -- dashboard/frontend/src dashboard/frontend/package.json dashboard/frontend/vite.config.js dashboard/frontend/index.html 2>/dev/null || echo "")"

NEEDS_BUILD=0
if [ ! -f "dashboard/frontend/dist/index.html" ]; then
  NEEDS_BUILD=1
  echo "Frontend not built yet."
elif [ ! -f "$BUILD_MARKER" ]; then
  NEEDS_BUILD=1
  echo "dist/ exists but has no build marker (built before this check existed) -- rebuilding once to establish one."
elif [ -n "$SOURCE_COMMIT" ] && [ "$(cat "$BUILD_MARKER")" != "$SOURCE_COMMIT" ]; then
  NEEDS_BUILD=1
  echo "Frontend source has moved since the last build ($(cat "$BUILD_MARKER" | cut -c1-8) -> ${SOURCE_COMMIT:0:8})."
fi

if [ "$NEEDS_BUILD" = "1" ]; then
  echo "Building frontend..."
  ( cd dashboard/frontend && npm install && npm run build )
  if [ -n "$SOURCE_COMMIT" ]; then
    echo "$SOURCE_COMMIT" > "$BUILD_MARKER"
  fi
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
