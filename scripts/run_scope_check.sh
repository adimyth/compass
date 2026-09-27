#!/usr/bin/env bash
set -euo pipefail

# Start the local Compass Scope Check dashboard. Pass the Compass model port as
# the first argument, for example: scripts/run_scope_check.sh 9000
compass_port=8000
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  compass_port="$1"
  shift
fi

exec uv run python -m compass.scope_check --compass-endpoint "http://127.0.0.1:${compass_port}/v1/systemone" "$@"
