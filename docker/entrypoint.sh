#!/usr/bin/env bash
set -euo pipefail

SERVICE="${SERVICE:-gateway}"
PORT="${PORT:-8000}"

echo "Starting BioForge service '${SERVICE}' on port ${PORT}"
exec python -m bioforge.cli serve "${SERVICE}" --port "${PORT}"
