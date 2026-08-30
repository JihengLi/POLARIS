#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --python 3.12
uv run python scripts/setup_baselines.py "$@"
