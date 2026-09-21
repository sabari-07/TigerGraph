#!/usr/bin/env bash
# Agentic GraphRAG - one-command setup + run (macOS / Linux).
#
# Usage:
#   ./setup.sh             # set up, test, and run the benchmark (offline mode)
#
# Runs fully offline by default (GRAPH_BACKEND=local, mock LLM) so it works with
# ZERO credentials. To use the live stack, edit .env afterwards (see README).

set -euo pipefail

echo "==> Creating virtual environment (.venv)..."
[ -d .venv ] || python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Installing dependencies..."
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

if [ ! -f .env ]; then
  echo "==> Creating .env from .env.example (offline defaults)..."
  cp .env.example .env
fi

export PYTHONPATH=src
export GRAPH_BACKEND=local          # offline, credential-free run
export EMBEDDING_PROVIDER=mock      # instant, no model download
export LLM_PROVIDER=mock            # deterministic, no API key / no network
export PYTHONIOENCODING=utf-8

echo "==> Running tests..."
python -m pytest -q

echo "==> Running the 3-pipeline benchmark (offline)..."
python -m agentic_graphrag.cli benchmark

echo ""
echo "Done. Open artifacts/dashboard_public.html for the metrics dashboard."
echo "Try a single question:"
echo "  python -m agentic_graphrag.cli ask \"Who won gold in the men's pole vault at the Summer Olympics before 2016?\""
