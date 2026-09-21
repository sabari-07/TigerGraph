# Agentic GraphRAG - one-command setup + run (Windows PowerShell).
#
# Usage:
#   ./setup.ps1            # set up, test, and run the benchmark (offline mode)
#
# Runs fully offline by default (GRAPH_BACKEND=local, mock LLM) so it works with
# ZERO credentials. To use the live stack, edit .env afterwards (see README).

$ErrorActionPreference = "Stop"

Write-Host "==> Creating virtual environment (.venv)..." -ForegroundColor Cyan
if (-not (Test-Path ".venv")) { python -m venv .venv }
& .\.venv\Scripts\Activate.ps1

Write-Host "==> Installing dependencies..." -ForegroundColor Cyan
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

if (-not (Test-Path ".env")) {
    Write-Host "==> Creating .env from .env.example (offline defaults)..." -ForegroundColor Cyan
    Copy-Item ".env.example" ".env"
}

$env:PYTHONPATH = "src"
$env:GRAPH_BACKEND = "local"          # offline, credential-free run
$env:EMBEDDING_PROVIDER = "mock"      # instant, no model download
$env:LLM_PROVIDER = "mock"            # deterministic, no API key / no network
$env:PYTHONIOENCODING = "utf-8"

Write-Host "==> Running tests..." -ForegroundColor Cyan
python -m pytest -q

Write-Host "==> Running the 3-pipeline benchmark (offline)..." -ForegroundColor Cyan
python -m agentic_graphrag.cli benchmark

Write-Host ""
Write-Host "Done. Open artifacts\dashboard_public.html for the metrics dashboard." -ForegroundColor Green
Write-Host "Try a single question:" -ForegroundColor Green
Write-Host '  python -m agentic_graphrag.cli ask "Who won gold in the men''s pole vault at the Summer Olympics before 2016?"'
