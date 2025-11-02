#!/usr/bin/env bash
set -euo pipefail

# Run from the backend folder
cd "$(dirname "$0")/.."

if [ ! -x venv/bin/python ]; then
  echo "[dev] creating virtualenv..."
  python3 -m venv venv
fi

# Upgrade packaging tools
venv/bin/python -m pip install --upgrade pip setuptools wheel

# Install deps
venv/bin/python -m pip install -r requirements.txt

# Export PORT if not set
export PORT="${PORT:-10000}"

echo "[dev] starting app on port ${PORT}"
exec venv/bin/python app.py
