#!/usr/bin/env bash
set -euo pipefail

# Puerto configurable (default 10000)
PORT="${PORT:-10000}"

echo "Verificando si el puerto ${PORT} está en uso..."
PIDS=$(lsof -t -i:"${PORT}" || true)

if [ -z "${PIDS}" ]; then
  echo "El puerto ${PORT} no está en uso."
else
  echo "Se encontró el/los PID: ${PIDS} usando el puerto ${PORT}. Finalizando procesos..."
  for PID in ${PIDS}; do
    kill -9 "${PID}" || true
    echo "Proceso ${PID} finalizado."
  done
fi

cd "$(dirname "$0")"

if [ ! -x venv/bin/python ]; then
  echo "Creando virtualenv local..."
  python3 -m venv venv
fi

venv/bin/python -m pip install --upgrade pip setuptools wheel
venv/bin/python -m pip install -r requirements.txt

echo "Iniciando app.py en puerto ${PORT}..."
exec env PORT="${PORT}" venv/bin/python app.py