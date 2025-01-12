#!/bin/bash

# Paso 1: Verificar qué PID está usando el puerto 5000
echo "Verificando si el puerto 5000 está en uso..."
PIDS=$(lsof -t -i:5000)

if [ -z "$PIDS" ]; then
  echo "El puerto 5000 no está en uso."
else
  echo "Se encontró el/los PID: $PIDS usando el puerto 5000. Finalizando procesos..."
  
  # Paso 2: Matar los procesos que están usando el puerto 5000
  for PID in $PIDS; do
    kill -9 $PID
    echo "Proceso $PID finalizado."
  done
fi

# Paso 3: Correr app.py
echo "Iniciando app.py..."
python app.py