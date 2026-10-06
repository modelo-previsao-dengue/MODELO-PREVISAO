#!/usr/bin/env bash
# Roda a pipeline v3 mesorregional inteira, em ordem, parando no primeiro erro.
# Uso: scripts/rodar_v3.sh [primeiro_script]   (ex.: scripts/rodar_v3.sh 27)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
INICIO="${1:-20}"
for s in scripts/2[0-9]_*.py scripts/3[0-3]_*.py; do
  n=$(basename "$s" | cut -d_ -f1)
  [[ "$n" < "$INICIO" ]] && continue
  echo; echo "########## $s  ($(date +%H:%M:%S))"
  "$PY" "$s"
done
echo; echo "########## fim ($(date +%H:%M:%S))"
