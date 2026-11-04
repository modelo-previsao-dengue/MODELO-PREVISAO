#!/usr/bin/env bash
# Roda a pipeline v4 (série completa) inteira, em ordem, parando no primeiro erro.
# Uso: scripts/rodar_v4.sh [primeiro_script]   (ex.: scripts/rodar_v4.sh 46)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
INICIO="${1:-40}"
for s in scripts/4[0-9]_*.py scripts/5[0-5]_*.py; do
  n=$(basename "$s" | cut -d_ -f1)
  [[ "$n" < "$INICIO" ]] && continue
  echo; echo "########## $s  ($(date +%H:%M:%S))"
  "$PY" "$s"
done
echo; echo "########## fim ($(date +%H:%M:%S))"
