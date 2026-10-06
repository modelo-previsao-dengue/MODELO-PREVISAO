#!/usr/bin/env bash
# Roda um comando com teto de memoria residente de verdade.
#
# Esta instalacao do WSL nao tem .wslconfig, entao a VM pode inflar ate toda a
# RAM do host. Quando um script exagera, o Windows derruba a VM inteira e o
# terminal junto, em vez de o Linux matar so o processo. Um cgroup com
# MemoryMax devolve esse comportamento: o processo morre, a sessao sobrevive.
#
# ulimit -v nao serve aqui: limita espaco virtual, e o XGBoost com 16 threads
# reserva dezenas de GB virtuais usando menos de um residente.
#
# Uso: scripts/rodar_com_teto.sh [-m 9G] -- comando args...
set -euo pipefail

TETO="${TETO_MEMORIA:-9G}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m) TETO="$2"; shift 2 ;;
    --) shift; break ;;
    *) break ;;
  esac
done

exec systemd-run --user --scope --quiet \
  -p MemoryMax="$TETO" -p MemorySwapMax=2G \
  -- "$@"
