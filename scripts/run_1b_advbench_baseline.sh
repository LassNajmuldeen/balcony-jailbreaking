#!/usr/bin/env bash
# run_1b_advbench_baseline.sh -- the one GCG cell that was never run.
#
# run_matrix.sh's PRIORITY 1 loop is `for m in 7b 8b`, so the 1B's GCG stages were
# never in it -- the 1B was attacked by hand on 2026-08-03 and the AdvBench
# no-balcony baseline was missed. Every other model/benchmark/arm combination has
# one. Without it the 1B's AdvBench column has nothing to compare against.
#
# Stage 2 re-judges the whole `results/` directory. judge_harmfulness.py globs
# *attacks*.json and scores stored completions -- it never loads the attacked
# model and never regenerates text, so this is cheap and cannot corrupt anything.
#
#   ./run_1b_advbench_baseline.sh --status
#
set -uo pipefail

REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
PYTHON="${PYTHON:-$REPO/AdversariaLLM/.venv/bin/python}"
STATE_DIR="$REPO/run_state"
LOG_ROOT="$REPO/run_logs"
mkdir -p "$STATE_DIR" "$LOG_ROOT"

# Copied verbatim from run_matrix.sh's 1b entries.
ENV_1B=(
  "BALCONY_MODEL=parsakaveh/Balcony-LLM-1B"
  "BASELINE_MODEL=parsakaveh/LLM-1B-Pretrained"
  "EXITS=4 8 12 full"
  "CHAT_TEMPLATE="
  "RESULTS_DIR=results"
  "N_BEHAVIORS=50"
  "NUM_STEPS=150"
)

STAGES=(
  "1b_gcg_advbench_baseline:experiments/attack_gcg_baseline.py:BENCHMARK=advbench:ATTACK_TARGET=baseline"
  "1b_judge_advbench:experiments/judge_harmfulness.py:BENCHMARK=advbench:ATTACK_TARGET=baseline"
)

if [ "${1:-}" = "--status" ]; then
  printf '%-30s %s\n' "STAGE" "STATE"
  for spec in "${STAGES[@]}"; do
    n="${spec%%:*}"
    if   [ -f "$STATE_DIR/$n.done" ];   then s=done
    elif [ -f "$STATE_DIR/$n.failed" ]; then s="failed(rc=$(cat "$STATE_DIR/$n.failed"))"
    else s=pending; fi
    printf '%-30s %s\n' "$n" "$s"
  done
  exit 0
fi

log() { echo "[$(date -Is)] $*"; }

for spec in "${STAGES[@]}"; do
  IFS=':' read -r name script e1 e2 <<< "$spec"

  if [ -f "$STATE_DIR/$name.done" ]; then
    log "SKIP $name (already completed)"
    continue
  fi

  log "=== START $name ($script)"
  rm -f "$STATE_DIR/$name.failed"
  t0=$(date +%s)

  env "${ENV_1B[@]}" "$e1" "$e2" \
      "$PYTHON" "$REPO/$script" > "$LOG_ROOT/$name.log" 2>&1
  rc=$?
  dt=$(( $(date +%s) - t0 ))

  if [ $rc -eq 0 ]; then
    touch "$STATE_DIR/$name.done"
    log "=== OK $name (${dt}s)"
  else
    echo "$rc" > "$STATE_DIR/$name.failed"
    log "=== FAILED $name (rc=$rc, ${dt}s)"
    # the judge is worthless if the attack did not produce a file
    [ "$name" = "1b_gcg_advbench_baseline" ] && { log "attack failed -- not judging"; exit 1; }
  fi
done

log "done -- results/gcg_no_balcony_advbench_attacks.json + refreshed judge_scores.json"
