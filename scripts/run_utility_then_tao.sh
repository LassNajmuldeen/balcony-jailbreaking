#!/usr/bin/env bash
# run_utility_then_tao.sh -- unattended re-prioritisation of the experiment queue.
#
# WHY THIS EXISTS: run_matrix.sh only ever defined attack + judge stages, so the
# per-exit utility evals (which the 1B has and which are what make an ASR number
# interpretable at all) were never going to run. TAO was scheduled ahead of them
# and takes ~9 days. This script puts utility first without editing run_matrix.sh
# while it is running (bash reads a script incrementally -- editing a running one
# can corrupt the queue).
#
# WHAT IT DOES, in order:
#   1. waits for the GCG phase to finish (run_state/8b_judge_gcg.done)
#   2. kills the matrix process group so TAO stops competing for the GPU
#   3. runs all six utility stages, most decision-relevant first
#   4. relaunches run_matrix.sh detached, which resumes with TAO
#
# Safe to re-run: every stage writes run_state/<name>.done and is skipped if
# that marker exists, exactly like run_matrix.sh. Killing a TAO stage mid-flight
# is cheap -- TAO checkpoints per behaviour and resumes.
#
#   ./run_utility_then_tao.sh --status     # per-stage state, no side effects
#   ./run_utility_then_tao.sh --no-tao     # utility only, do NOT relaunch the matrix
#
set -uo pipefail

REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
PYTHON="${PYTHON:-$REPO/AdversariaLLM/.venv/bin/python}"
STATE_DIR="$REPO/run_state"
LOG_ROOT="$REPO/run_logs"
GATE="$STATE_DIR/8b_judge_gcg.done"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-28800}"   # 8h ceiling so a hang cannot block forever

mkdir -p "$STATE_DIR" "$LOG_ROOT"

# model_key : balcony | baseline | exits | chat_template | results_dir
# Values copied verbatim from run_matrix.sh's BALCONY/BASELINE/MODEL_EXITS/TEMPLATE/OUTDIR.
declare -A BALCONY BASELINE MODEL_EXITS TEMPLATE OUTDIR
BALCONY[1b]="parsakaveh/Balcony-LLM-1B";                     BASELINE[1b]="parsakaveh/LLM-1B-Pretrained"
BALCONY[7b]="parsakaveh/Balcony-LLaMA2-7B";                  BASELINE[7b]="NousResearch/Llama-2-7b-hf"
BALCONY[8b]="adpretko/Balcony-Meta-LLaMA3-8B-exit-15-18-21"; BASELINE[8b]="NousResearch/Meta-Llama-3-8B"
MODEL_EXITS[1b]="4 8 12 full"; MODEL_EXITS[7b]="15 18 21 full"; MODEL_EXITS[8b]="15 18 21 full"
TEMPLATE[1b]="";               TEMPLATE[7b]="llama-2-chat";    TEMPLATE[8b]="llama-3-balcony"
OUTDIR[1b]="results";          OUTDIR[7b]="results_7b";        OUTDIR[8b]="results_8b"

# Ordered most-decision-relevant first: if this dies partway, the results that
# settle whether the 7B/8B have a story are the ones already on disk. 1B is last
# because it already has utility_eval.json from 2026-08-03; this only re-runs it
# so all three models are measured under the same lm_eval 0.4.12.
STAGES=(
  "7b_utility_clean:7b:experiments/eval_capabilities.py"
  "8b_utility_clean:8b:experiments/eval_capabilities.py"
  "7b_utility_attacked:7b:experiments/eval_capabilities_under_attack.py"
  "8b_utility_attacked:8b:experiments/eval_capabilities_under_attack.py"
  "1b_utility_clean:1b:experiments/eval_capabilities.py"
  "1b_utility_attacked:1b:experiments/eval_capabilities_under_attack.py"
)

stage_state() {
  if   [ -f "$STATE_DIR/$1.done" ];   then echo done
  elif [ -f "$STATE_DIR/$1.failed" ]; then echo "failed(rc=$(cat "$STATE_DIR/$1.failed"))"
  else echo pending; fi
}

if [ "${1:-}" = "--status" ]; then
  printf '%-34s %s\n' "STAGE" "STATE"
  for spec in "${STAGES[@]}"; do
    printf '%-34s %s\n' "${spec%%:*}" "$(stage_state "${spec%%:*}")"
  done
  exit 0
fi

NO_TAO=0
[ "${1:-}" = "--no-tao" ] && NO_TAO=1

log() { echo "[$(date -Is)] $*"; }

# --- 1. wait for the GCG phase --------------------------------------------
log "waiting for $GATE (ceiling ${MAX_WAIT_SECONDS}s)"
waited=0
while [ ! -f "$GATE" ]; do
  if [ -f "$STATE_DIR/8b_judge_gcg.failed" ]; then
    log "8b_judge_gcg FAILED -- proceeding to utility anyway (its completions are on disk and can be re-judged)"
    break
  fi
  # Anchored at end of cmdline on purpose: a bare `pgrep -f run_matrix.sh` also
  # matches shell wrappers that merely mention the name, and killing that process
  # group would take out something unrelated.
  if ! pgrep -f 'run_matrix\.sh$' >/dev/null; then
    log "run_matrix.sh is no longer running and the gate never appeared -- proceeding to utility"
    break
  fi
  if [ "$waited" -ge "$MAX_WAIT_SECONDS" ]; then
    log "hit the ${MAX_WAIT_SECONDS}s ceiling -- proceeding to utility"
    break
  fi
  sleep 60; waited=$((waited + 60))
done
log "gate cleared after ${waited}s"

# --- 2. stop the matrix so TAO is not competing for the GPU ---------------
MPID="$(pgrep -f 'run_matrix\.sh$' | head -1 || true)"
if [ -n "$MPID" ]; then
  PGID="$(ps -o pgid= -p "$MPID" | tr -d ' ')"
  log "stopping run_matrix.sh (pid $MPID, pgid $PGID) and any stage it is running"
  kill -TERM "-$PGID" 2>/dev/null || true
  sleep 10
  kill -KILL "-$PGID" 2>/dev/null || true
  sleep 5
else
  log "run_matrix.sh not running -- nothing to stop"
fi

# --- 3. utility stages ----------------------------------------------------
for spec in "${STAGES[@]}"; do
  name="${spec%%:*}"; rest="${spec#*:}"; m="${rest%%:*}"; script="${rest#*:}"

  if [ -f "$STATE_DIR/$name.done" ]; then
    log "SKIP $name (already completed)"
    continue
  fi

  log "=== START $name  ($script, $m)"
  rm -f "$STATE_DIR/$name.failed"
  t0=$(date +%s)

  env "BALCONY_MODEL=${BALCONY[$m]}" \
      "BASELINE_MODEL=${BASELINE[$m]}" \
      "EXITS=${MODEL_EXITS[$m]}" \
      "CHAT_TEMPLATE=${TEMPLATE[$m]}" \
      "RESULTS_DIR=${OUTDIR[$m]}" \
      "N_BEHAVIORS=50" \
      "NUM_STEPS=150" \
      "$PYTHON" "$REPO/$script" > "$LOG_ROOT/$name.log" 2>&1
  rc=$?
  dt=$(( $(date +%s) - t0 ))

  if [ $rc -eq 0 ]; then
    touch "$STATE_DIR/$name.done"
    log "=== OK $name (${dt}s)"
  else
    echo "$rc" > "$STATE_DIR/$name.failed"
    log "=== FAILED $name (rc=$rc, ${dt}s) -- continuing to the next stage"
  fi
done

# --- 4. hand the GPU back to the matrix (TAO) -----------------------------
if [ "$NO_TAO" -eq 1 ]; then
  log "utility phase finished -- --no-tao given, leaving the GPU idle"
  exit 0
fi

log "utility phase finished -- relaunching run_matrix.sh for TAO"
cd "$REPO" || exit 1
setsid nohup ./run_matrix.sh >> "$LOG_ROOT/matrix_restart.log" 2>&1 &
log "run_matrix.sh relaunched (pid $!)"
