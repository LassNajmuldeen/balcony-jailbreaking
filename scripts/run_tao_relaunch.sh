#!/usr/bin/env bash
# Finish the TAO run and everything downstream of it, unattended.
#
# Written 2026-08-09 after six HarmBench shards died of CUDA OOM caused by
# NEED_MB[7b]=30000 against 7B workers that actually peak at 41.16 GiB. That
# reservation is now fixed in run_tao_parallel.sh; this script drives the rest:
#
#   phase 1  wait for the in-flight run_tao_parallel.sh to exit
#   phase 2  re-run every shard that is not .done (this includes the six failed)
#   phase 3  merge shards into canonical files -- guarded, incomplete groups skip
#   phase 4  judge the newly published canonical TAO completions
#   phase 5  utility benchmarks under TAO suffixes, for models with complete TAO
#
# Why a separate script rather than edits to run_tao_parallel.sh: bash reads a
# script incrementally, so editing the live one can corrupt its queue. Same reason
# run_utility_then_tao.sh exists. This one only ever *invokes* the runner.
#
#   ./run_tao_relaunch.sh --status        # what each phase would do; safe any time
#   ./run_tao_relaunch.sh --dry-run       # print the exact commands, run nothing
#   ./run_tao_relaunch.sh --failed-only   # phase 2 runs ONLY the 6 failed shards
#   SKIP_WAIT=1 ./run_tao_relaunch.sh     # do not wait for a runner (it already exited)
#
# Detach it so it outlives the shell:
#   setsid nohup ./run_tao_relaunch.sh > run_logs/tao_relaunch_console.log 2>&1 < /dev/null &
#
# Resume-safe: every phase writes run_state/<name>.done and is skipped on re-run.

set -uo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."

REPO="$(pwd)"
PY="$REPO/AdversariaLLM/.venv/bin/python"
STATE="$REPO/run_state"
LOGS="$REPO/run_logs"
mkdir -p "$STATE" "$LOGS"

N_BEHAVIORS="${N_BEHAVIORS:-50}"
NUM_STEPS="${NUM_STEPS:-150}"

# Phase 2 concurrency. TWO, not three. The six outstanding shards are all 7B or
# 8B, and with honest reservations (44 GB each) only two fit in the 94.97 GiB
# card -- 44+44 = 88 GB leaves ~9 GB of headroom, while any third worker
# overcommits. run_tao_parallel.sh's admission gate would enforce this anyway;
# setting it here just stops the pool spinning on a worker it can never admit.
TAO_WORKERS_RELAUNCH="${TAO_WORKERS_RELAUNCH:-2}"

# Phase 5 concurrency. Utility stages are far smaller than TAO shards: lm_eval
# runs at a FIXED batch of 8 (simple_evaluate ignores batch_size="auto" when it
# is handed an instantiated LM, which this code does), so the footprint is
# weights + one small activation window rather than a 128-candidate sweep. All
# three models together reserve 72 GB and fit on the card at once.
UTIL_WORKERS="${UTIL_WORKERS:-3}"

WAIT_CEILING_H="${WAIT_CEILING_H:-96}"
SKIP_WAIT="${SKIP_WAIT:-0}"

# Kept in sync BY HAND with run_tao_parallel.sh and run_matrix.sh. If a checkpoint
# or a template changes there, change it here too.
declare -A BALCONY BASELINE MODEL_EXITS TEMPLATE OUTDIR UTIL_NEED_MB
BALCONY[1b]="parsakaveh/Balcony-LLM-1B";                     BASELINE[1b]="parsakaveh/LLM-1B-Pretrained"
BALCONY[7b]="parsakaveh/Balcony-LLaMA2-7B";                  BASELINE[7b]="NousResearch/Llama-2-7b-hf"
BALCONY[8b]="adpretko/Balcony-Meta-LLaMA3-8B-exit-15-18-21"; BASELINE[8b]="NousResearch/Meta-Llama-3-8B"
MODEL_EXITS[1b]="4 8 12 full";  MODEL_EXITS[7b]="15 18 21 full";  MODEL_EXITS[8b]="15 18 21 full"
TEMPLATE[1b]="";                TEMPLATE[7b]="llama-2-chat";      TEMPLATE[8b]="llama-3-balcony"
OUTDIR[1b]="results";           OUTDIR[7b]="results_7b";          OUTDIR[8b]="results_8b"
# weights + lm_eval at batch 8. The 8B is the outlier because its 128256-token
# vocab makes the per-batch logit tensor ~4x the 7B's 32000-token one.
UTIL_NEED_MB[1b]=12000;         UTIL_NEED_MB[7b]=26000;           UTIL_NEED_MB[8b]=34000

MODELS=(8b 7b 1b)

DRY_RUN=0
FAILED_ONLY=0
for a in "$@"; do
  case "$a" in
    --dry-run)     DRY_RUN=1 ;;
    --failed-only) FAILED_ONLY=1 ;;
  esac
done

say()  { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
run()  { if (( DRY_RUN )); then echo "    would run: $*"; else "$@"; fi; }

# The canonical TAO files a model needs before it can be judged or have its
# utility measured under TAO suffixes. Deliberately HarmBench-only: the existing
# GCG utility run used phase2_attacks.json, i.e. HarmBench suffixes, and the two
# have to be measured the same way to be comparable.
tao_canon_balcony() { echo "${OUTDIR[$1]}/tao_attacks.json"; }
tao_canon_baseline() { echo "${OUTDIR[$1]}/tao_no_balcony_harmbench_attacks.json"; }

# Only the BALCONY file is required. The baseline arm was dropped 2026-08-09 for
# the shards that had not yet run: the balcony trunk is numerically identical to
# the baseline checkpoint in all three models (291/291 tensors, max |delta| = 0
# after the bf16 cast the loader performs), so `no_balcony` is the same model as
# the `full` exit, not an independent control. Requiring it here would have
# silently skipped judging and utility for every model whose baseline was dropped.
tao_complete() {  # tao_complete <model> -> 0 if the HarmBench balcony file exists
  [[ -f "$(tao_canon_balcony "$1")" ]]
}

# --- other runners on the card --------------------------------------------
# Filtered against our own PID and against shell wrappers that merely mention the
# name: a bare match on this pattern also catches the Claude Code bash wrapper,
# which is how a previous session killed its own calling shell (exit 144).
runner_pids() {
  pgrep -af 'run_tao_parallel\.sh$' 2>/dev/null \
    | grep -v snapshot \
    | awk -v me="$$" '$1 != me {print $1}'
}

# --- status ----------------------------------------------------------------
if [[ "${1:-}" == "--status" ]]; then
  printf '%-40s %s\n' PHASE STATUS
  pids="$(runner_pids | tr '\n' ' ')"
  printf '%-40s %s\n' "1 in-flight run_tao_parallel.sh" \
    "$([[ -n "${pids// /}" ]] && echo "RUNNING (pids: $pids)" || echo "not running")"
  if [[ -x ./run_tao_parallel.sh ]]; then
    printf '%-40s %s\n' "2 tao shards" "$(./run_tao_parallel.sh --status | tail -1)"
  fi
  for m in "${MODELS[@]}"; do
    printf '%-40s %s\n' "3 tao canonical / $m" \
      "$(tao_complete "$m" && echo "published" || echo "not published (incomplete shards)")"
    printf '%-40s %s\n' "4 judge / $m" \
      "$([[ -f "$STATE/${m}_judge_tao.done" ]] && echo done || echo pending)"
    printf '%-40s %s\n' "5 utility tao / $m" \
      "$([[ -f "${OUTDIR[$m]}/utility_eval_attacked_tao.json" ]] && echo done || echo pending)"
  done
  exit 0
fi

# --- phase 1: wait for the in-flight runner --------------------------------
if (( SKIP_WAIT == 0 )); then
  say "phase 1: waiting for any in-flight run_tao_parallel.sh"
  waited=0
  while :; do
    pids="$(runner_pids | tr '\n' ' ')"
    [[ -z "${pids// /}" ]] && { say "  no runner on the card, continuing"; break; }
    (( waited % 30 == 0 )) && say "  still running (pids: $pids), waited ${waited}m"
    if (( waited >= WAIT_CEILING_H * 60 )); then
      say "  ceiling of ${WAIT_CEILING_H}h reached, giving up rather than racing it"
      exit 1
    fi
    (( DRY_RUN )) && { say "  (dry-run: would wait here)"; break; }
    sleep 60
    waited=$((waited + 1))
  done
else
  say "phase 1: SKIP_WAIT=1, not waiting"
fi

# --- phase 2: re-run every shard that is not done --------------------------
# run_tao_parallel.sh resumes by design: the pool skips only <name>.done, so both
# the six failed shards and anything still pending are picked up. Partial work
# survives -- TAO checkpoints per behaviour, so a shard that reached 33/50 starts
# at 34.
if (( FAILED_ONLY )); then
  say "phase 2: re-running ONLY the previously-failed shards ($TAO_WORKERS_RELAUNCH workers)"
  run env TAO_WORKERS="$TAO_WORKERS_RELAUNCH" ./run_tao_parallel.sh --failed
else
  say "phase 2: resuming all shards that are not done ($TAO_WORKERS_RELAUNCH workers)"
  run env TAO_WORKERS="$TAO_WORKERS_RELAUNCH" ./run_tao_parallel.sh
fi
say "phase 2 finished; shard table:"
(( DRY_RUN )) || ./run_tao_parallel.sh --status | tail -3

# --- phase 3: merge shards into canonical files ----------------------------
# The guard in merge_tao_shards.py refuses to publish a group that is missing an
# exit or short of N_BEHAVIORS, so a shard that failed again leaves the canonical
# name absent rather than publishing a short stage that downstream globs would
# score as if it were complete.
say "phase 3: merging shards (guarded)"
for m in "${MODELS[@]}"; do
  run env RESULTS_DIR="${OUTDIR[$m]}" EXITS="${MODEL_EXITS[$m]}" N_BEHAVIORS="$N_BEHAVIORS" \
      "$PY" experiments/merge_tao_shards.py
done

# --- phase 4: judge -------------------------------------------------------
# judge_harmfulness.py scores STORED completions with the HarmBench Mistral-7B
# classifier; it never loads the attacked model, so a mistake here costs a ~10
# minute rescoring pass and no attack compute. It re-scores every *attacks*.json
# in the directory, which also overwrites the stale single-behaviour TAO entries
# that were sitting in results_7b/judge_scores.json.
say "phase 4: judging"
for m in "${MODELS[@]}"; do
  if ! tao_complete "$m"; then
    say "  $m: no complete TAO group, skipping judge"
    continue
  fi
  if [[ -f "$STATE/${m}_judge_tao.done" ]]; then
    say "  $m: judge already done"
    continue
  fi
  say "  $m: judging"
  if run env BALCONY_MODEL="${BALCONY[$m]}" BASELINE_MODEL="${BASELINE[$m]}" \
        EXITS="${MODEL_EXITS[$m]}" CHAT_TEMPLATE="${TEMPLATE[$m]}" \
        RESULTS_DIR="${OUTDIR[$m]}" N_BEHAVIORS="$N_BEHAVIORS" NUM_STEPS="$NUM_STEPS" \
        "$PY" experiments/judge_harmfulness.py >> "$LOGS/${m}_judge_tao.log" 2>&1; then
    (( DRY_RUN )) || : > "$STATE/${m}_judge_tao.done"
  else
    say "  $m: judge FAILED, see $LOGS/${m}_judge_tao.log"
  fi
done

# --- phase 5: utility under TAO suffixes -----------------------------------
# Clean and GCG-attacked utility are already complete for all three models
# (verified 2026-08-09: all 5 arms x 6 tasks present in utility_eval.json and
# utility_eval_attacked.json). The clean run is the shared comparator for both
# attacks, so only the TAO-suffixed pass is missing.
say "phase 5: utility under TAO suffixes ($UTIL_WORKERS workers)"

util_stage() {
  local m="$1" log="$LOGS/${m}_utility_tao.log"
  say "  [start] $m utility (tao)"
  env BALCONY_MODEL="${BALCONY[$m]}" BASELINE_MODEL="${BASELINE[$m]}" \
      EXITS="${MODEL_EXITS[$m]}" CHAT_TEMPLATE="${TEMPLATE[$m]}" \
      RESULTS_DIR="${OUTDIR[$m]}" \
      "$PY" experiments/eval_capabilities_under_attack.py --attack tao >> "$log" 2>&1
  local rc=$?
  if (( rc == 0 )); then
    : > "$STATE/${m}_utility_tao.done"
    say "  [done]  $m utility (tao)"
  else
    echo "$rc" > "$STATE/${m}_utility_tao.failed"
    say "  [FAIL]  $m utility (tao) rc=$rc -- see $log"
  fi
  return 0
}

gpu_free_mb() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1
}

for m in "${MODELS[@]}"; do
  if ! tao_complete "$m"; then
    say "  $m: no complete TAO group, skipping utility"
    continue
  fi
  if [[ -f "$STATE/${m}_utility_tao.done" ]]; then
    say "  $m: utility (tao) already done"
    continue
  fi
  if (( DRY_RUN )); then
    echo "    would run: eval_capabilities_under_attack.py --attack tao  (RESULTS_DIR=${OUTDIR[$m]})"
    continue
  fi
  # same admission discipline as the TAO pool: never start a stage against
  # instantaneous free memory without checking it actually fits.
  while :; do
    running=$(jobs -r | wc -l)
    free="$(gpu_free_mb)"
    if (( running < UTIL_WORKERS )) && { [[ -z "$free" ]] || (( free >= UTIL_NEED_MB[$m] )); }; then
      break
    fi
    sleep 60
  done
  util_stage "$m" &
  sleep 90   # let it load before the next admission check
done
wait

say "=== relaunch pipeline finished ==="
(( DRY_RUN )) || ./run_tao_relaunch.sh --status
