#!/usr/bin/env bash
# Full experiment matrix, run as one resumable queue.
#
# Deliberately does NOT use `set -e`: every stage runs to completion or failure and
# the queue moves on either way, so one crash at 3am cannot cost the remaining
# days of GPU time. Per-stage exit codes are recorded and summarised at the end.
#
# Stages are ordered by the priority they were requested in:
#   PRIORITY 1  GCG on 7B and 8B -- Balcony (per exit) and the weight-matched
#               baseline, on HarmBench and AdvBench, 150 steps, 50 behaviours.
#               Loss comes from the attack scripts, ASR from the judge stage.
#   PRIORITY 2  TAO on 1B, 7B and 8B -- Balcony and baseline, both benchmarks,
#               50 behaviours, then ASR.
#
# Resume semantics: a stage that finished successfully writes a marker into
# run_state/ and is skipped on re-run. Re-running after a crash therefore picks up
# where it left off. TAO additionally checkpoints per behaviour, so even a
# half-finished TAO stage resumes mid-stage rather than restarting.
#
# usage:
#   ./run_matrix.sh                  # run everything not yet done
#   ./run_matrix.sh --status         # show what is done/failed/pending, run nothing
#   ./run_matrix.sh --only 8b_gcg    # run only stages whose name matches a substring
#   FORCE=1 ./run_matrix.sh          # ignore completion markers and redo
#   N_BEHAVIORS=2 NUM_STEPS=5 ./run_matrix.sh --only 7b   # quick pilot of a subset

set -uo pipefail

REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
PYTHON="${PYTHON:-$REPO/AdversariaLLM/.venv/bin/python}"

# RUN_TAG isolates a run completely: its own state dir, its own logs, and a
# suffix on every results dir. Set it for pilots so a 2-behaviour smoke test
# cannot write completion markers or partial json into the real matrix.
RUN_TAG="${RUN_TAG:-}"
STATE_DIR="$REPO/run_state${RUN_TAG:+_$RUN_TAG}"
LOG_ROOT="$REPO/run_logs${RUN_TAG:+_$RUN_TAG}"
mkdir -p "$STATE_DIR" "$LOG_ROOT"

[ -x "$PYTHON" ] || { echo "FATAL: no interpreter at $PYTHON (see SETUP.md 5)" >&2; exit 1; }

ONLY=""
STATUS_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY="${2:-}"; shift 2 ;;
    --status) STATUS_ONLY=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# --- model roster ----------------------------------------------------------
# Balcony checkpoint, weight-matched baseline, exits, chat template, results dir.
#
# The baselines are the NousResearch mirrors rather than the gated meta-llama
# repos. They are not a compromise: verify_baseline_weights.py confirms every
# backbone tensor is bit-identical to the Balcony backbone (291/291 for both
# 7B and 8B), which is the property the baseline!=full rule actually requires.
#
# Chat template: the 1B ships ChatML on both checkpoints so it is left alone. The
# 7B Balcony ships llama-2-chat and its bare baseline is given the same one
# (renders byte-identical). Neither 8B checkpoint ships a template, so both get
# llama-3-balcony -- the llama-3 header format from Balcony-LLaMA's own configs,
# which unlike AdversariaLLM's llama-3-instruct injects no system prompt. A
# baseline must use the same template as the Balcony model it is compared with,
# or the comparison measures prompt formatting instead of early exits.

declare -A BALCONY BASELINE MODEL_EXITS TEMPLATE OUTDIR
BALCONY[1b]="parsakaveh/Balcony-LLM-1B";                        BASELINE[1b]="parsakaveh/LLM-1B-Pretrained"
BALCONY[7b]="parsakaveh/Balcony-LLaMA2-7B";                     BASELINE[7b]="NousResearch/Llama-2-7b-hf"
BALCONY[8b]="adpretko/Balcony-Meta-LLaMA3-8B-exit-15-18-21";    BASELINE[8b]="NousResearch/Meta-Llama-3-8B"
MODEL_EXITS[1b]="4 8 12 full";  MODEL_EXITS[7b]="15 18 21 full";  MODEL_EXITS[8b]="15 18 21 full"
TEMPLATE[1b]="";                TEMPLATE[7b]="llama-2-chat";      TEMPLATE[8b]="llama-3-balcony"
OUTDIR[1b]="results";           OUTDIR[7b]="results_7b";          OUTDIR[8b]="results_8b"

N_BEHAVIORS="${N_BEHAVIORS:-50}"
NUM_STEPS="${NUM_STEPS:-150}"

# --- queue -----------------------------------------------------------------
STAGE_NAMES=(); STAGE_SCRIPTS=(); STAGE_ENVS=()

add_stage() {  # add_stage <name> <script> <env assignments...>
  STAGE_NAMES+=("$1"); STAGE_SCRIPTS+=("$2"); shift 2
  STAGE_ENVS+=("$(printf '%s\n' "$@" | paste -sd $'\x1f' -)")
}

model_env() {  # common env for a model size
  local m="$1"
  printf '%s\n' \
    "BALCONY_MODEL=${BALCONY[$m]}" \
    "BASELINE_MODEL=${BASELINE[$m]}" \
    "EXITS=${MODEL_EXITS[$m]}" \
    "CHAT_TEMPLATE=${TEMPLATE[$m]}" \
    "RESULTS_DIR=${OUTDIR[$m]}${RUN_TAG:+_$RUN_TAG}" \
    "N_BEHAVIORS=$N_BEHAVIORS" \
    "NUM_STEPS=$NUM_STEPS"
}

# ---- PRIORITY 1: GCG on 7B and 8B, both arms, both benchmarks ----
for m in 7b 8b; do
  mapfile -t base_env < <(model_env "$m")
  add_stage "${m}_gcg_harmbench_balcony"  experiments/attack_gcg_harmbench.py "${base_env[@]}" "BENCHMARK=harmbench" "ATTACK_TARGET=balcony"
  add_stage "${m}_gcg_advbench_balcony"   experiments/attack_gcg_advbench.py        "${base_env[@]}" "BENCHMARK=advbench"  "ATTACK_TARGET=balcony"
  add_stage "${m}_gcg_harmbench_baseline" experiments/attack_gcg_baseline.py      "${base_env[@]}" "BENCHMARK=harmbench" "ATTACK_TARGET=baseline"
  add_stage "${m}_gcg_advbench_baseline"  experiments/attack_gcg_baseline.py      "${base_env[@]}" "BENCHMARK=advbench"  "ATTACK_TARGET=baseline"
  add_stage "${m}_judge_gcg"              experiments/judge_harmfulness.py        "${base_env[@]}"
done

# ---- PRIORITY 2: TAO on 1B, 7B and 8B, both arms, both benchmarks ----
for m in 1b 7b 8b; do
  mapfile -t base_env < <(model_env "$m")
  add_stage "${m}_tao_harmbench_balcony"  experiments/attack_tao.py "${base_env[@]}" "BENCHMARK=harmbench" "ATTACK_TARGET=balcony"
  add_stage "${m}_tao_advbench_balcony"   experiments/attack_tao.py "${base_env[@]}" "BENCHMARK=advbench"  "ATTACK_TARGET=balcony"
  add_stage "${m}_tao_harmbench_baseline" experiments/attack_tao.py "${base_env[@]}" "BENCHMARK=harmbench" "ATTACK_TARGET=baseline"
  add_stage "${m}_tao_advbench_baseline"  experiments/attack_tao.py "${base_env[@]}" "BENCHMARK=advbench"  "ATTACK_TARGET=baseline"
done

# final judge pass per model, so TAO gets ASR alongside GCG
for m in 1b 7b 8b; do
  mapfile -t base_env < <(model_env "$m")
  add_stage "${m}_judge_final" experiments/judge_harmfulness.py "${base_env[@]}"
done

# --- status ----------------------------------------------------------------
stage_state() {
  local name="$1"
  if [ -f "$STATE_DIR/$name.done" ]; then echo done
  elif [ -f "$STATE_DIR/$name.failed" ]; then echo "failed(rc=$(cat "$STATE_DIR/$name.failed"))"
  else echo pending; fi
}

if [ "$STATUS_ONLY" = 1 ]; then
  printf '%-34s %s\n' "STAGE" "STATE"
  for i in "${!STAGE_NAMES[@]}"; do
    printf '%-34s %s\n' "${STAGE_NAMES[$i]}" "$(stage_state "${STAGE_NAMES[$i]}")"
  done
  exit 0
fi

# --- run -------------------------------------------------------------------
echo "repo        : $REPO"
echo "python      : $PYTHON"
echo "behaviours  : $N_BEHAVIORS   steps: $NUM_STEPS"
echo "stages      : ${#STAGE_NAMES[@]}${ONLY:+  (filtered by '$ONLY')}"
echo "state dir   : $STATE_DIR"
echo "logs        : $LOG_ROOT"
echo "started     : $(date -Is)"
echo

started_at=$(date +%s)
ran=0; ok=0; failed=0; skipped=0

for i in "${!STAGE_NAMES[@]}"; do
  name="${STAGE_NAMES[$i]}"; script="${STAGE_SCRIPTS[$i]}"
  IFS=$'\x1f' read -r -a envs <<< "${STAGE_ENVS[$i]}"

  if [ -n "$ONLY" ] && [[ "$name" != *"$ONLY"* ]]; then continue; fi

  if [ -f "$STATE_DIR/$name.done" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "SKIP    $name (already completed)"
    skipped=$((skipped+1)); continue
  fi

  log="$LOG_ROOT/$name.log"
  echo "=== START  $name  ($(date -Is))"
  echo "    script: $script"
  t0=$(date +%s)

  rm -f "$STATE_DIR/$name.failed"
  env "${envs[@]}" "$PYTHON" "$REPO/$script" > "$log" 2>&1
  rc=$?
  dt=$(( $(date +%s) - t0 ))
  ran=$((ran+1))

  if [ $rc -eq 0 ]; then
    touch "$STATE_DIR/$name.done"; ok=$((ok+1))
    echo "=== OK     $name  (${dt}s)"
  else
    echo "$rc" > "$STATE_DIR/$name.failed"; failed=$((failed+1))
    echo "=== FAILED $name  (rc=$rc, ${dt}s) -- continuing"
    echo "--- last 15 lines of $log ---"
    tail -15 "$log" | sed 's/^/    /'
    echo "--- end ---"
  fi
  echo
done

# --- summary ---------------------------------------------------------------
echo "================ SUMMARY ($(date -Is), $(( ($(date +%s) - started_at) / 60 )) min) ================"
printf '%-34s %s\n' "STAGE" "STATE"
for i in "${!STAGE_NAMES[@]}"; do
  name="${STAGE_NAMES[$i]}"
  if [ -n "$ONLY" ] && [[ "$name" != *"$ONLY"* ]]; then continue; fi
  printf '%-34s %s\n' "$name" "$(stage_state "$name")"
done
echo
echo "ran=$ran ok=$ok failed=$failed skipped=$skipped"
[ "$failed" -gt 0 ] && echo "re-run ./run_matrix.sh to retry the failed stages (completed ones are skipped)"
exit 0
