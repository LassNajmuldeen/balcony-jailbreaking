#!/usr/bin/env bash
# Full budget TAO runner: 7B and 8B only, sharded by (model, benchmark, exit,
# behaviour range), pulled off one global queue by a fixed pool of workers.
#
# This is a SEPARATE script from run_tao_parallel.sh on purpose. That one drove
# the run behind the workshop numbers at 150 steps / batch 128, and its run_state
# markers and results dirs are the provenance for those numbers. Overwriting it
# would make the published run unreproducible, so it stays exactly as it is.
#
# What changed here versus that script:
#
#   1. NO 1B. Scope decision 2026-09-03, the 1B was a toy project. The queue is
#      two models, which is also why the round robin below only interleaves two.
#   2. Sharded by BEHAVIOUR RANGE, not just by exit. At 500 steps one 50 behaviour
#      shard is about two days, and a pool whose smallest unit is two days spends
#      its entire tail with one worker running alone while the card sits idle.
#   3. ROUND MAJOR ordering. Every cell finishes behaviours 0-49 before any cell
#      starts 50-99. So the matrix is always BALANCED: kill this at any point and
#      what is on disk is a complete 16 cell matrix at whatever behaviour count
#      the last finished round reached, never a half of one cell and none of
#      another. Extra rounds only tighten the confidence intervals.
#   4. Steps and sampling are dials, not constants. Defaults are the published TAO
#      budget (experiments/configs/template.py in the TAO-Attack checkout:
#      n_steps=500, batch_size=512, topk=256), not our old 150/128.
#   5. Writes to results_7b_tao500 / results_8b_tao500, NOT the dirs the workshop
#      numbers live in. Merging 500 step results into a file of 150 step results
#      would silently mix two attack budgets under one ASR.
#
#   ./run_tao_full.sh              # run (resumes; skips completed shards)
#   ./run_tao_full.sh --status     # progress table, safe any time
#   ./run_tao_full.sh --dry-run    # print the queue and exit
#   TAO_ROUNDS=2 ./run_tao_full.sh # only the first 2 rounds (100 behaviours)
#
# Detach it so it outlives the shell:
#   setsid nohup ./run_tao_full.sh > run_logs/tao_full_console.log 2>&1 < /dev/null &

set -uo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."

REPO="$(pwd)"
PY="$REPO/AdversariaLLM/.venv/bin/python"
STATE="$REPO/run_state_tao500"
LOGS="$REPO/run_logs"
mkdir -p "$STATE" "$LOGS"

# --- the dials -------------------------------------------------------------
# Published TAO budget by default. Measured cost on this card, 7B, solo, per
# behaviour, AFTER the fused loss change:  t = steps * (2.91 + 0.00461 * batch)
# seconds. Note how flat that is in batch -- going 128 -> 512 costs +46% wall
# clock, going 150 -> 500 steps costs +233%. Sampling is nearly free here and
# steps are what you pay for, which is the opposite of what i assumed before i
# measured it.
NUM_STEPS="${NUM_STEPS:-500}"
TAO_BATCH_SIZE="${TAO_BATCH_SIZE:-512}"
TAO_TOPK="${TAO_TOPK:-256}"

# behaviours per round, and how many rounds to queue. 50 x 1 reproduces the old
# behaviour count. HarmBench has 400 usable behaviours and AdvBench 520, so the
# ceiling is 8 rounds before harmbench runs out.
ROUND_SIZE="${TAO_ROUND_SIZE:-50}"
ROUNDS="${TAO_ROUNDS:-1}"

WORKERS="${TAO_WORKERS:-3}"
WORKERS_FOREIGN="${TAO_WORKERS_FOREIGN:-1}"
FOREIGN_RESERVE_MB="${TAO_FOREIGN_RESERVE_MB:-72000}"

declare -A BALCONY BASELINE MODEL_EXITS TEMPLATE OUTDIR
BALCONY[7b]="parsakaveh/Balcony-LLaMA2-7B";                  BASELINE[7b]="NousResearch/Llama-2-7b-hf"
BALCONY[8b]="adpretko/Balcony-Meta-LLaMA3-8B-exit-15-18-21";  BASELINE[8b]="NousResearch/Meta-Llama-3-8B"
MODEL_EXITS[7b]="15 18 21 full";  MODEL_EXITS[8b]="15 18 21 full"
# CHAT_TEMPLATE IS LOAD-BEARING FOR TAO AS OF 2026-09-03. It used to do nothing
# here -- attack_tao.py set tokenizer.chat_template and then never applied it --
# so TAO and GCG were attacking differently formatted prompts. SimpleSuffixManager
# now goes through prepare_conversation, the same function GCG uses, and the two
# produce byte identical token sequences on both models. A TAO run with the wrong
# template is therefore now as wrong as a GCG one.
#
# 7B uses `plain`, NOT `llama-2-chat`: the shipped llama-2-chat template is the
# [INST] bug (trap 2 in local-notes/CLAUDE.md) and the corrected GCG run is the
# CHAT_TEMPLATE=plain one in results_7b_notemplate. Matching GCG means matching
# the CORRECTED GCG.
TEMPLATE[7b]="plain";             TEMPLATE[8b]="llama-3-balcony"
OUTDIR[7b]="results_7b_tao500";   OUTDIR[8b]="results_8b_tao500"

# Peak GPU per worker. MEASURED 2026-09-03 on an otherwise idle card, one
# behaviour at batch 512, AFTER freezing the model parameters:
#
#     7B  20.9 GB   (was 33.6 before the freeze)
#     8B  23.3 GB   (was 37.7 before the freeze)
#
# Almost exactly half of each old peak was a .grad buffer for every model
# parameter, which autograd was filling on every backward because nothing ever
# set requires_grad=False. See load_model_and_tokenizer in attack_tao.py.
#
# Peak is flat in TAO_BATCH_SIZE now: the 7B measured 20946 MB at batch 128 and
# 20948 MB at batch 512. Sampling costs wall clock, not memory.
#
# These are set ABOVE the measured peak on purpose. The 2026-08-09 run lost six
# shards to an estimate set BELOW the true peak -- three workers were admitted
# against a 30 GB guess and summed to 94.86 GiB on a 94.97 GiB card.
declare -A NEED_MB
NEED_MB[8b]="${TAO_NEED_MB_8B:-26000}"
NEED_MB[7b]="${TAO_NEED_MB_7B:-24000}"

# Per process allocator cap, so one worker's freed-but-cached blocks cannot starve
# the others. config.py reads this; run_tao_parallel.sh never set it, which is
# part of why that run could only ever admit two workers.
#
# RAISED 0.28 -> 0.31 on 2026-09-04. The first real launch OOMed a third worker
# inside 15 minutes: it hit the 26.59 GiB cap while the CARD still had 30.5 GiB
# free, so the cap was the binding constraint, not the hardware. My 23.3 GB
# measurement came from behaviour 0, which is short; harmbench behaviours that
# carry a ContextString paragraph need meaningfully more.
#
# 0.31 is about 30 GB. Three workers is 90 GB of the 97.9 GB card, which is tight,
# so it is paired with TAO_FORWARD_CHUNK=32 in attack_tao.py -- that halves the
# biggest transient (the shard that died was asking for one 2.08 GiB block) and is
# the real headroom. If a worker still OOMs, drop TAO_FORWARD_CHUNK to 16 before
# touching this number or the worker count.
GPU_MEM_FRACTION="${GPU_MEM_FRACTION:-0.31}"

shard_name() {  # model bench exit start end -> stage name
  printf '%s_tao_%s_exit%s_b%03d-%03d' "$1" "$2" "$3" "$4" "$(( $5 - 1 ))"
}

# --- build the queue -------------------------------------------------------
# Round major: the outer loop is the behaviour range, so round r+1 is only queued
# after every cell of round r. Within a round, 8b and 7b alternate so the two
# model sizes finish together instead of leaving all the 8B work for the end.
SHARDS=()
for (( r = 0; r < ROUNDS; r++ )); do
  start=$(( r * ROUND_SIZE ))
  end=$(( start + ROUND_SIZE ))
  for bench in harmbench advbench; do
    for ex in 15 18 21 full; do
      for m in 8b 7b; do
        SHARDS+=("${m}|${bench}|${ex}|${start}|${end}")
      done
    done
  done
done

beh_count() {
  # UNIQUE indices only. Logs are append mode across attempts, so a plain grep -c
  # double counts a shard that crashed and restarted.
  grep -oE '\[[0-9]+\] final_loss=' "$1" 2>/dev/null | sort -u | wc -l
}

# --- status ----------------------------------------------------------------
if [[ "${1:-}" == "--status" ]]; then
  printf '%-40s %-10s %s\n' STAGE STATUS DETAIL
  done_n=0; fail_n=0; run_n=0; todo_n=0
  for s in "${SHARDS[@]}"; do
    IFS='|' read -r m bench ex st en <<< "$s"
    name="$(shard_name "$m" "$bench" "$ex" "$st" "$en")"
    log="$LOGS/${name}.log"
    if [[ -f "$STATE/${name}.done" ]]; then
      state=done; ((done_n++)); detail="$(beh_count "$log") behaviors"
    elif [[ -f "$STATE/${name}.failed" ]]; then
      state=FAILED; ((fail_n++))
      # LAST error, not first: append mode logs would otherwise show a superseded
      # failure forever
      detail="$(grep -E '^[A-Za-z_.]*(Error|Exception)' "$log" 2>/dev/null | tail -1 | head -c 60)"
    elif [[ -f "$STATE/${name}.running" ]]; then
      state=running; ((run_n++)); detail="$(beh_count "$log")/${ROUND_SIZE} behaviors"
    else
      state=pending; ((todo_n++)); detail=""
    fi
    printf '%-40s %-10s %s\n' "$name" "$state" "$detail"
  done
  echo
  echo "done=$done_n running=$run_n pending=$todo_n failed=$fail_n  (of ${#SHARDS[@]})"
  echo "steps=$NUM_STEPS batch=$TAO_BATCH_SIZE rounds=$ROUNDS x $ROUND_SIZE behaviors"
  exit 0
fi

if [[ -n "${TAO_ONLY:-}" ]]; then
  _filtered=()
  for s in "${SHARDS[@]}"; do
    IFS='|' read -r m bench ex st en <<< "$s"
    n="$(shard_name "$m" "$bench" "$ex" "$st" "$en")"
    [[ " $TAO_ONLY " == *" $n "* ]] && _filtered+=("$s")
  done
  if (( ${#_filtered[@]} == 0 )); then
    echo "TAO_ONLY matched no shards: $TAO_ONLY" >&2
    exit 1
  fi
  SHARDS=("${_filtered[@]}")
  echo "queue filtered to ${#SHARDS[@]} shard(s)"
fi

if [[ "${1:-}" == "--dry-run" ]]; then
  # Cost model from the 2026-09-03 calibration, 7B solo. The 1.1 is the 8B/7B
  # average, measured off the old run's per behaviour elapsed times.
  per_beh=$($PY - <<PY
steps, batch = $NUM_STEPS, $TAO_BATCH_SIZE
print(round(1.1 * steps * (2.910 + 0.004609 * batch)))
PY
)
  total=$(( ${#SHARDS[@]} * ROUND_SIZE * per_beh ))
  echo "workers=$WORKERS shards=${#SHARDS[@]} steps=$NUM_STEPS batch=$TAO_BATCH_SIZE"
  echo "rounds=$ROUNDS x $ROUND_SIZE behaviors"
  echo "est ${per_beh}s per behavior, $(( total / 3600 )) GPU-hours serial"
  # 3 workers do not give 3x. Measured contention on the old run was 1.28x slowdown
  # per process at 2 workers, i.e. 1.56x throughput; 3 workers is about 1.8x.
  echo "est $(( total / 3600 * 10 / 18 )) hours wall at $WORKERS workers (~1.8x throughput)"
  for s in "${SHARDS[@]}"; do
    IFS='|' read -r m bench ex st en <<< "$s"
    echo "  $(shard_name "$m" "$bench" "$ex" "$st" "$en")"
  done
  exit 0
fi

# --- run one shard ---------------------------------------------------------
run_shard() {
  local s="$1"
  IFS='|' read -r m bench ex st en <<< "$s"
  local name; name="$(shard_name "$m" "$bench" "$ex" "$st" "$en")"
  local log="$LOGS/${name}.log"

  if [[ -f "$STATE/${name}.done" ]]; then
    echo "[skip] $name already done"
    return 0
  fi
  rm -f "$STATE/${name}.failed"
  : > "$STATE/${name}.running"

  echo "[start] $name  ($(date +%H:%M:%S))"
  {
    echo "=== shard start $(date -Is)  $name  (reserved ${NEED_MB[$m]}MB) ==="
    nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader 2>/dev/null
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null
  } >> "$log"

  # TAO_SHARD carries the behaviour range as well as the exit, so two rounds of the
  # same exit land in different shard files. merge_tao_shards.py needs no changes
  # for this: it already dedupes on behavior_idx and behavior_idx stays global.
  env BALCONY_MODEL="${BALCONY[$m]}" \
      BASELINE_MODEL="${BASELINE[$m]}" \
      EXITS="$ex" \
      CHAT_TEMPLATE="${TEMPLATE[$m]}" \
      RESULTS_DIR="${OUTDIR[$m]}" \
      BENCHMARK="$bench" \
      ATTACK_TARGET=balcony \
      TAO_SHARD="e${ex}.b${st}-${en}" \
      BEHAVIOR_START="$st" \
      BEHAVIOR_END="$en" \
      N_BEHAVIORS="$en" \
      NUM_STEPS="$NUM_STEPS" \
      TAO_BATCH_SIZE="$TAO_BATCH_SIZE" \
      TAO_TOPK="$TAO_TOPK" \
      GPU_MEM_FRACTION="$GPU_MEM_FRACTION" \
      TAO_FORWARD_CHUNK="${TAO_FORWARD_CHUNK:-32}" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$PY" experiments/attack_tao.py >> "$log" 2>&1
  local rc=$?

  rm -f "$STATE/${name}.running"
  if [[ $rc -eq 0 ]]; then
    : > "$STATE/${name}.done"
    echo "[done]  $name  ($(date +%H:%M:%S))"
  else
    echo "$rc" > "$STATE/${name}.failed"
    reason="other"
    grep -q 'torch.OutOfMemoryError' "$log" 2>/dev/null && \
      [[ "$(grep -E '^[A-Za-z_.]*(Error|Exception)' "$log" | tail -1)" == *OutOfMemory* ]] && reason="OOM"
    {
      echo "=== shard FAILED $(date -Is)  $name  rc=$rc  reason=$reason ==="
      nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader 2>/dev/null
    } >> "$log"
    echo "[FAIL]  $name rc=$rc reason=$reason  ($(date +%H:%M:%S)) -- see $log"
  fi
  return 0   # never let one shard kill the pool
}

# --- pool ------------------------------------------------------------------
echo "=== TAO full-budget run: ${#SHARDS[@]} shards, $WORKERS workers, started $(date) ==="
echo "steps=$NUM_STEPS batch=$TAO_BATCH_SIZE topk=$TAO_TOPK rounds=$ROUNDS x $ROUND_SIZE"
echo "resume-safe: completed shards are skipped, partial shards resume per behaviour"

foreign_count() {
  local foreign=0 p
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    p="${p//[[:space:]]/}"
    [[ -z "$p" ]] && continue
    if ! tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q 'attack_tao.py'; then
      foreign=$((foreign + 1))
    fi
  done
  echo "$foreign"
}

target_workers() {
  if (( $(foreign_count) > 0 )); then echo "$WORKERS_FOREIGN"; else echo "$WORKERS"; fi
}

gpu_free_mb() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1
}

mem_available_for() {
  local need="${NEED_MB[$1]}" free reserve=0
  # Free memory alone is not a safe admission test while a foreign job is on the
  # card: a GCG job swings between ~3 GB and ~70 GB as it builds and frees each
  # candidate batch, so admitting against a trough reading is exactly how an 8B
  # worker got OOM killed on 2026-08-06. Reserve the foreign job's PEAK.
  (( $(foreign_count) > 0 )) && reserve="$FOREIGN_RESERVE_MB"
  free="$(gpu_free_mb)"
  [[ -z "$free" ]] && return 0     # cannot read the GPU: do not deadlock the queue
  (( free >= need + reserve ))
}

next=0
total=${#SHARDS[@]}
last_note=""
while (( next < total )) || (( $(jobs -r | wc -l) > 0 )); do
  tw="$(target_workers)"
  while (( $(jobs -r | wc -l) < tw )) && (( next < total )); do
    IFS='|' read -r nm _ _ _ _ <<< "${SHARDS[$next]}"
    if ! mem_available_for "$nm"; then
      note="waiting on GPU memory for a $nm shard (need ${NEED_MB[$nm]}MB, free $(gpu_free_mb)MB)"
      [[ "$note" != "$last_note" ]] && echo "[pool] $note ($(date +%H:%M:%S))" && last_note="$note"
      break
    fi
    echo "[pool] admitting $nm shard; $(jobs -r | wc -l) running, free $(gpu_free_mb)MB, target $tw"
    run_shard "${SHARDS[$next]}" &
    ((next++))
    sleep 120   # model load + first steps, so its memory is real before the next check
    tw="$(target_workers)"
  done
  wait -n 2>/dev/null || sleep 30
done
wait

echo "=== all shards finished $(date); merging ==="
# N_BEHAVIORS is the merge completeness guard: a cell is only published once every
# exit holds this many behaviours, so a partial round never reaches the judge
# looking like a finished stage with a smaller denominator.
for d in results_7b_tao500 results_8b_tao500; do
  RESULTS_DIR="$d" N_BEHAVIORS=$(( ROUNDS * ROUND_SIZE )) "$PY" experiments/merge_tao_shards.py
done

echo "=== summary ==="
"$0" --status
