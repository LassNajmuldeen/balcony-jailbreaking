#!/usr/bin/env bash
# Parallel TAO runner: one process per (model, benchmark, exit-or-baseline) shard,
# pulled off a single global queue by a fixed pool of workers.
#
# Why sharded rather than the 12 whole stages run_matrix.sh defines: a balcony
# stage is 4 exits x 50 behaviours and a baseline stage is 50, so running whole
# stages concurrently leaves the pool badly unbalanced and ends with one 8B
# balcony stage running alone for hours. Sharding per exit makes every unit of
# work the same size (50 behaviours), so the pool drains evenly.
#
# One GLOBAL queue with N slots, not one pool per model. A per-model pool makes
# the slowest model the long pole while the GPU sits half idle behind the models
# that already finished; a global queue keeps every slot busy until the very end.
# The queue is ordered longest-job-first (8B, then 7B, then 1B) because
# longest-processing-time-first is what keeps the makespan tail short.
#
# N is bounded by GPU memory, not by cores: a worker peaks around PEAK_GB below,
# so N * PEAK_GB must leave room for anything else already on the card.
#
#   ./run_tao_parallel.sh              # run (resumes; skips completed shards)
#   ./run_tao_parallel.sh --status     # progress table, safe any time
#   ./run_tao_parallel.sh --dry-run    # print the queue and exit
#   TAO_WORKERS=2 ./run_tao_parallel.sh
#
# Detach it so it outlives the shell:
#   setsid nohup ./run_tao_parallel.sh > run_logs/tao_parallel_console.log 2>&1 < /dev/null &

set -uo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."

REPO="$(pwd)"
PY="$REPO/AdversariaLLM/.venv/bin/python"
STATE="$REPO/run_state"
LOGS="$REPO/run_logs"
mkdir -p "$STATE" "$LOGS"

WORKERS="${TAO_WORKERS:-3}"                  # with the GPU to ourselves
WORKERS_FOREIGN="${TAO_WORKERS_FOREIGN:-1}"  # while someone else's job is on the card
# Headroom held back for a foreign job's PEAK while one is running. 72 GB covers
# the 70.33 GB spike observed from run_1b_advbench_baseline.sh's GCG stage. The
# effect is that big shards simply wait out a neighbouring job rather than racing
# it; set to 0 to share the card aggressively.
FOREIGN_RESERVE_MB="${TAO_FOREIGN_RESERVE_MB:-72000}"
N_BEHAVIORS="${N_BEHAVIORS:-50}"
NUM_STEPS="${NUM_STEPS:-150}"

# Same model matrix as run_matrix.sh. Kept in sync by hand -- if run_matrix.sh
# changes a checkpoint or a template, change it here too.
declare -A BALCONY BASELINE MODEL_EXITS TEMPLATE OUTDIR
BALCONY[1b]="parsakaveh/Balcony-LLM-1B";                        BASELINE[1b]="parsakaveh/LLM-1B-Pretrained"
BALCONY[7b]="parsakaveh/Balcony-LLaMA2-7B";                     BASELINE[7b]="NousResearch/Llama-2-7b-hf"
BALCONY[8b]="adpretko/Balcony-Meta-LLaMA3-8B-exit-15-18-21";    BASELINE[8b]="NousResearch/Meta-Llama-3-8B"
MODEL_EXITS[1b]="4 8 12 full";  MODEL_EXITS[7b]="15 18 21 full";  MODEL_EXITS[8b]="15 18 21 full"
TEMPLATE[1b]="";                TEMPLATE[7b]="llama-2-chat";      TEMPLATE[8b]="llama-3-balcony"
OUTDIR[1b]="results";           OUTDIR[7b]="results_7b";          OUTDIR[8b]="results_8b"

# --- build the queue -------------------------------------------------------
# Round-robin 8b/7b/1b rather than all-8B-first. Peak GPU footprint per worker is
# roughly 8B 36 GB, 7B 30 GB, 1B 14 GB, so ten consecutive 8B shards would let only
# TWO workers fit in 97 GB, while a mixed 8B+7B+1B trio is ~80 GB and fits THREE.
# Interleaving also lands all three models near the same finish time instead of
# leaving the 1B untouched until the end.
declare -A PER_MODEL
for m in 8b 7b 1b; do
  list=()
  for bench in harmbench advbench; do
    for ex in ${MODEL_EXITS[$m]}; do
      list+=("${m}|${bench}|balcony|${ex}")
    done
    list+=("${m}|${bench}|baseline|no_balcony")
  done
  PER_MODEL[$m]="${list[*]}"
done

SHARDS=()
for (( i = 0; i < 10; i++ )); do          # each model contributes exactly 10 shards
  for m in 8b 7b 1b; do
    read -r -a arr <<< "${PER_MODEL[$m]}"
    [[ -n "${arr[$i]:-}" ]] && SHARDS+=("${arr[$i]}")
  done
done

# Peak GPU use per worker. REVISED 2026-08-09 from the OOM tracebacks of the six
# failed HarmBench shards, which report every resident process's usage at the
# moment of the crash. The 2026-08-06 numbers were taken early in a behaviour and
# badly understated the tail: a worker's footprint grows as the suffix lengthens.
#
#   observed per-process peaks   7B 41.16 GiB (42150 MB)   8B 38.78 GiB (39718 MB)
#
# NEED_MB[7b]=30000 was the direct cause of all six failures -- three workers were
# admitted against a 30 GB estimate, then summed to 94.86 GiB on a 94.97 GiB card.
# These values are set ABOVE the observed peak, so on this card they admit TWO
# workers, not three. That is the honest capacity: 44000+44000 = 88 GB leaves ~9 GB
# of headroom, while any third worker (even the 1B at 16 GB) overcommits it.
declare -A NEED_MB
NEED_MB[8b]=44000
NEED_MB[7b]=44000
NEED_MB[1b]=16000

shard_name() {  # model bench arm exit -> stage name
  local m="$1" bench="$2" arm="$3" ex="$4"
  if [[ "$arm" == "baseline" ]]; then
    echo "${m}_tao_${bench}_baseline"
  else
    echo "${m}_tao_${bench}_exit${ex}"
  fi
}

# Behaviours actually completed, counted as UNIQUE indices from the log's
# "  [N] final_loss=..." lines. Logs are append-mode across attempts, so a plain
# `grep -c` double-counts a shard that crashed and was re-run from behaviour 0 --
# 1b_tao_harmbench_baseline read "63 behaviors" for a clean 50-behaviour file.
beh_count() {
  grep -oE '\[[0-9]+\] final_loss=' "$1" 2>/dev/null | sort -u | wc -l
}

# --- status ----------------------------------------------------------------
if [[ "${1:-}" == "--status" ]]; then
  printf '%-34s %-10s %s\n' STAGE STATUS DETAIL
  done_n=0; fail_n=0; run_n=0; todo_n=0; skip_n=0
  for s in "${SHARDS[@]}"; do
    IFS='|' read -r m bench arm ex <<< "$s"
    name="$(shard_name "$m" "$bench" "$arm" "$ex")"
    log="$LOGS/${name}.log"
    if [[ -f "$STATE/${name}.skipped" ]]; then
      st=skipped; ((skip_n++))
      detail="$(head -1 "$STATE/${name}.skipped" 2>/dev/null | head -c 58)"
    elif [[ -f "$STATE/${name}.done" ]]; then
      st=done; ((done_n++))
      detail="$(beh_count "$log") behaviors"
    elif [[ -f "$STATE/${name}.failed" ]]; then
      st=FAILED; ((fail_n++))
      # LAST error, not first: logs are append-mode, so a shard that failed once
      # for reason A and later for reason B would otherwise show A forever. This
      # is why 7b_tao_harmbench_baseline displayed a stale ValueError from a
      # superseded revision while its actual last failure was an OOM.
      detail="$(grep -E '^[A-Za-z_.]*(Error|Exception)' "$log" 2>/dev/null | tail -1 | head -c 60)"
    elif [[ -f "$STATE/${name}.running" ]]; then
      st=running; ((run_n++))
      detail="$(beh_count "$log")/${N_BEHAVIORS} behaviors"
    else
      st=pending; ((todo_n++)); detail=""
    fi
    printf '%-34s %-10s %s\n' "$name" "$st" "$detail"
  done
  echo
  echo "done=$done_n running=$run_n pending=$todo_n failed=$fail_n skipped=$skip_n  (of ${#SHARDS[@]})"
  exit 0
fi

# --- deliberately skipped shards ------------------------------------------
# A shard with run_state/<name>.skipped is dropped from the queue entirely. Used
# 2026-08-09 for the TAO baseline arms: the balcony trunk is numerically identical
# to the baseline checkpoint in all three models (291/291 tensors, max |delta| = 0
# after the bf16 cast the loader performs), so `baseline` is the same model as the
# `full` exit rather than an independent control. A `.done` marker is written
# alongside so an ALREADY-RUNNING pool -- which cannot see this code -- also skips
# them, since run_shard() short-circuits on .done.
_kept=()
for s in "${SHARDS[@]}"; do
  IFS='|' read -r m bench arm ex <<< "$s"
  n="$(shard_name "$m" "$bench" "$arm" "$ex")"
  if [[ -f "$STATE/${n}.skipped" ]]; then
    echo "[skip] $n -- $(head -1 "$STATE/${n}.skipped")"
    continue
  fi
  _kept+=("$s")
done
SHARDS=("${_kept[@]}")

# --- optional queue filter -------------------------------------------------
# Re-running this script already resumes correctly -- the pool skips only
# <name>.done, so failed and pending shards are retried automatically. The filter
# is for when you want ONLY the previously-failed shards to run, e.g. to clear the
# six 2026-08-09 OOM casualties without also starting fresh AdvBench work.
#
#   ./run_tao_parallel.sh --failed
#   TAO_ONLY="7b_tao_harmbench_exit15 8b_tao_harmbench_exit18" ./run_tao_parallel.sh
#
# Placed after --status deliberately, so --status always reports all 30 shards
# regardless of what this run was scoped to.
if [[ "${1:-}" == "--failed" ]]; then
  _only=""
  for s in "${SHARDS[@]}"; do
    IFS='|' read -r m bench arm ex <<< "$s"
    _only+=" $(shard_name "$m" "$bench" "$arm" "$ex")"
  done
  TAO_ONLY=""
  for n in $_only; do
    [[ -f "$STATE/${n}.failed" ]] && TAO_ONLY+=" $n"
  done
  if [[ -z "${TAO_ONLY// /}" ]]; then
    echo "no failed shards to rerun"
    exit 0
  fi
  shift
fi

if [[ -n "${TAO_ONLY:-}" ]]; then
  _filtered=()
  for s in "${SHARDS[@]}"; do
    IFS='|' read -r m bench arm ex <<< "$s"
    n="$(shard_name "$m" "$bench" "$arm" "$ex")"
    [[ " $TAO_ONLY " == *" $n "* ]] && _filtered+=("$s")
  done
  if (( ${#_filtered[@]} == 0 )); then
    echo "TAO_ONLY matched no shards: $TAO_ONLY" >&2
    exit 1
  fi
  SHARDS=("${_filtered[@]}")
  echo "queue filtered to ${#SHARDS[@]} shard(s):$TAO_ONLY"
fi

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "workers=$WORKERS  shards=${#SHARDS[@]}  n_behaviors=$N_BEHAVIORS  steps=$NUM_STEPS"
  for s in "${SHARDS[@]}"; do
    IFS='|' read -r m bench arm ex <<< "$s"
    echo "  $(shard_name "$m" "$bench" "$arm" "$ex")"
  done
  exit 0
fi

# --- run one shard ---------------------------------------------------------
run_shard() {
  local s="$1"
  IFS='|' read -r m bench arm ex <<< "$s"
  local name; name="$(shard_name "$m" "$bench" "$arm" "$ex")"
  local log="$LOGS/${name}.log"

  if [[ -f "$STATE/${name}.done" ]]; then
    echo "[skip] $name already done"
    return 0
  fi
  rm -f "$STATE/${name}.failed"
  : > "$STATE/${name}.running"

  # balcony shards attack ONE exit; the baseline arm has no exits to split
  local exits_env shard_tag
  if [[ "$arm" == "baseline" ]]; then
    exits_env="${MODEL_EXITS[$m]}"   # ignored by the baseline path, kept for config sanity
    shard_tag="nb"
  else
    exits_env="$ex"
    shard_tag="e${ex}"
  fi

  echo "[start] $name  ($(date +%H:%M:%S))"
  # Snapshot the card at admission. Every failure in this run was an overcommit --
  # three workers summing past the card -- and reconstructing that afterwards meant
  # parsing per-process figures out of OOM tracebacks. Record it up front instead.
  {
    echo "=== shard start $(date -Is)  $name  (reserved ${NEED_MB[$m]}MB) ==="
    echo "--- gpu: total,used,free MiB ---"
    nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader 2>/dev/null
    echo "--- resident compute apps: pid,used ---"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null
  } >> "$log"
  env BALCONY_MODEL="${BALCONY[$m]}" \
      BASELINE_MODEL="${BASELINE[$m]}" \
      EXITS="$exits_env" \
      CHAT_TEMPLATE="${TEMPLATE[$m]}" \
      RESULTS_DIR="${OUTDIR[$m]}" \
      BENCHMARK="$bench" \
      ATTACK_TARGET="$arm" \
      TAO_SHARD="$shard_tag" \
      N_BEHAVIORS="$N_BEHAVIORS" \
      NUM_STEPS="$NUM_STEPS" \
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
      nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null
    } >> "$log"
    echo "[FAIL]  $name rc=$rc reason=$reason  ($(date +%H:%M:%S)) -- see $log"
  fi
  return 0   # never let one shard kill the pool
}
# No `export -f` needed: `run_shard ... &` forks THIS shell, so the background job
# already inherits every variable and function, associative arrays included.

# --- pool ------------------------------------------------------------------
echo "=== TAO parallel run: ${#SHARDS[@]} shards, $WORKERS workers, started $(date) ==="
echo "resume-safe: completed shards are skipped, partial shards resume per behaviour"

# Concurrency adapts to whoever else is on the card. An 8B worker peaks near 25 GB
# and the card holds 97 GB, so three workers fit -- but only with the GPU to
# ourselves. A foreign job (e.g. run_1b_advbench_baseline.sh's GCG stage) swings
# between ~5 GB and ~67 GB as it builds and frees each candidate batch, so sizing
# the pool off instantaneous free memory would admit three workers during a trough
# and then OOM both us AND the foreign job at its next peak. Counting foreign
# processes instead of free bytes is stable against that.
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

# `jobs -r` counts only still-running children. Counting with `kill -0` instead
# would count finished-but-unreaped children too -- a zombie still answers
# kill -0 -- which silently shrinks the pool to one worker over time.
# Admission gate. The parent forks workers serially, so checking free memory here
# needs no lock -- there is no other thread that could claim it in between. After
# forking we pause so the new worker actually allocates before the next check,
# otherwise three workers would all be admitted against the same free memory.
gpu_free_mb() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1
}

mem_available_for() {  # mem_available_for <model> -> 0 if it fits right now
  local need="${NEED_MB[$1]}" free reserve=0
  # Free memory alone is NOT a safe admission test while someone else is on the
  # card. A GCG job cycles between ~3 GB and ~70 GB as it builds and frees each
  # candidate batch, so admitting against a trough reading is how an 8B worker got
  # OOM-killed at 19:25 on 2026-08-06: admitted with 94 GB free, then the neighbour
  # spiked to 70.33 GB and there was nothing left. So reserve the foreign job's
  # PEAK, not its present size, for as long as any foreign process exists.
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
    IFS='|' read -r nm _ _ _ <<< "${SHARDS[$next]}"
    if ! mem_available_for "$nm"; then
      note="waiting on GPU memory for a $nm shard (need ${NEED_MB[$nm]}MB, free $(gpu_free_mb)MB, running $(jobs -r | wc -l))"
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
for d in results results_7b results_8b; do
  RESULTS_DIR="$d" "$PY" experiments/merge_tao_shards.py
done

echo "=== summary ==="
"$0" --status
