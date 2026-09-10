#!/usr/bin/env bash
# 7B, ALL REMAINING CELLS, NO CHAT TEMPLATE — parallel work queue (2026-08-12)
#
# Fills in every 7B cell contaminated by the [INST] template bug (BUGS.md 2.7).
# Already handled elsewhere and NOT queued here:
#     full / HarmBench   done, 44% (run_7b_notemplate.sh)
#     full / AdvBench    in flight (run_7b_notemplate_advbench.sh)
# So this queues the six exit cells: {15,18,21} x {HarmBench, AdvBench}.
#
# WHY PARALLEL: measured GPU utilisation of a single GCG process is ~37% with
# dips to 0% (CPU-bound candidate sampling), and peak memory is ~24 GB of 98 GB.
# Three concurrent processes fit with headroom. Sequential would be ~16.5 h;
# three-way lands around 8 h.
#
# WHAT IS DELIBERATELY NOT CHANGED: SEARCH_WIDTH (128) and TOPK (64). Raising
# them would fill the GPU but change the attack, and these numbers must stay
# comparable to the 1B and 8B cells. Parallelism is free; hyperparameters are not.
#
# OOM GUARD: a job only starts when nvidia-smi reports NEED_MB free. That is the
# same failure this repo already hit once (BUGS.md 3.5 admitted workers against
# instantaneous free memory) -- so we also re-check after a settle delay, and we
# never count memory that a just-launched sibling has not finished claiming.
set -u
cd "$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"  # repo root
PY=./AdversariaLLM/.venv/bin/python
OUT=results_7b_notemplate
mkdir -p run_logs run_state "$OUT"

MAXJOBS="${MAXJOBS:-3}"
FRACTION="${FRACTION:-0.30}"    # per-process cap, ~30 GB of the 102 GB card
NEED_MB="${NEED_MB:-32000}"     # must exceed the cap, or a job starts and dies
SETTLE=120                      # let a new job load weights and claim its share

# Count GPU compute processes, NOT pgrep. `pgrep -f <pattern>` matches the shell
# that runs it -- it reported "2 running" with zero jobs alive during testing,
# which is BUGS.md 4.1 in a new costume. nvidia-smi cannot match itself.
running() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . ; }
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

# one cell = one exit x one benchmark, in its own results dir so nothing collides
launch() {
  local exit="$1" bench="$2" script="$3" tag="notpl_cell_${bench}_${exit}"
  if [[ -f "run_state/${tag}.done" ]]; then echo "[$tag] done, skip"; return 0; fi
  echo "[$tag] launching $(date -Is)"
  setsid nohup env \
      BALCONY_MODEL=parsakaveh/Balcony-LLaMA2-7B \
      BASELINE_MODEL=NousResearch/Llama-2-7b-hf \
      EXITS="$exit" N_BEHAVIORS=50 NUM_STEPS=150 MAX_NEW_TOKENS=512 \
      CHAT_TEMPLATE=plain RESULTS_DIR="${OUT}/${tag}" \
      GPU_MEM_FRACTION="$FRACTION" \
      "$PY" "experiments/${script}" \
      > "run_logs/${tag}.log" 2>&1 < /dev/null &
  sleep "$SETTLE"
}

# queue: exit, benchmark label, script
QUEUE=(
  "15 harmbench attack_gcg_harmbench.py"
  "18 harmbench attack_gcg_harmbench.py"
  "21 harmbench attack_gcg_harmbench.py"
  "15 advbench  attack_gcg_advbench.py"
  "18 advbench  attack_gcg_advbench.py"
  "21 advbench  attack_gcg_advbench.py"
)

echo "=== 7B no-template queue started $(date -Is) (maxjobs=$MAXJOBS need=${NEED_MB}MB) ==="
for job in "${QUEUE[@]}"; do
  read -r ex bench script <<< "$job"
  # wait for a free slot AND enough memory
  while :; do
    n=$(running)
    f=$(free_mb)
    if (( n < MAXJOBS )) && (( f > NEED_MB )); then
      sleep 10                                  # re-check: guard against a sibling still claiming
      f=$(free_mb); n=$(running)
      (( n < MAXJOBS )) && (( f > NEED_MB )) && break
    fi
    sleep 60
  done
  launch "$ex" "$bench" "$script"
done

echo "=== all queued, waiting for stragglers $(date -Is) ==="
while (( $(running) > 0 )); do sleep 60; done

# mark done only for cells that actually produced output
for job in "${QUEUE[@]}"; do
  read -r ex bench script <<< "$job"
  tag="notpl_cell_${bench}_${ex}"
  if ls "${OUT}/${tag}"/*attacks*.json > /dev/null 2>&1; then
    touch "run_state/${tag}.done"
  else
    touch "run_state/${tag}.failed"; echo "[$tag] NO OUTPUT — see run_logs/${tag}.log"
  fi
done

# ---- judge every finished cell with the study's classifier --------------------
echo "=== judging $(date -Is) ==="
for job in "${QUEUE[@]}"; do
  read -r ex bench script <<< "$job"
  tag="notpl_cell_${bench}_${ex}"
  [[ -f "run_state/${tag}.done" ]] || continue
  env BALCONY_MODEL=parsakaveh/Balcony-LLaMA2-7B \
      BASELINE_MODEL=NousResearch/Llama-2-7b-hf \
      EXITS="$ex" N_BEHAVIORS=50 RESULTS_DIR="${OUT}/${tag}" CHAT_TEMPLATE=plain \
      JUDGE_CLASSIFIER=llama-2-13b COMPLETION_FIELD=own_exit_completion \
      COMPLETION_GLOB='*attacks*.json' JUDGE_OUTPUT=judge_scores_13b_notemplate.json \
      "$PY" experiments/judge_harmfulness.py >> "run_logs/${tag}_judge.log" 2>&1 \
    || echo "[$tag] JUDGE FAILED"
done
echo "ALL DONE $(date -Is)"
