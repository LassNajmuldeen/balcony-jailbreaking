#!/usr/bin/env bash
# waits for run_regen_512.sh to finish, then judges the regenerated completions
# with HarmBench's llama-2-13b classifier -- twice per model:
#
#   judge_scores_512.json    own_exit_completion        (512 tokens)
#   judge_scores_256.json    own_exit_completion_short  (same run, cut at 256)
#
# both arms come from ONE generation, so the pair isolates completion length from
# everything else. the pre-existing judge_scores.json (mistral-7b, on-disk 256-token
# text) is left untouched, so nothing that already works is disturbed.
#
#   nohup ./run_judge_512.sh > run_logs/judge512_console.log 2>&1 &
set -u

cd "$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"  # repo root
PY=./AdversariaLLM/.venv/bin/python
mkdir -p run_logs run_state

# --- wait for the regeneration --------------------------------------------
# poll the done/failed markers rather than a PID, so this survives the regen being
# restarted and refuses to judge output that was never produced.
echo "waiting for regeneration to finish..."
while :; do
  done_n=0
  for m in 8b 7b 1b; do
    [[ -f "run_state/regen512_${m}.done" ]] && done_n=$((done_n + 1))
    if [[ -f "run_state/regen512_${m}.failed" ]]; then
      echo "ABORT: regen512_${m} failed -- see run_logs/regen512_${m}.log"
      exit 1
    fi
  done
  [[ $done_n -eq 3 ]] && break
  sleep 60
done
echo "regeneration complete $(date -Is)"

# free the GPU before loading a 13B judge
sleep 10

judge_one() {
  local tag="$1" balcony="$2" baseline="$3" exits="$4" outdir="$5" template="$6"
  local field="$7" out="$8" label="$9"

  if [[ -f "run_state/judge512_${tag}_${label}.done" ]]; then
    echo "[$tag/$label] already done, skipping"
    return 0
  fi
  echo "[$tag/$label] judging $field -> $out  $(date -Is)"
  env BALCONY_MODEL="$balcony" BASELINE_MODEL="$baseline" \
      EXITS="$exits" RESULTS_DIR="$outdir" CHAT_TEMPLATE="$template" \
      N_BEHAVIORS=50 \
      JUDGE_CLASSIFIER=llama-2-13b \
      COMPLETION_FIELD="$field" \
      COMPLETION_GLOB='*attacks*_512.json' \
      JUDGE_OUTPUT="$out" \
      "$PY" experiments/judge_harmfulness.py \
      > "run_logs/judge512_${tag}_${label}.log" 2>&1
  if [[ $? -eq 0 ]]; then
    touch "run_state/judge512_${tag}_${label}.done"
    echo "[$tag/$label] done $(date -Is)"
  else
    touch "run_state/judge512_${tag}_${label}.failed"
    echo "[$tag/$label] FAILED -- see run_logs/judge512_${tag}_${label}.log"
  fi
}

run_model() {
  local tag="$1" balcony="$2" baseline="$3" exits="$4" outdir="$5" template="$6"
  judge_one "$tag" "$balcony" "$baseline" "$exits" "$outdir" "$template" \
            own_exit_completion       judge_scores_512.json 512
  judge_one "$tag" "$balcony" "$baseline" "$exits" "$outdir" "$template" \
            own_exit_completion_short judge_scores_256.json 256
}

run_model 8b adpretko/Balcony-Meta-LLaMA3-8B-exit-15-18-21 NousResearch/Meta-Llama-3-8B "15 18 21 full" results_8b llama-3-balcony
run_model 7b parsakaveh/Balcony-LLaMA2-7B                  NousResearch/Llama-2-7b-hf   "15 18 21 full" results_7b llama-2-chat
run_model 1b parsakaveh/Balcony-LLM-1B                     parsakaveh/LLM-1B-Pretrained "4 8 12 full"   results     ""

echo "ALL JUDGING DONE $(date -Is)"
echo
echo "=== ASR: 256 vs 512, llama-2-13b classifier ==="
"$PY" experiments/compare_256_512.py 2>&1 | tail -60
