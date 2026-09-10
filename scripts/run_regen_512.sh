#!/usr/bin/env bash
# regenerates every judged completion at 512 tokens from the stored suffixes.
# no attack compute. writes results*/<name>_512.json, never touches the originals.
# see experiments/regen_completions.py for why the 256 comparison arm comes from
# truncating these outputs rather than from the on-disk text.
set -u

cd "$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"  # repo root
PY=./AdversariaLLM/.venv/bin/python
mkdir -p run_logs run_state

run_model() {
  local tag="$1" balcony="$2" baseline="$3" exits="$4" outdir="$5" template="$6"
  if [[ -f "run_state/regen512_${tag}.done" ]]; then
    echo "[$tag] already done, skipping"
    return 0
  fi
  echo "[$tag] starting $(date -Is)"
  env BALCONY_MODEL="$balcony" BASELINE_MODEL="$baseline" \
      EXITS="$exits" RESULTS_DIR="$outdir" CHAT_TEMPLATE="$template" \
      MAX_NEW_TOKENS=512 N_BEHAVIORS=50 \
      "$PY" experiments/regen_completions.py > "run_logs/regen512_${tag}.log" 2>&1
  if [[ $? -eq 0 ]]; then
    touch "run_state/regen512_${tag}.done"
    echo "[$tag] done $(date -Is)"
  else
    touch "run_state/regen512_${tag}.failed"
    echo "[$tag] FAILED $(date -Is) -- see run_logs/regen512_${tag}.log"
  fi
}

run_model 8b adpretko/Balcony-Meta-LLaMA3-8B-exit-15-18-21 NousResearch/Meta-Llama-3-8B "15 18 21 full" results_8b llama-3-balcony
run_model 7b parsakaveh/Balcony-LLaMA2-7B                  NousResearch/Llama-2-7b-hf   "15 18 21 full" results_7b llama-2-chat
run_model 1b parsakaveh/Balcony-LLM-1B                     parsakaveh/LLM-1B-Pretrained "4 8 12 full"   results     ""

echo "ALL DONE $(date -Is)"
