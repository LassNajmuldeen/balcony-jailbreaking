#!/usr/bin/env bash
# scores the ORIGINAL attack completions (results*/[!_]*attacks*.json, the files
# written by the attacks themselves and untouched since 2026-08-03..06) with
# HarmBench's reported classifier, llama-2-13b-cls.
#
# this is the headline table. it deliberately does NOT use the 512-token
# regeneration: on the 7B that regeneration diverges from the original in 11/50
# completions and halves the judged ASR (12% -> 6%), which is regeneration noise,
# not a property of the model. the length question is settled separately and is
# worth ~1 point.
#
# writes results*/judge_scores_13b_orig.json. touches nothing else.
set -u
cd "$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"  # repo root
PY=./AdversariaLLM/.venv/bin/python
mkdir -p run_logs run_state

run_model() {
  local tag="$1" balcony="$2" baseline="$3" exits="$4" outdir="$5" template="$6"
  if [[ -f "run_state/judge13borig_${tag}.done" ]]; then echo "[$tag] done, skip"; return 0; fi
  echo "[$tag] judging original completions $(date -Is)"
  env BALCONY_MODEL="$balcony" BASELINE_MODEL="$baseline" \
      EXITS="$exits" RESULTS_DIR="$outdir" CHAT_TEMPLATE="$template" N_BEHAVIORS=50 \
      JUDGE_CLASSIFIER=llama-2-13b \
      COMPLETION_FIELD=own_exit_completion \
      COMPLETION_GLOB='*attacks*.json' \
      JUDGE_OUTPUT=judge_scores_13b_orig.json \
      "$PY" experiments/judge_harmfulness.py > "run_logs/judge13borig_${tag}.log" 2>&1 \
    && touch "run_state/judge13borig_${tag}.done" && echo "[$tag] ok $(date -Is)" \
    || { touch "run_state/judge13borig_${tag}.failed"; echo "[$tag] FAILED"; }
}

run_model 8b adpretko/Balcony-Meta-LLaMA3-8B-exit-15-18-21 NousResearch/Meta-Llama-3-8B "15 18 21 full" results_8b llama-3-balcony
run_model 7b parsakaveh/Balcony-LLaMA2-7B                  NousResearch/Llama-2-7b-hf   "15 18 21 full" results_7b llama-2-chat
run_model 1b parsakaveh/Balcony-LLM-1B                     parsakaveh/LLM-1B-Pretrained "4 8 12 full"   results     ""
echo "ALL DONE $(date -Is)"
