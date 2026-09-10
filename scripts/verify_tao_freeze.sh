#!/usr/bin/env bash
cd /path/to/project
CAL=/tmp/claude-1000/-home-deepevolve-adversarial-attack-experiments/2109b682-dc59-4a97-80a6-690b86c80d16/scratchpad/cal
PY=./AdversariaLLM/.venv/bin/python
one() {  # tag model baseline template batch
  rm -rf "$CAL/fz_$1"; mkdir -p "$CAL/fz_$1"
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 2; done ) > "$CAL/fzmem_$1.txt" &
  local MON=$!
  env BALCONY_MODEL="$2" BASELINE_MODEL="$3" EXITS=full CHAT_TEMPLATE="$4" \
      RESULTS_DIR="$CAL/fz_$1" BENCHMARK=harmbench ATTACK_TARGET=balcony \
      N_BEHAVIORS=1 NUM_STEPS=20 TAO_BATCH_SIZE="$5" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      $PY experiments/attack_tao.py > "$CAL/fzlog_$1.txt" 2>&1
  kill $MON 2>/dev/null
  echo "$1: peak=$(sort -n "$CAL/fzmem_$1.txt" | tail -1)MiB  $(grep -oE 'final_loss=[-0-9.]+ stage=[0-9]+ elapsed=[0-9.]+' "$CAL/fzlog_$1.txt" | tail -1)"
}
one 7b128 parsakaveh/Balcony-LLaMA2-7B NousResearch/Llama-2-7b-hf llama-2-chat 128
one 7b512 parsakaveh/Balcony-LLaMA2-7B NousResearch/Llama-2-7b-hf llama-2-chat 512
one 8b512 adpretko/Balcony-Meta-LLaMA3-8B-exit-15-18-21 NousResearch/Meta-Llama-3-8B llama-3-balcony 512
echo "--- BEFORE the freeze:"
echo "  7b128 peak=33792MiB final_loss=0.6172 elapsed=70.0"
echo "  7b512 peak=33652MiB final_loss=0.4570 elapsed=105.4"
echo "  8b512 peak=37654MiB final_loss=0.0195 elapsed=95.2"
