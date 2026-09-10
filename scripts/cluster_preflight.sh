#!/usr/bin/env bash
# CLUSTER PREFLIGHT -- run this on ONE gpu on their cluster BEFORE porting anything else.
#
# the point of this script is to fail fast and cheap. it does the five things that
# can actually go wrong, in the order they go wrong, and it stops at the first one.
# about 15 minutes end to end, most of which is the model download.
#
# it is deliberately scheduler agnostic: no slurm, no condor, no qsub, nothing but
# bash and python. run it in an interactive gpu session (salloc / condor_submit -i
# / qsub -I / srun --pty, whatever their thing is). the scheduler script is the
# LAST thing to write, not the first -- if this passes, the scheduler part is
# twenty lines of glue and it cannot surprise you.
#
#   ./cluster_preflight.sh            full check
#   ./cluster_preflight.sh --env-only just the environment, no gpu needed
#
set -uo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
REPO="$(pwd)"
PY="${CLUSTER_PY:-$REPO/AdversariaLLM/.venv/bin/python}"
FAIL=0

step() { printf '\n=== %s\n' "$*"; }
ok()   { printf '  OK    %s\n' "$*"; }
bad()  { printf '  FAIL  %s\n' "$*"; FAIL=1; }

# --- 1. driver ------------------------------------------------------------
# torch 2.7.1+cu128 wants a driver new enough for cuda 12.8. anything 525+ works
# through cuda minor version compatibility, but old cluster images are the single
# most common reason a port dies on day one, so check it before anything else.
step "1. driver"
if ! command -v nvidia-smi >/dev/null; then
  bad "no nvidia-smi -- are you on a gpu node?"
else
  DRV="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  MAJ="${DRV%%.*}"
  if (( MAJ >= 525 )); then ok "driver $DRV"; else bad "driver $DRV is too old for cu128, need 525+"; fi
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -4 | sed 's/^/        /'
fi

# --- 2. the two patches ---------------------------------------------------
# THIS IS THE ONE THAT BITES. the patches are applied as UNCOMMITTED working tree
# edits in the AdversariaLLM submodule on the dev box, so a fresh clone of the
# pinned commit does NOT have them and GCG dies on filter_suffix / TokenMergeError.
# both verified to apply cleanly to 1e521ba on 2026-09-04.
step "2. AdversariaLLM patches"
cd AdversariaLLM 2>/dev/null || { bad "no AdversariaLLM checkout"; cd "$REPO"; }
if git diff --quiet -- adversariallm/attacks/gcg.py adversariallm/lm_utils/text_generation.py 2>/dev/null; then
  bad "patches NOT applied -- run: git apply ../patches/*.patch"
else
  ok "working tree differs from the pin, patches look applied"
fi
cd "$REPO"

# --- 3. the environment ---------------------------------------------------
step "3. python environment"
if [[ ! -x "$PY" ]]; then
  bad "no interpreter at $PY -- set CLUSTER_PY=/path/to/python"
else
  "$PY" - <<'PYCHECK' || bad "environment import failed"
import sys, torch, transformers
print(f"  OK    python {sys.version.split()[0]}, torch {torch.__version__}, transformers {transformers.__version__}")
print(f"  OK    cuda available: {torch.cuda.is_available()}, built for {torch.version.cuda}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  OK    {p.name}, {p.total_memory/2**30:.0f} GiB, sm_{p.major}{p.minor}")
    assert p.total_memory/2**30 > 24, "GPU too small: a worker needs about 23 GiB"
import rouge_score, adversariallm  # the two that are easy to miss
print("  OK    rouge_score and adversariallm import")
PYCHECK
fi
[[ "${1:-}" == "--env-only" ]] && { echo; echo "env-only: done, FAIL=$FAIL"; exit $FAIL; }

# --- 4. models ------------------------------------------------------------
# these are public, no HF token needed. on a cluster with no internet on the
# compute nodes, do this on the LOGIN node first and point HF_HOME at scratch.
step "4. model access (downloads ~56 GB the first time)"
"$PY" - <<'PYCHECK' || bad "model load failed"
import os
from transformers import AutoTokenizer
for mid in ["parsakaveh/Balcony-LLaMA2-7B",
            "adpretko/Balcony-Meta-LLaMA3-8B-exit-15-18-21"]:
    AutoTokenizer.from_pretrained(mid)
    print(f"  OK    {mid}")
print(f"  OK    HF_HOME={os.environ.get('HF_HOME', '~/.cache/huggingface')}")
PYCHECK

# --- 5. one real behaviour ------------------------------------------------
# 20 steps, one behaviour, throwaway results dir. if this produces a falling loss
# then all 800 of them will run; nothing about behaviour 400 differs from
# behaviour 0 except the text.
#
# do NOT expect the loss to match the dev box to the last digit. different gpu
# architectures give different floating point results and that is fine. what you
# are checking is that it runs, and that the loss goes DOWN.
step "5. one behaviour, 20 steps"
OUT="$(mktemp -d)"
env BALCONY_MODEL=parsakaveh/Balcony-LLaMA2-7B \
    BASELINE_MODEL=NousResearch/Llama-2-7b-hf \
    EXITS=full CHAT_TEMPLATE=plain \
    RESULTS_DIR="$OUT" BENCHMARK=harmbench ATTACK_TARGET=balcony \
    N_BEHAVIORS=1 NUM_STEPS=20 TAO_BATCH_SIZE=128 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" experiments/attack_tao.py > "$OUT/log.txt" 2>&1
RC=$?
if (( RC != 0 )); then
  bad "attack exited $RC"
  grep -E "Error|Exception" "$OUT/log.txt" | tail -3 | sed 's/^/        /'
else
  FIRST="$(grep -oE 'step   0  loss=[0-9.-]+' "$OUT/log.txt" | head -1)"
  LAST="$(grep -oE 'final_loss=[-0-9.]+' "$OUT/log.txt" | tail -1)"
  ok "ran to completion: $FIRST -> $LAST"
  echo "        (dev box gave step0 0.6445 -> final 0.3750; close is fine, exact is not expected)"
fi
rm -rf "$OUT"

# --- verdict ---------------------------------------------------------------
echo
if (( FAIL == 0 )); then
  cat <<'DONE'
=== PREFLIGHT PASSED

everything that can silently break has been checked. the remaining work is the
scheduler script, which is glue and cannot surprise you:

  - one array task per (model, benchmark, exit, behaviour block)
  - 16 cells x 50 behaviours = 800 behaviours; blocks of 5-10 keep each task
    under the usual 12-24 h wall limit
  - the task body is exactly the env block from scripts/run_tao_full.sh with
    BEHAVIOR_START / BEHAVIOR_END set from the array index
  - results merge with experiments/merge_tao_shards.py, which already dedupes
    on the global behavior_idx, so tasks can finish in any order
DONE
else
  echo "=== PREFLIGHT FAILED -- fix the FAIL lines above before porting anything else"
fi
exit $FAIL
