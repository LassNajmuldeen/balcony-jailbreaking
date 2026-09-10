#!/usr/bin/env bash
# builds a tarball with the MINIMUM needed to run the TAO attack somewhere else.
#
# the point: you do not clone this repo onto someone else's cluster. the cluster
# only needs to turn public models + public benchmarks into raw suffix json. all
# the parts that are actually yours -- results, RESULTS.md, the paper, local-notes,
# the judging, the figures -- stay here and never leave the box.
#
# what goes in (everything in here is either public already, or is attack code
# that reimplements a published attack):
#     experiments/attack_tao.py       the attack
#     experiments/config.py           env driven knobs
#     experiments/behavior_datasets.py
#     adapter/                        the balcony exit wrapper
#     chat_templates/*.jinja          load-bearing, see CLAUDE.md trap 2
#     data/                           harmbench + advbench behaviours and targets
#     run_one.sh                      runs ONE shard, scheduler agnostic
#     README_CLUSTER.md               setup, 4 steps
#
# what does NOT go in: results*/, RESULTS.md, local-notes/, paper_*, run_logs/,
# scripts/, patches applied to a private fork, anything with a finding in it.
#
#   ./make_cluster_bundle.sh              writes tao_cluster_bundle.tar.gz
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
REPO="$(pwd)"
STAGE="$(mktemp -d)"
B="$STAGE/tao_cluster"
mkdir -p "$B"/{experiments,adapter,chat_templates,data/behavior_datasets/extra_behavior_datasets,data/optimizer_targets/extra_targets}

cp experiments/attack_tao.py experiments/config.py experiments/behavior_datasets.py "$B/experiments/"
cp -r adapter/balcony_llama adapter/exit_wrapper.py adapter/__init__.py "$B/adapter/"
cp chat_templates/*.jinja "$B/chat_templates/"
cp AdversariaLLM/data/behavior_datasets/harmbench_behaviors_text_all.csv "$B/data/behavior_datasets/"
cp AdversariaLLM/data/behavior_datasets/extra_behavior_datasets/advbench_behaviors.csv "$B/data/behavior_datasets/extra_behavior_datasets/"
cp AdversariaLLM/data/optimizer_targets/harmbench_targets_text.json "$B/data/optimizer_targets/"
cp AdversariaLLM/data/optimizer_targets/extra_targets/advbench_targets.json "$B/data/optimizer_targets/extra_targets/"
cp patches/*.patch "$B/"
find "$B" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

# --- the one-shard runner -------------------------------------------------
cat > "$B/run_one.sh" <<'RUNONE'
#!/usr/bin/env bash
# runs ONE task. knows nothing about any scheduler -- drive it from a slurm array,
# a condor queue, a pbs job, or a bare for loop.
#
#   ./run_one.sh <7b|8b> <harmbench|advbench> <15|18|21|full> <start> <end>
#
# THERE IS NO SEPARATE SMOKE TEST. every task self checks its gpu and environment
# first, because on a shared queue you cannot sit at a prompt and try things. a
# task that fails its own check writes status/<task>.FAILED and exits non zero,
# and since every task is an INDEPENDENT job that costs you exactly one task, not
# the run. per behaviour failures inside a task are caught too (attack_tao.py) so
# behaviour 3 blowing up still leaves 4 and 5 attempted.
set -uo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
M="$1"; BENCH="$2"; EX="$3"; START="$4"; END="$5"
TASK="${M}_${BENCH}_e${EX}_b${START}-${END}"
PY="${CLUSTER_PY:-python}"
export DATA_ROOT="$(pwd)"
mkdir -p status logs "results/$M"

fail() { echo "PRECHECK FAIL: $*" | tee "status/${TASK}.FAILED"; exit 1; }

# --- self check, about 10 seconds -----------------------------------------
command -v "$PY" >/dev/null 2>&1 || [[ -x "$PY" ]] || fail "no interpreter at $PY (set CLUSTER_PY)"
"$PY" - <<'PRECHECK' || fail "environment or gpu check failed"
import sys
import torch, transformers, rouge_score, pandas  # noqa: F401
assert torch.cuda.is_available(), "no CUDA device visible to this task"
p = torch.cuda.get_device_properties(0)
gb = p.total_memory / 2**30
print(f"  precheck: {p.name}, {gb:.0f} GiB, sm_{p.major}{p.minor}, torch {torch.__version__}")
assert gb > 23, f"gpu has {gb:.0f} GiB, a worker needs about 23"
# actually touch the card. a device that enumerates but cannot allocate is a
# thing that happens on shared clusters, and finding out 4 hours in is worse
x = torch.zeros(1024, 1024, device="cuda"); del x
torch.cuda.empty_cache()
print("  precheck: allocation ok")
PRECHECK

case "$M" in
  7b) BAL=parsakaveh/Balcony-LLaMA2-7B;                     BASE=NousResearch/Llama-2-7b-hf;      TPL=plain ;;
  8b) BAL=adpretko/Balcony-Meta-LLaMA3-8B-exit-15-18-21;    BASE=NousResearch/Meta-Llama-3-8B;    TPL=llama-3-balcony ;;
  *)  fail "unknown model $M" ;;
esac

# --- the actual work -------------------------------------------------------
env BALCONY_MODEL="$BAL" BASELINE_MODEL="$BASE" EXITS="$EX" CHAT_TEMPLATE="$TPL" \
    RESULTS_DIR="results/$M" BENCHMARK="$BENCH" ATTACK_TARGET=balcony \
    TAO_SHARD="e${EX}.b${START}-${END}" \
    BEHAVIOR_START="$START" BEHAVIOR_END="$END" N_BEHAVIORS="$END" \
    NUM_STEPS="${NUM_STEPS:-500}" TAO_BATCH_SIZE="${TAO_BATCH_SIZE:-512}" \
    TAO_FORWARD_CHUNK="${TAO_FORWARD_CHUNK:-32}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" experiments/attack_tao.py 2>&1 | tee "logs/${TASK}.log"
RC="${PIPESTATUS[0]}"

if (( RC == 0 )); then
  : > "status/${TASK}.DONE"
  echo "TASK OK  $TASK"
else
  echo "rc=$RC" > "status/${TASK}.FAILED"
  echo "TASK FAIL $TASK rc=$RC -- see logs/${TASK}.log"
fi
# exit 0 on purpose: a nonzero exit makes some schedulers hold or requeue the
# whole array. the .DONE / .FAILED markers are the record, and status.sh reads them
exit 0
RUNONE
chmod +x "$B/run_one.sh"


# --- setup.sh: run once on a login node ------------------------------------
cat > "$B/setup.sh" <<'SETUP'
#!/usr/bin/env bash
# run ONCE on a login node. installs the environment and downloads the models.
# no gpu needed and nothing is queued -- this is setup, not a test.
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

echo "=== 1/3 python env"
python -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q torch==2.7.1 transformers==4.57.1 rouge-score pandas numpy accelerate

echo "=== 2/3 AdversariaLLM + the two required patches"
[[ -d AdversariaLLM ]] || git clone -q https://github.com/LLM-QC/AdversariaLLM.git
cd AdversariaLLM
git checkout -q 1e521ba
# REQUIRED. without these GCG dies on filter_suffix and TokenMergeError.
# verified to apply cleanly to this commit. || true so a re-run is not an error
git apply ../adversariallm-*.patch 2>/dev/null || echo "  (patches already applied)"
pip install -q -e .
cd ..

echo "=== 3/3 models, about 56 GB"
echo "  HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}"
echo "  if that is your NFS home, ctrl-C and re-run with HF_HOME=/scratch/you/hf"
huggingface-cli download parsakaveh/Balcony-LLaMA2-7B                  >/dev/null
huggingface-cli download adpretko/Balcony-Meta-LLaMA3-8B-exit-15-18-21 >/dev/null
huggingface-cli download NousResearch/Llama-2-7b-hf                    >/dev/null
huggingface-cli download NousResearch/Meta-Llama-3-8B                  >/dev/null

echo
echo "=== SETUP DONE"
echo "now submit tasks.txt (160 lines). use CLUSTER_PY=$(pwd)/.venv/bin/python"
SETUP
chmod +x "$B/setup.sh"

# --- status.sh: what worked, what to retry ---------------------------------
cat > "$B/status.sh" <<'STATUS'
#!/usr/bin/env bash
# what finished, what failed, and what to resubmit. safe to run any time.
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
total=$(wc -l < tasks.txt)
done_n=$(ls status/*.DONE 2>/dev/null | wc -l)
fail_n=$(ls status/*.FAILED 2>/dev/null | wc -l)
printf 'done=%s failed=%s pending=%s (of %s tasks)\n' \
       "$done_n" "$fail_n" "$(( total - done_n - fail_n ))" "$total"

if (( fail_n )); then
  echo; echo "failures:"
  for f in status/*.FAILED; do printf '  %-40s %s\n' "$(basename "${f%.FAILED}")" "$(head -1 "$f")"; done
fi

# rebuild a task list of everything not DONE, so a resubmit only redoes the gaps.
# behaviours already finished inside a partly done task are skipped by the
# checkpoint in attack_tao.py, so resubmitting is cheap and never duplicates work.
: > retry.txt
n=0
while read -r m b e s t; do
  n=$((n+1))
  [[ -f "status/${m}_${b}_e${e}_b${s}-${t}.DONE" ]] || sed -n "${n}p" tasks.txt >> retry.txt
done < tasks.txt
echo; echo "retry.txt written: $(wc -l < retry.txt) task(s) still to do"

echo; echo "behaviours that failed INSIDE a task (task survived, these did not):"
grep -ho '"error": "[^"]*"' results/*/tao_shards/*.json 2>/dev/null | sort | uniq -c | sed 's/^/  /' || echo "  none"
STATUS
chmod +x "$B/status.sh"

# --- the task list --------------------------------------------------------
# 800 behaviours as 160 tasks of 5. one line per task, so ANY scheduler can drive
# it: array index -> sed -n "${i}p" tasks.txt -> xargs ./run_one.sh
{
  for bench in harmbench advbench; do
    for ex in 15 18 21 full; do
      for m in 8b 7b; do
        for (( s = 0; s < 50; s += 5 )); do
          echo "$m $bench $ex $s $(( s + 5 ))"
        done
      done
    done
  done
} > "$B/tasks.txt"

cat > "$B/README_CLUSTER.md" <<'CLUSTERDOC'
# TAO attack, portable bundle

Turns public model weights + public benchmarks into raw adversarial-suffix JSON.
No analysis, no judging, no results from the main project are in here.

## Setup, 4 steps

1. **Environment** (once)

       python -m venv .venv && . .venv/bin/activate
       pip install torch==2.7.1 transformers==4.57.1 rouge-score pandas numpy
       git clone https://github.com/LLM-QC/AdversariaLLM.git
       cd AdversariaLLM && git checkout 1e521ba && git apply ../*.patch && pip install -e . && cd ..

   The two patches are REQUIRED. Without them GCG crashes on `filter_suffix` and
   `TokenMergeError`. They apply cleanly to `1e521ba`.

2. **Models** (once, on a login node if compute nodes have no internet)

       export HF_HOME=/your/scratch/hf          # NOT your NFS home, 56 GB
       huggingface-cli download parsakaveh/Balcony-LLaMA2-7B
       huggingface-cli download adpretko/Balcony-Meta-LLaMA3-8B-exit-15-18-21

   Both are public, no token needed. On compute nodes set `HF_HUB_OFFLINE=1`.

3. **Smoke test** one task, about 4 minutes

       NUM_STEPS=20 ./run_one.sh 7b harmbench full 0 1

   The loss should fall. It will NOT match another machine to the last digit --
   different GPUs give different floating point results, that is fine.

4. **Queue it.** `tasks.txt` holds 160 tasks of 5 behaviours each (~4 h per task
   at 500 steps, comfortably inside a 12 h wall limit). Task N is:

       read -r m b e s t <<< "$(sed -n "${N}p" tasks.txt)"; ./run_one.sh $m $b $e $s $t

   Any scheduler can drive that. One GPU per task, **23 GB** each, no
   inter-task communication -- they are fully independent and can finish in any
   order.

## Sending results back

`results/7b/tao_shards/*.json` and `results/8b/tao_shards/*.json`. That is all
that needs to come back; nothing else in this directory changes.
CLUSTERDOC

tar -czf "$REPO/tao_cluster_bundle.tar.gz" -C "$STAGE" tao_cluster
rm -rf "$STAGE"
echo "wrote $REPO/tao_cluster_bundle.tar.gz  ($(du -h "$REPO/tao_cluster_bundle.tar.gz" | cut -f1))"
echo "contents:"
tar -tzf "$REPO/tao_cluster_bundle.tar.gz" | grep -v '/$' | sed 's/^/  /' | head -20
