#!/usr/bin/env bash
# full attack/eval pipeline for one model size (SETUP.md 8.3).
#
# two fixes vs the original version of this script:
#   1. a failing stage is now FATAL. the original caught failures and carried on,
#      which is exactly why the TAO crash went unnoticed for so long: the pipeline
#      "finished", produced the other outputs, and the plotting code silently
#      skipped the missing file (SETUP.md 9b).
#   2. it is parameterised by RESULTS_DIR / BALCONY_MODEL / BASELINE_MODEL / EXITS
#      instead of needing every experiments/*.py edited per model (SETUP.md 8.1).
#
# usage:
#   ./run_all.sh                                  # 1B, the defaults
#
#   RESULTS_DIR=results_7b \
#   BALCONY_MODEL=parsakaveh/Balcony-LLaMA2-7B \
#   BASELINE_MODEL=meta-llama/Llama-2-7b-hf \
#   EXITS="15 18 21 full" ./run_all.sh
#
#   N_BEHAVIORS=2 NUM_STEPS=10 RESULTS_DIR=results_pilot ./run_all.sh   # pilot

set -euo pipefail

REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
PYTHON="${PYTHON:-$REPO/AdversariaLLM/.venv/bin/python}"

# exported so experiments/config.py picks them up in every stage
export RESULTS_DIR="${RESULTS_DIR:-results}"
export BALCONY_MODEL="${BALCONY_MODEL:-parsakaveh/Balcony-LLM-1B}"
export BASELINE_MODEL="${BASELINE_MODEL:-parsakaveh/LLM-1B-Pretrained}"
export EXITS="${EXITS:-4 8 12 full}"

LOG_DIR="$REPO/${RESULTS_DIR}/logs"
mkdir -p "$LOG_DIR"

[ -x "$PYTHON" ] || { echo "FATAL: no interpreter at $PYTHON (see SETUP.md 5)" >&2; exit 1; }

echo "repo          : $REPO"
echo "python        : $PYTHON"
echo "balcony model : $BALCONY_MODEL"
echo "baseline model: $BASELINE_MODEL"
echo "exits         : $EXITS"
echo "results dir   : $RESULTS_DIR"
echo "n_behaviors   : ${N_BEHAVIORS:-50} (steps: ${NUM_STEPS:-150})"
echo

run() {
  local name="$1" script="$2"
  echo "=== START $name ($(date -Is)) ==="
  # pipefail makes the python exit status win over tee's, so a crash is fatal
  "$PYTHON" "$REPO/$script" 2>&1 | tee "$LOG_DIR/$name.log"
  echo "=== OK $name ($(date -Is)) ==="
}

run baseline_control      experiments/measure_clean_loss.py        # unattacked control
run phase2_gcg_harmbench  experiments/attack_gcg_harmbench.py # GCG, HarmBench, per-exit
run no_balcony_baseline   experiments/attack_gcg_baseline.py     # GCG, SEPARATE baseline model
run advbench_gcg          experiments/attack_gcg_advbench.py       # GCG, AdvBench, per-exit
run tao_harmbench         experiments/attack_tao.py              # TAO, HarmBench, per-exit
run judge_scoring         experiments/judge_harmfulness.py       # ASR for every set
run utility_eval          experiments/eval_capabilities.py
run utility_eval_attacked experiments/eval_capabilities_under_attack.py
run visualize             experiments/visualize_results.py

# --- output completeness check (SETUP.md 9b) --------------------------------
# the pipeline "succeeding" is not enough: verify each expected file exists and
# holds one entry per behavior for every exit, so a silently-short run is caught.
echo
echo "=== output completeness check ==="
REPO="$REPO" RESULTS_DIR="$RESULTS_DIR" EXITS="$EXITS" \
N_BEHAVIORS="${N_BEHAVIORS:-50}" "$PYTHON" - <<'PY'
import json, os, sys
from pathlib import Path

results = Path(os.environ["RESULTS_DIR"])
if not results.is_absolute():
    results = Path(os.environ["REPO"]) / results
exits = [t if t == "full" else int(t) for t in os.environ["EXITS"].split()]
n_expected = int(os.environ["N_BEHAVIORS"])

expected = {
    "phase2_attacks.json":     [str(e) for e in exits],
    "advbench_attacks.json":   [str(e) for e in exits],
    "no_balcony_attacks.json": ["no_balcony"],
    "tao_attacks.json":        [str(e) for e in exits],
    "baseline_control.json":   None,   # shape varies, presence-only
    "judge_scores.json":       None,
    "utility_eval.json":       None,
}

problems = []
for name, keys in expected.items():
    path = results / name
    if not path.exists():
        problems.append(f"MISSING  {name}")
        continue
    if keys is None:
        print(f"ok       {name}")
        continue
    data = json.load(open(path))
    bad = []
    for k in keys:
        if k not in data:
            bad.append(f"MISSING  {name} -> key {k!r}")
        elif len(data[k]) != n_expected:
            bad.append(f"SHORT    {name} -> key {k!r}: {len(data[k])}/{n_expected} behaviors")
    problems.extend(bad)
    if not bad:
        print(f"ok       {name}")

if problems:
    print("\nincomplete outputs:")
    for p in problems:
        print("  " + p)
    sys.exit(1)
print("\nall expected outputs present and complete.")
PY

echo
echo "=== PIPELINE COMPLETE ($RESULTS_DIR) ==="
