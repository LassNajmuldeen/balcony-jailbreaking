#!/usr/bin/env bash
# Figures + result tables from whatever has finished so far.
#
# Safe to run at any time, including while run_matrix.sh is still going: it only
# reads the results json and skips model sizes that have not produced anything
# yet. Run it again later to pick up newly finished stages.
#
# Deliberately separate from run_matrix.sh rather than a stage inside it, so it
# can be run on demand without waiting for the queue, and so run_matrix.sh does
# not need editing while it is mid-run.
#
# writes:
#   results*/figures/*.png     per-model diagnostic figures
#   paper_tables/results_table.{md,csv,tex}   consolidated loss + ASR
#
# usage:
#   ./make_figures.sh            # every model with results
#   ./make_figures.sh 7b 8b      # just these

set -uo pipefail

REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
PYTHON="${PYTHON:-$REPO/AdversariaLLM/.venv/bin/python}"
[ -x "$PYTHON" ] || { echo "FATAL: no interpreter at $PYTHON" >&2; exit 1; }

declare -A OUTDIR MODEL_EXITS
OUTDIR[1b]="results";      MODEL_EXITS[1b]="4 8 12 full"
OUTDIR[7b]="results_7b";   MODEL_EXITS[7b]="15 18 21 full"
OUTDIR[8b]="results_8b";   MODEL_EXITS[8b]="15 18 21 full"

models=("$@")
[ ${#models[@]} -eq 0 ] && models=(1b 7b 8b)

made=()
for m in "${models[@]}"; do
  dir="$REPO/${OUTDIR[$m]}"
  if [ ! -d "$dir" ] || ! ls "$dir"/*attacks*.json >/dev/null 2>&1; then
    echo "SKIP $m: no results in ${OUTDIR[$m]} yet"
    continue
  fi
  echo "=== figures for $m (${OUTDIR[$m]}, exits ${MODEL_EXITS[$m]}) ==="
  if RESULTS_DIR="${OUTDIR[$m]}" EXITS="${MODEL_EXITS[$m]}" \
       "$PYTHON" "$REPO/experiments/visualize_results.py"; then
    made+=("$m")
    echo "    -> ${OUTDIR[$m]}/figures/"
  else
    echo "    !! figure generation failed for $m (continuing)"
  fi
  echo
done

echo "=== consolidated results table ==="
"$PYTHON" "$REPO/experiments/export_results.py"

echo
echo "figures generated for: ${made[*]:-none}"
echo "tables: $REPO/paper_tables/results_table.{md,csv,tex}"
