#!/usr/bin/env bash
# recovery chain for the 2026-08-11 512-token regeneration.
#
# the first run_regen_512.sh launched with a load_model() that called
# wrap_for_single_exit on the BASELINE checkpoints too, which is a hard
# AttributeError on stock HF llama (see attack_gcg_baseline.py:72-79). That is
# fixed, but the already-running process holds the old code in memory, so it will
# still fail on every baseline arm it reaches.
#
# so: wait for it to exit, clear the failure markers, and re-run. the regen now
# resumes per-file (complete_output()), so the balcony arms it DID finish are not
# recomputed -- only the baseline arms are left. then hand off to judging.
set -u

cd "$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"  # repo root
mkdir -p run_logs run_state

echo "=== recovery chain started $(date -Is) ==="

# 1. wait for the in-flight regen to exit on its own. don't kill it: it is still
#    producing usable balcony-arm output for the 7B and 1B.
while pgrep -f 'run_regen_512\.sh$' > /dev/null; do
  sleep 30
done
echo "first regen pass exited $(date -Is)"

# 2. clear failure markers so the re-run reattempts those models. .done markers
#    are left alone -- a model that genuinely finished stays finished.
rm -f run_state/regen512_*.failed
rm -f run_state/judge512_*.failed

# 3. re-run. resumes; only unfinished files cost anything.
echo "=== relaunching regen $(date -Is) ==="
./run_regen_512.sh >> run_logs/regen512_console.log 2>&1

if ls run_state/regen512_*.failed > /dev/null 2>&1; then
  echo "ABORT: regen still failing after fix -- see run_logs/regen512_*.log"
  exit 1
fi

# 4. judge. this script waits on the .done markers itself, which are all set now.
echo "=== launching judging $(date -Is) ==="
./run_judge_512.sh >> run_logs/judge512_console.log 2>&1

echo "=== recovery chain complete $(date -Is) ==="
