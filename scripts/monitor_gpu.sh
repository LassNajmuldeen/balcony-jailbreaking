#!/usr/bin/env bash
# Sample GPU memory every INTERVAL seconds into run_logs/gpu_mem.tsv, with each
# resident process resolved to the SHARD it is running.
#
# Written 2026-08-09. Every failure in the TAO run was an overcommit -- three
# workers whose footprints summed past the 94.97 GiB card -- and diagnosing that
# after the fact meant parsing per-process figures out of OOM tracebacks, which
# only exist for the process that happened to lose the race. This records the
# whole card continuously so the next one is visible while it builds, not after.
#
#   ./monitor_gpu.sh                     # foreground
#   setsid nohup ./monitor_gpu.sh > /dev/null 2>&1 < /dev/null &
#   ./monitor_gpu.sh --report            # summarise what has been collected
#
# Columns: iso_time  total_mib  used_mib  free_mib  n_procs  pid:shard=mib,...

set -uo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
LOGS="$(pwd)/run_logs"
OUT="$LOGS/gpu_mem.tsv"
INTERVAL="${INTERVAL:-60}"
mkdir -p "$LOGS"

if [[ "${1:-}" == "--report" ]]; then
  [[ -f "$OUT" ]] || { echo "no samples yet at $OUT"; exit 0; }
  echo "samples: $(wc -l < "$OUT")   window: $(head -1 "$OUT" | cut -f1) .. $(tail -1 "$OUT" | cut -f1)"
  awk -F'\t' 'NR>1{u=$3+0; if(u>mx){mx=u; mt=$1; ml=$6} s+=u; n++}
    END{printf "peak used: %.1f GiB at %s\n  -> %s\nmean used: %.1f GiB over %d samples\n",
        mx/1024, mt, ml, (s/n)/1024, n}' "$OUT"
  echo
  echo "closest approaches to the card limit:"
  awk -F'\t' 'NR>1{print $4"\t"$1"\t"$6}' "$OUT" | sort -n | head -5 |
    awk -F'\t' '{printf "  free %6.2f GiB  %s  %s\n", $1/1024, $2, $3}'
  exit 0
fi

# pid -> shard name, by reading the shard tag out of the worker's own environment.
# /proc/<pid>/environ is NUL-separated; TAO_SHARD and RESULTS_DIR together identify
# the shard uniquely, and both are set by run_tao_parallel.sh's run_shard().
label_for() {
  local pid="$1" env shard results
  env="$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null)" || { echo "unknown"; return; }
  shard="$(grep -m1 '^TAO_SHARD=' <<< "$env" | cut -d= -f2-)"
  results="$(grep -m1 '^RESULTS_DIR=' <<< "$env" | cut -d= -f2-)"
  bench="$(grep -m1 '^BENCHMARK=' <<< "$env" | cut -d= -f2-)"
  if [[ -z "$shard" ]]; then
    # not a TAO worker: name it by argv so foreign jobs are still identifiable
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | awk '{print $NF}' | xargs -r basename
    return
  fi
  echo "${results:-?}/${bench:-?}/${shard}"
}

[[ -f "$OUT" ]] || printf 'time\ttotal_mib\tused_mib\tfree_mib\tn_procs\tprocs\n' > "$OUT"

while :; do
  read -r total used free < <(nvidia-smi --query-gpu=memory.total,memory.used,memory.free \
      --format=csv,noheader,nounits 2>/dev/null | tr -d ',' | head -1)
  [[ -z "${total:-}" ]] && { sleep "$INTERVAL"; continue; }

  procs=""; n=0
  while IFS=, read -r pid mib; do
    pid="${pid//[[:space:]]/}"; mib="${mib//[^0-9]/}"
    [[ -z "$pid" ]] && continue
    procs+="${procs:+,}${pid}:$(label_for "$pid")=${mib}"
    n=$((n+1))
  done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null)

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$total" "$used" "$free" "$n" "${procs:--}" >> "$OUT"
  sleep "$INTERVAL"
done
