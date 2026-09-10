# scripts/

Orchestration drivers. **Not tracked by git** (`.gitignore`: `scripts/`) — they
are machine- and run-specific, and several are one-off scheduling interventions
rather than program code.

Each script resolves the repo root by following its own symlink and going up one
level, so it works whether you call it as `./scripts/x.sh` or through a root shim:

```bash
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
```

## Pipeline drivers

| script | what it does |
| --- | --- |
| `run_all.sh` | full attack/eval pipeline for **one** model size (SETUP.md §8.3). Failure-fatal, with an output completeness check. |
| `run_matrix.sh` | multi-model driver over the 1B/7B/8B matrix (SETUP.md §8). |
| `run_tao_parallel.sh` | the TAO worker pool — GPU admission gate, per-shard resume, `--status`/`--dry-run`/`--failed`/`TAO_ONLY=`. |
| `make_figures.sh` | regenerates the figures from the result JSONs. |

## Diagnostics

| script | what it does |
| --- | --- |
| `monitor_gpu.sh` | samples GPU memory into `run_logs/gpu_mem.tsv`, resolving each resident process to the shard it is running. Written after the overcommit OOMs, which were only diagnosable from whichever process happened to lose the race. |

## Spent one-offs — kept for provenance, not for reuse

These record *how* specific numbers were produced. Their headers explain why each
was needed. None of them should be run again as-is.

| script | why it existed |
| --- | --- |
| `run_tao_relaunch.sh` | drives shards → merge → judge → utility after the 3-worker pool OOMed. Exists as a separate script only because bash reads a running script incrementally, so the live runner could not be edited. Spent once the current run drains. |
| `run_utility_then_tao.sh` | re-prioritised a live queue so the per-exit utility evals ran before TAO's ~9 days. The queue it reorders no longer exists. |
| `run_1b_advbench_baseline.sh` | filled the one GCG cell `run_matrix.sh` never covered (its priority loop was `for m in 7b 8b`). Already run — see `run_state/1b_gcg_advbench_baseline.done`. |

## Temporary root shims

`/run_tao_parallel.sh` and `/run_tao_relaunch.sh` are symlinks into this folder.
The in-flight `run_tao_relaunch.sh` (started before the move) calls
`./run_tao_parallel.sh` by relative path from the repo root at five sites, and it
still has merge → judge → utility ahead of it. **Delete both symlinks once the run
finishes** — nothing else depends on them.
