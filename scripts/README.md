# Commands

Two entry points: `setup.sh` installs the environment; `run.py` runs experiments
and analysis. Commands resolve paths from the repository, so they also work when
called from another directory.

## Experiments

```bash
.venv/bin/python scripts/run.py experiment --model 7b
.venv/bin/python scripts/run.py experiment --model 8b --stage tao --benchmarks advbench
.venv/bin/python scripts/run.py experiment --model 7b --stage gcg --plain --results-dir runs/7b-plain
```

| Stage | Work |
| --- | --- |
| `all` | Clean loss → GCG and TAO on both benchmarks → judge → clean and attacked utility |
| `gcg`, `tao` | Attack the selected exits and benchmarks |
| `control`, `judge` | Measure unattacked loss, or score saved completions |
| `utility`, `utility-attacked` | Evaluate ordinary tasks, clean or with an attack suffix |
| `regenerate`, `merge`, `transfer`, `verify-weights` | Replay completions, combine TAO shards, test cross-exit transfer, or compare checkpoint weights |

Choose `--exits 15 full` for a subset. Budgets default to 50 behaviors, 150 steps,
and 512 generated tokens. Use `--n-behaviors`, `--num-steps`, and
`--max-new-tokens` to change them.

New runs go to `runs/<model>/`; use a new `--results-dir` for different settings.
The runner stops on a failed stage and rejects incomplete attack outputs.
`--resume` skips completed attack stages with matching settings; TAO also saves
progress after each behavior. `--dry-run` prints commands without loading models.

For a separate baseline, pass `--target baseline --baseline-model <checkpoint>`.
Its default output directory is `runs/<model>-baseline/`.
Baseline verification also requires `--baseline-model`.

## Figures and exports

```bash
.venv/bin/python scripts/run.py figures
.venv/bin/python scripts/run.py export
.venv/bin/python scripts/run.py export --loss-curves
```

Figures go to `paper_figures/`; tables and CSVs go to `paper_tables/`.
These commands read the archived data, including its original prompt variants.
The optional loss-curve export uses the corrected template-free 7B reruns.
To plot or export a new run, set `RESULTS_DIR_1B`, `RESULTS_DIR_7B`, or
`RESULTS_DIR_8B` to its output directory. Every exporter uses the same mapping.

## Implementation

`experiments/config.py` holds shared settings and output names.
`attack_gcg.py` handles both benchmarks and the optional baseline; `attack_tao.py`
implements TAO with behavior checkpoints. Evaluation and plotting remain in
separate modules so their dependencies and numerical methods stay easy to inspect.
