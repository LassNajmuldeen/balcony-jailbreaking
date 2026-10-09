# Commands

Two entry points: `setup.sh` installs the environment; `run.py` runs experiments
and analysis. Commands resolve paths from the repository, so they also work when
called from another directory.

## Experiments

```bash
.venv/bin/python scripts/run.py experiment --model 7b
.venv/bin/python scripts/run.py experiment --model 7b --stage gcg --plain --results-dir runs/7b-plain
```

| Stage | Work |
| --- | --- |
| `all` | Clean loss → GCG on both benchmarks → judge → clean and GCG-attacked utility |
| `gcg` | Attack the selected exits and benchmarks |
| `tao` | Experimental TAO extension; select explicitly |
| `control`, `judge` | Measure unattacked loss, or score saved completions |
| `utility`, `utility-attacked` | Evaluate ordinary tasks, clean or with an attack suffix |
| `regenerate`, `merge`, `transfer`, `verify-weights` | Replay completions, combine TAO shards, test cross-exit transfer, or compare checkpoint weights |

Active models are `7b` and `8b`. Choose `--exits 15 full` for a subset. Budgets default to 50 behaviors, 150 steps,
and 512 generated tokens. Use `--n-behaviors`, `--num-steps`, and
`--max-new-tokens` to change them.

New runs go to `runs/<model>/`; use a new `--results-dir` for different settings.
The runner stops on a failed stage and rejects incomplete attack outputs.
`--resume` skips completed attack stages with matching settings; TAO also saves
progress after each behavior. `--dry-run` prints commands without loading models.

For a separate baseline, pass `--target baseline --baseline-model <checkpoint>`.
Its default output directory is `runs/<model>-baseline/`.
Baseline verification also requires `--baseline-model`.

## Experimental TAO

TAO is a planned extension. The code can be exercised explicitly with
`--stage tao`; existing checkpoints do not establish validated study results.
For attacked utility with a TAO suffix, use `--stage utility-attacked --attack tao`.

## Paper figures and historical exports

Extract the paper figures and tables with Poppler installed:

```bash
python3 scripts/run.py figures --paper /path/to/paper.pdf
```

Crops go to `paper_figures/published/` and `paper_tables/published/`.
The PDF remains local; the source manifest records its hash and crop coordinates.
The crop layout is specific to the ten-page paper supplied for this study.

To inspect older exports:

```bash
.venv/bin/python scripts/run.py figures --archive
.venv/bin/python scripts/run.py export --archive
.venv/bin/python scripts/run.py export --archive --loss-curves
```

The `--archive` flag selects historical inputs. Figures go to `paper_figures/`;
tables and CSVs go to `paper_tables/`, outside the `published/` directories.
These commands reproduce historical artifacts, including the old 1B tests and
unvalidated TAO outputs. They do not regenerate the current study figures.
The optional loss-curve export uses the corrected template-free 7B reruns.
To plot or export a new run, set `RESULTS_DIR_1B`, `RESULTS_DIR_7B`, or
`RESULTS_DIR_8B` to its output directory. Every exporter uses the same mapping.

## Implementation

`experiments/config.py` holds shared settings and output names.
`attack_gcg.py` handles both benchmarks and the optional baseline; `attack_tao.py`
implements TAO with behavior checkpoints. Evaluation and plotting remain in
separate modules so their dependencies and numerical methods stay easy to inspect.
