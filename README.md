# Jailbreaks across model depth

How does early exit change a language model's vulnerability to jailbreaks?
We study GCG attacks on Balcony-LLaMA2-7B and Balcony-LLaMA3-8B at layers
15, 18, 21, and full depth, using HarmBench and AdvBench.

TAO is a planned extension. Its implementation is experimental; we do not report
validated TAO results. The earlier 1B runs were development tests and are outside
the current study.

## Run

Follow [setup](SETUP.md), then start with a small GCG run:

```bash
.venv/bin/python scripts/run.py experiment --model 7b --stage gcg \
  --benchmarks harmbench --n-behaviors 2 --num-steps 5 --results-dir runs/pilot
```

The default pipeline runs GCG, judging, and utility evaluation for one model:

```bash
.venv/bin/python scripts/run.py experiment --model 7b
```

New outputs go to `runs/<model>/`. Use `--dry-run` to inspect commands.
[Command reference](scripts/README.md).

## Results

The old figures have been removed from this README while the current plots and
their data sources are reconciled. [Result status](RESULTS.md).

## Repository

| Path | Contents |
| --- | --- |
| `adapter/` | Early-exit model code and wrappers |
| `experiments/` | Attack, evaluation, and analysis implementations |
| `scripts/` | Setup and the shared runner |
| `results*/` | Historical run artifacts; not a release of validated study results |
| `paper_figures/`, `paper_tables/` | Historical exports, superseded by the current analysis |

The Git submodules pin [AdversariaLLM](https://github.com/LLM-QC/AdversariaLLM),
[Balcony-LLaMA](https://github.com/benyaminjami/Balcony-LLaMA), and
[TAO-Attack](https://github.com/ZevineXu/TAO-Attack). Toolkit fixes are in `patches/`.
