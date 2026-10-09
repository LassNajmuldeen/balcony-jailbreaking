# Jailbreaks across model depth

How does early exit change a language model's vulnerability to jailbreaks?
We study GCG attacks on Balcony-LLaMA2-7B and Balcony-LLaMA3-8B at layers
15, 18, 21, and full depth, using HarmBench and AdvBench.

TAO is a planned extension. Its implementation is experimental; we do not report
validated TAO results. The earlier 1B runs were development tests and are outside
the current study.

## Results

The paper evaluates GCG over 150 steps on 50 behaviors per benchmark.
The figures below are extracted directly from the paper. [Results and tables](RESULTS.md).

![Evaluation pipeline with early exits at layers 15, 18, 21, and full depth.](paper_figures/published/pipeline.png)

![GCG attack success rates by exit depth for the 7B and 8B models on HarmBench and AdvBench.](paper_figures/published/attack_success.png)

<details open>
<summary>Loss curves and loss versus attack success</summary>

![GCG loss over 150 steps, by model, benchmark, and exit depth.](paper_figures/published/loss_curves.png)

![Final GCG loss versus attack success rate on HarmBench.](paper_figures/published/loss_vs_asr.png)

</details>

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

## Repository

| Path | Contents |
| --- | --- |
| `adapter/` | Early-exit model code and wrappers |
| `experiments/` | Attack, evaluation, and analysis implementations |
| `scripts/` | Setup and the shared runner |
| `results*/` | Historical run artifacts; not a release of validated study results |
| `paper_figures/published/`, `paper_tables/published/` | Figures and tables extracted from the paper |

Other files in `paper_figures/` and `paper_tables/` are historical exports.

The Git submodules pin [AdversariaLLM](https://github.com/LLM-QC/AdversariaLLM),
[Balcony-LLaMA](https://github.com/benyaminjami/Balcony-LLaMA), and
[TAO-Attack](https://github.com/ZevineXu/TAO-Attack). Toolkit fixes are in `patches/`.
