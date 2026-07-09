# adversarial-balcony

investigating adversarial robustness across early-exit points in a balcony-style language model. this is the experiment repo for our NeurIPS workshop paper.

## overview

balcony models add prediction heads ("exits") at intermediate transformer layers so the model can output predictions before all layers have finished processing. we attack each exit independently with GCG and measure how attack success varies with exit depth, and whether attacks transfer between exits.

**models**:
- `parsakaveh/Balcony-LLM-1B` -- 1B param SmolLM with exit heads at layers 4, 8, 12, and the full model output at layer 16
- `parsakaveh/LLM-1B-Pretrained` -- same architecture, no exit heads. all shared weights are bit-identical to the balcony model

**attack**: GCG (Greedy Coordinate Gradient), 100 steps, 6 HarmBench behaviors, evaluated with both teacher-forced loss and HarmBench Mistral-7B judge scoring.

## repo structure

```
adapter/
  balcony_llama/                        # locally vendored model code (avoids trust_remote_code)
  exit_wrapper.py                       # wraps balcony model to expose one exit at a time

experiments/
  phase2_per_exit_transfer.py           # main experiment: per-exit GCG attack + transfer matrix
  no_balcony_baseline.py                # same attack on the pre-balcony baseline
  baseline_control.py                   # pre-attack loss measurements (no suffix + filler)
  judge_completions.py                  # HarmBench judge scoring on completions
  transfer_baseline.py                  # extends transfer matrix with the baseline model
  eval_utility.py                       # lm-eval benchmarks per exit (hellaswag, arc, mmlu, etc.)
  visualize_results.py                  # generates all figures from json results

results/                                # raw json data + generated figures
RESULTS.md                              # full writeup with tables and figures
```

## dependencies

clone these two repos alongside this one (they are gitignored):

```
git clone https://github.com/parsakaveh/Balcony-LLaMA.git
git clone <adversariallm-repo-url> AdversariaLLM
```

## running experiments

```bash
python experiments/phase2_per_exit_transfer.py    # main attack + transfer matrix
python experiments/no_balcony_baseline.py          # baseline attack
python experiments/baseline_control.py             # pre-attack losses
python experiments/judge_completions.py            # judge scoring
python experiments/visualize_results.py            # regenerate all figures
python experiments/eval_utility.py                 # utility benchmarks (requires lm-eval)
```

all experiments were run on a single NVIDIA A4000 (16GB VRAM), Ubuntu, CUDA 12.4.

## key results

| exit | GCG-attacked loss | judge score (p_harmful) |
|---|---|---|
| baseline (no balconies) | 0.246 | 0.99 |
| exit 4 (shallowest) | 0.578 | 0.19 |
| exit 8 | 0.491 | 0.32 |
| exit 12 | 0.334 | 0.78 |
| full (deepest) | 0.194 | 0.92 |

shallow exits resist jailbreaking not by resisting the optimization, but because they lack the coherence to follow through on harmful content during free generation. see [RESULTS.md](RESULTS.md) for the full analysis.
