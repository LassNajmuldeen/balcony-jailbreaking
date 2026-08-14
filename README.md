# adversarial-balcony

investigating adversarial robustness across early-exit points in a balcony-style language model. this is the experiment repo for our NeurIPS workshop paper.

## overview

balcony models add prediction heads ("exits") at intermediate transformer layers so the model can output predictions before all layers have finished processing. we attack each exit independently with GCG and TAO, measure how attack success varies with exit depth, and whether attacks transfer between exits.

**models**:
- `parsakaveh/Balcony-LLM-1B` -- 1B param SmolLM with exit heads at layers 4, 8, 12, and the full model output at layer 16
- `parsakaveh/LLM-1B-Pretrained` -- same architecture, no exit heads. all shared weights are bit-identical to the balcony model

**attacks**:
- GCG (Greedy Coordinate Gradient), 150 steps, 50 HarmBench behaviors
- TAO (Two-stage loss + Direction-Priority Token Optimization), 150 steps, 50 HarmBench behaviors

**benchmarks**:
- HarmBench (50 behaviors) -- standard adversarial robustness
- AdvBench (50 behaviors) -- additional adversarial benchmark
- Utility: hellaswag, arc_challenge, arc_easy, mmlu, winogrande, piqa (via lm-eval)

**metrics**: teacher-forced loss, HarmBench judge p_harmful, ASR% (attack success rate)

## repo structure

```
adapter/
  balcony_llama/                        # locally vendored model code (avoids trust_remote_code)
  exit_wrapper.py                       # wraps balcony model to expose one exit at a time

experiments/
  attack_gcg_harmbench.py           # main experiment: per-exit GCG attack + transfer matrix (HarmBench)
  attack_gcg_baseline.py                # same GCG attack on the pre-balcony baseline
  measure_clean_loss.py                   # pre-attack loss measurements (no suffix + filler)
  judge_harmfulness.py                  # HarmBench judge scoring + ASR% computation
  transfer_matrix_baseline.py                  # extends transfer matrix with the baseline model
  attack_gcg_advbench.py                  # GCG attack using AdvBench behaviors
  attack_tao.py                         # TAO attack on all exits (HarmBench)
  eval_capabilities.py                       # lm-eval benchmarks per exit (hellaswag, arc, mmlu, etc.)
  visualize_results.py                  # generates all figures from json results

results/                                # raw json data + generated figures
RESULTS.md                              # full writeup with tables and figures
```

## dependencies

clone these repos alongside this one (they are gitignored):

```
git clone https://github.com/parsakaveh/Balcony-LLaMA.git
git clone <adversariallm-repo-url> AdversariaLLM
git clone https://github.com/ZevineXu/TAO-Attack.git
```

install lm-eval in the AdversariaLLM venv for utility benchmarks:
```
uv pip install --python AdversariaLLM/.venv/bin/python lm-eval
```

## running experiments

all scripts use the AdversariaLLM venv. run with:
```bash
AdversariaLLM/.venv/bin/python experiments/<script>.py
```

```bash
# GCG attacks (HarmBench, 50 behaviors, 150 steps)
python experiments/attack_gcg_harmbench.py    # main attack + transfer matrix
python experiments/attack_gcg_baseline.py          # baseline attack
python experiments/measure_clean_loss.py             # pre-attack losses

# AdvBench attacks (50 behaviors, 150 steps)
python experiments/attack_gcg_advbench.py            # GCG on AdvBench behaviors

# TAO attack (HarmBench, 50 behaviors, 150 steps)
python experiments/attack_tao.py                   # TAO on all exits

# judge scoring + ASR
python experiments/judge_harmfulness.py            # judge scoring with ASR%

# utility benchmarks
python experiments/eval_capabilities.py                 # hellaswag, arc, mmlu, winogrande, piqa

# visualization
python experiments/visualize_results.py            # regenerate all figures
```

all experiments were run on a single NVIDIA A4000 (16GB VRAM), Ubuntu, CUDA 12.4.

## key results

see [RESULTS.md](RESULTS.md) for the full analysis with tables, figures, and ASR% breakdowns.
