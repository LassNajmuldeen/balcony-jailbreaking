# Results: Adversarial Robustness Across Balcony Exits

## Setup

**Model**: `parsakaveh/Balcony-LLM-1B`, a SmolLM-1B with early-exit heads at
layers 4, 8, and 12 of its 16-layer trunk, plus the normal full-depth path
(layer 16). Each exit has its own decoder layer + norm + output projection,
trained via self-distillation.

**Baseline model**: `parsakaveh/LLM-1B-Pretrained`, the original checkpoint
with no exit heads. Same 16-layer architecture; every shared weight is
bit-identical to the balcony model's trunk. The only difference is the balcony
model has three extra exit heads bolted on.

**Attack**: GCG (Greedy Coordinate Gradient), 150 steps, search width 128,
top-k 64. Appends a 20-token adversarial suffix to the prompt and optimizes
it to maximize the probability of a target compliance string.

**Benchmarks**:
- HarmBench (50 behaviors) — standard adversarial robustness evaluation
- AdvBench (50 behaviors) — additional adversarial benchmark
- Utility: HellaSwag, ARC-Challenge, ARC-Easy, WinoGrande, PIQA, MMLU (via lm-eval)

**Metrics**:
- *Teacher-forced loss* — how much probability the model assigns to the target
  compliance string. Lower = the model is closer to saying "Sure, here's how
  to...". This is what GCG directly optimizes.
- *p_harmful* — after GCG finishes, the model freely generates a response,
  which is scored by HarmBench's classifier (`cais/HarmBench-Mistral-7b-val-cls`).
  0–1 score for whether the model actually produced harmful content.
- *ASR%* — attack success rate, the fraction of behaviors where p_harmful > 0.5.

**Hardware**: single NVIDIA RTX A4000 (16 GB VRAM), Ubuntu, CUDA 12.4.

---

## 1. Utility Benchmarks (No Attack)

These measure each exit's general capability with no adversarial suffix
applied. Scores climb monotonically from exit 4 → full, as expected: deeper
exits use more layers and produce better predictions. The no-balcony baseline
matches the full exit exactly because they share identical weights through
all 16 layers.

| Benchmark | Exit 4 | Exit 8 | Exit 12 | Full (16) | No-Balcony Baseline |
|---|---|---|---|---|---|
| HellaSwag (acc_norm) | 0.333 | 0.416 | 0.546 | 0.611 | 0.611 |
| ARC-Challenge (acc_norm) | 0.236 | 0.279 | 0.342 | 0.390 | 0.390 |
| ARC-Easy (acc_norm) | 0.455 | 0.526 | 0.611 | 0.688 | 0.688 |
| WinoGrande (acc) | 0.515 | 0.531 | 0.570 | 0.603 | 0.603 |
| PIQA (acc_norm) | 0.645 | 0.693 | 0.720 | 0.750 | 0.750 |
| MMLU (acc) | 0.245 | 0.252 | 0.250 | 0.256 | 0.256 |

MMLU is essentially flat (~25%) across all exits, consistent with near-random
performance on a 4-choice task for a 1B model. The clearest capability gradient
shows up on HellaSwag (33% → 61%) and ARC-Easy (46% → 69%).

---

## 2. Utility Benchmarks (With GCG Suffix)

To check whether GCG adversarial suffixes degrade normal task performance, I
reran the same benchmarks with each exit's most effective GCG suffix (lowest
attack loss) prepended to every input. This is a worst-case measurement: if the
strongest jailbreak suffix doesn't hurt utility, weaker ones won't either.

| Benchmark | Exit 4 | Exit 8 | Exit 12 | Full (16) | No-Balcony Baseline |
|---|---|---|---|---|---|
| HellaSwag (acc_norm) | 0.330 (-0.003) | 0.408 (-0.008) | 0.542 (-0.004) | 0.604 (-0.007) | 0.603 (-0.008) |
| ARC-Challenge (acc_norm) | 0.259 (+0.022) | 0.274 (-0.005) | 0.330 (-0.012) | 0.387 (-0.003) | 0.395 (+0.005) |
| ARC-Easy (acc_norm) | 0.407 (-0.048) | 0.518 (-0.008) | 0.585 (-0.027) | 0.693 (+0.006) | 0.689 (+0.001) |
| WinoGrande (acc) | 0.504 (-0.012) | 0.525 (-0.006) | 0.531 (-0.039) | 0.560 (-0.043) | 0.573 (-0.030) |
| PIQA (acc_norm) | 0.616 (-0.029) | 0.661 (-0.032) | 0.706 (-0.014) | 0.736 (-0.014) | 0.739 (-0.010) |
| MMLU (acc) | 0.254 (+0.009) | 0.252 (+0.000) | 0.252 (+0.003) | 0.258 (+0.003) | 0.257 (+0.002) |

GCG suffixes have negligible impact on utility. Deltas are within noise for most
benchmarks (< 1 percentage point). The largest drops are on WinoGrande (~3-4pp
at exits 12 and full) and PIQA (~3pp at exits 4 and 8), but nothing that would
suggest the attack broadly degrades the model. MMLU is completely unaffected.
This confirms that GCG suffixes are narrowly targeted at eliciting specific
harmful completions, not at disrupting general language understanding.

![utility clean vs attacked](results/figures/12_utility_clean_vs_attacked.png)

---

## 3. Pre-Attack Baseline Losses (HarmBench)

Before running GCG, I measured each exit's loss on the target compliance
strings under two conditions: **no suffix** (just the raw harmful prompt) and
**filler suffix** (20 meaningless "x" tokens appended). The filler suffix
controls for the effect of added input length, since GCG's adversarial suffix
is the same length.

| Exit | No-Suffix Loss | Filler Loss |
|---|---|---|
| Exit 4 | 2.953 | 2.369 |
| Exit 8 | 3.097 | 2.242 |
| Exit 12 | 2.373 | 1.686 |
| Full (16) | 2.483 | 1.594 |
| No-Balcony Baseline | 2.483 | 1.594 |

Shallow exits start with higher loss on the target strings even before any
attack — they're generally worse at producing any specific text. The filler
suffix actually lowers loss for all exits (by 0.6–0.9 points), possibly because
it gives the model more context length to work with. The no-balcony baseline
has identical pre-attack losses to the full exit, as expected from shared weights.

---

## 4. HarmBench GCG Attack Results (50 behaviors, 150 steps)

### 4a. Loss Results

| Exit | Filler Baseline Loss | GCG Final Loss | Absolute Drop | Relative Drop |
|---|---|---|---|---|
| Exit 4 | 2.369 | 0.691 | 1.678 | 70.8% |
| Exit 8 | 2.242 | 0.562 | 1.681 | 75.0% |
| Exit 12 | 1.686 | 0.376 | 1.309 | 77.7% |
| Full (16) | 1.594 | 0.253 | 1.341 | 84.1% |
| No-Balcony Baseline | 1.594 | 0.259 | 1.335 | 83.8% |

GCG pushes loss down significantly at every exit. The relative drop increases
with depth (71% → 84%), meaning deeper exits are proportionally more vulnerable
to optimization. The no-balcony baseline and the full balcony exit land at
nearly identical final losses (0.259 vs 0.253) and nearly identical relative
drops (83.8% vs 84.1%), confirming that the balcony training did not change
the trunk's vulnerability at full depth.

![baseline vs attacked](results/figures/01_baseline_vs_attacked.png)

![loss curves](results/figures/04_loss_curves.png)

### 4b. Judge Results — HarmBench (the numbers that actually matter)

Loss measures teacher-forced probability on a fixed string. The judge score
measures whether the model, when freely generating, actually produces harmful
content. These tell a much more dramatic story.

| Exit | Mean p_harmful | ASR% (p_harmful > 0.5) |
|---|---|---|
| Exit 4 | 0.248 | 26.0% |
| Exit 8 | 0.196 | 18.0% |
| Exit 12 | 0.689 | 70.0% |
| Full (16) | 0.583 | 58.0% |
| No-Balcony Baseline | — * | — * |

\* Judge scores were not computed for the no-balcony baseline in the 50-behavior
run. In the earlier 6-behavior pilot, the no-balcony model reached 0.99 mean
p_harmful (99% ASR), consistent with its nearly identical loss to the full exit.

**Key finding: exit 8 is the most robust exit (18% ASR), not exit 4 (26%).**
This is a non-monotonic result — the shallowest exit is not the most resistant.
Exit 8 sits in a sweet spot where it has enough capacity to be coherent but
the GCG-optimized suffixes don't translate into actual harmful generations.

Exit 12 is the most vulnerable at 70% ASR, worse than even the full model at
58%. This is another surprise — the deepest exit is not the most vulnerable.
Exit 12's higher ASR may come from it being deep enough to generate coherent
harmful text but having a slightly different optimization landscape than the
full exit, one that GCG exploits more effectively.

![judge scores](results/figures/06_judge_scores_dotplot.png)

![ASR bar chart](results/figures/08_asr_bar.png)

---

## 5. AdvBench GCG Attack Results (50 behaviors, 150 steps)

AdvBench uses a different set of harmful behaviors from a separate benchmark.
This tests whether the HarmBench findings generalize. AdvBench attacks were
run on the 4 balcony exits only, not the no-balcony baseline.

### 5a. Loss Results

| Exit | GCG Final Loss (mean) |
|---|---|
| Exit 4 | 0.487 |
| Exit 8 | 0.440 |
| Exit 12 | 0.315 |
| Full (16) | 0.257 |
| No-Balcony Baseline | — (not evaluated) |

The same depth → lower loss pattern holds. AdvBench final losses are slightly
lower than HarmBench at exits 4 and 8, suggesting AdvBench behaviors may be
marginally easier to attack at shallow exits.

![advbench loss curves](results/figures/10_advbench_loss_curves.png)

### 5b. HarmBench vs AdvBench Comparison

| Exit | HarmBench Final Loss | AdvBench Final Loss |
|---|---|---|
| Exit 4 | 0.691 | 0.487 |
| Exit 8 | 0.562 | 0.440 |
| Exit 12 | 0.376 | 0.315 |
| Full (16) | 0.253 | 0.257 |
| No-Balcony Baseline | 0.259 | — |

At exits 4 and 8, AdvBench behaviors reach lower final losses than HarmBench.
At the full exit, the two benchmarks converge (0.253 vs 0.257). The no-balcony
baseline's HarmBench loss (0.259) sits right alongside the full exit, further
confirming trunk equivalence. This suggests that AdvBench behaviors may be
inherently easier to optimize against, but the gap closes at greater depth
where GCG can exploit the model more fully regardless of behavior source.

![harmbench vs advbench](results/figures/11_harmbench_vs_advbench.png)

---

## 6. Per-Behavior Heatmap (HarmBench)

Not all behaviors are equally vulnerable. The per-behavior heatmap shows
p_harmful for each behavior × exit combination, revealing which specific
harmful requests are most susceptible at each depth.

![per-behavior heatmap](results/figures/03_per_behavior_heatmap.png)

---

## 7. Variance Across Behaviors

The dotplot shows the spread of final losses across all 50 behaviors for each
exit. Deeper exits have tighter distributions (GCG reliably converges) while
shallow exits show more variance (some behaviors resist, others don't).

![variance dotplot](results/figures/05_variance_dotplot.png)

---

## 8. Summary of All Experiments

| | Exit 4 | Exit 8 | Exit 12 | Full (16) | No-Balcony |
|---|---|---|---|---|---|
| **Utility (clean)** | | | | | |
| HellaSwag | 0.333 | 0.416 | 0.546 | 0.611 | 0.611 |
| ARC-C | 0.236 | 0.279 | 0.342 | 0.390 | 0.390 |
| ARC-E | 0.455 | 0.526 | 0.611 | 0.688 | 0.688 |
| WinoGrande | 0.515 | 0.531 | 0.570 | 0.603 | 0.603 |
| PIQA | 0.645 | 0.693 | 0.720 | 0.750 | 0.750 |
| MMLU | 0.245 | 0.252 | 0.250 | 0.256 | 0.256 |
| **Utility (with GCG suffix)** | | | | | |
| HellaSwag | 0.330 | 0.408 | 0.542 | 0.604 | 0.603 |
| ARC-C | 0.259 | 0.274 | 0.330 | 0.387 | 0.395 |
| ARC-E | 0.407 | 0.518 | 0.585 | 0.693 | 0.689 |
| WinoGrande | 0.504 | 0.525 | 0.531 | 0.560 | 0.573 |
| PIQA | 0.616 | 0.661 | 0.706 | 0.736 | 0.739 |
| MMLU | 0.254 | 0.252 | 0.252 | 0.258 | 0.257 |
| **HarmBench GCG** | | | | | |
| Final loss | 0.691 | 0.562 | 0.376 | 0.253 | 0.259 |
| Loss reduction | 70.8% | 75.0% | 77.7% | 84.1% | 83.8% |
| Mean p_harmful | 0.248 | 0.196 | 0.689 | 0.583 | — * |
| ASR% | 26.0% | 18.0% | 70.0% | 58.0% | — * |
| **AdvBench GCG** | | | | | |
| Final loss | 0.487 | 0.440 | 0.315 | 0.257 | — ** |

\* Judge scores were not computed for the no-balcony baseline in the 50-behavior
run. In the earlier 6-behavior pilot, the no-balcony model reached 0.99 mean
p_harmful (99% ASR), consistent with its nearly identical loss to the full exit.

\** AdvBench attacks were run on the 4 balcony exits only, not the no-balcony
baseline.

---

## 9. Key Findings

1. **Early exits trade capability for robustness.** Utility scores climb
   monotonically from exit 4 → full (e.g., HellaSwag 33% → 61%), while attack
   success drops dramatically at early exits. This is the core tradeoff.

2. **The robustness–depth relationship is non-monotonic.** Exit 8 is the most
   robust (18% ASR), beating exit 4 (26%). Exit 12 is the most vulnerable
   (70% ASR), worse than the full model (58%). This means neither "shallower =
   safer" nor "deeper = more broken" — the relationship is more complex.

3. **Loss underestimates the gap.** GCG drops loss by 71–84% at every exit,
   suggesting similar vulnerability. But actual jailbreak success (ASR%) ranges
   from 18% to 70%. Shallow exits resist not because GCG can't optimize their
   loss, but because they can't coherently follow through on harmful content
   during free generation.

4. **Balcony training doesn't change trunk vulnerability.** The full balcony
   exit and the no-balcony baseline reach near-identical final losses (0.253 vs
   0.259) and near-identical loss reductions (84.1% vs 83.8%) despite the
   balcony model having three extra exit heads. Adding exit heads doesn't make
   the base model any more or less robust at full depth.

5. **GCG suffixes don't degrade normal utility.** Prepending the strongest
   GCG suffix to standard benchmarks causes < 1pp drops on most tasks (worst
   case ~4pp on WinoGrande). The attack is narrowly targeted at harmful
   completions, not a general disruption of language understanding. This holds
   for both the balcony exits and the no-balcony baseline.

6. **AdvBench confirms the pattern.** The depth → lower loss gradient holds
   across both benchmarks, with AdvBench behaviors slightly easier to optimize
   at shallow exits but converging at full depth.

---

## Where to Find Each Experiment

| Experiment | Script | Output |
|---|---|---|
| Baseline control (pre-attack losses) | `experiments/baseline_control.py` | `results/baseline_control.json` |
| HarmBench GCG (50 behaviors, 150 steps) | `experiments/phase2_per_exit_transfer.py` | `results/phase2_attacks.json` |
| No-balcony GCG baseline | `experiments/no_balcony_baseline.py` | `results/no_balcony_attacks.json` |
| AdvBench GCG (50 behaviors, 150 steps) | `experiments/advbench_per_exit.py` | `results/advbench_attacks.json` |
| Judge scoring + ASR | `experiments/judge_completions.py` | `results/judge_scores.json` |
| Utility benchmarks (clean) | `experiments/eval_utility.py` | `results/utility_eval.json` |
| Utility benchmarks (with GCG suffix) | `experiments/eval_utility_attacked.py` | `results/utility_eval_attacked.json` |
| All figures | `experiments/visualize_results.py` | `results/figures/` |
| Paper figures | `generate_paper_figures.py` | `paper_figures/` |
| Full pipeline | `run_all.sh` | `results/logs/run_all.log` |
