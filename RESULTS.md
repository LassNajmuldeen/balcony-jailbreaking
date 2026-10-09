# Results

The tables and figures summarize the recorded outputs. Attack success rate (ASR)
is the percentage of completions with a HarmBench judge score above 0.5.
Lower attack success at a shallow exit should be read alongside its task performance.

## GCG attack success

| Model | Exits, shallow to full | HarmBench ASR (%) | AdvBench ASR (%) |
| --- | --- | --- | --- |
| Balcony 1B | 4, 8, 12, full | 26, 18, 72, 58 | 26, 32, 62, 68 |
| LLaMA 2 7B | 15, 18, 21, full | 24, 28, 26, 38 | 30, 36, 34, 34 |
| LLaMA 3 8B | 15, 18, 21, full | 4, 12, 12, 28 | 2, 0, 12, 28 |

Loss, TAO results, and separate baseline arms are in the
[full table](paper_tables/results_table.md). Figures are in the [README](README.md#figures).

## Data notes

- `results/`, `results_7b/`, and `results_8b/` are the sources for the figures above.
- `results_7b_notemplate/` contains the separate, corrected template-free GCG reruns, split by exit.
- `*_512.json` contains regenerated completions. Keep these separate from the original completions and their judge scores.
- `results_*_tao500/` contains partial TAO shards with a different attack budget. These are not complete benchmark runs.
- Archived judge files do not consistently record classifier and generation settings. Avoid mixing score variants.

TAO reports a two-stage objective, which can be negative. Its final loss is not
directly comparable with GCG cross-entropy.

Rebuild the tables with `.venv/bin/python scripts/run.py export`.
Rebuild the figures with `.venv/bin/python scripts/run.py figures`.
