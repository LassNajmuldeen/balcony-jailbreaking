# Results

The paper is the source of truth: *Exploring Jailbreak Vulnerability Across
Depth-Based Dynamic Inference in Large Language Models*. Figures and tables in
the `published/` directories are extracted from its pages without recalculating
values. [Source manifest](paper_figures/published/source.json).

## Attack success

GCG, 150 optimization steps, 50 behaviors per benchmark. Values are percentages
from Figure 3; full depth is layer 32.

| Exit | 7B HarmBench | 7B AdvBench | 8B HarmBench | 8B AdvBench |
| --- | ---: | ---: | ---: | ---: |
| 15 | 36 | 46 | 4 | 2 |
| 18 | 40 | 42 | 12 | 0 |
| 21 | 40 | 48 | 12 | 12 |
| Full | 44 | 54 | 28 | 28 |

## Utility

Clean utility (Table 2):

![Clean utility across six benchmarks for both models and all exits.](paper_tables/published/utility_clean.png)

With each exit's strongest GCG suffix (Table 3):

![Utility with GCG suffixes across the same benchmarks, models, and exits.](paper_tables/published/utility_attacked.png)

[Model configuration (Table 1)](paper_tables/published/model_configuration.png).

## Historical outputs

`results*/` and the older exports are retained for provenance. They are not
synchronized with the paper; run variants, judge settings, and exit labels can
differ. In particular, 7B runs with and without chat templates must stay separate.

TAO is a planned extension with experimental code. The 1B runs were development
tests. Neither contributes results to the current paper.
