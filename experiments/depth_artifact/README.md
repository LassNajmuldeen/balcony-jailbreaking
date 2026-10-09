# Depth explorer

`balcony-depth.html` is a standalone view of the archived results, including
attack success, utility, loss, and paired McNemar comparisons.

Rebuild from the repository root:

```bash
python3 experiments/depth_artifact/recompute.py
python3 experiments/depth_artifact/util.py
python3 experiments/depth_artifact/loss.py
python3 experiments/depth_artifact/build_depth.py
```

The first three scripts derive the four JSON inputs from `results*/`.
The last assembles the HTML. This view uses the original result sets; the
template-free 7B reruns are separate.
