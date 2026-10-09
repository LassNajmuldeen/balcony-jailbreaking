# Depth explorer

`results/archive/depth_explorer/depth_explorer.html` is a standalone snapshot of
the archived results, including
attack success, utility, loss, and paired McNemar comparisons.

Rebuild from the repository root:

```bash
python3 scripts/run.py figures --archive --depth-explorer
```

The runner derives JSON inputs from `results/archive/` and assembles the HTML
in `results/generated/depth_explorer/`, which stays local. This historical view
uses the original result sets; the template-free 7B reruns are separate.
