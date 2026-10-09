"""Export mean GCG loss curves to results/generated/tables/loss_curves/. Uses the template-free 7B
reruns. Requires every model/benchmark/exit to have 50 behaviors and 150 steps."""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "results/generated/tables" / "loss_curves"
N_STEPS = 150
N_BEHAVIOURS = 50

# model -> (csv stem, exits shallowest-first, {benchmark: {exit: (dir, file)}})
MODELS = {
    "1B": {
        "stem": "llm_1b",
        "exits": ["4", "8", "12", "full"],
        "sources": {
            "Harmbench": {e: ("results/archive/1b", "phase2_attacks.json") for e in ["4", "8", "12", "full"]},
            "Advbench":  {e: ("results/archive/1b", "advbench_attacks.json") for e in ["4", "8", "12", "full"]},
        },
    },
    "7B": {
        "stem": "llama2_7b",
        "exits": ["15", "18", "21", "full"],
        "sources": {
            # the overnight repair wrote one directory per cell; the two full-exit
            # cells predate it and sit at the top level.
            "Harmbench": {
                "15":   ("results/archive/7b_plain/harmbench_exit_15", "phase2_attacks.json"),
                "18":   ("results/archive/7b_plain/harmbench_exit_18", "phase2_attacks.json"),
                "21":   ("results/archive/7b_plain/harmbench_exit_21", "phase2_attacks.json"),
                "full": ("results/archive/7b_plain",                         "phase2_attacks.json"),
            },
            "Advbench": {
                "15":   ("results/archive/7b_plain/advbench_exit_15", "advbench_attacks.json"),
                "18":   ("results/archive/7b_plain/advbench_exit_18", "advbench_attacks.json"),
                "21":   ("results/archive/7b_plain/advbench_exit_21", "advbench_attacks.json"),
                "full": ("results/archive/7b_plain",                        "advbench_attacks.json"),
            },
        },
    },
    "8B": {
        "stem": "llama3_8b",
        "exits": ["15", "18", "21", "full"],
        "sources": {
            "Harmbench": {e: ("results/archive/8b", "phase2_attacks.json") for e in ["15", "18", "21", "full"]},
            "Advbench":  {e: ("results/archive/8b", "advbench_attacks.json") for e in ["15", "18", "21", "full"]},
        },
    },
}


def mean_curve(dirname, fname, exit_key):
    """Mean loss at each step over the behaviours, with the shape checks that
    caught the NaN-locked runs in 3.2."""
    path = ROOT / dirname / fname
    entries = json.load(open(path)).get(exit_key)
    if not entries:
        raise KeyError(f"no exit {exit_key!r} in {path}")
    if len(entries) != N_BEHAVIOURS:
        raise ValueError(f"{path} exit {exit_key}: {len(entries)} behaviours, "
                         f"expected {N_BEHAVIOURS}")
    curves = [e["loss_curve"] for e in entries if len(e["loss_curve"]) == N_STEPS]
    if len(curves) != len(entries):
        raise ValueError(f"{path} exit {exit_key}: "
                         f"{len(entries) - len(curves)} short loss curves")
    n = len(curves)
    return [sum(c[i] for c in curves) / n for i in range(N_STEPS)]


def build(model):
    """{benchmark: {exit: curve}}, raising before anything is written."""
    out = {}
    for bench, spec in model["sources"].items():
        out[bench] = {e: mean_curve(d, f, e) for e, (d, f) in spec.items()}
    return out


def rows(curves, exits):
    """One two-column block per exit, side by side, with a blank spacer column
    between blocks so the file is readable in a spreadsheet."""
    header = []
    for e in exits:
        if header:
            header.append("")
        header += [f"exit_{e} iteration", f"exit_{e} loss"]
    out = [header]
    for i in range(N_STEPS):
        row = []
        for e in exits:
            if row:
                row.append("")
            row += [i, repr(curves[e][i])]
        out.append(row)
    return out


def main():
    missing = []
    for name, model in MODELS.items():
        for bench, spec in model["sources"].items():
            for exit_key, (dirname, fname) in spec.items():
                if not (ROOT / dirname / fname).exists():
                    missing.append(f"{name} {bench} exit {exit_key}: {dirname}/{fname}")
    if missing:
        print("refusing to write, source data not on disk:")
        for m in missing:
            print(f"  {m}")
        return 1

    built = {name: build(model) for name, model in MODELS.items()}

    OUTDIR.mkdir(parents=True, exist_ok=True)
    for name, model in MODELS.items():
        for bench, curves in built[name].items():
            target = OUTDIR / f"{model['stem']}_{bench.lower()}.csv"
            with open(target, "w", newline="") as f:
                csv.writer(f, lineterminator="\n").writerows(rows(curves, model["exits"]))
            print(f"{target.name}: {N_STEPS} rows, "
                  f"{len(model['exits'])} exit blocks, "
                  f"{N_BEHAVIOURS} behaviours each")
            print("  final loss  " +
                  "  ".join(f"exit_{e}={curves[e][-1]:.3f}" for e in model["exits"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
