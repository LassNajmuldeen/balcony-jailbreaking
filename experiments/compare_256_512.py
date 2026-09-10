# prints the 256-vs-512 ASR table once run_judge_512.sh has produced both score
# files. pure json, no torch -- safe to run any time, including mid-GPU-job.
#
# both arms come from the SAME generation (512 tokens, truncated to 256 for the
# short arm), so a difference here is completion length and nothing else. it is
# NOT comparable to the old judge_scores.json, which used a different classifier
# AND a different batch composition.

import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

# same env override as export_paper_csv.py, so this can be pointed at a rerun
MODELS = [
    ("1B", os.environ.get("RESULTS_DIR_1B") or "results"),
    ("7B", os.environ.get("RESULTS_DIR_7B") or "results_7b"),
    ("8B", os.environ.get("RESULTS_DIR_8B") or "results_8b"),
]


def load(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def main():
    rows = []
    for name, rdir in MODELS:
        d = _REPO / rdir
        short, full = load(d / "judge_scores_256.json"), load(d / "judge_scores_512.json")
        if not short or not full:
            print(f"{name}: missing score files, skipping", file=sys.stderr)
            continue
        for label in sorted(set(short["asr"]) & set(full["asr"])):
            for key in sorted(set(short["asr"][label]) & set(full["asr"][label])):
                a = short["asr"][label][key]
                b = full["asr"][label][key]
                if a["asr_pct"] is None or b["asr_pct"] is None:
                    continue
                rows.append((name, label, key, a["n"], a["asr_pct"], b["asr_pct"],
                             b["asr_pct"] - a["asr_pct"]))

    if not rows:
        print("no comparable cells found")
        return

    print(f"{'model':>5} | {'arm':>26} | {'exit':>6} | {'n':>3} | "
          f"{'ASR@256':>8} | {'ASR@512':>8} | {'delta':>7}")
    print("-" * 84)
    for r in rows:
        print(f"{r[0]:>5} | {r[1]:>26} | {r[2]:>6} | {r[3]:>3} | "
              f"{r[4]:7.1f}% | {r[5]:7.1f}% | {r[6]:+6.1f}pt")

    deltas = [r[6] for r in rows]
    print("-" * 84)
    print(f"cells: {len(deltas)}   mean delta {sum(deltas) / len(deltas):+.2f}pt   "
          f"max {max(deltas):+.1f}pt   min {min(deltas):+.1f}pt")
    print(f"cells that moved > 4pt (the 8B run-to-run noise floor): "
          f"{sum(1 for x in deltas if abs(x) > 4)}/{len(deltas)}")


if __name__ == "__main__":
    main()
