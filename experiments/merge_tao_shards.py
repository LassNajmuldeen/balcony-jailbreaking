"""folds the per exit TAO shard checkpoints back into the canonical *_attacks.json.

the parallel runner splits one (model, benchmark, arm) TAO stage into one process
per exit, and each of them writes RESULTS_DIR/tao_shards/<base>.<shard>.json.
nothing downstream knows shards exist: judge_harmfulness.py and export_results.py
both glob "*attacks*.json" straight in RESULTS_DIR. this script merges the shards
into that one file, keyed by exit exactly the way an unsharded run would have
written it.

its idempotent and safe to re-run mid flight. it only ever adds behaviors, and a
shard thats still being written just contributes whatever it has so far.

A GROUP ONLY GETS PUBLISHED ONCE IT IS COMPLETE. publishing is the whole risk
here: the canonical name is what the judge and the exports glob, so an incomplete
merge does not look incomplete downstream, it looks like a finished stage with a
smaller ASR denominator. on 2026-08-09 the 7B had only exit 21 plus a 31/50 exit
15 and would have been published as a whole 7B TAO harmbench stage. so a group
missing an exit, or holding fewer than N_BEHAVIORS behaviors on any exit, gets
skipped with a reason and the canonical file is left alone. MERGE_FORCE=1
overrides that, for when you deliberately want to look at a partial run.

  RESULTS_DIR=results_7b ./AdversariaLLM/.venv/bin/python experiments/merge_tao_shards.py
  RESULTS_DIR=results_7b MERGE_FORCE=1 ... # publish anyway (rarely what you want)
"""
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "AdversariaLLM"))

from config import RESULTS_DIR, N_BEHAVIORS  # noqa: E402

# which exits each results dir is meant to contain. this is deliberately NOT
# config.ALL_EXITS: run_tao_parallel.sh's merge loop only sets RESULTS_DIR, so
# EXITS falls back to its "4 8 12 full" default and every 7B/8B group would be
# called complete the moment four exits existed under the wrong names. keying off
# the directory instead means the guard is right no matter how this gets invoked.
EXPECTED_EXITS = {
    "results": ["4", "8", "12", "full"],
    "results_7b": ["15", "18", "21", "full"],
    "results_8b": ["15", "18", "21", "full"],
    # the full budget rerun (run_tao_full.sh, 500 steps / batch 512). deliberately
    # a seperate dir from results_7b/results_8b: those hold 150 step results and
    # merging the two would put two attack budgets under one ASR without anything
    # downstream being able to tell
    "results_7b_tao500": ["15", "18", "21", "full"],
    "results_8b_tao500": ["15", "18", "21", "full"],
}


def expected_keys(base, results_dir):
    """the exit keys a complete <base>.json has to hold."""
    # the baseline arm is a single un-exited run, so it only ever has one key
    if "no_balcony" in base:
        return ["no_balcony"]
    exits = EXPECTED_EXITS.get(results_dir.name)
    if exits is None:
        return None      # unknown layout, we cant judge completeness at all
    return list(exits)


def completeness_problems(base, merged, results_dir, n_behaviors):
    """empty list means its publishable. otherwise its the list of reasons why
    not, which get printed so the skip is never mysterious."""
    wanted = expected_keys(base, results_dir)
    if wanted is None:
        return [f"unknown results dir {results_dir.name}, no expected exit set"]

    problems = []
    for exit_key in wanted:
        if exit_key not in merged:
            problems.append(f"missing exit {exit_key}")
            continue
        # counted as a SET of behavior_idx, because two shards overlapping would
        # otherwise inflate the count and look complete when its not
        n_unique = len({b["behavior_idx"] for b in merged[exit_key]})
        if n_unique < n_behaviors:
            problems.append(f"exit {exit_key} has {n_unique}/{n_behaviors}")
    # an exit we never asked for usually means the wrong EXITS were set for the
    # run, so treat it as a problem too rather than merging it in silently
    for exit_key in merged:
        if exit_key not in wanted:
            problems.append(f"unexpected exit {exit_key}")
    return problems


def merge():
    shard_dir = RESULTS_DIR / "tao_shards"
    if not shard_dir.is_dir():
        print(f"no shard dir at {shard_dir} -- nothing to merge")
        return

    # group the <base>.<shard>.json files by their <base>.json
    groups = {}
    for shard_path in sorted(shard_dir.glob("*.json")):
        # a running attack writes .json.tmp then renames, so anything still tmp is
        # mid write and not ours to read
        if shard_path.name.endswith(".json.tmp"):
            continue
        stem = shard_path.name[: -len(".json")]
        if "." not in stem:
            print(f"  skipping {shard_path.name}: no .<shard> component")
            continue
        base = stem.rsplit(".", 1)[0] + ".json"
        groups.setdefault(base, []).append(shard_path)

    if not groups:
        print(f"no shards found in {shard_dir}")
        return

    force = os.environ.get("MERGE_FORCE") == "1"
    dry_run = "--dry-run" in sys.argv
    n_published = n_skipped = 0

    for base, shard_paths in sorted(groups.items()):
        out_path = RESULTS_DIR / base
        merged = {}
        # start from whats already published, thats what makes re-running cheap
        if out_path.exists():
            try:
                merged = json.load(open(out_path))
            except json.JSONDecodeError:
                print(f"  {base}: existing file unreadable, rebuilding from shards")
                merged = {}

        n_new = 0
        for shard_path in shard_paths:
            try:
                data = json.load(open(shard_path))
            except json.JSONDecodeError:
                # half written shard. next pass will pick it up, no harm done
                print(f"  {shard_path.name}: mid-write or corrupt, skipped this pass")
                continue
            for exit_key, behaviors in data.items():
                dest = merged.setdefault(exit_key, [])
                seen_idx = {b["behavior_idx"] for b in dest}
                for behavior in behaviors:
                    # first writer wins, so re-running never duplicates a behavior
                    if behavior["behavior_idx"] not in seen_idx:
                        dest.append(behavior)
                        seen_idx.add(behavior["behavior_idx"])
                        n_new += 1

        # sorted by behavior index so the merged file reads like a serial run did it
        for exit_key in merged:
            merged[exit_key].sort(key=lambda b: b["behavior_idx"])

        counts = ", ".join(f"{k}:{len(v)}" for k, v in sorted(merged.items(), key=str))
        problems = completeness_problems(base, merged, RESULTS_DIR, N_BEHAVIORS)
        if problems and not force:
            print(f"  {base}: SKIPPED, incomplete -- {'; '.join(problems)}")
            print(f"      (have {counts}; canonical file left untouched)")
            n_skipped += 1
            continue

        if not dry_run:
            # tmp then rename, so a kill mid write cant leave a truncated
            # canonical file behind for the judge to score
            tmp = out_path.with_suffix(".json.tmp")
            with open(tmp, "w") as f:
                json.dump(merged, f, indent=2)
            tmp.replace(out_path)
        n_published += 1

        flag = "  [FORCED, INCOMPLETE]" if problems else ""
        flag += "  [dry-run, not written]" if dry_run else ""
        print(f"  {base}: +{n_new} new  ->  {counts}{flag}")

    print(f"merge summary: published={n_published} skipped={n_skipped}")


if __name__ == "__main__":
    print(f"merging TAO shards into {RESULTS_DIR}")
    merge()
