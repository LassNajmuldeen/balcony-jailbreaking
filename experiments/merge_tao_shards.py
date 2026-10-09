"""Merge TAO behavior checkpoints into canonical attack files. Publishes only complete exit
sets with valid behavior indices and successful results. Existing records are retained;
writes are atomic."""

import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "third_party" / "adversariallm"))

from config import RESULTS_DIR, N_BEHAVIORS, EXITS  # noqa: E402

# Archive names identify their historical exit sets; new runs use explicit EXITS.
EXPECTED_EXITS = {
    "1b": ["4", "8", "12", "full"],
    "7b": ["15", "18", "21", "full"],
    "8b": ["15", "18", "21", "full"],
    "7b_tao_500": ["15", "18", "21", "full"],
    "8b_tao_500": ["15", "18", "21", "full"],
}


def expected_keys(base, results_dir):
    """the exit keys a complete <base>.json has to hold."""
    # the baseline arm is a single un-exited run, so it only ever has one key
    if "no_balcony" in base:
        return ["no_balcony"]
    exits = EXPECTED_EXITS.get(results_dir.name, list(map(str, EXITS)))
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
        indices = {b["behavior_idx"] for b in merged[exit_key]}
        n_unique = len(indices)
        if indices != set(range(n_behaviors)):
            problems.append(f"exit {exit_key} has {n_unique}/{n_behaviors}")
        if any(b.get("error") or b.get("final_loss") is None for b in merged[exit_key]):
            problems.append(f"exit {exit_key} contains failed behaviors")
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
