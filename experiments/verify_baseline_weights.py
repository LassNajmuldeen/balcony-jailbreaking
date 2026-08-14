# checks that a "baseline" checkpoint really is the model the balcony backbone was
# frozen from (SETUP.md 6, the baseline != full rule).
#
# balcony trains with freeze_model=true, so every backbone tensor in the balcony
# checkpoint has to be bit identical to the stock pretrained model. if they differ
# then the baseline is the wrong checkpoint and every baseline vs exit comparison
# in the paper is comparing two unrelated models.
#
# it reads tensors straight out of the safetensors shards rather than building the
# models, so checking an 8B pair costs a few hundred MB of RAM instead of ~32 GB.
#
# usage:
#   python experiments/verify_baseline_weights.py                 # all pairs
#   python experiments/verify_baseline_weights.py 8b              # just the one

import json
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

# label goes to (balcony checkpoint, the baseline we claim it was frozen from)
PAIRS = {
    "1b": ("parsakaveh/Balcony-LLM-1B", "parsakaveh/LLM-1B-Pretrained"),
    "7b": ("parsakaveh/Balcony-LLaMA2-7B", "NousResearch/Llama-2-7b-hf"),
    "8b": ("adpretko/Balcony-Meta-LLaMA3-8B-exit-15-18-21", "NousResearch/Meta-Llama-3-8B"),
}

# the tensors balcony adds on top of the frozen backbone. these are excluded from
# the comparison, obviously, since the baseline has no such thing
EXIT_PREFIXES = ("model.exit_modules",)


def shard_map(repo):
    """{tensor_name: local shard path} for a repo, sharded or single file."""
    try:
        # big repos ship an index that says which shard holds which tensor.
        # hf_hub_download caches, so the same shard isnt pulled twice
        index = json.load(open(hf_hub_download(repo, "model.safetensors.index.json")))
        return {
            name: hf_hub_download(repo, shard)
            for name, shard in index["weight_map"].items()
        }
    except Exception:
        # no index means the whole model is one file, e.g. the 1B pair
        path = hf_hub_download(repo, "model.safetensors")
        with safe_open(path, framework="pt") as f:
            return {name: path for name in f.keys()}


def load_tensor(path_by_name, name):
    # opened per tensor on purpose. safetensors mmaps, so this is cheap, and it
    # means we never hold more than one tensor pair in memory at a time
    with safe_open(path_by_name[name], framework="pt") as f:
        return f.get_tensor(name)


def compare(label, balcony_repo, baseline_repo):
    print(f"\n=== {label}: {balcony_repo}")
    print(f"    vs baseline: {baseline_repo}")

    balcony_shards = shard_map(balcony_repo)
    baseline_shards = shard_map(baseline_repo)

    # only tensors present in BOTH and not added by balcony are comparable
    backbone = sorted(
        n for n in balcony_shards
        if not n.startswith(EXIT_PREFIXES) and n in baseline_shards
    )
    only_balcony = sorted(
        n for n in balcony_shards
        if n not in baseline_shards and not n.startswith(EXIT_PREFIXES)
    )
    only_baseline = sorted(n for n in baseline_shards if n not in balcony_shards)
    exit_tensors = sorted(n for n in balcony_shards if n.startswith(EXIT_PREFIXES))

    print(f"    backbone tensors compared : {len(backbone)}")
    print(f"    balcony exit tensors      : {len(exit_tensors)} (excluded)")
    # a non empty either of these usually means a renamed layout, not a real
    # difference, so they are printed rather than treated as a failure
    if only_balcony:
        print(f"    only in balcony ({len(only_balcony)}): {only_balcony[:4]}{' ...' if len(only_balcony) > 4 else ''}")
    if only_baseline:
        print(f"    only in baseline ({len(only_baseline)}): {only_baseline[:4]}{' ...' if len(only_baseline) > 4 else ''}")

    # no overlap at all means the two names dont even use the same layout, which
    # is a fail and not a pass on a technicality
    if not backbone:
        print("    RESULT: FAIL -- no overlapping backbone tensors at all")
        return False

    identical, mismatched, worst_diff, worst_name = 0, [], 0.0, None
    for name in backbone:
        balcony_tensor = load_tensor(balcony_shards, name)
        baseline_tensor = load_tensor(baseline_shards, name)
        if balcony_tensor.shape != baseline_tensor.shape:
            mismatched.append((name, f"shape {tuple(balcony_tensor.shape)} vs {tuple(baseline_tensor.shape)}"))
            continue
        # torch.equal, not allclose. frozen means frozen, any difference at all is
        # a difference worth knowing about
        if torch.equal(balcony_tensor, baseline_tensor):
            identical += 1
        else:
            # .float() first because bf16 subtraction would round the very
            # difference we are trying to report
            diff = (balcony_tensor.float() - baseline_tensor.float()).abs().max().item()
            mismatched.append((name, f"max|diff|={diff:.3e}"))
            if diff > worst_diff:
                worst_diff, worst_name = diff, name
        del balcony_tensor, baseline_tensor      # keeps peak RAM at one pair

    print(f"    bit-identical             : {identical}/{len(backbone)}")
    if mismatched:
        print(f"    MISMATCHED                : {len(mismatched)}")
        # only the first handfull, a genuinely wrong pair mismatches on everything
        # and 300 lines of that tells you nothing extra
        for name, why in mismatched[:8]:
            print(f"      {name}: {why}")
        if len(mismatched) > 8:
            print(f"      ... and {len(mismatched) - 8} more")
        print(f"    worst: {worst_name} max|diff|={worst_diff:.3e}")

    ok = identical == len(backbone)
    print(f"    RESULT: {'PASS -- backbone is bit-identical, baseline is correct' if ok else 'FAIL -- backbone differs'}")
    return ok


def main():
    wanted = sys.argv[1:] or list(PAIRS)
    results = {}
    for label in wanted:
        if label not in PAIRS:
            print(f"unknown pair {label!r}; known: {list(PAIRS)}")
            return 2
        results[label] = compare(label, *PAIRS[label])

    print("\n=== SUMMARY ===")
    for label, ok in results.items():
        print(f"  {label}: {'PASS' if ok else 'FAIL'}")
    # exit code, so this can sit in a runner script as a gate before any attack
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
