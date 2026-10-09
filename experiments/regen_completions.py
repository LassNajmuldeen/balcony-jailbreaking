# regenerates the judged completions from ALREADY-STORED adversarial suffixes,
# at a longer max_new_tokens. no attack compute: the suffix is read off disk and
# only generation is replayed.
#
# why: the completions in results*/ were generated at 256 (GCG, AdversariaLLM's
# GCGConfig default) and 200 (TAO), while HarmBench mandates 512
# ("We use max_new_tokens=512 for all experiments", docs/evaluation_pipeline.md).
# 72-100% of stored GCG completions sit exactly at the 256 cap, i.e. the model was
# still writing when we cut it off, so ASR is undercounted.
#
# uses the same path GCG used (prepare_conversation -> cat(tokens[:5]) ->
# generate_ragged_batched, gcg.py:386-398), but the stored completions are NOT
# bit-reproducible and must not be used as the 256-token comparison arm.
#
# WHY (measured 2026-08-11): GCG generated with generate_completions="all", so it
# batched all 150 step-variants of a behaviour into one ragged batch. That padding
# cannot be recreated without re-running the attack, and in bf16 a tied logit then
# falls the other way -- one observed case split on "Sure!" vs "Sure," after 127
# identical characters and diverged from there. Exact-match rate against disk is
# ~31% at batch 8 and ~12% at batch 1, i.e. batch composition, not a wrong prompt
# (the behaviours that do match, match to the character over 1400+ chars).
#
# SO: generate ONCE at 512 and derive the 256 arm by truncating that same output to
# 256 tokens. Greedy decoding is prefix-deterministic within a single run, so the
# truncation IS the 256-token completion of that run -- exactly, at zero extra GPU
# cost. Both arms then come from one generation and differ ONLY in length, which is
# the comparison we actually want. Judge both; do not judge the on-disk text.
#
#   ./AdversariaLLM/.venv/bin/python experiments/regen_completions.py --verify -n 5
#   ./AdversariaLLM/.venv/bin/python experiments/regen_completions.py
#
# writes <name>_512.json beside each input. never touches the originals.

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "AdversariaLLM"))

import torch

import adapter.balcony_llama  # noqa: F401 (registers "nested_llama")
from adapter.exit_wrapper import wrap_for_single_exit, wrap_plain_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from adversariallm.lm_utils import generate_ragged_batched, prepare_conversation

from config import (  # noqa: E402
    ensure_chat_template,
    MAX_NEW_TOKENS,
    RESULTS_DIR,
    BALCONY_MODEL,
    BASELINE_MODEL,
)

# the cap each file was originally generated at, used by --verify. GCG files came
# from GCGConfig.generation_config (256); TAO files from attack_tao.py's own call (200).
ORIGINAL_CAP = {"gcg": 256, "tao": 200}

# files whose completions came from the BASELINE checkpoint, not the balcony one.
# they carry a single "no_balcony" arm and must be generated with the other model.
BASELINE_FILES = {
    "no_balcony_attacks.json",
    "no_balcony_baseline.json",
    "no_balcony_advbench_baseline.json",
    "gcg_no_balcony_advbench_attacks.json",
    "tao_no_balcony_harmbench_attacks.json",
}


def attack_kind(name):
    return "tao" if "tao" in name else "gcg"


def complete_output(src, out):
    """True if `out` already holds a finished regeneration of `src`.

    "finished" means same arms, same row counts, and every row that had a suffix
    carries a 512 completion -- not merely that the file exists. A run killed
    mid-write would otherwise be treated as done and silently leave a short file
    in the judged set.
    """
    if not out.exists():
        return False
    try:
        a, b = json.loads(src.read_text()), json.loads(out.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if set(a) != set(b):
        return False
    for key, rows in a.items():
        if not isinstance(rows, list):
            continue
        if len(b.get(key, [])) != len(rows):
            return False
        for r in b[key]:
            # presence, NOT truthiness: an empty completion is a real result (8B
            # exit 15 under TAO emits nothing for one behaviour), and treating ""
            # as unfinished makes the file permanently un-resumable.
            if r.get("suffixed_input") and "own_exit_completion_short" not in r:
                return False
    return True


def load_model(model_id, is_baseline):
    """Use the same model wrapping as attack_gcg.py.

    the two arms need DIFFERENT wrapping and getting it wrong is a hard crash:
    wrap_for_single_exit reads config.output_exit_layers, which only a NestedLlama
    balcony checkpoint has, so calling it on the stock 7B/8B baselines raises
    AttributeError. the 1B baseline IS nested (with exit_layer_indices=[]) and
    needs wrap_plain_model to unwrap its 1-element forward; the 7B/8B baselines are
    stock HF llama and must be left completely alone.
    """
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    tok = ensure_chat_template(tok)
    if not tok.pad_token:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=False, dtype=torch.bfloat16,
    ).to("cuda").eval()
    if is_baseline:
        if model.config.model_type == "nested_llama":
            model = wrap_plain_model(model)
        return model, tok
    return wrap_for_single_exit(model), tok


@torch.no_grad()
def generate_batch(model, tok, suffixed_inputs, targets, max_new_tokens, batch_size=8):
    """mirrors gcg.py:386-398 -- prepare_conversation, drop the target chunk,
    generate. the target text is still needed because prepare_conversation splits
    on it, but tokens[:5] excludes it so it is never fed to the model."""
    out = []
    for lo in range(0, len(suffixed_inputs), batch_size):
        token_list = []
        for text, target in zip(suffixed_inputs[lo:lo + batch_size], targets[lo:lo + batch_size]):
            conv = [
                {"role": "user", "content": text},
                {"role": "assistant", "content": target},
            ]
            tokens = prepare_conversation(tok, conv)[0]
            token_list.append(torch.cat(tokens[:5]))
        completions = generate_ragged_batched(
            model, tok,
            token_list=token_list,
            initial_batch_size=len(token_list),
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            num_return_sequences=1,
        )
        out.extend(c[0] for c in completions)
    return out


def process_file(path, model, tok, max_new_tokens, limit=None, verify=False,
                 is_baseline=False):
    data = json.loads(path.read_text())
    changed = {}
    stats = []
    for exit_key, rows in data.items():
        if not isinstance(rows, list):
            changed[exit_key] = rows
            continue
        rows = rows[:limit] if limit else rows
        usable = [r for r in rows if r.get("suffixed_input") and r.get("target")]
        if not usable:
            changed[exit_key] = rows
            continue
        # baseline files carry a single "no_balcony" arm on a model that has no
        # exits to select -- the stock 7B/8B checkpoints do not even have the method.
        if not is_baseline:
            model.set_active_exit(int(exit_key) if exit_key.isdigit() else exit_key)
        t0 = time.time()
        comps = generate_batch(
            model, tok,
            [r["suffixed_input"] for r in usable],
            [r["target"] for r in usable],
            max_new_tokens,
        )
        if verify:
            same = sum(
                (c or "").strip() == (r.get("own_exit_completion") or "").strip()
                for c, r in zip(comps, usable)
            )
            stats.append((exit_key, same, len(usable)))
            print(f"    exit {exit_key}: reproduced {same}/{len(usable)} "
                  f"({100 * same / len(usable):.0f}%)  {time.time() - t0:.0f}s", flush=True)
        else:
            # rebuild preserving order. own_exit_completion is the full 512-token
            # generation; own_exit_completion_short is THE SAME generation truncated
            # to the old cap, which is what that run would have produced at the old
            # setting. judging both isolates length from every other difference.
            # own_exit_completion_orig keeps the on-disk text for reference only --
            # it is from a different batch composition, so it is not the control.
            short_cap = ORIGINAL_CAP[attack_kind(path.name)]
            it = iter(comps)
            new_rows = []
            for r in rows:
                r = dict(r)
                if r.get("suffixed_input") and r.get("target"):
                    full = next(it)
                    ids = tok.encode(full, add_special_tokens=False)
                    r["own_exit_completion_orig"] = r.get("own_exit_completion")
                    r["own_exit_completion_short"] = tok.decode(
                        ids[:short_cap], skip_special_tokens=True
                    )
                    r["own_exit_completion"] = full
                new_rows.append(r)
            changed[exit_key] = new_rows
            print(f"    exit {exit_key}: {len(usable)} regenerated  "
                  f"{time.time() - t0:.0f}s", flush=True)
    return changed, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="regenerate at the ORIGINAL cap and diff against disk")
    ap.add_argument("-n", type=int, default=None, help="limit behaviours per exit")
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--force", action="store_true",
                    help="regenerate even where a complete output already exists")
    args = ap.parse_args()

    results = Path(RESULTS_DIR)
    files = ([results / f for f in args.files] if args.files
             else sorted(results.glob("*attacks*.json")) + sorted(results.glob("*baseline*.json")))
    # exclude .bak AND this script's own outputs -- <name>_512.json still matches
    # the *attacks* glob, so a second run would regenerate from its own output and
    # silently overwrite own_exit_completion_orig with a regenerated string.
    files = [f for f in files
             if f.exists()
             and ".bak" not in f.name
             and not f.stem.endswith(f"_{MAX_NEW_TOKENS}")]

    # group by which checkpoint generated them so each model loads once
    groups = {}
    skipped_baseline = []
    for f in files:
        # baseline files were generated by the seperate stock checkpoint. with no
        # BASELINE_MODEL set they get skipped out loud, not regenerated with the
        # balcony model and not quietly dropped either
        if f.name in BASELINE_FILES and not BASELINE_MODEL:
            skipped_baseline.append(f.name)
            continue
        groups.setdefault(BASELINE_MODEL if f.name in BASELINE_FILES else BALCONY_MODEL, []).append(f)
    if skipped_baseline:
        print(f"skipping {len(skipped_baseline)} baseline file(s), BASELINE_MODEL is not set: "
              f"{', '.join(skipped_baseline)}", flush=True)

    all_stats = []
    for model_id, group in groups.items():
        is_baseline = model_id == BASELINE_MODEL
        # resume: a completed output means that file is done. checked before the
        # model loads so a re-run after a mid-way failure does not pay for a
        # checkpoint it has no work for.
        if not args.verify and not args.force:
            group = [p for p in group
                     if not complete_output(p, p.with_name(p.stem + f"_{MAX_NEW_TOKENS}.json"))]
        if not group:
            print(f"\n=== {model_id}: all outputs present, skipping ===", flush=True)
            continue
        print(f"\n=== loading {model_id} (baseline={is_baseline}) ===", flush=True)
        model, tok = load_model(model_id, is_baseline)
        for path in group:
            cap = ORIGINAL_CAP[attack_kind(path.name)] if args.verify else MAX_NEW_TOKENS
            print(f"  {path.name}  (max_new_tokens={cap})", flush=True)
            changed, stats = process_file(path, model, tok, cap, args.n, args.verify,
                                          is_baseline)
            all_stats.extend((path.name, *s) for s in stats)
            if not args.verify:
                out = path.with_name(path.stem + f"_{MAX_NEW_TOKENS}.json")
                out.write_text(json.dumps(changed, indent=2))
                print(f"  -> {out.name}", flush=True)
        del model
        torch.cuda.empty_cache()

    if args.verify and all_stats:
        # stats tuples are (filename, exit_key, same, n)
        print("\n=== verify summary ===")
        total_same = sum(st[2] for st in all_stats)
        total_n = sum(st[3] for st in all_stats)
        print(f"exact match against disk: {total_same}/{total_n} "
              f"= {100 * total_same / total_n:.1f}%")
        print("Diagnostic only. <100% is EXPECTED and is batch-composition "
              "nondeterminism, not a wrong prompt (see module docstring). The "
              "controlled 256-vs-512 comparison comes from own_exit_completion_short "
              "vs own_exit_completion in the same output file, not from disk.")


if __name__ == "__main__":
    main()
