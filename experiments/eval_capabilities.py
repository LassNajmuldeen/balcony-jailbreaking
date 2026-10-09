"""Evaluate ordinary language tasks at each exit with lm-eval. Writes utility_eval.json,
the clean comparator for attacked utility."""

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "AdversariaLLM"))

import torch

import adapter.balcony_llama  # noqa: F401
from adapter.exit_wrapper import wrap_for_single_exit, wrap_plain_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import lm_eval
from lm_eval.models.huggingface import HFLM

from config import (  # noqa: E402
    ensure_chat_template,
    ALL_EXITS,
    RESULTS_DIR,
    BALCONY_MODEL as BALCONY_ID,
    BASELINE_MODEL as BASELINE_ID,
    require_baseline_model,
)

DEFAULT_TASKS = ["hellaswag", "arc_challenge", "arc_easy", "mmlu", "winogrande", "piqa"]


class BalconyExitLM(HFLM):
    """wraps our SingleExitCausalLM so lm-eval will accept it.

    lm-eval's HFLM expects a normal HF model. our wrapper already makes one exit
    look like a normal model, so all thats needed is handing over the loaded
    objects instead of letting HFLM load them from a model id itself.
    """

    def __init__(self, model, tokenizer, batch_size=8):
        # HFLM.__init__ would normally do from_pretrained here. passing the live
        # objects skips that, and more importantly keeps the exit we already set
        super().__init__(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            dtype="bfloat16",
        )


def load_balcony_model():
    tokenizer = AutoTokenizer.from_pretrained(BALCONY_ID, trust_remote_code=False)
    tokenizer = ensure_chat_template(tokenizer)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        BALCONY_ID, trust_remote_code=False, dtype=torch.bfloat16,
    ).to("cuda").eval()
    model = wrap_for_single_exit(model)
    return model, tokenizer


def load_baseline_model():
    # only reachable when no_balcony was asked for by hand (--exits no_balcony),
    # and without a BASELINE_MODEL that stops here with a clear message
    require_baseline_model()
    tokenizer = AutoTokenizer.from_pretrained(BASELINE_ID, trust_remote_code=False)
    tokenizer = ensure_chat_template(tokenizer)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        BASELINE_ID, trust_remote_code=False, dtype=torch.bfloat16,
    ).to("cuda").eval()
    model = wrap_plain_model(model)
    return model, tokenizer


def run_eval(lm, tasks):
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        batch_size="auto",
        # num_fewshot=None means each task keeps its own standard shot count,
        # which is what makes these numbers comparable to published ones
        num_fewshot=None,
    )
    return results["results"]


def extract_scores(raw_results):
    """pulls the main accuracy metric out of each task's results.

    the order below is the priority order: acc_norm where a task has it (its the
    length normalised one hellaswag and arc report), then plain acc, then
    exact_match for the generative tasks.
    """
    scores = {}
    for task_name, metrics in raw_results.items():
        for key in ["acc_norm,none", "acc,none", "exact_match,none"]:
            if key in metrics:
                scores[task_name] = {
                    "metric": key.split(",")[0],
                    "value": metrics[key],
                }
                break
        if task_name not in scores:
            # fallback, just take the first numeric metric that isnt a stderr.
            # never hit for the default task list, its here so an unusual task
            # doesnt silently vanish from the table
            for key, val in metrics.items():
                if isinstance(val, (int, float)) and "stderr" not in key:
                    scores[task_name] = {"metric": key, "value": val}
                    break
    return scores


def save_results(all_results):
    """merges into utility_eval.json and writes it out.

    this gets called after EVERY exit, not just at the end. on 2026-08-05 the 7B
    and 8B runs evaluated all four exits and then died while loading the baseline,
    and since the only write happened after that step every single number was
    thrown away. re-reading and merging each time also keeps it idempotent, so a
    resumed run doesnt clobber what the first attempt got.
    """
    out_path = RESULTS_DIR / "utility_eval.json"
    merged = {}
    if out_path.exists():
        merged = json.load(open(out_path))
    merged.update(all_results)
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--exits", nargs="+", default=None,
                        help="which exits to eval (default: all). e.g. --exits 4 12 full no_balcony")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    # exits arrive as strings off the command line but the model indexes hidden
    # states with real ints, so everything except the two sentinels gets cast
    exits_to_run = ALL_EXITS
    if args.exits:
        exits_to_run = []
        for name in args.exits:
            if name == "no_balcony":
                exits_to_run.append("no_balcony")
            elif name == "full":
                exits_to_run.append("full")
            else:
                exits_to_run.append(int(name))

    all_results = {}

    # the two checkpoints dont fit on the card together, so work out up front
    # which ones are actually needed and load them one after the other
    need_balcony = any(e != "no_balcony" for e in exits_to_run)
    need_baseline = "no_balcony" in exits_to_run

    if need_balcony:
        print("Loading balcony model...", flush=True)
        balcony_model, balcony_tok = load_balcony_model()

        for exit_layer in exits_to_run:
            if exit_layer == "no_balcony":
                continue      # different checkpoint, handled further down
            print(f"\n{'='*50}", flush=True)
            print(f"Evaluating exit {exit_layer}", flush=True)
            print(f"{'='*50}", flush=True)

            # the exit has to be set BEFORE the HFLM wrapper is built around it
            balcony_model.set_active_exit(exit_layer)
            lm = BalconyExitLM(balcony_model, balcony_tok, batch_size=args.batch_size)

            raw_results = run_eval(lm, args.tasks)
            scores = extract_scores(raw_results)
            all_results[str(exit_layer)] = scores

            print(f"\nExit {exit_layer} results:")
            for task, info in scores.items():
                print(f"  {task}: {info['value']:.4f} ({info['metric']})")
            print(f"Saved exit {exit_layer} to {save_results(all_results)}", flush=True)

        # let go of the balcony model before the baseline gets loaded
        if need_baseline:
            del balcony_model, balcony_tok
            torch.cuda.empty_cache()

    if need_baseline:
        print(f"\n{'='*50}", flush=True)
        print("Evaluating no-balcony baseline", flush=True)
        print(f"{'='*50}", flush=True)

        baseline_model, baseline_tok = load_baseline_model()
        lm = BalconyExitLM(baseline_model, baseline_tok, batch_size=args.batch_size)

        raw_results = run_eval(lm, args.tasks)
        scores = extract_scores(raw_results)
        all_results["no_balcony"] = scores

        print(f"\nBaseline results:")
        for task, info in scores.items():
            print(f"  {task}: {info['value']:.4f} ({info['metric']})")

    # re-read after saving, so the summary shows everything on disk and not only
    # the arms this particular invocation happened to run
    out_path = save_results(all_results)
    all_results = json.load(open(out_path))
    print(f"\nSaved to {out_path}", flush=True)

    # summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    tasks_seen = set()
    for scores in all_results.values():
        tasks_seen.update(scores.keys())
    tasks_sorted = sorted(tasks_seen)

    header = f"{'exit':>14}" + "".join(f"{t:>16}" for t in tasks_sorted)
    print(header)
    # NOTE these row names are the 1B exits. on a 7B/8B run the json is still
    # written correctly, but this table prints only the no_balcony and full rows.
    # read utility_eval.json directly for those, or fix this to walk ALL_EXITS
    for exit_key in ["no_balcony", "4", "8", "12", "full"]:
        if exit_key not in all_results:
            continue
        row = f"{exit_key:>14}"
        for task in tasks_sorted:
            if task in all_results[exit_key]:
                row += f"{all_results[exit_key][task]['value']:>16.4f}"
            else:
                row += f"{'--':>16}"
        print(row)


if __name__ == "__main__":
    main()
