# runs lm-evaluation-harness benchmarks on each balcony exit + the
# no-balcony baseline to measure how useful each exit is for normal
# tasks (not adversarial). this gives us the other half of the story:
# "shallow exits resist attacks, but are they still useful?"
#
# requires: pip install lm-eval
#
# usage:
#   python experiments/eval_utility.py                  # all benchmarks
#   python experiments/eval_utility.py --tasks hellaswag  # just one
#   python experiments/eval_utility.py --exits 4 full   # just two exits

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/path/to/project")
sys.path.insert(0, "/path/to/project/AdversariaLLM")

import torch

import adapter.balcony_llama  # noqa: F401
from adapter.exit_wrapper import wrap_for_single_exit, wrap_plain_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import lm_eval
from lm_eval.models.huggingface import HFLM

BALCONY_ID = "parsakaveh/Balcony-LLM-1B"
PLAIN_ID = "parsakaveh/LLM-1B-Pretrained"
RESULTS_DIR = Path("/path/to/project/results")
RESULTS_DIR.mkdir(exist_ok=True)

DEFAULT_TASKS = ["hellaswag", "arc_challenge", "arc_easy", "mmlu", "winogrande", "piqa"]
ALL_EXITS = ["no_balcony", 4, 8, 12, "full"]


class BalconyExitLM(HFLM):
    """wraps our SingleExitCausalLM so lm-eval can use it.
    lm-eval's HFLM expects a standard HF model. our wrapper already
    makes each exit look like a standard model, so we just need to
    pass the pre-loaded model and tokenizer instead of letting HFLM
    load them from a model id."""

    def __init__(self, model, tokenizer, batch_size=8):
        # HFLM.__init__ normally loads the model from pretrained.
        # we skip that by passing the already-loaded objects.
        super().__init__(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            dtype="bfloat16",
        )


def load_balcony_model():
    tok = AutoTokenizer.from_pretrained(BALCONY_ID, trust_remote_code=False)
    if not tok.pad_token:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        BALCONY_ID, trust_remote_code=False, dtype=torch.bfloat16,
    ).to("cuda").eval()
    model = wrap_for_single_exit(model)
    return model, tok


def load_plain_model():
    tok = AutoTokenizer.from_pretrained(PLAIN_ID, trust_remote_code=False)
    if not tok.pad_token:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        PLAIN_ID, trust_remote_code=False, dtype=torch.bfloat16,
    ).to("cuda").eval()
    model = wrap_plain_model(model)
    return model, tok


def run_eval(lm, tasks):
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        batch_size="auto",
        num_fewshot=None,
    )
    return results["results"]


def extract_scores(raw_results):
    """pull out the main accuracy metric from each task's results."""
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
            # fallback: grab the first numeric metric
            for key, val in metrics.items():
                if isinstance(val, (int, float)) and "stderr" not in key:
                    scores[task_name] = {"metric": key, "value": val}
                    break
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--exits", nargs="+", default=None,
                        help="which exits to eval (default: all). e.g. --exits 4 12 full no_balcony")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    exits_to_run = ALL_EXITS
    if args.exits:
        exits_to_run = []
        for e in args.exits:
            if e == "no_balcony":
                exits_to_run.append("no_balcony")
            elif e == "full":
                exits_to_run.append("full")
            else:
                exits_to_run.append(int(e))

    all_results = {}

    # figure out which models we need
    need_balcony = any(e != "no_balcony" for e in exits_to_run)
    need_plain = "no_balcony" in exits_to_run

    if need_balcony:
        print("Loading balcony model...", flush=True)
        balcony_model, balcony_tok = load_balcony_model()

        for exit_layer in exits_to_run:
            if exit_layer == "no_balcony":
                continue
            print(f"\n{'='*50}", flush=True)
            print(f"Evaluating exit {exit_layer}", flush=True)
            print(f"{'='*50}", flush=True)

            balcony_model.set_active_exit(exit_layer)
            lm = BalconyExitLM(balcony_model, balcony_tok, batch_size=args.batch_size)

            raw = run_eval(lm, args.tasks)
            scores = extract_scores(raw)
            all_results[str(exit_layer)] = scores

            print(f"\nExit {exit_layer} results:")
            for task, info in scores.items():
                print(f"  {task}: {info['value']:.4f} ({info['metric']})")

        # free balcony model before loading plain model
        if need_plain:
            del balcony_model, balcony_tok
            torch.cuda.empty_cache()

    if need_plain:
        print(f"\n{'='*50}", flush=True)
        print("Evaluating no-balcony baseline", flush=True)
        print(f"{'='*50}", flush=True)

        plain_model, plain_tok = load_plain_model()
        lm = BalconyExitLM(plain_model, plain_tok, batch_size=args.batch_size)

        raw = run_eval(lm, args.tasks)
        scores = extract_scores(raw)
        all_results["no_balcony"] = scores

        print(f"\nBaseline results:")
        for task, info in scores.items():
            print(f"  {task}: {info['value']:.4f} ({info['metric']})")

    # save results
    out_path = RESULTS_DIR / "utility_eval.json"
    if out_path.exists():
        existing = json.load(open(out_path))
        existing.update(all_results)
        all_results = existing
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)

    # print summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    tasks_seen = set()
    for scores in all_results.values():
        tasks_seen.update(scores.keys())
    tasks_sorted = sorted(tasks_seen)

    header = f"{'exit':>14}" + "".join(f"{t:>16}" for t in tasks_sorted)
    print(header)
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
