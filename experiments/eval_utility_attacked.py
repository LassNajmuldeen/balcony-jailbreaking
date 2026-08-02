# measures utility benchmarks WITH a GCG adversarial suffix prepended
# to every input. this answers: "does the GCG suffix degrade normal
# task performance, not just enable jailbreaks?"
#
# for each exit we pick the suffix that achieved the lowest attack loss
# (most effective attack) and prepend it to every benchmark prompt.
# this is a worst-case measurement.
#
# usage:
#   python experiments/eval_utility_attacked.py
#   python experiments/eval_utility_attacked.py --tasks hellaswag --exits 4 full

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

DEFAULT_TASKS = ["hellaswag", "arc_challenge", "arc_easy", "mmlu", "winogrande", "piqa"]
ALL_EXITS = ["no_balcony", 4, 8, 12, "full"]


def get_best_suffix(attacks_data, exit_key):
    """pick the suffix from the attack that achieved the lowest loss."""
    entries = attacks_data[str(exit_key)]
    best = min(entries, key=lambda x: x["final_loss"])
    user_content = best["user_content"]
    suffixed = best["suffixed_input"]
    suffix = suffixed[len(user_content):]
    return suffix.strip()


class SuffixPrependLM(HFLM):
    """wraps HFLM to prepend a fixed adversarial suffix to every input.
    the suffix tokens are added at the start of every context so
    lm-eval sees a model that always has garbage tokens at the front."""

    def __init__(self, model, tokenizer, suffix_text, batch_size=8):
        super().__init__(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            dtype="bfloat16",
        )
        self._suffix_ids = self.tokenizer.encode(
            suffix_text, add_special_tokens=False
        )

    def tok_encode(self, string, **kwargs):
        encoded = super().tok_encode(string, **kwargs)
        return self._suffix_ids + encoded

    def _encode_pair(self, context, continuation):
        context_enc = self.tok_encode(context)
        continuation_enc = self.tok_encode(continuation)
        # only the context gets the suffix; continuation stays clean
        # since tok_encode already prepends, we need to strip it from continuation
        raw_cont = self.tokenizer.encode(continuation, add_special_tokens=False)
        return context_enc, raw_cont


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
            for key, val in metrics.items():
                if isinstance(val, (int, float)) and "stderr" not in key:
                    scores[task_name] = {"metric": key, "value": val}
                    break
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--exits", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    exits_to_run = ALL_EXITS
    if args.exits:
        exits_to_run = []
        for e in args.exits:
            if e in ("no_balcony", "full"):
                exits_to_run.append(e if e == "no_balcony" else "full")
            else:
                exits_to_run.append(int(e))

    # load attack results to extract suffixes
    attacks = json.load(open(RESULTS_DIR / "phase2_attacks.json"))
    nb_attacks = json.load(open(RESULTS_DIR / "no_balcony_attacks.json"))

    # pick best suffix per exit
    suffixes = {}
    for exit_key in [4, 8, 12, "full"]:
        suffixes[str(exit_key)] = get_best_suffix(attacks, exit_key)
    suffixes["no_balcony"] = get_best_suffix(nb_attacks, "no_balcony")

    all_results = {}

    need_balcony = any(e != "no_balcony" for e in exits_to_run)
    need_plain = "no_balcony" in exits_to_run

    if need_balcony:
        print("Loading balcony model...", flush=True)
        balcony_model, balcony_tok = load_balcony_model()

        for exit_layer in exits_to_run:
            if exit_layer == "no_balcony":
                continue
            print(f"\n{'='*50}", flush=True)
            print(f"Evaluating exit {exit_layer} WITH GCG suffix", flush=True)
            print(f"Suffix: {suffixes[str(exit_layer)][:60]}...", flush=True)
            print(f"{'='*50}", flush=True)

            balcony_model.set_active_exit(exit_layer)
            lm = SuffixPrependLM(
                balcony_model, balcony_tok,
                suffixes[str(exit_layer)],
                batch_size=args.batch_size,
            )

            raw = run_eval(lm, args.tasks)
            scores = extract_scores(raw)
            all_results[str(exit_layer)] = scores

            print(f"\nExit {exit_layer} (attacked) results:")
            for task, info in scores.items():
                print(f"  {task}: {info['value']:.4f} ({info['metric']})")

        if need_plain:
            del balcony_model, balcony_tok
            torch.cuda.empty_cache()

    if need_plain:
        print(f"\n{'='*50}", flush=True)
        print("Evaluating no-balcony baseline WITH GCG suffix", flush=True)
        print(f"Suffix: {suffixes['no_balcony'][:60]}...", flush=True)
        print(f"{'='*50}", flush=True)

        plain_model, plain_tok = load_plain_model()
        lm = SuffixPrependLM(
            plain_model, plain_tok,
            suffixes["no_balcony"],
            batch_size=args.batch_size,
        )

        raw = run_eval(lm, args.tasks)
        scores = extract_scores(raw)
        all_results["no_balcony"] = scores

        print(f"\nBaseline (attacked) results:")
        for task, info in scores.items():
            print(f"  {task}: {info['value']:.4f} ({info['metric']})")

    # save
    out_path = RESULTS_DIR / "utility_eval_attacked.json"
    if out_path.exists():
        existing = json.load(open(out_path))
        existing.update(all_results)
        all_results = existing
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)

    # comparison summary
    clean = json.load(open(RESULTS_DIR / "utility_eval.json"))
    core_benchmarks = ["hellaswag", "arc_challenge", "arc_easy", "winogrande", "piqa", "mmlu"]
    print(f"\n{'='*70}")
    print("CLEAN vs ATTACKED UTILITY")
    print(f"{'='*70}")
    print(f"{'exit':>14} {'benchmark':>16} {'clean':>8} {'attacked':>10} {'delta':>8}")
    for exit_key in ["no_balcony", "4", "8", "12", "full"]:
        if exit_key not in all_results or exit_key not in clean:
            continue
        for bench in core_benchmarks:
            if bench in all_results[exit_key] and bench in clean[exit_key]:
                c = clean[exit_key][bench]["value"]
                a = all_results[exit_key][bench]["value"]
                print(f"{exit_key:>14} {bench:>16} {c:8.4f} {a:10.4f} {a-c:+8.4f}")


if __name__ == "__main__":
    main()
