# the same utility benchmarks as eval_capabilities.py, but WITH an adversarial
# suffix stuck on the front of every input. it answers: does the attack suffix
# also wreck normal task performance, or does it only buy a jailbreak?
#
# for each exit we take the suffix that reached the lowest attack loss, i.e. the
# most effective attack we found, and prepend it to every benchmark prompt. so
# this is a worst case measurement, not an average one.
#
# --attack picks whose suffixes get used. the clean comparator is utility_eval.json
# either way, so a gcg run and a tao run share one baseline.
#
#   gcg   writes utility_eval_attacked.json      (the original behaviour)
#   tao   writes utility_eval_attacked_tao.json
#
# CAVEAT on --attack tao: suffixes are ranked by final_loss, and TAO's final_loss
# is target_loss - alpha*neg_loss, which is NOT a target loss. it goes negative,
# and the most negative value can mean "diverged hardest from the refusal" rather
# than "matched the target best". so the TAO suffix we pick is the one TAO itself
# thought was best, the same suffix that produced own_exit_completion, and it is
# NOT necessarily the strongest jailbreak. dont put a TAO utility delta next to a
# GCG one as if the two suffixes were chosen by the same criterion.
#
# usage:
#   python experiments/eval_capabilities_under_attack.py
#   python experiments/eval_capabilities_under_attack.py --attack tao
#   python experiments/eval_capabilities_under_attack.py --tasks hellaswag --exits 4 full

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
    result_filename,
    BALCONY_MODEL as BALCONY_ID,
    BASELINE_MODEL as BASELINE_ID,
)

DEFAULT_TASKS = ["hellaswag", "arc_challenge", "arc_easy", "mmlu", "winogrande", "piqa"]

OUT_NAME = {
    "gcg": "utility_eval_attacked.json",   # historical name, dont touch it
    "tao": "utility_eval_attacked_tao.json",
}


def get_best_suffix(attacks_data, exit_key):
    """the suffix from whichever behavior reached the lowest loss on this exit."""
    entries = attacks_data[str(exit_key)]
    best = min(entries, key=lambda x: x["final_loss"])
    # TAO records the winning suffix on its own, GCG doesnt, so for GCG it has to
    # be sliced back off the suffixed input. prefer the recorded field wherever it
    # exists, its exactly the suffix that produced own_exit_completion and it
    # doesnt depend on suffixed_input starting with user_content
    if best.get("best_suffix"):
        return best["best_suffix"].strip()
    user_content = best["user_content"]
    suffixed = best["suffixed_input"]
    # loud failure rather than a silently mis-sliced suffix, which would look like
    # a real measurement and quietly be nonsense
    if not suffixed.startswith(user_content):
        raise ValueError(
            f"exit {exit_key}: suffixed_input does not start with user_content, "
            "cannot recover the suffix by slicing"
        )
    suffix = suffixed[len(user_content):]
    return suffix.strip()


def save_results(all_results, out_path):
    """merges into the output file and writes it.

    called after every exit rather than once at the end. eval_capabilities.py got
    this same fix on 2026-08-05 after a crash on the baseline threw away four
    already evaluated exits, this script kept the end only write until 2026-08-09.
    re-reading and merging each time keeps it idempotent.
    """
    merged = {}
    if out_path.exists():
        merged = json.load(open(out_path))
    merged.update(all_results)
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    return out_path


class SuffixPrependLM(HFLM):
    """HFLM that sticks a fixed adversarial suffix onto every input.

    the suffix tokens go at the START of every context, so lm-eval ends up
    evaluating a model that permanently has a pile of garbage tokens in front of
    the actual question.
    """

    def __init__(self, model, tokenizer, suffix_text, batch_size=8):
        super().__init__(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            dtype="bfloat16",
        )
        # tokenized once at construction. add_special_tokens=False so we dont
        # inject a second BOS into the middle of every prompt
        self._suffix_ids = self.tokenizer.encode(
            suffix_text, add_special_tokens=False
        )

    def tok_encode(self, string, **kwargs):
        encoded = super().tok_encode(string, **kwargs)
        return self._suffix_ids + encoded

    def _encode_pair(self, context, continuation):
        context_enc = self.tok_encode(context)
        continuation_enc = self.tok_encode(continuation)
        # only the context is supposed to carry the suffix, the continuation has to
        # stay clean. our tok_encode above prepends unconditionally, so the
        # continuation is re-encoded here straight off the tokenizer instead.
        # (continuation_enc is deliberately left unused, its the polluted one)
        raw_cont = self.tokenizer.encode(continuation, add_special_tokens=False)
        return context_enc, raw_cont


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
        num_fewshot=None,     # each task keeps its own standard shot count
    )
    return results["results"]


def extract_scores(raw_results):
    # same priority order as eval_capabilities.py, and it has to stay that way or
    # the clean vs attacked delta would be comparing two different metrics
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
    parser.add_argument("--attack", choices=["gcg", "tao"], default="gcg",
                        help="which attack's suffixes to prepend (default gcg)")
    parser.add_argument("--suffix-benchmark", choices=["harmbench", "advbench"],
                        default="harmbench",
                        help="which benchmark's attack run the suffixes come from")
    args = parser.parse_args()

    exits_to_run = ALL_EXITS
    if args.exits:
        exits_to_run = []
        for name in args.exits:
            # the two sentinels stay strings, real layer numbers become ints
            if name in ("no_balcony", "full"):
                exits_to_run.append(name)
            else:
                exits_to_run.append(int(name))

    # the attack results are where the suffixes come from. resolved through
    # result_filename so gcg/tao and harmbench/advbench all land on the same names
    # the attack scripts wrote, instead of the two hardcoded GCG harmbench paths
    # this script used to carry around
    balcony_path = RESULTS_DIR / result_filename(args.attack, "balcony", args.suffix_benchmark)
    baseline_path = RESULTS_DIR / result_filename(args.attack, "baseline", args.suffix_benchmark)
    if not balcony_path.exists():
        raise SystemExit(
            f"missing {balcony_path}\n"
            f"  --attack {args.attack} needs the merged canonical file, not shards. "
            f"run experiments/merge_tao_shards.py first if this is a TAO run."
        )

    # the baseline arm is OPTIONAL here. its TAO shards were dropped on 2026-08-09
    # once the trunks were shown to be numerically identical to the balcony
    # checkpoints: `no_balcony` is then the same model as `full`, so measuring it
    # again is duplicate work and not a control. where the file is missing the arm
    # gets skipped rather than made up, so nothing in the output claims to be a
    # measurement that never happened
    baseline_attacks = None
    if baseline_path.exists():
        baseline_attacks = json.load(open(baseline_path))
        print(f"suffix source: {balcony_path.name} / {baseline_path.name}", flush=True)
    else:
        print(f"suffix source: {balcony_path.name}", flush=True)
        print(f"  no {baseline_path.name}: skipping the no_balcony arm. Its trunk is "
              f"identical to the full exit, so read 'full' for it.", flush=True)
        exits_to_run = [e for e in exits_to_run if e != "no_balcony"]
    attacks = json.load(open(balcony_path))

    out_path = RESULTS_DIR / OUT_NAME[args.attack]

    # one suffix per exit, chosen before any model is loaded so a missing field
    # fails immediately instead of 40 minutes into an eval
    suffixes = {}
    for exit_key in (e for e in ALL_EXITS if e != "no_balcony"):
        suffixes[str(exit_key)] = get_best_suffix(attacks, exit_key)
    if baseline_attacks is not None:
        suffixes["no_balcony"] = get_best_suffix(baseline_attacks, "no_balcony")

    all_results = {}

    need_balcony = any(e != "no_balcony" for e in exits_to_run)
    need_baseline = "no_balcony" in exits_to_run

    if need_balcony:
        print("Loading balcony model...", flush=True)
        balcony_model, balcony_tok = load_balcony_model()

        for exit_layer in exits_to_run:
            if exit_layer == "no_balcony":
                continue      # different checkpoint, done further down
            print(f"\n{'='*50}", flush=True)
            print(f"Evaluating exit {exit_layer} WITH {args.attack.upper()} suffix", flush=True)
            print(f"Suffix: {suffixes[str(exit_layer)][:60]}...", flush=True)
            print(f"{'='*50}", flush=True)

            # each exit gets ITS OWN best suffix, not a shared one. the point is
            # the worst case for that exit, not how one exit's suffix travels
            balcony_model.set_active_exit(exit_layer)
            lm = SuffixPrependLM(
                balcony_model, balcony_tok,
                suffixes[str(exit_layer)],
                batch_size=args.batch_size,
            )

            raw_results = run_eval(lm, args.tasks)
            scores = extract_scores(raw_results)
            all_results[str(exit_layer)] = scores

            print(f"\nExit {exit_layer} (attacked) results:")
            for task, info in scores.items():
                print(f"  {task}: {info['value']:.4f} ({info['metric']})")
            save_results(all_results, out_path)      # save per exit, see above

        if need_baseline:
            del balcony_model, balcony_tok
            torch.cuda.empty_cache()

    if need_baseline:
        print(f"\n{'='*50}", flush=True)
        print(f"Evaluating no-balcony baseline WITH {args.attack.upper()} suffix", flush=True)
        print(f"Suffix: {suffixes['no_balcony'][:60]}...", flush=True)
        print(f"{'='*50}", flush=True)

        baseline_model, baseline_tok = load_baseline_model()
        lm = SuffixPrependLM(
            baseline_model, baseline_tok,
            suffixes["no_balcony"],
            batch_size=args.batch_size,
        )

        raw_results = run_eval(lm, args.tasks)
        scores = extract_scores(raw_results)
        all_results["no_balcony"] = scores

        print(f"\nBaseline (attacked) results:")
        for task, info in scores.items():
            print(f"  {task}: {info['value']:.4f} ({info['metric']})")
        save_results(all_results, out_path)

    save_results(all_results, out_path)
    # re-read so the comparison below covers everything on disk, not just this run
    all_results = json.load(open(out_path))
    print(f"\nSaved to {out_path}", flush=True)

    # clean vs attacked, side by side. the clean side always comes from
    # utility_eval.json, which is why eval_capabilities.py has to have run first
    clean = json.load(open(RESULTS_DIR / "utility_eval.json"))
    core_benchmarks = ["hellaswag", "arc_challenge", "arc_easy", "winogrande", "piqa", "mmlu"]
    print(f"\n{'='*70}")
    print("CLEAN vs ATTACKED UTILITY")
    print(f"{'='*70}")
    print(f"{'exit':>14} {'benchmark':>16} {'clean':>8} {'attacked':>10} {'delta':>8}")
    for exit_key in [str(e) for e in ALL_EXITS]:
        if exit_key not in all_results or exit_key not in clean:
            continue
        for bench in core_benchmarks:
            # only print a row when BOTH sides have the number, a one sided delta
            # would be meaningless
            if bench in all_results[exit_key] and bench in clean[exit_key]:
                clean_value = clean[exit_key][bench]["value"]
                attacked_value = all_results[exit_key][bench]["value"]
                print(f"{exit_key:>14} {bench:>16} {clean_value:8.4f} {attacked_value:10.4f} {attacked_value-clean_value:+8.4f}")


if __name__ == "__main__":
    main()
