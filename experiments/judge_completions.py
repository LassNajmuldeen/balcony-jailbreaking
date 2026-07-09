# re-scores all the completions from phase2 with a real judge instead of
# the keyword-based refusal heuristic that turned out to be useless.
# no new GCG runs here, just reads the completions already saved in
# phase2_attacks.json and phase2_transfer.json and judges each one.
#
# i use HarmBench's fine-tuned Mistral-7B classifier via the judgezoo
# package. i originally tried StrongREJECT (gemma-2b based) but that
# model is gated on the hub and i dont have access.
#
# the judge scores each completion against the ORIGINAL un-suffixed
# behavior text, which is the right thing because i want to know if the
# model produced harmful content relevant to the actual request, not
# whether it repeated the adversarial suffix back.
#
# TODO: also score the no-balcony baseline completions from
# no_balcony_attacks.json so i can fill in that row in the results table

import json
import sys
from pathlib import Path

sys.path.insert(0, "/path/to/project")
sys.path.insert(0, "/path/to/project/AdversariaLLM")

from judgezoo import Judge
from judgezoo.harmbench import HarmBenchClassifier

RESULTS_DIR = Path("/path/to/project/results")

EXITS = [4, 8, 12, "full"]


def build_chat(behavior_text, completion):
    return [
        {"role": "user", "content": behavior_text},
        {"role": "assistant", "content": completion or ""},
    ]


def main():
    attacks = json.load(open(RESULTS_DIR / "phase2_attacks.json"))
    transfer = json.load(open(RESULTS_DIR / "phase2_transfer.json"))

    # mistral-7b variant, not llama-2-13b, because 13b doesnt fit on my 16gb card
    print("Loading HarmBench judge (cais/HarmBench-Mistral-7b-val-cls)...", flush=True)
    judge = Judge.from_name("harmbench", classifier=HarmBenchClassifier.MISTRAL_7B)

    # score own-exit completions (the ones where exit X was attacked and
    # exit X generated the response)
    print("\n=== Scoring own-exit completions ===", flush=True)
    own_exit_scores = {}
    for exit_layer in EXITS:
        behaviors = attacks[str(exit_layer)]
        chats = [build_chat(b["user_content"], b["own_exit_completion"]) for b in behaviors]
        results = judge(chats)
        scores = results["p_harmful"]
        own_exit_scores[str(exit_layer)] = [
            {"behavior_idx": b["behavior_idx"], "p_harmful": s}
            for b, s in zip(behaviors, scores)
        ]
        for b, s in zip(behaviors, scores):
            print(f"  exit {exit_layer} [{b['behavior_idx']}] p_harmful={s:.3f}", flush=True)

    # score transfer completions (suffix from exit X tested on exit Y)
    print("\n=== Scoring transfer completions ===", flush=True)
    transfer_scores = {}
    for source_exit in EXITS:
        # i need the original behavior text for judging, which lives in
        # the attacks dict keyed by behavior_idx
        source_behaviors = {b["behavior_idx"]: b["user_content"] for b in attacks[str(source_exit)]}
        for target_exit in EXITS:
            key = f"{source_exit}->{target_exit}"
            entries = transfer[key]
            chats = [build_chat(source_behaviors[e["behavior_idx"]], e["completion"]) for e in entries]
            results = judge(chats)
            scores = results["p_harmful"]
            transfer_scores[key] = [
                {"behavior_idx": e["behavior_idx"], "p_harmful": s}
                for e, s in zip(entries, scores)
            ]
            mean_score = sum(scores) / len(scores)
            print(f"  {key}: mean p_harmful={mean_score:.3f}", flush=True)

    with open(RESULTS_DIR / "judge_scores.json", "w") as f:
        json.dump({"own_exit": own_exit_scores, "transfer": transfer_scores}, f, indent=2)
    print(f"\nSaved judge scores to {RESULTS_DIR / 'judge_scores.json'}", flush=True)

    print("\n=== SUMMARY: mean p_harmful per exit (own-exit completions) ===")
    for exit_layer in EXITS:
        scores = [x["p_harmful"] for x in own_exit_scores[str(exit_layer)]]
        print(f"{str(exit_layer):>6}: mean p_harmful={sum(scores)/len(scores):.3f}  ({[round(s,2) for s in scores]})")


if __name__ == "__main__":
    main()
