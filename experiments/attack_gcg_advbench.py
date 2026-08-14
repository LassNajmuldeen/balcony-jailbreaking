# same GCG attack as attack_gcg_harmbench.py, but pointed at advbench instead.
# advbench is the second benchmark so that the per exit story doesnt rest on one
# behavior set. everything else (steps, search width, topk, completion length) is
# imported from config.py so the two runs stay comparable.
#
# writes results/advbench_attacks.json.

import copy
import json
import sys
import time
from pathlib import Path

# repo + AdversariaLLM on the path before anything from either gets imported
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "AdversariaLLM"))

import pandas as pd
import torch

import adapter.balcony_llama  # noqa: F401
from adapter.exit_wrapper import wrap_for_single_exit
from transformers import AutoModelForCausalLM, AutoTokenizer

from adversariallm.attacks import Attack
from adversariallm.attacks.attack import GenerationConfig
from adversariallm.attacks.gcg import GCGConfig, compute_loss
from adversariallm.defenses.base import UndefendedTarget
from adversariallm.lm_utils import get_disallowed_ids, prepare_conversation

from config import (  # noqa: E402
    ensure_chat_template,
    EXITS,
    MAX_NEW_TOKENS,
    N_BEHAVIORS,
    NUM_STEPS,
    RESULTS_DIR,
    ROOT,
    SEARCH_WIDTH,
    TOPK,
    BALCONY_MODEL as MODEL_ID,
)


class AdvBenchDataset:
    """loads advbench behaviors + targets into the conversation format
    AdversariaLLM expects, i.e. a list of [user_msg, assistant_msg] pairs.

    this is the original of what later became BehaviorDataset in
    behavior_datasets.py. it stays here so this script keeps producing byte for
    byte the same behavior ordering the published advbench numbers came from.
    """

    def __init__(self, messages_path, targets_path, n_behaviors=None):
        self.messages = pd.read_csv(messages_path, header=0)
        targets = pd.read_json(targets_path, typ="series")
        self.targets = self.messages["BehaviorID"].map(targets)
        # no target means the behavior cant be attacked, so drop those rows first
        valid = self.targets.notna()
        self.messages = self.messages[valid].reset_index(drop=True)
        self.targets = self.targets[valid].reset_index(drop=True)
        if n_behaviors is not None:
            self.messages = self.messages.iloc[:n_behaviors].reset_index(drop=True)
            self.targets = self.targets.iloc[:n_behaviors].reset_index(drop=True)

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        # returned fresh each call rather than precomputed, the attack only ever
        # touches one behavior at a time so theres no point building a list
        msg = self.messages.iloc[idx]
        target = self.targets.iloc[idx]
        content = msg["Behavior"]
        # advbench normally has no ContextString, the check is here so the same
        # class can eat a harmbench csv without changes
        if isinstance(msg.get("ContextString"), str) and msg["ContextString"]:
            content = msg["ContextString"] + "\n\n" + content
        return [
            {"role": "user", "content": content},
            {"role": "assistant", "content": target},
        ]


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
    tokenizer = ensure_chat_template(tokenizer)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=False, dtype=torch.bfloat16,
    ).to("cuda").eval()
    # gives us set_active_exit(), same as in the harmbench script
    model = wrap_for_single_exit(model)
    return model, tokenizer


def run_attacks(model, tokenizer, dataset):
    """one independent GCG run per exit. fresh Attack object each time so the RNG
    restarts from seed 0 and the exits dont influence eachother."""
    attacks = {}
    for exit_layer in EXITS:
        print(f"\n=== Attacking exit {exit_layer} (AdvBench) ===", flush=True)
        model.set_active_exit(exit_layer)

        gcg_config = GCGConfig(
            num_steps=NUM_STEPS,
            search_width=SEARCH_WIDTH,
            topk=TOPK,
            # prefix cache on, the instruction never changes within a behavior so
            # theres no reason to recompute it for all 128 candidates every step
            use_prefix_cache=True,
            verbosity="WARNING",
            # this is the completion the judge scores later, so the length is a
            # science knob and not a cosmetic one. see config.py
            generation_config=GenerationConfig(max_new_tokens=MAX_NEW_TOKENS),
        )
        target_system = UndefendedTarget(model=model, tokenizer=tokenizer)
        attack = Attack.from_name("gcg")(gcg_config)

        started = time.time()
        attack_result = attack.run(target_system, dataset)
        elapsed = time.time() - started
        print(f"exit {exit_layer}: {len(attack_result.runs)} behaviors in {elapsed:.1f}s", flush=True)

        per_behavior = []
        for i, behavior_run in enumerate(attack_result.runs):
            last_step = behavior_run.steps[-1]     # final step holds the winning suffix
            attacked_user_content = last_step.model_input[0]["content"]
            per_behavior.append({
                "behavior_idx": i,
                "user_content": dataset[i][0]["content"],
                "target": dataset[i][1]["content"],
                "final_loss": last_step.loss,
                "loss_curve": [s.loss for s in behavior_run.steps],
                "suffixed_input": attacked_user_content,
                "own_exit_completion": last_step.model_completions[0] if last_step.model_completions else None,
            })
            print(
                f"  [{i}] final_loss={last_step.loss:.4f}",
                flush=True,
            )
        attacks[str(exit_layer)] = per_behavior
    return attacks


@torch.no_grad()
def transfer_loss(model, tokenizer, exit_layer, user_content_with_suffix, target_text, not_allowed_ids):
    """teacher forced loss for one suffix on one exit. same tokenization path GCG
    uses internally, so these numbers can sit next to final_loss."""
    model.set_active_exit(exit_layer)
    conversation = [
        {"role": "user", "content": user_content_with_suffix},
        {"role": "assistant", "content": target_text},
    ]
    pre_ids, attack_prefix_ids, prompt_ids, attack_suffix_ids, post_ids, target_ids = prepare_conversation(
        tokenizer, conversation
    )[0]
    full_ids = torch.cat(
        [pre_ids, attack_prefix_ids, prompt_ids, attack_suffix_ids, post_ids, target_ids], dim=0
    ).unsqueeze(0).to(model.device)
    outputs = model(input_ids=full_ids)
    logits = outputs.logits
    target_len = target_ids.size(0)
    # shifted by one: position t predicts token t+1, so the target logits start one
    # step earlier than the target ids do
    shift_logits = logits[:, -target_len - 1:-1, :]
    shift_labels = full_ids[:, -target_len:]
    loss = compute_loss(shift_logits, shift_labels, "ce", not_allowed_ids)
    return loss.item()


def transfer_completion(target_system, user_content_with_suffix, exit_layer, model):
    model.set_active_exit(exit_layer)
    conv = [{"role": "user", "content": user_content_with_suffix}]
    # greedy, so the matrix is reproducible
    res = target_system.generate([conv], max_new_tokens=100, temperature=0.0)
    return res.gen0[0]


def run_transfer_matrix(model, tokenizer, attacks, not_allowed_ids):
    """every source exit's suffixes evaluated on every target exit."""
    target_system = UndefendedTarget(model=model, tokenizer=tokenizer)
    transfer = {}
    for source_exit in EXITS:
        for behavior in attacks[str(source_exit)]:
            suffixed_input = behavior["suffixed_input"]
            target_text = behavior["target"]
            for target_exit in EXITS:
                key = f"{source_exit}->{target_exit}"
                transfer.setdefault(key, [])
                loss = transfer_loss(model, tokenizer, target_exit, suffixed_input, target_text, not_allowed_ids)
                completion = transfer_completion(target_system, suffixed_input, target_exit, model)
                transfer[key].append({
                    "behavior_idx": behavior["behavior_idx"],
                    "loss": loss,
                    "completion": completion,
                })
                print(
                    f"  {key} behavior={behavior['behavior_idx']} loss={loss:.4f}",
                    flush=True,
                )
    return transfer


def summarize(attacks, transfer):
    print("\n\n=== SUMMARY: per-exit attack difficulty (AdvBench) ===")
    print(f"{'exit':>6} | {'mean final loss':>16}")
    for exit_layer in EXITS:
        behaviors = attacks[str(exit_layer)]
        mean_loss = sum(b["final_loss"] for b in behaviors) / len(behaviors)
        print(f"{str(exit_layer):>6} | {mean_loss:16.4f}")

    print("\n=== SUMMARY: transfer matrix (mean loss) ===")
    # rows are where the suffix came from, columns are where it was tested
    header = " " * 8 + "".join(f"{str(e):>10}" for e in EXITS)
    print(header)
    for source_exit in EXITS:
        row = f"{str(source_exit):>8}"
        for target_exit in EXITS:
            key = f"{source_exit}->{target_exit}"
            entries = transfer[key]
            mean_loss = sum(e["loss"] for e in entries) / len(entries)
            row += f"{mean_loss:10.3f}"
        print(row)


def main():
    print("Loading model + tokenizer...", flush=True)
    model, tokenizer = load_model_and_tokenizer()

    dataset = AdvBenchDataset(
        messages_path=f"{ROOT}/data/behavior_datasets/extra_behavior_datasets/advbench_behaviors.csv",
        targets_path=f"{ROOT}/data/optimizer_targets/extra_targets/advbench_targets.json",
        n_behaviors=N_BEHAVIORS,
    )
    print(f"Loaded {len(dataset)} AdvBench behaviors", flush=True)

    # tokens GCG may not use in the suffix, clipped to this model's vocab because
    # the disallowed list can run past the embedding table
    not_allowed_ids = get_disallowed_ids(tokenizer, False, False).to(model.device)
    embed_matrix = model.get_input_embeddings().weight
    not_allowed_ids = not_allowed_ids[not_allowed_ids < embed_matrix.size(0)]

    attacks = run_attacks(model, tokenizer, dataset)
    with open(RESULTS_DIR / "advbench_attacks.json", "w") as f:
        json.dump(attacks, f, indent=2)
    print(f"\nSaved attack results to {RESULTS_DIR / 'advbench_attacks.json'}", flush=True)

    # only the per exit part gets printed here. the transfer matrix is expensive
    # and i ran it seperately, so summarize() above isnt called from main
    print("\n\n=== SUMMARY: per-exit attack difficulty (AdvBench) ===")
    print(f"{'exit':>6} | {'mean final loss':>16}")
    for exit_layer in EXITS:
        behaviors = attacks[str(exit_layer)]
        mean_loss = sum(b["final_loss"] for b in behaviors) / len(behaviors)
        print(f"{str(exit_layer):>6} | {mean_loss:16.4f}")


if __name__ == "__main__":
    main()
