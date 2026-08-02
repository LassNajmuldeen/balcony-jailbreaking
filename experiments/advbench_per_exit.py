import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/path/to/project")
sys.path.insert(0, "/path/to/project/AdversariaLLM")

import pandas as pd
import torch

import adapter.balcony_llama  # noqa: F401
from adapter.exit_wrapper import wrap_for_single_exit
from transformers import AutoModelForCausalLM, AutoTokenizer

from adversariallm.attacks import Attack
from adversariallm.attacks.gcg import GCGConfig, compute_loss
from adversariallm.defenses.base import UndefendedTarget
from adversariallm.lm_utils import get_disallowed_ids, prepare_conversation

ROOT = "/path/to/project/AdversariaLLM"
MODEL_ID = "parsakaveh/Balcony-LLM-1B"
RESULTS_DIR = Path("/path/to/project/results")
RESULTS_DIR.mkdir(exist_ok=True)

EXITS = [4, 8, 12, "full"]
N_BEHAVIORS = 50
NUM_STEPS = 150
SEARCH_WIDTH = 128
TOPK = 64


class AdvBenchDataset:
    """Loads AdvBench behaviors + targets into the same conversation format
    that AdversariaLLM expects: list of [user_msg, assistant_msg] pairs."""

    def __init__(self, messages_path, targets_path, n_behaviors=None):
        self.messages = pd.read_csv(messages_path, header=0)
        targets = pd.read_json(targets_path, typ="series")
        self.targets = self.messages["BehaviorID"].map(targets)
        valid = self.targets.notna()
        self.messages = self.messages[valid].reset_index(drop=True)
        self.targets = self.targets[valid].reset_index(drop=True)
        if n_behaviors is not None:
            self.messages = self.messages.iloc[:n_behaviors].reset_index(drop=True)
            self.targets = self.targets.iloc[:n_behaviors].reset_index(drop=True)

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        msg = self.messages.iloc[idx]
        target = self.targets.iloc[idx]
        content = msg["Behavior"]
        if isinstance(msg.get("ContextString"), str) and msg["ContextString"]:
            content = msg["ContextString"] + "\n\n" + content
        return [
            {"role": "user", "content": content},
            {"role": "assistant", "content": target},
        ]


def load_model_and_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
    if not tok.pad_token:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=False, dtype=torch.bfloat16,
    ).to("cuda").eval()
    model = wrap_for_single_exit(model)
    return model, tok


def run_attacks(model, tok, dataset):
    attacks = {}
    for exit_layer in EXITS:
        print(f"\n=== Attacking exit {exit_layer} (AdvBench) ===", flush=True)
        model.set_active_exit(exit_layer)

        gcg_config = GCGConfig(
            num_steps=NUM_STEPS,
            search_width=SEARCH_WIDTH,
            topk=TOPK,
            use_prefix_cache=True,
            verbosity="WARNING",
        )
        target = UndefendedTarget(model=model, tokenizer=tok)
        attack = Attack.from_name("gcg")(gcg_config)

        t0 = time.time()
        result = attack.run(target, dataset)
        elapsed = time.time() - t0
        print(f"exit {exit_layer}: {len(result.runs)} behaviors in {elapsed:.1f}s", flush=True)

        per_behavior = []
        for i, run in enumerate(result.runs):
            last_step = run.steps[-1]
            suffix_user_content = last_step.model_input[0]["content"]
            per_behavior.append({
                "behavior_idx": i,
                "user_content": dataset[i][0]["content"],
                "target": dataset[i][1]["content"],
                "final_loss": last_step.loss,
                "loss_curve": [s.loss for s in run.steps],
                "suffixed_input": suffix_user_content,
                "own_exit_completion": last_step.model_completions[0] if last_step.model_completions else None,
            })
            print(
                f"  [{i}] final_loss={last_step.loss:.4f}",
                flush=True,
            )
        attacks[str(exit_layer)] = per_behavior
    return attacks


@torch.no_grad()
def transfer_loss(model, tok, exit_layer, user_content_with_suffix, target_text, not_allowed_ids):
    model.set_active_exit(exit_layer)
    conversation = [
        {"role": "user", "content": user_content_with_suffix},
        {"role": "assistant", "content": target_text},
    ]
    pre_ids, attack_prefix_ids, prompt_ids, attack_suffix_ids, post_ids, target_ids = prepare_conversation(
        tok, conversation
    )[0]
    full_ids = torch.cat(
        [pre_ids, attack_prefix_ids, prompt_ids, attack_suffix_ids, post_ids, target_ids], dim=0
    ).unsqueeze(0).to(model.device)
    outputs = model(input_ids=full_ids)
    logits = outputs.logits
    target_len = target_ids.size(0)
    shift_logits = logits[:, -target_len - 1:-1, :]
    shift_labels = full_ids[:, -target_len:]
    loss = compute_loss(shift_logits, shift_labels, "ce", not_allowed_ids)
    return loss.item()


def transfer_completion(target_system, user_content_with_suffix, exit_layer, model):
    model.set_active_exit(exit_layer)
    conv = [{"role": "user", "content": user_content_with_suffix}]
    res = target_system.generate([conv], max_new_tokens=100, temperature=0.0)
    return res.gen0[0]


def run_transfer_matrix(model, tok, attacks, not_allowed_ids):
    target = UndefendedTarget(model=model, tokenizer=tok)
    transfer = {}
    for source_exit in EXITS:
        for behavior in attacks[str(source_exit)]:
            suffixed_input = behavior["suffixed_input"]
            target_text = behavior["target"]
            for target_exit in EXITS:
                key = f"{source_exit}->{target_exit}"
                transfer.setdefault(key, [])
                loss = transfer_loss(model, tok, target_exit, suffixed_input, target_text, not_allowed_ids)
                completion = transfer_completion(target, suffixed_input, target_exit, model)
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
    model, tok = load_model_and_tokenizer()

    dataset = AdvBenchDataset(
        messages_path=f"{ROOT}/data/behavior_datasets/extra_behavior_datasets/advbench_behaviors.csv",
        targets_path=f"{ROOT}/data/optimizer_targets/extra_targets/advbench_targets.json",
        n_behaviors=N_BEHAVIORS,
    )
    print(f"Loaded {len(dataset)} AdvBench behaviors", flush=True)

    not_allowed_ids = get_disallowed_ids(tok, False, False).to(model.device)
    embeddings = model.get_input_embeddings().weight
    not_allowed_ids = not_allowed_ids[not_allowed_ids < embeddings.size(0)]

    attacks = run_attacks(model, tok, dataset)
    with open(RESULTS_DIR / "advbench_attacks.json", "w") as f:
        json.dump(attacks, f, indent=2)
    print(f"\nSaved attack results to {RESULTS_DIR / 'advbench_attacks.json'}", flush=True)

    print("\n\n=== SUMMARY: per-exit attack difficulty (AdvBench) ===")
    print(f"{'exit':>6} | {'mean final loss':>16}")
    for exit_layer in EXITS:
        behaviors = attacks[str(exit_layer)]
        mean_loss = sum(b["final_loss"] for b in behaviors) / len(behaviors)
        print(f"{str(exit_layer):>6} | {mean_loss:16.4f}")


if __name__ == "__main__":
    main()
