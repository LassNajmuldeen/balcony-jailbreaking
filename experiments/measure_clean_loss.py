"""Measure clean and filler-suffix target loss at each exit. Compare filler_loss with
attacked final_loss to hold prompt length approximately constant."""

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "AdversariaLLM"))

import torch

import adapter.balcony_llama  # noqa: F401 (registers "nested_llama" locally)
from adapter.exit_wrapper import wrap_for_single_exit
from transformers import AutoModelForCausalLM, AutoTokenizer

from adversariallm.attacks.gcg import GCGConfig, compute_loss
from adversariallm.dataset.adv_behaviors import AdvBehaviorsConfig, AdvBehaviorsDataset
from adversariallm.lm_utils import TokenMergeError, get_disallowed_ids, prepare_conversation

from config import (  # noqa: E402
    ensure_chat_template,
    EXITS,
    N_BEHAVIORS,
    RESULTS_DIR,
    ROOT,
    BALCONY_MODEL as MODEL_ID,
)
# read off GCGConfig rather than typed out, so if the upstream default ever
# changes this control still uses whatever GCG really starts from
FILLER_SUFFIX = GCGConfig().optim_str_init


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
    tokenizer = ensure_chat_template(tokenizer)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=False, dtype=torch.bfloat16,
    ).to("cuda").eval()
    model = wrap_for_single_exit(model)
    return model, tokenizer


@torch.no_grad()
def teacher_forced_loss(model, tokenizer, exit_layer, user_content, target_text, not_allowed_ids):
    """same loss computation as the transfer_loss() in the attack scripts, except
    it takes a plain user_content string since theres no suffix optimisation
    going on here."""
    model.set_active_exit(exit_layer)
    conversation = [
        {"role": "user", "content": user_content},
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
    # the usual off by one shift, logits at t predict the token at t+1
    shift_logits = logits[:, -target_len - 1:-1, :]
    shift_labels = full_ids[:, -target_len:]

    loss = compute_loss(shift_logits, shift_labels, "ce", not_allowed_ids)
    return loss.item()


def filler_user_content(base_content):
    # glue the filler straight onto the user message, same as GCG does internally
    # in _attack_single_conversation
    return base_content + FILLER_SUFFIX


def filler_user_content_spaced(base_content):
    # fallback for when the direct concat trips a tokenization merge
    return base_content + " " + FILLER_SUFFIX


def main():
    print("Loading model + tokenizer...", flush=True)
    model, tokenizer = load_model_and_tokenizer()

    # shuffle=False so behavior i here is behavior i in the attack files. the
    # control is only meaningfull if its measuring the same behaviors
    dataset = AdvBehaviorsDataset(AdvBehaviorsConfig(
        messages_path=f"{ROOT}/data/behavior_datasets/harmbench_behaviors_text_all.csv",
        targets_path=f"{ROOT}/data/optimizer_targets/harmbench_targets_text.json",
        idx=list(range(N_BEHAVIORS)),
        shuffle=False,
    ))
    print(f"Loaded {len(dataset)} behaviors", flush=True)

    # same clipping as the attack scripts: the disallowed id list can hold ids
    # past the end of this model's embedding table
    not_allowed_ids = get_disallowed_ids(tokenizer, False, False).to(model.device)
    embed_matrix = model.get_input_embeddings().weight
    not_allowed_ids = not_allowed_ids[not_allowed_ids < embed_matrix.size(0)]

    results = {}
    for exit_layer in EXITS:
        print(f"\n=== Baseline control for exit {exit_layer} ===", flush=True)
        per_behavior = []
        for i in range(len(dataset)):
            user_msg, target_msg = dataset[i]
            base_content = user_msg["content"]
            target_text = target_msg["content"]

            no_suffix_loss = teacher_forced_loss(
                model, tokenizer, exit_layer, base_content, target_text, not_allowed_ids
            )
            # concatenating the filler directly sometimes raises TokenMergeError,
            # because the last token of the prompt merges with the first token of
            # the suffix. adding a space sorts it out. GCG handles this internally
            # the same way
            try:
                filler_loss = teacher_forced_loss(
                    model, tokenizer, exit_layer, filler_user_content(base_content), target_text, not_allowed_ids
                )
            except TokenMergeError:
                filler_loss = teacher_forced_loss(
                    model, tokenizer, exit_layer, filler_user_content_spaced(base_content), target_text, not_allowed_ids
                )

            per_behavior.append({
                "behavior_idx": i,
                "no_suffix_loss": no_suffix_loss,
                "filler_loss": filler_loss,
            })
            print(
                f"  [{i}] no_suffix_loss={no_suffix_loss:.4f} filler_loss={filler_loss:.4f}",
                flush=True,
            )
        results[str(exit_layer)] = per_behavior

    with open(RESULTS_DIR / "baseline_control.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved baseline results to {RESULTS_DIR / 'baseline_control.json'}", flush=True)

    # the interesting column is mean filler_loss per exit. if it climbs as the
    # exits get shallower then the "shallow exits resist attacks" story is at
    # least partly just a starting point difference
    print("\n=== SUMMARY: mean baseline losses per exit ===")
    print(f"{'exit':>6} | {'mean no_suffix_loss':>20} | {'mean filler_loss':>18}")
    for exit_layer in EXITS:
        behaviors = results[str(exit_layer)]
        mean_no_suffix = sum(b["no_suffix_loss"] for b in behaviors) / len(behaviors)
        mean_filler = sum(b["filler_loss"] for b in behaviors) / len(behaviors)
        print(f"{str(exit_layer):>6} | {mean_no_suffix:20.4f} | {mean_filler:18.4f}")


if __name__ == "__main__":
    main()
