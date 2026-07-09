# measures the model's loss on the target compliance string WITHOUT any
# optimized adversarial suffix. this is the control experiment that tells
# me whether shallow exits are genuinely harder to attack or if they just
# start from a higher loss baseline (spoiler: its mostly the latter).
#
# two conditions per (exit, behavior):
#   no_suffix_loss  = nothing appended at all
#   filler_loss     = GCG's starting suffix ("x x x x..." x20) with zero
#                     optimization steps run on it
#
# results go to results/baseline_control.json

import json
import sys
from pathlib import Path

sys.path.insert(0, "/path/to/project")
sys.path.insert(0, "/path/to/project/AdversariaLLM")

import torch

import adapter.balcony_llama  # noqa: F401 (registers "nested_llama" locally)
from adapter.exit_wrapper import wrap_for_single_exit
from transformers import AutoModelForCausalLM, AutoTokenizer

from adversariallm.attacks.gcg import GCGConfig, compute_loss
from adversariallm.dataset.adv_behaviors import AdvBehaviorsConfig, AdvBehaviorsDataset
from adversariallm.lm_utils import TokenMergeError, get_disallowed_ids, prepare_conversation

ROOT = "/path/to/project/AdversariaLLM"
MODEL_ID = "parsakaveh/Balcony-LLM-1B"
RESULTS_DIR = Path("/path/to/project/results")
RESULTS_DIR.mkdir(exist_ok=True)

EXITS = [4, 8, 12, "full"]
N_BEHAVIORS = 6
FILLER_SUFFIX = GCGConfig().optim_str_init


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


@torch.no_grad()
def teacher_forced_loss(model, tok, exit_layer, user_content, target_text, not_allowed_ids):
    """same loss computation as phase2's transfer_loss() but takes a plain
    user_content string since i'm not doing any suffix optimization here."""
    model.set_active_exit(exit_layer)
    conversation = [
        {"role": "user", "content": user_content},
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


def filler_user_content(base_content):
    # just glue the filler suffix onto the user message, same way GCG does
    # internally in _attack_single_conversation
    return base_content + FILLER_SUFFIX


def filler_user_content_spaced(base_content):
    # fallback if direct concat causes a tokenization merge error
    return base_content + " " + FILLER_SUFFIX


def main():
    print("Loading model + tokenizer...", flush=True)
    model, tok = load_model_and_tokenizer()

    dataset = AdvBehaviorsDataset(AdvBehaviorsConfig(
        messages_path=f"{ROOT}/data/behavior_datasets/harmbench_behaviors_text_all.csv",
        targets_path=f"{ROOT}/data/optimizer_targets/harmbench_targets_text.json",
        idx=list(range(N_BEHAVIORS)),
        shuffle=False,
    ))
    print(f"Loaded {len(dataset)} behaviors", flush=True)

    not_allowed_ids = get_disallowed_ids(tok, False, False).to(model.device)
    embeddings = model.get_input_embeddings().weight
    not_allowed_ids = not_allowed_ids[not_allowed_ids < embeddings.size(0)]

    results = {}
    for exit_layer in EXITS:
        print(f"\n=== Baseline control for exit {exit_layer} ===", flush=True)
        per_behavior = []
        for i in range(len(dataset)):
            user_msg, target_msg = dataset[i]
            base_content = user_msg["content"]
            target_text = target_msg["content"]

            no_suffix_loss = teacher_forced_loss(
                model, tok, exit_layer, base_content, target_text, not_allowed_ids
            )
            # sometimes concatenating the filler directly causes a TokenMergeError
            # because the last token of the prompt merges with the first token of
            # the suffix. adding a space fixes it. GCG handles this internally too
            try:
                filler_loss = teacher_forced_loss(
                    model, tok, exit_layer, filler_user_content(base_content), target_text, not_allowed_ids
                )
            except TokenMergeError:
                filler_loss = teacher_forced_loss(
                    model, tok, exit_layer, filler_user_content_spaced(base_content), target_text, not_allowed_ids
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

    print("\n=== SUMMARY: mean baseline losses per exit ===")
    print(f"{'exit':>6} | {'mean no_suffix_loss':>20} | {'mean filler_loss':>18}")
    for exit_layer in EXITS:
        behaviors = results[str(exit_layer)]
        mean_no_suffix = sum(b["no_suffix_loss"] for b in behaviors) / len(behaviors)
        mean_filler = sum(b["filler_loss"] for b in behaviors) / len(behaviors)
        print(f"{str(exit_layer):>6} | {mean_no_suffix:20.4f} | {mean_filler:18.4f}")


if __name__ == "__main__":
    main()
