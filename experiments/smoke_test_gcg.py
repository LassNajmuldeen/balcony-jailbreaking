"""Small GCG diagnostic for the early-exit model. For a configurable pilot, use
scripts/run.py experiment with a small behavior and step budget."""

import sys
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "third_party" / "adversariallm"))

import torch
import adapter.balcony_llama  # registers "nested_llama" locally
from adapter.exit_wrapper import wrap_for_single_exit
from transformers import AutoModelForCausalLM, AutoTokenizer

from adversariallm.attacks import Attack
from adversariallm.dataset.adv_behaviors import AdvBehaviorsConfig, AdvBehaviorsDataset
from adversariallm.defenses.base import UndefendedTarget

from config import ensure_chat_template, ROOT, BALCONY_MODEL as MODEL_ID  # noqa: E402

print("Loading model + tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
tok = ensure_chat_template(tok)
if not tok.pad_token:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, trust_remote_code=False, dtype=torch.bfloat16,
).to("cuda").eval()
model = wrap_for_single_exit(model)
model.set_active_exit(12)
print("active exit:", model._active_exit, "available:", model.available_exits())

dataset = AdvBehaviorsDataset(AdvBehaviorsConfig(
    messages_path=f"{ROOT}/data/behavior_datasets/harmbench_behaviors_text_all.csv",
    targets_path=f"{ROOT}/data/optimizer_targets/harmbench_targets_text.json",
    idx=[0, 1],
    shuffle=False,
))
print(f"dataset size: {len(dataset)}")
for i in range(len(dataset)):
    print(f"  [{i}] user={dataset[i][0]['content']!r}")
    print(f"      target={dataset[i][1]['content']!r}")

GCGConfig = Attack.from_name("gcg")
from adversariallm.attacks.gcg import GCGConfig as _GCGConfig

gcg_config = _GCGConfig(
    num_steps=60,
    search_width=128,
    topk=64,
    use_prefix_cache=True,
    verbosity="INFO",
)

target = UndefendedTarget(model=model, tokenizer=tok)
attack = Attack.from_name("gcg")(gcg_config)

print("Running GCG against exit 12...")
result = attack.run(target, dataset)

for i, run in enumerate(result.runs):
    last_step = run.steps[-1]
    print(f"--- behavior {i} ---")
    print("final loss:", last_step.loss)
    print("final suffix input:", last_step.model_input)
    print("model completion:", last_step.model_completions)
