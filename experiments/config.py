"""Shared run settings. Paths are relative to the repository unless absolute.

The public runner sets these variables explicitly. Direct module use requires
BALCONY_MODEL; a separate baseline is loaded only when BASELINE_MODEL is set.
"""

import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ROOT = str(PROJECT_ROOT / "AdversariaLLM")

DATA_ROOT = os.environ.get("DATA_ROOT") or ROOT


def _env_str(name, default):
    raw = os.environ.get(name)
    return raw if raw else default


def _env_int(name, default):
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _env_float(name, default):
    raw = os.environ.get(name)
    return float(raw) if raw else default


def _parse_exits(raw):
    """Parse integer layer indices and the full-depth sentinel."""
    parsed = []
    for piece in raw.replace(",", " ").split():
        parsed.append(piece if piece == "full" else int(piece))
    return parsed


if os.environ.get("BALCONY_MODEL"):
    BALCONY_MODEL = os.environ["BALCONY_MODEL"]

BASELINE_MODEL = os.environ.get("BASELINE_MODEL") or None


def require_baseline_model():
    """Require an explicitly selected separate baseline checkpoint."""
    if not BASELINE_MODEL:
        raise SystemExit("Set BASELINE_MODEL to use a separate baseline checkpoint.")
    return BASELINE_MODEL


EXITS = _parse_exits(_env_str("EXITS", "15 18 21 full"))

ALL_EXITS = (["no_balcony"] if BASELINE_MODEL else []) + EXITS


_results = Path(_env_str("RESULTS_DIR", "results"))
RESULTS_DIR = _results if _results.is_absolute() else PROJECT_ROOT / _results

FIG_DIR = RESULTS_DIR / "figures"


N_BEHAVIORS = _env_int("N_BEHAVIORS", 50)
NUM_STEPS = _env_int("NUM_STEPS", 150)

BEHAVIOR_START = _env_int("BEHAVIOR_START", 0)
BEHAVIOR_END = _env_int("BEHAVIOR_END", N_BEHAVIORS)

MAX_NEW_TOKENS = _env_int("MAX_NEW_TOKENS", 512)

SEARCH_WIDTH = _env_int("SEARCH_WIDTH", 128)
TOPK = _env_int("TOPK", 64)

TAO_BATCH_SIZE = _env_int("TAO_BATCH_SIZE", 128)
TAO_TOPK = _env_int("TAO_TOPK", 256)
TAO_ALPHA = _env_float("TAO_ALPHA", 0.2)

GPU_MEM_FRACTION = _env_float("GPU_MEM_FRACTION", 0.0)

if GPU_MEM_FRACTION > 0:
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(GPU_MEM_FRACTION)


BENCHMARK = _env_str("BENCHMARK", "harmbench")

ATTACK_TARGET = _env_str("ATTACK_TARGET", "balcony")


def result_filename(attack, target=None, benchmark=None):
    """Resolve an output name while preserving the archived file conventions."""
    target = target or ATTACK_TARGET
    benchmark = benchmark or BENCHMARK
    legacy = {
        ("gcg", "balcony", "harmbench"): "phase2_attacks.json",
        ("gcg", "balcony", "advbench"): "advbench_attacks.json",
        ("gcg", "baseline", "harmbench"): "no_balcony_attacks.json",
        ("tao", "balcony", "harmbench"): "tao_attacks.json",
    }
    key = (attack, target, benchmark)
    if key in legacy:
        return legacy[key]
    arm = "" if target == "balcony" else "_no_balcony"
    return f"{attack}{arm}_{benchmark}_attacks.json"


JUDGE_LABELS = {
    "phase2_attacks.json": "harmbench_gcg",
    "advbench_attacks.json": "advbench_gcg",
    "no_balcony_attacks.json": "baseline_gcg",
    "tao_attacks.json": "harmbench_tao",
}


def judge_label(filename):
    """Use the same judge label for original and regenerated completions."""
    stem = re.sub(r"_(\d+)\.json$", ".json", filename)
    return JUDGE_LABELS.get(stem, stem.replace("_attacks.json", ""))


JUDGE_CLASSIFIER = _env_str("JUDGE_CLASSIFIER", "llama-2-13b")
COMPLETION_FIELD = _env_str("COMPLETION_FIELD", "own_exit_completion")
COMPLETION_GLOB = _env_str("COMPLETION_GLOB", "*attacks*.json")
JUDGE_OUTPUT = _env_str("JUDGE_OUTPUT", "judge_scores.json")


CHAT_TEMPLATE = _env_str("CHAT_TEMPLATE", "")


def resolve_chat_template(name):
    """Find a template here, then in AdversariaLLM."""
    local = PROJECT_ROOT / "chat_templates" / f"{name}.jinja"
    if local.exists():
        return local.read_text()
    from adversariallm.io_utils import load_chat_template

    return load_chat_template(name)


def ensure_chat_template(tokenizer):
    """Apply the selected template, or keep the checkpoint template."""
    if CHAT_TEMPLATE:
        tokenizer.chat_template = resolve_chat_template(CHAT_TEMPLATE)
    return tokenizer


HARMBENCH_MESSAGES = f"{DATA_ROOT}/data/behavior_datasets/harmbench_behaviors_text_all.csv"
HARMBENCH_TARGETS = f"{DATA_ROOT}/data/optimizer_targets/harmbench_targets_text.json"


def describe():
    """Describe the model, exits, output directory, and attack budget."""
    return (
        f"balcony={globals().get('BALCONY_MODEL', 'unset')} baseline={BASELINE_MODEL} "
        f"exits={EXITS} results={RESULTS_DIR} "
        f"n_behaviors={N_BEHAVIORS} steps={NUM_STEPS}"
    )


def __getattr__(name):
    if name == "BALCONY_MODEL":
        raise ImportError(
            "BALCONY_MODEL is required. For example:\n"
            "  BALCONY_MODEL=parsakaveh/Balcony-LLaMA2-7B RESULTS_DIR=results_7b\n"
            "  BALCONY_MODEL=adpretko/Balcony-Meta-LLaMA3-8B-exit-15-18-21 RESULTS_DIR=results_8b"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
