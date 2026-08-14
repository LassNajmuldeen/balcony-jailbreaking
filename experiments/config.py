# one place that says what a run is actually pointed at: which balcony
# checkpoint, which exits, which baseline, and where the results land. every
# script in experiments/ imports from here instead of keeping its own copy of the
# same constants, so moving the suite from 1B to 7B/8B is an env var change and
# not an edit in ten files (SETUP.md 8.1).
#
# the defaults reproduce the original 1B runs exactly, so a script run with no env
# vars set behaves the same as it did before i pulled all this out.
#
# override per run, e.g.:
#   BALCONY_MODEL=parsakaveh/Balcony-LLaMA2-7B \
#   BASELINE_MODEL=meta-llama/Llama-2-7b-hf \
#   EXITS="15 18 21 full" \
#   RESULTS_DIR=results_7b \
#     ./run_all.sh

import os
import re
from pathlib import Path

# repo root worked out from where this file lives, so no script has to carry an
# absolute path anymore (SETUP.md 7, this is the thing that used to need a sed
# over every file each time the repo moved)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# the AdversariaLLM checkout. it holds the venv and the behavior/target datasets.
# the attack scripts call this one ROOT.
ROOT = str(PROJECT_ROOT / "AdversariaLLM")


def _env_str(name, default):
    raw = os.environ.get(name)
    # empty string counts as unset, otherwise EXITS="" would blow up _parse_exits
    return raw if raw else default


def _env_int(name, default):
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _env_float(name, default):
    raw = os.environ.get(name)
    return float(raw) if raw else default


def _parse_exits(raw):
    """turns '15 18 21 full' into [15, 18, 21, 'full'].

    the ints have to stay ints because the model code indexes hidden states with
    them. 'full' stays a string, its just a sentinel for the deepest exit.
    """
    parsed = []
    for piece in raw.replace(",", " ").split():
        parsed.append(piece if piece == "full" else int(piece))
    return parsed


# --- what we are attacking ------------------------------------------------

# the balcony (early exit) checkpoint under attack
BALCONY_MODEL = _env_str("BALCONY_MODEL", "parsakaveh/Balcony-LLM-1B")

# the weight matched, NON balcony control. its a seperate pretrained checkpoint,
# NOT the 'full' exit. balcony freezes the backbone while training, so the stock
# model is the correct no-early-exit control (SETUP.md 6).
BASELINE_MODEL = _env_str("BASELINE_MODEL", "parsakaveh/LLM-1B-Pretrained")

# the backbone layers the balcony heads hang off of, plus 'full' for the deepest
# exit. 1B: [4, 8, 12, 'full'] (16 layers), 7B/8B: [15, 18, 21, 'full'] (32 layers)
EXITS = _parse_exits(_env_str("EXITS", "4 8 12 full"))

# the utility evals also score the seperate baseline model, so they want the exits
# plus that extra arm in one list
ALL_EXITS = ["no_balcony"] + EXITS

# --- where results go -----------------------------------------------------

# relative names get resolved under the repo root, thats what makes
# RESULTS_DIR=results_7b work from anywhere
_results = Path(_env_str("RESULTS_DIR", "results"))
RESULTS_DIR = _results if _results.is_absolute() else PROJECT_ROOT / _results
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FIG_DIR = RESULTS_DIR / "figures"

# --- attack hyperparameters ----------------------------------------------
# these are kept identical across attacks and model sizes, otherwise the
# comparison isnt fair (SETUP.md 8.2). N_BEHAVIORS is env overridable mostly so a
# pilot run can do 2 behaviors and finish in a minute.

N_BEHAVIORS = _env_int("N_BEHAVIORS", 50)
NUM_STEPS = _env_int("NUM_STEPS", 150)

# how long the completions we hand to the judge are allowed to be. harmbench
# mandates 512 ("We use max_new_tokens=512 for all experiments",
# docs/evaluation_pipeline.md). we ran 256 (AdversariaLLM's GCGConfig default) and
# 200 (TAO) up until 2026-08-11, which cut off 72 to 100% of the completions mid
# sentence and therefore undercounts ASR.
MAX_NEW_TOKENS = _env_int("MAX_NEW_TOKENS", 512)

# GCG
SEARCH_WIDTH = _env_int("SEARCH_WIDTH", 128)
TOPK = _env_int("TOPK", 64)

# TAO. heavier per step than GCG because of the two stage forward
TAO_BATCH_SIZE = _env_int("TAO_BATCH_SIZE", 128)
TAO_TOPK = _env_int("TAO_TOPK", 256)
TAO_ALPHA = _env_float("TAO_ALPHA", 0.2)

# --- optional per process GPU memory cap (for running cells in parallel) ------
# one 7B GCG process reserves about 81 GB of the 98 GB card even though the live
# tensors are nowhere near that. thats pytorch's caching allocator grabbing freed
# blocks and never giving them back. harmless when one job owns the GPU, fatal
# when three do, because the first process starves the others before they even
# finish loading.
#
# GPU_MEM_FRACTION caps *this* process's share so a few of them can coexist. its
# opt in and unset by default so every existing single job script behaves exactly
# like it always did. do not give this a default value.
#
#   GPU_MEM_FRACTION=0.30   about 29 GB, which i measured to be comfortably above
#                           the ~25 GB live footprint of a 7B GCG cell
GPU_MEM_FRACTION = _env_float("GPU_MEM_FRACTION", 0.0)

if GPU_MEM_FRACTION > 0:
    # imported lazily, the whole point is that scripts which dont need a GPU
    # (merge, export) can still import config without paying for torch
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(GPU_MEM_FRACTION)

# --- which benchmark, which arm -------------------------------------------

# "harmbench" or "advbench". behavior_datasets.py maps these onto files
BENCHMARK = _env_str("BENCHMARK", "harmbench")

# "balcony" attacks BALCONY_MODEL one exit at a time. "baseline" attacks the
# seperate weight matched non balcony checkpoint as a single arm.
ATTACK_TARGET = _env_str("ATTACK_TARGET", "balcony")


def result_filename(attack, target=None, benchmark=None):
    """the results filename for one (attack, arm, benchmark) combination.

    the four original 1B outputs keep their old names so the results already on
    disk and the figure scripts carry on working. anything new gets suffixed.
    """
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


# stable labels for those same four original 1B outputs. the figure scripts key
# off "harmbench_gcg" and "baseline_gcg" so these must not drift.
JUDGE_LABELS = {
    "phase2_attacks.json": "harmbench_gcg",
    "advbench_attacks.json": "advbench_gcg",
    "no_balcony_attacks.json": "baseline_gcg",
    "tao_attacks.json": "harmbench_tao",
}


def judge_label(filename):
    """the label judge_scores.json should use for one *_attacks.json file.

    regen_completions.py writes <name>_512.json, so the length suffix gets
    stripped first. without that the label drifts to "phase2_attacks_512.json"
    and the figure scripts, which look for "harmbench_gcg", quietly find nothing
    at all.
    """
    stem = re.sub(r"_(\d+)\.json$", ".json", filename)
    return JUDGE_LABELS.get(stem, stem.replace("_attacks.json", ""))


# --- judging --------------------------------------------------------------
# which classifier, which completion field, which files. all env driven so the
# same script can score the 512 token completions and then their 256 token
# truncations without me editing code in between the two passes.
#
# JUDGE_CLASSIFIER: "llama-2-13b" is the classifier harmbench reports its results
# with, so thats the default now. "mistral-7b" is the *validation* classifier we
# used up to 2026-08-11, kept around so the old numbers can still be reproduced.
JUDGE_CLASSIFIER = _env_str("JUDGE_CLASSIFIER", "llama-2-13b")
COMPLETION_FIELD = _env_str("COMPLETION_FIELD", "own_exit_completion")
COMPLETION_GLOB = _env_str("COMPLETION_GLOB", "*attacks*.json")
JUDGE_OUTPUT = _env_str("JUDGE_OUTPUT", "judge_scores.json")


# --- chat template --------------------------------------------------------
# balcony 7B/8B sit on top of *base* llama checkpoints, which ship no chat
# template at all, and the third party 8B upload didnt carry one over either.
# AdversariaLLM's prepare_conversation() calls tokenizer.apply_chat_template(),
# so a missing template is a hard crash and not a silent default.
#
# the rule that actually matters for the science: a baseline has to use the SAME
# template as the balcony model its being compared to, otherwise "baseline vs
# exit" is measuring prompt formatting rather than the early exit machinery.
#
#   1B  balcony + baseline : both already ship ChatML       so leave them alone
#   7B  balcony            : ships llama-2-chat
#   7B  baseline           : bare, give it "llama-2-chat" (renders byte identical)
#   8B  balcony + baseline : both bare, give them "llama-3-balcony"
#
# "llama-3-balcony" is the llama-3 header format lifted from Balcony-LLaMA's own
# experiment configs. it is deliberately NOT AdversariaLLM's "llama-3-instruct",
# which injects a default system prompt ("You are a helpful assistant" plus a
# knowledge cutoff line) that neither the 1B nor the 7B template adds. that
# asymmetry would confound the cross model comparison.
CHAT_TEMPLATE = _env_str("CHAT_TEMPLATE", "")


def resolve_chat_template(name):
    """look in the repo's own chat_templates/ first, then fall back to the ones
    AdversariaLLM ships."""
    local = PROJECT_ROOT / "chat_templates" / f"{name}.jinja"
    if local.exists():
        return local.read_text()
    # imported down here because the repo local file usually wins and importing
    # adversariallm at module import time drags in a lot for no reason
    from adversariallm.io_utils import load_chat_template

    return load_chat_template(name)


def ensure_chat_template(tokenizer):
    """apply CHAT_TEMPLATE if the run asked for one. does nothing when its unset,
    so the 1B path keeps using whatever the checkpoint came with."""
    if CHAT_TEMPLATE:
        tokenizer.chat_template = resolve_chat_template(CHAT_TEMPLATE)
    return tokenizer


# --- benchmark data -------------------------------------------------------

HARMBENCH_MESSAGES = f"{ROOT}/data/behavior_datasets/harmbench_behaviors_text_all.csv"
HARMBENCH_TARGETS = f"{ROOT}/data/optimizer_targets/harmbench_targets_text.json"


def describe():
    """one line banner so every log actually says what it was run against. worth
    the two lines of code, i have lost hours to logs that didnt."""
    return (
        f"balcony={BALCONY_MODEL} baseline={BASELINE_MODEL} "
        f"exits={EXITS} results={RESULTS_DIR} "
        f"n_behaviors={N_BEHAVIORS} steps={NUM_STEPS}"
    )
