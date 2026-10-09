#!/usr/bin/env python3
"""Run experiments, rebuild figures, or export the recorded results."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
from models import MODELS, ACTIVE_MODELS

STAGES = ("all", "gcg", "tao", "control", "judge", "utility",
          "utility-attacked", "regenerate", "merge", "transfer", "verify-weights")
# These variables are controlled by this CLI, so an old shell environment cannot
# silently select a different model, shard, classifier, or output directory.
RUN_ENV = (
    "BALCONY_MODEL", "BASELINE_MODEL", "EXITS", "CHAT_TEMPLATE", "RESULTS_DIR",
    "N_BEHAVIORS", "NUM_STEPS", "MAX_NEW_TOKENS", "BENCHMARK", "ATTACK_TARGET",
    "BEHAVIOR_START", "BEHAVIOR_END", "TAO_SHARD", "COMPLETION_FIELD",
    "COMPLETION_GLOB", "JUDGE_CLASSIFIER", "JUDGE_OUTPUT", "MERGE_FORCE",
    "SEARCH_WIDTH", "TOPK", "TAO_BATCH_SIZE", "TAO_TOPK", "TAO_ALPHA",
    "TAO_FORWARD_CHUNK", "GPU_MEM_FRACTION", "DATA_ROOT",
)


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python", default=sys.executable, help="interpreter for worker processes")
    commands = ap.add_subparsers(dest="command", required=True)
    run = commands.add_parser("experiment", help="run one model, or one stage")
    run.add_argument("--model", choices=ACTIVE_MODELS, required=True)
    run.add_argument("--stage", choices=STAGES, default="all",
                     help="all runs GCG only; tao is an experimental extension")
    run.add_argument("--attack", choices=("gcg", "tao"), default="gcg",
                     help="suffix for utility-attacked (TAO is experimental)")
    run.add_argument("--benchmarks", nargs="+", choices=("harmbench", "advbench"),
                     default=["harmbench", "advbench"])
    run.add_argument("--target", choices=("balcony", "baseline"), default="balcony")
    run.add_argument("--baseline-model", help="opt in to a separate baseline checkpoint")
    run.add_argument("--results-dir", help="output directory (default: runs/<model>)")
    run.add_argument("--exits", nargs="+", help="subset of the model's exits, including full")
    run.add_argument("--plain", action="store_true", help="use the template-free prompt format")
    run.add_argument("--n-behaviors", type=positive, default=50)
    run.add_argument("--num-steps", type=positive, default=150)
    run.add_argument("--max-new-tokens", type=positive, default=512)
    run.add_argument("--search-width", type=positive, default=128)
    run.add_argument("--topk", type=positive, default=64)
    run.add_argument("--tao-batch-size", type=positive, default=128)
    run.add_argument("--tao-topk", type=positive, default=256)
    run.add_argument("--tao-alpha", type=float, default=0.2)
    run.add_argument("--tao-forward-chunk", type=positive, default=32)
    run.add_argument("--gpu-memory-fraction", type=float, default=0.0)
    run.add_argument("--data-root", default=str(REPO / "AdversariaLLM"))
    run.add_argument("--classifier", choices=("llama-2-13b", "mistral-7b"), default="llama-2-13b")
    run.add_argument("--resume", action="store_true", help="skip attack stages completed with the same settings")
    for name in ("figures", "export"):
        sub = commands.add_parser(name, help="extract paper figures" if name == "figures" else "inspect historical exports")
        sub.add_argument("--archive", action="store_true",
                         help="explicitly use superseded inputs, including 1B development runs")
        if name == "figures":
            sub.add_argument("--paper", type=Path, help="extract the published figures and tables from a local paper PDF")
        else:
            sub.add_argument("--loss-curves", action="store_true", help="also export the corrected GCG curves")
    for sub in (run, *[commands.choices[n] for n in ("figures", "export")]):
        sub.add_argument("--dry-run", action="store_true", help="show commands without running them")
    return ap


def result_name(attack, target, benchmark):
    # Import only the pure naming function. No model or GPU imports are needed.
    from config import result_filename
    return result_filename(attack, target, benchmark)


def plan(args):
    """Return (stage name, script, arguments, environment) in execution order."""
    if args.command == "figures":
        if args.paper and args.archive:
            raise ValueError("choose --paper or --archive")
        if args.paper:
            return [("paper-figures", "extract_paper_assets.py", [str(args.paper.resolve())], {})]
        if not args.archive:
            raise ValueError("select --paper PDF for published figures or --archive for historical outputs")
        return [("archived-figures", "make_paper_assets.py", [], {})]
    if args.command == "export":
        if not args.archive:
            raise ValueError("use --archive to inspect historical exports; published results are in RESULTS.md")
        jobs = [("tables", "export_results.py", [], {}),
                ("csv", "export_paper_csv.py", [], {})]
        if args.loss_curves:
            jobs.append(("loss-curves", "export_loss_curves.py", [], {}))
        return jobs

    spec = MODELS[args.model]
    exits = args.exits or [str(e) for e in spec.exits]
    if len(set(exits)) != len(exits) or any(e not in map(str, spec.exits) for e in exits):
        raise ValueError(f"--exits must be a unique subset of: {' '.join(map(str, spec.exits))}")
    if args.target == "baseline" and not args.baseline_model:
        raise ValueError("--target baseline requires --baseline-model")
    if args.target == "baseline" and args.stage in ("control", "transfer"):
        raise ValueError(f"{args.stage} uses the balcony exits; select --target balcony")
    if args.stage == "verify-weights" and not args.baseline_model:
        raise ValueError("verify-weights requires --baseline-model")
    if not 0 <= args.gpu_memory_fraction <= 1:
        raise ValueError("--gpu-memory-fraction must be between 0 and 1")
    if not math.isfinite(args.tao_alpha) or args.tao_alpha < 0:
        raise ValueError("--tao-alpha must be finite and nonnegative")
    suffix = "-baseline" if args.target == "baseline" else ""
    out = Path(args.results_dir or f"runs/{args.model}{suffix}")
    out = (REPO / out).resolve()
    # Keep the published data intact. A fresh directory also isolates pilot runs.
    if any(out == p.resolve() or p.resolve() in out.parents
           for p in REPO.glob("results*")) or out == REPO:
        raise ValueError("choose a new output directory, such as runs/7b; recorded results are read-only")
    env = {
        "BALCONY_MODEL": spec.checkpoint, "BASELINE_MODEL": args.baseline_model or "",
        "EXITS": " ".join(exits), "CHAT_TEMPLATE": "plain" if args.plain else spec.template,
        "RESULTS_DIR": str(out), "N_BEHAVIORS": str(args.n_behaviors),
        "NUM_STEPS": str(args.num_steps), "MAX_NEW_TOKENS": str(args.max_new_tokens),
        "SEARCH_WIDTH": str(args.search_width), "TOPK": str(args.topk),
        "TAO_BATCH_SIZE": str(args.tao_batch_size), "TAO_TOPK": str(args.tao_topk),
        "TAO_ALPHA": str(args.tao_alpha), "TAO_FORWARD_CHUNK": str(args.tao_forward_chunk),
        "GPU_MEM_FRACTION": str(args.gpu_memory_fraction),
        "DATA_ROOT": str((REPO / args.data_root).resolve()),
        "ATTACK_TARGET": args.target, "JUDGE_CLASSIFIER": args.classifier,
        "JUDGE_OUTPUT": "judge_scores.json", "COMPLETION_FIELD": "own_exit_completion",
        "COMPLETION_GLOB": "*attacks.json",
    }
    jobs = []

    def add(name, script, argv=(), **overrides):
        jobs.append((name, script + ".py", list(argv), {**env, **overrides}))

    if args.stage in ("all", "control") and args.target == "balcony":
        add("control", "measure_clean_loss", BENCHMARK="harmbench")
    for benchmark in dict.fromkeys(args.benchmarks):
        for attack in ("gcg", "tao"):
            if args.stage == attack or (args.stage == "all" and attack == "gcg"):
                add(f"{attack}-{benchmark}", f"attack_{attack}", BENCHMARK=benchmark)
    if args.stage in ("all", "judge"):
        add("judge", "judge_harmfulness")
    if args.stage in ("all", "utility"):
        add("utility", "eval_capabilities", ["--exits", "no_balcony"] if args.target == "baseline" else [])
    if args.stage in ("all", "utility-attacked"):
        for attack in (["gcg"] if args.stage == "all" else [args.attack]):
            argv = ["--attack", attack, "--suffix-benchmark", args.benchmarks[0]]
            if args.target == "baseline":
                argv += ["--exits", "no_balcony"]
            add(f"utility-{attack}", "eval_capabilities_under_attack", argv,
                BENCHMARK=args.benchmarks[0])
    if args.stage == "regenerate":
        add("regenerate", "regen_completions")
    if args.stage == "merge":
        add("merge", "merge_tao_shards")
    if args.stage == "transfer":
        add("transfer", "attack_gcg", ["--transfer"], BENCHMARK="harmbench")
        if args.baseline_model:
            add("transfer-baseline", "transfer_matrix_baseline")
    if args.stage == "verify-weights":
        add("verify-weights", "verify_baseline_weights", [args.model])
    return jobs


def validate_attack(path, keys, n):
    """Catch short runs, duplicate indices, failed behaviors, and nonfinite loss."""
    data = json.loads(path.read_text())
    for key in keys:
        entries = data.get(key, [])
        if len(entries) != n or {b.get("behavior_idx") for b in entries} != set(range(n)):
            raise ValueError(f"{path.name}: exit {key} is incomplete (expected {n} unique behaviors)")
        for b in entries:
            loss = b.get("final_loss")
            if b.get("error") or not isinstance(loss, (int, float)) or not math.isfinite(loss):
                raise ValueError(f"{path.name}: exit {key}, behavior {b.get('behavior_idx')} failed")


def execute(args, jobs):
    for name, script, argv, overrides in jobs:
        env = {k: v for k, v in os.environ.items() if k not in RUN_ENV}
        env.update(overrides)
        command = [args.python, str(REPO / "experiments" / script), *argv]
        print(f"[{name}] {shlex.join(command)}", flush=True)
        if args.dry_run:
            print("  " + " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(overrides.items())))
            continue
        marker = None
        expected = None
        if overrides:
            out = Path(overrides["RESULTS_DIR"])
            digest = hashlib.sha256(json.dumps([command, sorted(env.items())]).encode()).hexdigest()[:16]
            marker = out / ".state" / f"{name}-{digest}.done"
            if name.startswith(("gcg-", "tao-")):
                attack, benchmark = name.split("-", 1)
                expected = out / result_name(attack, args.target, benchmark)
            keys = ["no_balcony"] if args.target == "baseline" else overrides["EXITS"].split()
            if args.resume and marker.exists() and expected is not None:
                validate_attack(expected, keys, args.n_behaviors)
                print("  completed; skipping", flush=True)
                continue
            # TAO has its own per-behavior checkpoints; refuse mismatched settings.
            metadata = out / "run.json"
            settings = {k: v for k, v in overrides.items()
                        if k not in ("BENCHMARK", "COMPLETION_GLOB", "COMPLETION_FIELD", "JUDGE_OUTPUT")}
            if metadata.exists() and json.loads(metadata.read_text()) != settings:
                raise ValueError("output directory contains different run settings; choose a new --results-dir")
            out.mkdir(parents=True, exist_ok=True)
            metadata.write_text(json.dumps(settings, indent=2) + "\n")
        subprocess.run(command, env=env, cwd=REPO, check=True)
        if expected is not None:
            validate_attack(expected, keys, args.n_behaviors)
        if marker is not None:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    try:
        execute(args, plan(args))
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        ap.exit(1, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
