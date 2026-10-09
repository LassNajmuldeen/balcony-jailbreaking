"""CPU-only checks for experiment routing and output integrity."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("runner", REPO / "scripts" / "run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def args(self, *extra):
        return runner.parser().parse_args(["experiment", "--model", "7b", *extra])

    def test_full_plan_has_both_attacks_and_benchmarks(self):
        jobs = runner.plan(self.args())
        self.assertEqual([j[0] for j in jobs], [
            "control", "gcg-harmbench", "tao-harmbench", "gcg-advbench", "tao-advbench",
            "judge", "utility", "utility-gcg", "utility-tao",
        ])
        self.assertTrue(all(j[3]["BASELINE_MODEL"] == "" for j in jobs))
        self.assertTrue(all(Path(j[3]["RESULTS_DIR"]) == REPO / "runs/7b" for j in jobs))

    def test_baseline_requires_explicit_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "requires --baseline-model"):
            runner.plan(self.args("--target", "baseline"))

    def test_baseline_utility_uses_only_baseline_and_selected_benchmark(self):
        jobs = runner.plan(self.args("--target", "baseline", "--baseline-model", "org/model",
                                     "--benchmarks", "advbench"))
        self.assertNotIn("control", [j[0] for j in jobs])
        for name, _, argv, _ in jobs:
            if name.startswith("utility"):
                self.assertIn("no_balcony", argv)
            if name in ("utility-gcg", "utility-tao"):
                self.assertIn("advbench", argv)

    def test_recorded_data_is_protected(self):
        for name in ("results", "results_7b", "results_8b/figures", "."):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "new output directory"):
                runner.plan(self.args("--results-dir", name))

    def test_model_presets_and_plain_prompt(self):
        for model, exits in (("1b", "4 8 12 full"), ("7b", "15 18 21 full"), ("8b", "15 18 21 full")):
            args = runner.parser().parse_args(["experiment", "--model", model, "--plain"])
            env = runner.plan(args)[0][3]
            self.assertEqual(env["EXITS"], exits)
            self.assertEqual(env["CHAT_TEMPLATE"], "plain")
        with self.assertRaisesRegex(ValueError, "unique subset"):
            runner.plan(self.args("--exits", "4", "full"))

    def test_attack_completeness_checks(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "attacks.json"
            good = [{"behavior_idx": i, "final_loss": 0.2} for i in range(2)]
            path.write_text(json.dumps({"full": good}))
            runner.validate_attack(path, ["full"], 2)
            for entries in (good[:1], [good[0], good[0]],
                            [good[0], {"behavior_idx": 1, "final_loss": None, "error": "failed"}],
                            [good[0], {"behavior_idx": 1, "final_loss": float("nan")} ]):
                path.write_text(json.dumps({"full": entries}))
                with self.assertRaises(ValueError):
                    runner.validate_attack(path, ["full"], 2)

    def test_runner_stops_on_worker_failure(self):
        args = self.args("--stage", "gcg")
        with tempfile.TemporaryDirectory() as temp:
            args.results_dir = temp
            with patch.object(runner.subprocess, "run", side_effect=subprocess.CalledProcessError(3, "worker")) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    runner.execute(args, runner.plan(args))
                self.assertEqual(run.call_count, 1)
                self.assertFalse((Path(temp) / ".state").exists())

    def test_dry_run_has_no_filesystem_or_worker_effects(self):
        args = self.args("--dry-run", "--stage", "gcg")
        with tempfile.TemporaryDirectory() as temp:
            args.results_dir = str(Path(temp) / "unused")
            with patch.object(runner.subprocess, "run") as run:
                runner.execute(args, runner.plan(args))
                run.assert_not_called()
            self.assertFalse(Path(args.results_dir).exists())

    def test_conflicting_settings_do_not_overwrite_existing_run(self):
        args = self.args("--stage", "gcg")
        with tempfile.TemporaryDirectory() as temp:
            args.results_dir = temp
            path = Path(temp) / "run.json"
            path.write_text('{"old": "settings"}')
            with patch.object(runner.subprocess, "run") as run:
                with self.assertRaisesRegex(ValueError, "different run settings"):
                    runner.execute(args, runner.plan(args))
                run.assert_not_called()
                self.assertEqual(json.loads(path.read_text()), {"old": "settings"})

    def test_cli_works_outside_repository_without_model_downloads(self):
        result = subprocess.run([sys.executable, "-B", str(REPO / "scripts/run.py"),
                                 "experiment", "--model", "8b", "--stage", "tao", "--dry-run"],
                                cwd=tempfile.gettempdir(), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("llama-3-balcony", result.stdout)
        self.assertNotIn("1B", result.stdout)

    def test_stale_environment_cannot_select_shard_or_old_budget(self):
        args = self.args("--stage", "gcg", "--benchmarks", "harmbench")
        with tempfile.TemporaryDirectory() as temp:
            args.results_dir = temp
            with patch.dict(os.environ, {"TAO_SHARD": "old", "BEHAVIOR_START": "40", "SEARCH_WIDTH": "999"}):
                with patch.object(runner.subprocess, "run", side_effect=RuntimeError("captured")) as run:
                    with self.assertRaises(RuntimeError):
                        runner.execute(args, runner.plan(args))
                    env = run.call_args.kwargs["env"]
                    self.assertNotIn("TAO_SHARD", env)
                    self.assertNotIn("BEHAVIOR_START", env)
                    self.assertEqual(env["SEARCH_WIDTH"], "128")


if __name__ == "__main__":
    unittest.main()
