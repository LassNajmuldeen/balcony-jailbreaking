# benchmark loading, shared between the attack scripts.
#
# harmbench and advbench ship different columns overall, but both csvs carry
# Behavior / BehaviorID / ContextString and both target files are a flat
# {BehaviorID: target_string} mapping, so one loader covers the two of them. this
# is the same loader attack_gcg_advbench.py already had inline, just lifted out
# here so the baseline and TAO scripts can run on either benchmark instead of
# being hardwired to harmbench.

import pandas as pd

from config import ROOT

# name goes to (behaviors csv, optimizer targets json)
BENCHMARKS = {
    "harmbench": (
        f"{ROOT}/data/behavior_datasets/harmbench_behaviors_text_all.csv",
        f"{ROOT}/data/optimizer_targets/harmbench_targets_text.json",
    ),
    "advbench": (
        f"{ROOT}/data/behavior_datasets/extra_behavior_datasets/advbench_behaviors.csv",
        f"{ROOT}/data/optimizer_targets/extra_targets/advbench_targets.json",
    ),
}


class BehaviorDataset:
    """behaviors + targets, handed back as the [user_msg, assistant_msg] pairs
    AdversariaLLM wants to see."""

    def __init__(self, messages_path, targets_path, n_behaviors=None):
        self.messages = pd.read_csv(messages_path, header=0)
        targets = pd.read_json(targets_path, typ="series")
        # the targets json is keyed by BehaviorID, so map it onto the csv order
        self.targets = self.messages["BehaviorID"].map(targets)

        # a behavior with no optimizer target cant be attacked at all, so drop it
        # here instead of dragging a NaN target into the loss later on
        valid = self.targets.notna()
        self.messages = self.messages[valid].reset_index(drop=True)
        self.targets = self.targets[valid].reset_index(drop=True)

        # truncation happens AFTER the drop, so n_behaviors=50 always means 50
        # usable behaviors and not "50 rows, some of which were junk"
        if n_behaviors is not None:
            self.messages = self.messages.iloc[:n_behaviors].reset_index(drop=True)
            self.targets = self.targets.iloc[:n_behaviors].reset_index(drop=True)

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        msg = self.messages.iloc[idx]
        content = msg["Behavior"]
        # some harmbench behaviors only make sense with their context paragraph in
        # front of them, advbench never has one. the .get is because advbench's csv
        # doesnt even have the column
        if isinstance(msg.get("ContextString"), str) and msg["ContextString"]:
            content = msg["ContextString"] + "\n\n" + content
        return [
            {"role": "user", "content": content},
            {"role": "assistant", "content": self.targets.iloc[idx]},
        ]


def load_benchmark(name, n_behaviors=None):
    if name not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {name!r}; known: {sorted(BENCHMARKS)}")
    messages_path, targets_path = BENCHMARKS[name]
    return BehaviorDataset(messages_path, targets_path, n_behaviors=n_behaviors)
