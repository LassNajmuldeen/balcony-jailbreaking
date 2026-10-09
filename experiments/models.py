"""Checkpoint and exit presets shared by the command line tools."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    checkpoint: str
    baseline: str
    exits: tuple
    layers: int
    template: str
    results: str


MODELS = {
    "1b": ModelSpec("parsakaveh/Balcony-LLM-1B", "parsakaveh/LLM-1B-Pretrained",
                    (4, 8, 12, "full"), 16, "", "results"),
    "7b": ModelSpec("parsakaveh/Balcony-LLaMA2-7B", "NousResearch/Llama-2-7b-hf",
                    (15, 18, 21, "full"), 32, "llama-2-chat", "results_7b"),
    "8b": ModelSpec("adpretko/Balcony-Meta-LLaMA3-8B-exit-15-18-21",
                    "NousResearch/Meta-Llama-3-8B", (15, 18, 21, "full"),
                    32, "llama-3-balcony", "results_8b"),
}
