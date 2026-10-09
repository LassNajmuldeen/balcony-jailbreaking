# Setup

GPU experiments need Linux, Python 3.12, an NVIDIA GPU, and a driver compatible
with the PyTorch CUDA 12.8 build. Memory needs depend on the model and search
width; begin with the small run in the README.

## Install

From a clone of this repository:

```bash
bash scripts/setup.sh
```

This initializes the pinned submodules, applies the two AdversariaLLM patches,
and installs the environment in `.venv/`. Repeating setup skips patches that
are already applied. Additional experiment dependencies are in `requirements.txt`.

For checkpoints that require access, accept their Hugging Face licenses and
authenticate with `.venv/bin/hf auth login`.

## Check

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.venv/bin/python scripts/run.py experiment --model 7b --stage gcg --dry-run
```

The first command should report CUDA availability. The second only prints the
planned commands and settings.

## Choose a model

| Preset | Checkpoint | Exits |
| --- | --- | --- |
| `1b` | `parsakaveh/Balcony-LLM-1B` | 4, 8, 12, full |
| `7b` | `parsakaveh/Balcony-LLaMA2-7B` | 15, 18, 21, full |
| `8b` | `adpretko/Balcony-Meta-LLaMA3-8B-exit-15-18-21` | 15, 18, 21, full |

Presets live in `experiments/models.py`. Each model uses its matching chat
template. Use `--plain` for the template-free prompt variant.

The full-depth exit is included in every preset. A separate stock checkpoint is
optional: select it with `--target baseline --baseline-model <checkpoint>`.
Use the same prompt template for comparisons between checkpoints.

## Figures only

Plotting the recorded results needs no GPU or model downloads:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install 'numpy>=1.26,<2' 'matplotlib>=3.10,<4'
.venv/bin/python scripts/run.py figures
```

See the [command reference](scripts/README.md) for individual stages and exports.
