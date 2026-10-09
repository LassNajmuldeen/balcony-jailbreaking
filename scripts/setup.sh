#!/usr/bin/env bash
# Install the pinned experiment environment on Linux with a CUDA GPU.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
git submodule update --init --recursive
for patch in "$REPO"/patches/*.patch; do
  if git -C AdversariaLLM apply --reverse --check "$patch" 2>/dev/null; then
    continue
  fi
  git -C AdversariaLLM apply --check "$patch"
  git -C AdversariaLLM apply "$patch"
done
"${PYTHON:-python3.12}" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install 'torch==2.7.1' --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -e ./AdversariaLLM -r requirements.txt
printf '\nEnvironment ready. Run .venv/bin/python scripts/run.py --help\n'
