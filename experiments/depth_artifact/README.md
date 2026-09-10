# depth artifact pipeline

Rebuilds the published page at
balcony-depth.html

Run in order, from anywhere:

    ../../AdversariaLLM/.venv/bin/python recompute.py   # -> asr.json, mcnemar.json
    ../../AdversariaLLM/.venv/bin/python util.py        # -> util.json
    ../../AdversariaLLM/.venv/bin/python loss.py        # -> loss.json
    ../../AdversariaLLM/.venv/bin/python build_depth.py # -> balcony-depth.html

Every number on the page comes from these four JSONs, which are derived from the
per-behaviour scores in results*/judge_scores.json and the loss curves in
results*/*attacks*.json. Nothing is hand-entered, so the page cannot drift from
the tables the way the 2026-08-05 version did.

Republish by passing the artifact URL above as `url`, otherwise a new
conversation mints a fresh URL instead of updating this one.
