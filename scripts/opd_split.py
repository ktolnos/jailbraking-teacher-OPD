"""The one canonical SciHazard train/eval split. Imported by both sides.

Previously the training draw lived in `opd_train.load_prompts` and the eval
draw reconstructed its complement by *reproducing* that code. That worked only
while the two stayed byte-identical, and it silently caps the training set:
the old draw took 40/category (280 prompts) and trained on all of them, so
"use more data" cannot be expressed in it at all.

This module owns the split instead:

  * eval  = `eval_per_cat` prompts from every CBRN category, balanced, so no
            category dominates the judged comparison (the raw pool is very
            unbalanced -- one category holds ~35% of it);
  * train = everything else CBRN.

Both halves are derived in one pass from one seed, so they are disjoint by
construction rather than by a reconstruction that could drift. Scenario-driven
categories are excluded on both sides, as in every earlier script.

Prints and returns counts only -- never question text.
"""
from __future__ import annotations

import json
import os
import random
import urllib.request

SRC = ("https://huggingface.co/datasets/SciHazardBench/SciHazardBench/"
       "resolve/main/2400unsafe_final.jsonl")
EVAL_PER_CAT = 60
SEED = 7


def load_rows():
    req = urllib.request.Request(
        SRC, headers={"Authorization": "Bearer " + os.environ["HF_TOKEN"]})
    raw = urllib.request.urlopen(req, timeout=120).read().decode()
    rows = [json.loads(l) for l in raw.splitlines() if l.strip()]
    if rows and "question" not in rows[0]:
        raise KeyError(f"SciHazardBench rows have no 'question'; keys are "
                       f"{sorted(rows[0].keys())}")
    return rows


def split(eval_per_cat: int = EVAL_PER_CAT, seed: int = SEED):
    """Return (train_rows, eval_rows), disjoint, CBRN only."""
    rows = load_rows()
    cbrn = [r for r in rows if not r["category"].startswith("Scenario-driven")]
    rng = random.Random(seed)
    train, ev = [], []
    for c in sorted({r["category"] for r in cbrn}):
        pool = sorted([r for r in cbrn if r["category"] == c],
                      key=lambda r: r["question"])  # stable before shuffling
        rng.shuffle(pool)
        ev += pool[:eval_per_cat]
        train += pool[eval_per_cat:]
    rng.shuffle(train)
    rng.shuffle(ev)
    # Disjointness is the whole contract; assert it rather than trust it.
    qt = {r["question"] for r in train}
    qe = {r["question"] for r in ev}
    if qt & qe:
        raise AssertionError(f"train/eval overlap: {len(qt & qe)} prompts")
    return train, ev


if __name__ == "__main__":
    tr, ev = split()
    print(f"CBRN train {len(tr)} | eval {len(ev)} | disjoint OK")
