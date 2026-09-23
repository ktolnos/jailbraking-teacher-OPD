"""Replay captured alignment failures CPU-side. No GPU, no trainer.

`ProxyTeacherClient._dump_failure` writes each failing rollout to
`$OUTDIR/opd_align_failures.jsonl` with everything needed to reproduce it:
the student `completion_ids`, the teacher's returned `teacher_ids`, and the
reconstructed prompt split. This loads them and characterises the divergence
so the fix comes from the real failing data, not a hypothesis.

    bash -lc '.venv/bin/python scripts/opd_align_debug.py [path.jsonl]'
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STUDENT = "Qwen/Qwen3.5-4B"


def main() -> int:
    from transformers import AutoTokenizer

    from opd_teacher_client import _find_run

    path = (sys.argv[1] if len(sys.argv) > 1
            else os.path.join(os.environ.get("OUTDIR", "."),
                              "opd_align_failures.jsonl"))
    if not os.path.exists(path):
        print(f"no failure dump at {path}")
        return 0
    tok = AutoTokenizer.from_pretrained(STUDENT)
    rows = [json.loads(l) for l in open(path) if l.strip()]
    print(f"{len(rows)} captured failures in {path}\n")

    for idx, r in enumerate(rows):
        comp = r["completion_ids"]
        tids = r["teacher_ids"]
        n = len(comp)

        # Is the completion its own canonical tokenization? (decode->encode)
        text = tok.decode(comp, skip_special_tokens=False)
        canon = tok(text, add_special_tokens=False)["input_ids"]
        is_canon = canon == comp

        # How much of the completion appears as a contiguous run, from the tail?
        best_drop = None
        for drop in range(n):
            if _find_run(tids, comp[drop:]) is not None:
                best_drop = drop
                break

        # Where does the student vs canonical tokenization first diverge?
        div = next((i for i, (a, b) in enumerate(zip(comp, canon)) if a != b),
                   None)

        print(f"[{idx}] n={n} canonical={is_canon} "
              f"leading_tokens_unmatched={best_drop} "
              f"first_noncanon_pos={div}")
        if not is_canon:
            print(f"     student[{div}:{div + 4}]  = {comp[div:div + 4]} "
                  f"{[tok.decode([t]) for t in comp[div:div + 4]]}")
            print(f"     canonical[{div}:{div + 4}] = {canon[div:div + 4]} "
                  f"{[tok.decode([t]) for t in canon[div:div + 4]]}")
        if best_drop:
            print(f"     dropped leading: {comp[:best_drop]} "
                  f"{[tok.decode([t]) for t in comp[:best_drop]]}")
        print(f"     completion head: {text[:80]!r}")
        print()

    # Verdict: is the failure explained by non-canonical generation, or by a
    # boundary effect (small leading drop), or neither?
    noncanon = sum(1 for r in rows
                   if tok(tok.decode(r["completion_ids"],
                                     skip_special_tokens=False),
                          add_special_tokens=False)["input_ids"]
                   != r["completion_ids"])
    print(f"summary: {noncanon}/{len(rows)} failures are non-canonical "
          "generations (decode->encode changed the tokens)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
