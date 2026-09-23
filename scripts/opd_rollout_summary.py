"""CPU-only summary and integrity check for OPD training rollout audits.

Usage: python scripts/opd_rollout_summary.py PATH/rollouts-*.jsonl --window 10
Prints aggregate counts only; prompt and completion text stay on disk.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def summarize(path: Path, window: int) -> dict:
    manifest_path = path.with_suffix(".meta.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    steps = defaultdict(list)
    seen = set()
    duplicates = 0
    bad = 0
    for lineno, line in enumerate(path.open(encoding="utf-8"), 1):
        try:
            r = json.loads(line)
            ids = r["completion_ids"]
            lps = r["teacher_actual_logprobs"]
            topk = r["teacher_topk"]
            if (r["schema_version"] != 1 or not ids or
                    (manifest and r["run_id"] != manifest["run_id"]) or
                    len(ids) != len(lps) or len(ids) != len(topk) or
                    not all(pos for pos in topk) or
                    not all(math.isfinite(x) for x in lps) or
                    r["status"] != "teacher_scored_loss_computed"):
                raise ValueError("invalid schema, alignment, or score")
            key = (r["run_id"], r["step_before_update"], r["microbatch"],
                   r["process_rank"], r["sample_index"])
            if key in seen:
                duplicates += 1
            seen.add(key)
            steps[r["step_before_update"]].append(r)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
            bad += 1
            print(f"invalid audit row {lineno}: {e}")
    expected_per_step = manifest.get("effective_batch_size")
    expected_steps = manifest.get("max_steps")
    step_log = path.parent / manifest["step_log"] if "step_log" in manifest else None
    committed = set()
    if step_log is not None and step_log.exists():
        for line in step_log.open(encoding="utf-8"):
            marker = json.loads(line)
            if marker["run_id"] != manifest["run_id"]:
                raise ValueError("optimizer step marker has a different run_id")
            committed.add(marker["completed_step"] - 1)
    missing_steps = []
    wrong_size_steps = []
    if expected_steps is not None:
        missing_steps = [i for i in range(expected_steps) if i not in steps]
    if expected_per_step is not None:
        wrong_size_steps = [(i, len(rows)) for i, rows in sorted(steps.items())
                            if len(rows) != expected_per_step]
    windows = []
    for start in range(0, max(steps, default=-1) + 1, window):
        rows = [r for i in range(start, start + window) for r in steps.get(i, [])]
        if not rows:
            continue
        n_tokens = sum(len(r["completion_ids"]) for r in rows)
        windows.append({
            "steps": [start, start + window - 1],
            "rollouts": len(rows),
            "opener_rate": sum(r["starts_with_opener"] for r in rows) / len(rows),
            "mean_completion_tokens": n_tokens / len(rows),
            "mean_teacher_nll": -sum(sum(r["teacher_actual_logprobs"]) for r in rows) / n_tokens,
        })
    return {
        "run_id": manifest.get("run_id"),
        "rows": sum(map(len, steps.values())),
        "steps_observed": len(steps),
        "expected_steps": expected_steps,
        "expected_per_step": expected_per_step,
        "duplicate_keys": duplicates,
        "invalid_rows": bad,
        "committed_steps": len(committed),
        "uncommitted_rollout_steps": sorted(set(steps) - committed) if step_log else [],
        "missing_steps": missing_steps,
        "wrong_size_steps": wrong_size_steps,
        "windows": windows,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("audit", type=Path)
    p.add_argument("--window", type=int, default=10)
    args = p.parse_args()
    if args.window < 1:
        p.error("--window must be positive")
    print(json.dumps(summarize(args.audit, args.window), indent=2))
