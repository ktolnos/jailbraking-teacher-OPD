#!/usr/bin/env python3
"""Check which HF datasets this token can actually READ.

Why this exists: neither of the two obvious checks is reliable.

  - `/api/datasets/{id}` returns 200 and a full file listing for gated repos
    you have NOT been granted, so a 200 proves existence, not access.
  - The datasets-server (`/info`, `/rows`) 404s on repos that read fine when
    its viewer cannot parse the files -- wildjailbreak (TSV), ScaleAI/mhj,
    HEx-PHI and jinjh0123/bbg are all in this category.

The only reliable test is a ranged GET of an actual data file, which is what
this does. Run under `bash -lc` so HF_TOKEN is in the environment.

    bash -lc 'python3 scripts/check_dataset_access.py [repo_id ...]'
"""
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT = [
    "AlignmentResearch/ClearHarm", "carolinewei/ClearHarm_prefills",
    "allenai/wildjailbreak", "allenai/wildguardmix",
    "sorry-bench/sorry-bench-202503",
    "sorry-bench/sorry-bench-human-judgment-202503",
    "OpenSafetyLab/Salad-Data", "IS2Lab/S-Eval", "Babelscape/ALERT",
    "declare-lab/CategoricalHarmfulQA", "LLM-Tuning-Safety/HEx-PHI",
    "ScaleAI/mhj", "Mechanistic-Anomaly-Detection/llama3-jailbreaks",
    "JailbreakBench/JBB-Behaviors", "walledai/AdvBench",
    "cais/wmdp", "futurehouse/lab-bench", "Idavidrein/gpqa",
    "walledai/CyberSecEval", "PKU-Alignment/PKU-SafeRLHF",
    "oskarvanderwal/bbq", "jinjh0123/bbg", "Anthropic/discrim-eval",
    "bench-llm/or-bench", "natolambert/xstest-v2-copy",
    "stanford-crfm/air-bench-2024",
]
DATA_EXT = ("tsv", "csv", "jsonl", "parquet", "json", "txt")


def _req(url, token, **headers):
    r = urllib.request.Request(url)
    r.add_header("Authorization", f"Bearer {token}")
    for k, v in headers.items():
        r.add_header(k, v)
    return r


def check(ds, token):
    try:
        info = json.load(urllib.request.urlopen(
            _req(f"https://huggingface.co/api/datasets/{ds}", token), timeout=45))
    except urllib.error.HTTPError as e:
        return f"{ds:52s} repo not visible (API {e.code})"

    files = [s["rfilename"] for s in info.get("siblings", [])]
    data = [f for f in files if f.rsplit(".", 1)[-1] in DATA_EXT]
    if not data:
        return f"{ds:52s} exists, no data file found among {len(files)} files"

    target = data[0]
    try:
        urllib.request.urlopen(
            _req(f"https://huggingface.co/datasets/{ds}/resolve/main/{target}",
                 token, Range="bytes=0-64"), timeout=60)
        return f"{ds:52s} READABLE       ({len(data)} data files, e.g. {target})"
    except urllib.error.HTTPError as e:
        gate = " -- accept terms at the repo page" if e.code == 403 else ""
        return f"{ds:52s} NOT READABLE {e.code}{gate}"


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN unset. Run under `bash -lc` -- a non-login shell can "
                 "inherit a stale environment without it.")
    for ds in (sys.argv[1:] or DEFAULT):
        print(check(ds, token))


if __name__ == "__main__":
    main()
