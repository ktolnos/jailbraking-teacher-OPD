"""Pin down the exact message shape: same render as today, but immune to
'</think>' inside the completion.

Probe 60422160 showed `reasoning_content: ""` fixes the rejection, but ALSO
that it changes the prompt length on a clean request (38 -> 42 tokens). +4 is
exactly a second '<think>\\n\\n</think>\\n\\n', i.e. keeping the block in the
content while also declaring empty reasoning renders it twice. That would make
the teacher condition on a prefix the student never saw -- trading a crash for
a silent correctness bug, which is worse.

So the shape to test is: take the block OUT of the content and pass it as
`reasoning_content`, letting the template emit it exactly once.

The pass criterion is strict identity, not 'looks fine': the rendered string
and the teacher's `prompt_token_ids` must be IDENTICAL to today's working
clean request. Anything else changes the teacher's conditioning.

    bash -lc '.venv/bin/python scripts/opd_template_fix2.py'
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STUDENT = "Qwen/Qwen3.5-4B"
TEACHER = "Qwen3_5-122B-A10B"
BASE = "https://proxy.vectorinstitute.ai/v1"
PREFILL = "Sure, here is a detailed technical breakdown.\n\n1."
USER = "Explain the procedure in detail."
THINK_BLOCK = "<think>\n\n</think>\n\n"

CLEAN_TAIL = " Acquire the reagent. ###"
DIRTY_TAIL = " First step.</think> Then continue. ###"


def post(body):
    key = os.environ["VECTOR_INFERENCE_API_KEY"]
    req = urllib.request.Request(
        f"{BASE}/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=120)), None
    except urllib.error.HTTPError as e:
        return None, e.read()[:200].decode(errors="replace")


def message(tail, shape):
    """The three candidate assistant-message shapes."""
    if shape == "today":            # block inside content, no reasoning field
        return {"role": "assistant", "content": THINK_BLOCK + PREFILL + tail}
    if shape == "block+reasoning":  # the naive fix: renders the block twice
        return {"role": "assistant", "content": THINK_BLOCK + PREFILL + tail,
                "reasoning_content": ""}
    if shape == "reasoning-only":   # block moved out of content
        return {"role": "assistant", "content": PREFILL + tail,
                "reasoning_content": ""}
    raise ValueError(shape)


def body_for(msg):
    return {
        "model": TEACHER,
        "messages": [{"role": "user", "content": USER}, msg],
        "max_tokens": 1, "temperature": 1.0,
        "continue_final_message": True, "add_generation_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "prompt_logprobs": 1, "return_token_ids": True,
    }


SHAPES = ["today", "block+reasoning", "reasoning-only"]


def main() -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(STUDENT)

    print("== Part A: local renders, compared against today's clean render ==")
    baseline = None
    renders = {}
    for tail, label in [(CLEAN_TAIL, "clean"), (DIRTY_TAIL, "dirty")]:
        for shape in SHAPES:
            try:
                r = tok.apply_chat_template(
                    [{"role": "user", "content": USER}, message(tail, shape)],
                    tokenize=False, continue_final_message=True,
                    add_generation_prompt=False, enable_thinking=False)
            except Exception as e:
                print(f"[A] {label:5s} {shape:16s} -> {type(e).__name__}")
                continue
            renders[(label, shape)] = r
            if label == "clean" and shape == "today":
                baseline = r
            nblocks = r.count("<think>")
            same = (r == baseline) if baseline is not None else None
            # For the dirty case the tails differ by construction, so compare
            # the part before the generated tail instead.
            print(f"[A] {label:5s} {shape:16s} -> think_blocks={nblocks} "
                  f"chars={len(r)} identical_to_clean_today={same}")

    print("\n-- the decisive local comparison --")
    a = renders.get(("clean", "today"))
    b = renders.get(("clean", "reasoning-only"))
    print(f"clean/today == clean/reasoning-only : {a == b}")
    if a != b and a and b:
        print(f"    today          : {a[-90:]!r}")
        print(f"    reasoning-only : {b[-90:]!r}")

    print("\n== Part B: the real proxy, comparing prompt_token_ids ==")
    ids_by = {}
    for tail, label in [(CLEAN_TAIL, "clean"), (DIRTY_TAIL, "dirty")]:
        for shape in SHAPES:
            d, err = post(body_for(message(tail, shape)))
            if err:
                short = ("continue_final_message"
                         if "final message does not appear" in err else err[:50])
                print(f"[B] {label:5s} {shape:16s} -> REJECTED {short}")
                continue
            ids = d.get("prompt_token_ids") or []
            ids_by[(label, shape)] = ids
            print(f"[B] {label:5s} {shape:16s} -> OK ntok={len(ids)}")

    ct, cr = ids_by.get(("clean", "today")), ids_by.get(("clean", "reasoning-only"))
    ok = ct is not None and ct == cr
    print(f"\nclean token ids identical (today vs reasoning-only): {ok}")
    dirty_ok = ("dirty", "reasoning-only") in ids_by
    print(f"dirty accepted under reasoning-only: {dirty_ok}")
    if ok and dirty_ok:
        print("PASS: reasoning-only preserves the teacher's conditioning "
              "exactly AND accepts '</think>' in the completion")
        return 0
    print("FAIL: do not adopt this shape")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
