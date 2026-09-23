"""Test the fix for the '</think>-in-completion' rejection, and check whether
the teacher has been conditioning on the same prefix as the student.

The trigger (probe 60421918, local and server agreeing 8/8): the Qwen3.5
template rewrites assistant content when '</think>' occurs in it, so a
completion that emits '</think>' makes the content vanish from the render and
`continue_final_message` fails.

Candidate fix: the template only takes that branch when the message has no
explicit `reasoning_content`:

    {%- if message.reasoning_content is string %}
        {%- set reasoning_content = message.reasoning_content %}
    {%- else %}
        {%- if '</think>' in content %}   <-- the branch that breaks us

So sending `reasoning_content: ""` should make the template treat the content
as opaque regardless of what the student generated.

Second question, independent of the bug: because that branch DOES fire today,
the template strips our leading '<think>\\n\\n</think>\\n\\n' out of the render.
If so, the teacher has been scoring completions conditioned on a prefix the
student never saw -- a silent teacher/student mismatch that would weaken every
logprob we have used so far. The probe reads back `prompt_token_ids` and checks
for the block directly rather than reasoning about the template.

    bash -lc '.venv/bin/python scripts/opd_template_fix_probe.py'
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


def request_body(content, reasoning=None):
    msg = {"role": "assistant", "content": content}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    return {
        "model": TEACHER,
        "messages": [{"role": "user", "content": USER}, msg],
        "max_tokens": 1, "temperature": 1.0,
        "continue_final_message": True, "add_generation_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "prompt_logprobs": 1, "return_token_ids": True,
    }


def main() -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(STUDENT)

    clean = THINK_BLOCK + PREFILL + " Acquire the reagent. ###"
    dirty = THINK_BLOCK + PREFILL + " First step.</think> Then continue. ###"

    print("== Part A: local template, with and without reasoning_content ==")
    for label, content in [("clean", clean), ("dirty(</think>)", dirty)]:
        for reasoning in (None, ""):
            msg = {"role": "assistant", "content": content}
            if reasoning is not None:
                msg["reasoning_content"] = reasoning
            try:
                r = tok.apply_chat_template(
                    [{"role": "user", "content": USER}, msg],
                    tokenize=False, continue_final_message=True,
                    add_generation_prompt=False, enable_thinking=False)
                has_block = THINK_BLOCK in r
                res = f"OK  think_block_in_render={has_block}"
            except Exception as e:
                res = f"{type(e).__name__}"
            tag = "reasoning_content=''" if reasoning is not None else "no reasoning_content"
            print(f"[A] {label:16s} {tag:22s} -> {res}")

    print("\n== Part B: the real proxy ==")
    for label, content in [("clean", clean), ("dirty(</think>)", dirty)]:
        for reasoning in (None, ""):
            d, err = post(request_body(content, reasoning))
            tag = "reasoning_content=''" if reasoning is not None else "no reasoning_content"
            if err:
                short = ("continue_final_message"
                         if "final message does not appear" in err
                         else err[:60])
                print(f"[B] {label:16s} {tag:22s} -> REJECTED {short}")
                continue
            ids = d.get("prompt_token_ids") or []
            text = tok.decode(ids, skip_special_tokens=False)
            print(f"[B] {label:16s} {tag:22s} -> OK  "
                  f"teacher_saw_think_block={THINK_BLOCK in text}  "
                  f"prefill_in_prompt={PREFILL in text}  ntok={len(ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
