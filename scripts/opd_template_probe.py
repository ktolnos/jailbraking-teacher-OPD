"""Isolate the HTTP 400 'final message does not appear in the chat'.

Job 60419803 trained two real steps, then the teacher proxy rejected a request
with:

    continue_final_message is set but the final message does not appear in the
    chat after applying the chat template!

The Qwen3.5 template rewrites assistant content whenever '</think>' occurs in
it (tokenizer_config.json chat_template, the `'</think>' in content` branch):
it splits the string and re-emits the parts, so the original content is no
longer a substring of the render and transformers' `rindex(final_message)`
check fails. Our assistant content ALWAYS begins with the empty think block,
so that branch should fire on every request -- yet most requests succeed.
That contradiction means the mechanism is not yet understood, so this probe
measures it instead of assuming.

Part A replicates the check locally with the same tokenizer and template, so
the trigger is identified without a GPU or the network. Part B sends the same
payloads to the real proxy, because the server's transformers version and
vLLM's own preprocessing may differ from ours -- if A and B disagree, that
difference is itself the answer.

    bash -lc '.venv/bin/python scripts/opd_template_probe.py'
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STUDENT = "Qwen/Qwen3.5-4B"
PREFILL = "Sure, here is a detailed technical breakdown.\n\n1."
USER = "Explain the procedure in detail."
THINK_BLOCK = "<think>\n\n</think>\n\n"

# Completion shapes. The markers are the suspects: the template branches on
# '</think>', and '<think>' / '<|im_end|>' reach other branches.
CASES = [
    ("plain", " Acquire the reagent and stir at 0 C."),
    ("trailing-newlines", " Acquire the reagent and stir.\n\n"),
    ("contains-close-think", " First step.</think> Then continue."),
    ("contains-open-think", " First step.<think> Then continue."),
    ("contains-both-think", " A <think>reasoning</think> B."),
    ("contains-im-end", " First step.<|im_end|> Then continue."),
    ("contains-im-start", " First step.<|im_start|> Then continue."),
    ("long-256ish", " Step. " * 60),
]


def local_check(tok, content: str):
    """Replicate transformers' continue_final_message check locally."""
    messages = [
        {"role": "user", "content": USER},
        {"role": "assistant", "content": content},
    ]
    try:
        rendered = tok.apply_chat_template(
            messages, tokenize=False, continue_final_message=True,
            add_generation_prompt=False, enable_thinking=False)
    except Exception as e:  # the ValueError we are chasing
        return False, f"{type(e).__name__}: {str(e)[:60]}"
    return True, f"rendered {len(rendered)} chars"


def main() -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(STUDENT)

    print("== Part A: local apply_chat_template (same template as the server) ==")
    local = {}
    for name, comp in CASES:
        content = THINK_BLOCK + PREFILL + comp + " ###"
        ok, detail = local_check(tok, content)
        local[name] = ok
        print(f"[A] {name:22s} local_ok={str(ok):5s} {detail}")

    # Does the think block alone trigger it? This is the control that decides
    # whether the mechanism is our prefix or the generated text.
    print("\n[A] controls:")
    for label, content in [
        ("think-block only", THINK_BLOCK + PREFILL + " Acquire. ###"),
        ("no think block", PREFILL + " Acquire. ###"),
    ]:
        ok, detail = local_check(tok, content)
        print(f"    {label:22s} local_ok={str(ok):5s} {detail}")

    print("\n== Part B: the real proxy ==")
    try:
        from opd_teacher_client import ProxyTeacherClient
        client = ProxyTeacherClient(tok)
    except Exception as e:
        print(f"could not build client: {e}")
        return 1

    gen = tok.apply_chat_template(
        [{"role": "user", "content": USER}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt_ids = tok(gen + PREFILL, add_special_tokens=False)["input_ids"]

    disagree = 0
    for name, comp in CASES:
        comp_ids = tok(comp, add_special_tokens=False)["input_ids"]
        try:
            client.get_sequence_logprobs(
                [prompt_ids + comp_ids], [len(prompt_ids)],
                top_logprobs=1, temperature=1.0)
            server_ok, detail = True, "OK"
        except Exception as e:
            msg = str(e)
            server_ok = False
            if "final message does not appear" in msg:
                detail = "HTTP400 continue_final_message"
            else:
                detail = f"{type(e).__name__}: {msg[:70]}"
        if server_ok != local[name]:
            disagree += 1
        print(f"[B] {name:22s} server_ok={str(server_ok):5s} "
              f"local_ok={str(local[name]):5s} {detail}")

    print(f"\nlocal/server disagreements: {disagree}/{len(CASES)}")
    print("client stats:", client.stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
