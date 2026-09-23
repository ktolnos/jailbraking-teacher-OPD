"""CPU+network test for canonicalize-before-loss. No GPU, no student model.

`opd_align_test.py` could not have caught the bug that killed job 60405196:
it builds every completion with `tok(completion_text)`, i.e. canonically, so
the text round trip was the identity by construction. The real student samples
token *sequences*, which need not be the canonical encoding of their own text
(observed: `['yl','ether']` where canonical is `['yle','ther']`).

This test therefore builds deliberately NON-canonical completions -- by
tokenizing a string in pieces and concatenating, which is how a sampler can
land off the greedy-BPE path -- and asserts:

  Part 1 (tokenizer only)
    * the constructed cases really are non-canonical (else the test is vacuous
      and fails loudly rather than passing for the wrong reason);
    * `canonicalize_pair` preserves the decoded text exactly;
    * its output completion is canonical in context.

  Part 2 (real proxy)
    * every canonicalized rollout gets a finite teacher logprob on every
      completion token -- the condition `iw_opd_trainer.py:1488` enforces.
    * the same rollouts WITHOUT canonicalization are also run, so the log
      shows the fix changing the outcome rather than a green tick alone.

    bash -lc '.venv/bin/python scripts/opd_canon_test.py'
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STUDENT = "Qwen/Qwen3.5-4B"
PREFILL = "Sure, here is a detailed technical breakdown.\n\n1."
USER = "Explain the procedure in detail."

# (text, kind). For "noncanonical" the test *searches* for a split of the text
# whose piecewise tokenization differs from the canonical one -- hand-picked
# splits are unreliable (the first attempt at this produced canonical ids and
# the test correctly refused to run). This reproduces the real phenomenon
# (a sampled sequence that is not its own text's greedy-BPE encoding) without
# needing hazardous content.
CASES = [
    (" Decabromodiphenyl ether is the precursor.", "noncanonical"),
    (" Heat to 80C and filter under vacuum.", "noncanonical"),
    (" The ratio is 3.2:1, yielding ~45%.", "noncanonical"),
    (" Acquire 23 g of the reagent and stir at 0 C.\n\n", "canonical"),
    (" The first step is to prepare the precursor.", "canonical"),
    ("Mix compound A with B, maintaining pH 7.4.", "fused-boundary"),
    ("H2SO4 and HNO3 combined.", "fused-boundary"),
    # Regression for the HTTP 400 that killed job 60419803: a student that
    # generates '</think>' made the chat template rewrite the assistant
    # content, so `continue_final_message` could not find it. Fixed by
    # sending the think block as `reasoning_content` (_split_reasoning).
    (" First step.</think> Then continue with the procedure.", "close-think"),
    (" A <think>aside</think> B, then heat to 80C.", "close-think"),
    # Real interior SPECIAL TOKEN, injected as an id rather than as text:
    # job 60422731's gate hit one in 6 rollouts. Identical text does not imply
    # identical ids for these, so canonicalize_pair checks them explicitly.
    (" First step.|SPECIAL| Then continue.", "interior-special"),
]

# Special tokens the student can plausibly emit mid-completion.
SPECIAL_PROBES = ["</think>", "<think>", "<|im_end|>"]


def noncanonical_ids(tok, text):
    """Find a split of `text` whose piecewise tokenization is NOT canonical.

    Returns the non-canonical id list, or None if every split re-encodes
    canonically (in which case the case cannot be used and the test says so
    instead of passing vacuously).
    """
    canon = tok(text, add_special_tokens=False)["input_ids"]
    for i in range(1, len(text)):
        a = tok(text[:i], add_special_tokens=False)["input_ids"]
        b = tok(text[i:], add_special_tokens=False)["input_ids"]
        ids = a + b
        if ids != canon and tok.decode(ids, skip_special_tokens=False) == text:
            return ids
    return None


def build(tok, text, kind):
    """Return (prompt_ids, completion_ids), non-canonical where asked for."""
    gen = tok.apply_chat_template(
        [{"role": "user", "content": USER}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt_ids = tok(gen + PREFILL, add_special_tokens=False)["input_ids"]
    if kind == "noncanonical":
        completion_ids = noncanonical_ids(tok, text)
    elif kind == "interior-special":
        # Build the completion the way the model would: a real special-token
        # ID in the middle, not the literal characters.
        sid = tok.convert_tokens_to_ids("</think>")
        left, right = text.split("|SPECIAL|")
        completion_ids = (tok(left, add_special_tokens=False)["input_ids"]
                          + [sid]
                          + tok(right, add_special_tokens=False)["input_ids"])
    else:
        completion_ids = tok(text, add_special_tokens=False)["input_ids"]
    return prompt_ids, completion_ids


def score(client, prompt_ids, completion_ids):
    """Return (ok, detail) for one rollout against the real teacher."""
    from opd_teacher_client import TeacherAlignmentError
    n = len(completion_ids)
    try:
        res = client.get_sequence_logprobs(
            [list(prompt_ids) + list(completion_ids)], [len(prompt_ids)],
            top_logprobs=1, temperature=1.0)
    except TeacherAlignmentError:
        return False, "MISALIGNED"
    actual = res["actual_logprobs"][0]
    finite = sum(1 for v in actual if v[0] != -math.inf)
    if len(actual) == n and finite == n:
        return True, "OK"
    return False, f"FAIL(finite={finite}/{n})"


def main() -> int:
    from transformers import AutoTokenizer

    from opd_canonicalize import canonicalize_pair
    from opd_teacher_client import ProxyTeacherClient

    tok = AutoTokenizer.from_pretrained(STUDENT)

    # ---- Part 1: tokenizer-only invariants -----------------------------
    print("== Part 0: which markers are 'special' to this tokenizer? ==")
    for m in SPECIAL_PROBES:
        i = tok.convert_tokens_to_ids(m)
        print(f"[0] {m:12s} id={i} in_all_special_ids={i in set(tok.all_special_ids)}")
    print()

    print("== Part 1: canonicalize_pair invariants ==")
    prepared, noncanon_built = [], 0
    for idx, (case_text, kind) in enumerate(CASES):
        p_ids, c_ids = build(tok, case_text, kind)
        if c_ids is None:
            print(f"[{idx}] SETUP FAIL: no non-canonical split exists for "
                  f"this text; pick a different one")
            return 1
        text = tok.decode(c_ids, skip_special_tokens=False)
        is_canon = tok(text, add_special_tokens=False)["input_ids"] == c_ids
        if kind == "noncanonical":
            if is_canon:
                print(f"[{idx}] SETUP FAIL: case marked noncanonical is canonical")
                return 1
            noncanon_built += 1

        np_ids, nc_ids, moved = canonicalize_pair(tok, p_ids, c_ids)

        before = tok.decode(list(p_ids) + list(c_ids), skip_special_tokens=False)
        after = tok.decode(list(np_ids) + list(nc_ids), skip_special_tokens=False)
        if before != after:
            print(f"[{idx}] FAIL: text not preserved")
            return 1
        joint = tok(tok.decode(np_ids, skip_special_tokens=False)
                    + tok.decode(nc_ids, skip_special_tokens=False),
                    add_special_tokens=False)["input_ids"]
        in_context_canon = joint == list(np_ids) + list(nc_ids)

        prompt_intact = list(np_ids) == list(p_ids)
        print(f"[{idx}] {kind:14s} n={len(c_ids):3d}->{len(nc_ids):3d} "
              f"was_canonical={str(is_canon):5s} moved={moved} "
              f"in_context_canonical={in_context_canon} "
              f"prompt_intact={prompt_intact}")
        if not in_context_canon:
            print(f"[{idx}] FAIL: output is still not canonical in context")
            return 1
        # Without boundary fusion the prompt must come back bit-identical.
        # If it does not, the chat template's special tokens ("<think>",
        # "<|im_start|>") are not round-tripping through decode/encode, which
        # would silently reshape the prompt block.
        if moved == 0 and not prompt_intact:
            print(f"[{idx}] FAIL: prompt ids changed with no boundary fusion "
                  "-- special tokens are not round-tripping")
            return 1
        prepared.append((idx, kind, p_ids, c_ids, np_ids, nc_ids))

    if not noncanon_built:
        print("SETUP FAIL: no genuinely non-canonical case was constructed")
        return 1
    print(f"part 1 OK ({noncanon_built} genuinely non-canonical cases built)\n")

    # ---- Part 2: real proxy, before vs after ---------------------------
    print("== Part 2: teacher alignment, raw vs canonicalized ==")
    client = ProxyTeacherClient(tok)
    fixed = raw_ok = canon_ok = 0
    failures = []
    for idx, kind, p_ids, c_ids, np_ids, nc_ids in prepared:
        ok_raw, det_raw = score(client, p_ids, c_ids)
        ok_canon, det_canon = score(client, np_ids, nc_ids)
        raw_ok += ok_raw
        canon_ok += ok_canon
        if ok_canon and not ok_raw:
            fixed += 1
        if not ok_canon:
            failures.append((idx, kind, det_canon))
        print(f"[{idx}] {kind:14s} raw={det_raw:18s} canonicalized={det_canon}")

    total = len(prepared)
    print(f"\nclient stats: {client.stats}")
    print(f"raw:           {raw_ok}/{total} aligned")
    print(f"canonicalized: {canon_ok}/{total} aligned  ({fixed} fixed by the change)")
    if failures:
        for idx, kind, det in failures:
            print(f"FAIL [{idx}] {kind}: {det}")
        return 1
    print("PASS: every rollout -> finite teacher logprob on all completion tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
