"""Canonicalize student rollouts so the teacher scores the tokens we train on.

The problem this solves
-----------------------
The Vector proxy exposes only `/v1/chat/completions`, which takes *text*. So
scoring a rollout means `student ids -> text -> teacher re-tokenizes`. That
round trip is the identity only when the student's sampled token sequence is
the canonical (greedy-BPE) tokenization of its own text. Sampling does not
guarantee that: job 60405196 produced `['yl','ether']` (3791, 2636) where the
canonical encoding of the same characters is `['yle','ther']` (937, 677). The
teacher then scores a different segmentation and per-token alignment is
impossible -- not approximately, but in principle.

The fix
-------
Teacher and student share one tokenizer (both Qwen3.5). So we can compute,
client-side and exactly, the tokenization the teacher *will* produce, and make
the student's ids equal to it before the forward pass and before the teacher
call. Concretely, for each rollout we re-tokenize `head + body` as one string
(`head` = the rendered prompt incl. the prefill, `body` = the generated text)
and cut it at the first token boundary at or after `len(head)`.

Two things fall out of doing it jointly rather than on the completion alone:

  * **interior** non-canonicality is gone by construction -- the completion is
    now a contiguous run of the teacher's own tokenization;
  * **boundary fusion** (a completion starting with a bare word right after the
    prefill's "1.", so its first token merges with the prefill) stops being an
    error. The fused token simply falls on the prompt side of the cut, where it
    is masked from the loss and never sent to the teacher. We lose <=1 token of
    training signal on those rollouts instead of losing the rollout.

What this costs
---------------
The gradient is taken on the canonical tokenization of the sampled *text*
rather than on the exact sampled token sequence. That is a real approximation:
for a non-canonical rollout the student is trained on a segmentation it did not
itself emit. It is off-policy by exactly one re-segmentation. The honest fix is
to send token ids to the teacher (`/v1/completions`, see
VECTOR_COMPLETIONS_REQUEST.md) -- this module exists so the pilot can run
before that lands, and should be deleted when it does.

Note `rollout_logprobs` is *not* affected: the `_generate_with_model` path does
not record generation-time logprobs, so `_compute_iw_opd_loss` falls back to
the fresh forward-pass student logprobs (`iw_opd_trainer.py:1249-1250`). The
importance weight therefore compares teacher and student on the same canonical
ids, with nothing stale in between. If a vLLM student-generation path is ever
enabled, `rollout_logprobs` *would* go stale here and must be recomputed.
"""

from __future__ import annotations

import json
import math
import os

import torch
import torch.nn.functional as F

from trl.experimental.iw_opd.iw_opd_trainer import IWOPDTrainer


class CanonicalizationError(RuntimeError):
    """Raised when a rollout cannot be re-tokenized without changing its text."""


def canonicalize_pair(tokenizer, prompt_ids: list[int], completion_ids: list[int]):
    """Re-cut one (prompt, completion) pair onto canonical token boundaries.

    Returns `(new_prompt_ids, new_completion_ids, moved)` where `moved` is the
    number of completion tokens absorbed into the prompt by boundary fusion
    (0 in the overwhelmingly common case). The concatenation is guaranteed to
    decode to exactly the same text as the input -- that invariant is asserted,
    not assumed.
    """
    special = set(tokenizer.all_special_ids)

    # Trailing specials (EOS) are kept verbatim: round-tripping them through
    # text would depend on the tokenizer re-parsing "<|im_end|>" as an id.
    cut = len(completion_ids)
    while cut > 0 and completion_ids[cut - 1] in special:
        cut -= 1
    body_ids, tail_ids = list(completion_ids[:cut]), list(completion_ids[cut:])

    head = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    body = tokenizer.decode(body_ids, skip_special_tokens=False)

    enc = tokenizer(head + body, add_special_tokens=False,
                    return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]

    start = len(head)
    i0 = len(ids)
    for i, (a, _b) in enumerate(offsets):
        if a >= start:
            i0 = i
            break

    new_prompt = list(ids[:i0])
    new_completion = list(ids[i0:]) + tail_ids

    # Text preservation is the whole contract. Check it rather than trust it.
    before = tokenizer.decode(list(prompt_ids) + list(completion_ids),
                              skip_special_tokens=False)
    after = tokenizer.decode(new_prompt + new_completion,
                             skip_special_tokens=False)
    if before != after:
        raise CanonicalizationError(
            "re-tokenization changed the text "
            f"({len(before)} chars -> {len(after)} chars)"
        )

    # Interior special tokens (the student does emit them -- job 60422731 saw
    # one in 6 gate rollouts) survive the round trip only if the tokenizer
    # re-parses e.g. "</think>" back to its own id rather than to literal
    # pieces. Identical text does not imply identical ids here, so check the
    # special-token subsequence explicitly: same ids, same order. Refusing
    # outright would be safe but leaves a hole -- a rollout that is BOTH
    # special-bearing and non-canonical would fall back to raw ids and then
    # fail alignment.
    def specials(seq):
        return [i for i in seq if i in special]

    if specials(body_ids) != specials(new_completion[:len(new_completion)
                                                     - len(tail_ids)]):
        names = [tokenizer.decode([i]) for i in specials(body_ids)]
        raise CanonicalizationError(
            f"special tokens did not survive re-tokenization: {names} "
            f"(ids {specials(body_ids)}); the student's ids and the teacher's "
            "would disagree despite identical text"
        )

    # How many tokens of real completion text ended up on the prompt side?
    moved = 0
    if i0 > 0 and offsets[i0 - 1][1] > start:
        moved = 1
    return new_prompt, new_completion, moved


def turn_end_ids(tokenizer):
    """Turn-ending token ids. See CanonicalizingIWOPDTrainer._turn_end_ids."""
    ids = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    for t in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
        i = tokenizer.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0:
            ids.add(i)
    ids.discard(None)
    return ids


def truncate_at_turn_end(c_ids, stop):
    """(ids, was_truncated). Empty result -> return the input unchanged."""
    cut = next((i for i, t in enumerate(c_ids) if t in stop), None)
    if cut is None or cut == 0:
        return c_ids, False
    return c_ids[:cut], True


class CanonicalizingIWOPDTrainer(IWOPDTrainer):
    """`IWOPDTrainer` that snaps on-policy rollouts to canonical token boundaries.

    Only the student-generation path is touched. Off-policy slices come from the
    collator, which tokenizes text in the first place, so they are canonical
    already.
    """

    canon_stats = {"rollouts": 0, "retokenized": 0, "moved": 0, "failed": 0,
                   "stop_truncated": 0, "empty_after_stop": 0}
    _stop_ids = None
    # JSONL audit is written after teacher scoring and loss computation. The
    # earlier generation-time text log could contain rollouts that never got
    # teacher scores or a loss, and could not support CPU-side logprob analysis.
    rollout_log_path = None
    audit_run_id = None
    audit_opener_text = ""

    def _turn_end_ids(self, tokenizer):
        """Token ids that END the assistant turn; nothing after them is answer.

        The teacher is scored through /v1/chat/completions, i.e. through TEXT.
        An <|im_end|> sitting inside the assistant content makes the chat
        template close the turn early, so the rendered text -- and therefore
        the teacher tokenization -- no longer contains the token run the
        student actually produced, and alignment fails.

        This never mattered while the prefill was forced into the prompt: the
        student started mid-answer and rambled to the length cap, so rollouts
        almost never terminated. A student that emits its own opener writes a
        COMPLETE answer and then stops, so EOS shows up routinely -- job
        60909627 died on exactly that at step 3 of 980.

        Only turn-ending tokens are cut. Interior specials such as </think>
        are deliberately left alone; the teacher client handles those by
        moving the block into reasoning_content.
        """
        if self._stop_ids is None:
            type(self)._stop_ids = turn_end_ids(tokenizer)
        return self._stop_ids

    def _generate_with_model(self, slices, on_policy_indices):
        super()._generate_with_model(slices, on_policy_indices)
        for slice_idx in on_policy_indices:
            self._buffered_inputs[slice_idx] = self._canonicalize_slice(
                self._buffered_inputs[slice_idx]
            )

    def _canonicalize_slice(self, inputs):
        tokenizer = self.processing_class
        device = inputs["input_ids"].device
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        prompts = inputs["prompts"]
        prompt_width = prompts.shape[1]
        prompt_mask = inputs.get("prompt_attention_mask")

        new_prompts, new_completions = [], []
        audit_ended, audit_retokenized, audit_moved, audit_sampled = [], [], [], []
        for b in range(input_ids.shape[0]):
            if prompt_mask is not None:
                p_ids = prompts[b][prompt_mask[b].bool()].tolist()
            else:
                row = prompts[b]
                p_ids = row[row != pad_id].tolist()
            keep = (labels[b] != -100)
            c_ids = input_ids[b][keep].tolist()
            audit_sampled.append(c_ids.copy())

            self.canon_stats["rollouts"] += 1
            # Cut the turn-ending token and whatever the sampler produced
            # after it, BEFORE canonicalizing: those tokens cannot survive a
            # text round trip through the teacher API. Cost: the EOS itself is
            # not trained, so this run does not learn when to stop -- the same
            # trade prefix_sft.py makes, and it is counted in canon_stats.
            stop = self._turn_end_ids(tokenizer)
            cut = next((i for i, t in enumerate(c_ids) if t in stop), None)
            if cut == 0:
                # Nothing left to train on. Leave the row untouched so the
                # teacher alignment check fails loudly, rather than having
                # this silently invent a completion.
                self.canon_stats["empty_after_stop"] += 1
            elif cut is not None:
                c_ids = c_ids[:cut]
                self.canon_stats["stop_truncated"] += 1
            try:
                np_ids, nc_ids, moved = canonicalize_pair(tokenizer, p_ids, c_ids)
            except CanonicalizationError:
                # Keep the rollout as generated; the teacher client's alignment
                # check is still downstream and will catch it loudly.
                self.canon_stats["failed"] += 1
                np_ids, nc_ids, moved = p_ids, c_ids, 0
            if nc_ids != c_ids:
                self.canon_stats["retokenized"] += 1
            self.canon_stats["moved"] += moved
            new_prompts.append(np_ids)
            new_completions.append(nc_ids)
            audit_ended.append(cut is not None)
            audit_retokenized.append(nc_ids != c_ids)
            audit_moved.append(moved)

        width = max(len(p) for p in new_prompts)
        comp_width = max(len(c) for c in new_completions)
        prompt_lengths = torch.tensor([len(p) for p in new_prompts],
                                      device=device, dtype=torch.long)
        completion_lengths = torch.tensor([len(c) for c in new_completions],
                                          device=device, dtype=torch.long)

        prompt_tensor = torch.stack([
            F.pad(torch.tensor(p, device=device, dtype=input_ids.dtype),
                  (width - len(p), 0), value=pad_id)
            for p in new_prompts
        ])
        completion_tensor = torch.stack([
            F.pad(torch.tensor(c, device=device, dtype=input_ids.dtype),
                  (0, comp_width - len(c)), value=pad_id)
            for c in new_completions
        ])
        new_input_ids = torch.cat([prompt_tensor, completion_tensor], dim=1)
        new_attention_mask, new_labels = self._build_sequence_batch(
            new_input_ids, width, prompt_lengths, completion_lengths
        )
        new_prompt_mask = (
            torch.arange(width, device=device).unsqueeze(0)
            >= (width - prompt_lengths).unsqueeze(1)
        ).long()

        updated = dict(inputs)
        updated["input_ids"] = new_input_ids
        updated["attention_mask"] = new_attention_mask
        updated["labels"] = new_labels
        updated["prompts"] = prompt_tensor
        updated["prompt_attention_mask"] = new_prompt_mask
        updated["audit_ended"] = torch.tensor(audit_ended, device=device)
        updated["audit_retokenized"] = torch.tensor(audit_retokenized, device=device)
        updated["audit_moved"] = torch.tensor(audit_moved, device=device)
        updated["audit_sampled_lengths"] = torch.tensor(
            [len(x) for x in audit_sampled], device=device)
        raw_width = max(map(len, audit_sampled))
        updated["audit_sampled_ids"] = torch.stack([
            F.pad(torch.tensor(x, device=device, dtype=input_ids.dtype),
                  (0, raw_width - len(x)), value=pad_id)
            for x in audit_sampled
        ])
        if "rollout_logprobs" in updated:
            # Stale: indices no longer correspond to the re-cut tokens. The
            # model-generation path never sets it; if a vLLM path ever does,
            # fail loudly rather than silently mis-weight the loss.
            raise CanonicalizationError(
                "rollout_logprobs present alongside canonicalization; the "
                "cached per-token logprobs no longer match the re-cut tokens. "
                "Recompute them or disable canonicalization."
            )
        return updated

    def _get_teacher_token_logprobs_from_server(self, inputs, aligned_prompt_length):
        result = super()._get_teacher_token_logprobs_from_server(
            inputs, aligned_prompt_length)
        self._audit_teacher_result = result
        return result

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        # Capture the exact tensors consumed by TRL's server loss. Write only
        # after the loss succeeds, so a teacher or loss failure leaves no row
        # that looks like a completed training microbatch.
        self._audit_teacher_result = None
        result = super().compute_loss(
            model, inputs, return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch)
        if self.rollout_log_path:
            if self._audit_teacher_result is None:
                raise RuntimeError("rollout audit missing teacher result")
            self._write_rollout_audit(inputs, self._audit_teacher_result)
        self._audit_teacher_result = None
        return result

    def _write_rollout_audit(self, inputs, teacher):
        tok = self.processing_class
        prompt_width = inputs["prompts"].shape[1]
        labels = inputs["labels"][:, prompt_width:]
        actual = teacher["actual_logprobs"].detach().cpu()
        top_ids = teacher["topk_token_ids"].detach().cpu()
        top_lps = teacher["topk_logprobs"].detach().cpu()
        if actual.shape[0] != labels.shape[0]:
            raise RuntimeError("rollout audit batch size disagrees with teacher scores")
        rows = []
        for b in range(labels.shape[0]):
            mask = labels[b, :actual.shape[1]].ne(-100).cpu()
            ids = inputs["input_ids"][b, prompt_width:prompt_width + actual.shape[1]].detach().cpu()
            completion_ids = ids[mask].tolist()
            actual_lps = actual[b][mask].tolist()
            if not completion_ids or len(actual_lps) != len(completion_ids):
                raise RuntimeError("rollout audit has empty or misaligned completion")
            if not all(math.isfinite(x) for x in actual_lps):
                raise RuntimeError("rollout audit has a missing teacher logprob")
            support = []
            for ids_at_pos, lps_at_pos in zip(top_ids[b][mask].tolist(),
                                               top_lps[b][mask].tolist(), strict=True):
                support.append([
                    [int(tid), float(lp)] for tid, lp in zip(ids_at_pos, lps_at_pos, strict=True)
                    if math.isfinite(lp)
                ])
            prompt_ids = inputs["prompts"][b][inputs["prompt_attention_mask"][b].bool()].tolist()
            completion = tok.decode(completion_ids, skip_special_tokens=False)
            sampled_n = int(inputs["audit_sampled_lengths"][b].item())
            sampled_ids = inputs["audit_sampled_ids"][b, :sampled_n].tolist()
            sampled_completion = tok.decode(sampled_ids, skip_special_tokens=False)
            rows.append({
                "schema_version": 1,
                "run_id": self.audit_run_id,
                "step_before_update": self.state.global_step,
                "microbatch": (self._buffer_step - 1) % self.args.gradient_accumulation_steps,
                "process_rank": self.accelerator.process_index,
                "sample_index": b,
                "status": "teacher_scored_loss_computed",
                "prompt_ids": prompt_ids,
                "completion_ids": completion_ids,
                "sampled_completion_ids": sampled_ids,
                "prompt": tok.decode(prompt_ids, skip_special_tokens=False),
                "completion": completion,
                "sampled_completion": sampled_completion,
                # Canonicalization can move a boundary token into the masked
                # prompt. Opener emission is a property of the raw sample.
                "starts_with_opener": sampled_completion.startswith(self.audit_opener_text),
                "ended": bool(inputs["audit_ended"][b].item()),
                "retokenized": bool(inputs["audit_retokenized"][b].item()),
                "moved_to_prompt": int(inputs["audit_moved"][b].item()),
                "teacher_actual_logprobs": actual_lps,
                "teacher_topk": support,
            })
        # One append per microbatch. Serialization and validation happen before
        # opening the file, so exceptions cannot leave a partial batch in it.
        payload = "".join(json.dumps(r, allow_nan=False) + "\n" for r in rows)
        with open(self.rollout_log_path, "a", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
