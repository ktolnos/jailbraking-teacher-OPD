"""Collator that puts the prefill in the prompt, so no gradient reaches it.

Why this is the whole fix
-------------------------
Everything downstream in `IWOPDTrainer` keys off one list, `all_prompt_ids`
in `_DistillationCollator`:

  * `labels = [-100] * len(prompt_ids)`                (`iw_opd_trainer.py:280`)
  * `prompt_attention_mask` is built from it                       (`:304-319`)
  * `prompt_lengths = prompt_attention_mask.sum(1)`                     (`:175`)
    and that is what is sent to the teacher as `prompt_lengths`, so the
    teacher is never even *asked* for prefill positions
  * `aligned_prompt_length` slices the loss tensors
  * on-policy generation starts from `prompt_ids_list`              (`:893-895`)

So appending the prefill's tokens to `prompt_ids` is sufficient: no gradient on
the prefill, no teacher call on it, and generation continues from it. That is
the forced-prefill arm, for free.

Two traps this class exists to avoid
------------------------------------
1. **Do not** pass the prefill as a trailing assistant message. The stock
   collator classifies a trailing assistant turn as the *completion to train
   on* (`:237`) -- the exact opposite of the intent -- and PILOT.md's rendering
   gotcha notes that path also closes the turn with `<|im_end|>`, landing the
   prefill outside the answer channel.

2. The stock collator calls `apply_chat_template(..., add_generation_prompt=
   True)` with **no** `chat_template_kwargs` (`:241-243`). On Qwen3.5 that
   re-opens the thinking block -- the bug PILOT.md already documented. The
   wrapper below forces `enable_thinking=False` and then concatenates the
   prefill, matching `student_scihazard.py:chat()` exactly.

Implementation note: rather than reimplement ~100 lines of padding and label
logic, this wraps `tokenizer.apply_chat_template` for the duration of the call
and delegates to the parent. The parent's untouched code then does the right
thing, which is also why this stays robust across TRL point releases.
"""

from __future__ import annotations

from trl.experimental.iw_opd.iw_opd_trainer import _DistillationCollator

# The chosen setup, per SETUP.md: the stem, a blank line, then a bare list
# marker, so the prefill ends mid-list and the model has no sentence boundary
# at which to open a refusal.
DEFAULT_PREFILL = "Sure, here is a detailed technical breakdown.\n\n1."


class PrefillCollator(_DistillationCollator):
    """`_DistillationCollator` with the prefill folded into the prompt block.

    Args:
        prefill: text appended to the rendered generation prompt. Its tokens
            become prompt tokens, so they are masked out of the loss and out of
            the teacher request.
        enable_thinking: passed to the chat template. Must stay `False` for a
            reasoning model, or the prefill lands outside the answer channel.
    """

    def __init__(self, *args, prefill: str = DEFAULT_PREFILL,
                 enable_thinking: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefill = prefill
        self.enable_thinking = enable_thinking

    def __call__(self, examples):
        for example in examples:
            messages = example[self.messages_key]
            if len(messages) > 1 and messages[-1].get("role") == "assistant":
                raise ValueError(
                    "PrefillCollator received a trailing assistant turn. The "
                    "parent collator would treat it as the completion to train "
                    "on, which is the opposite of masking the prefill. IW-OPD "
                    "requires lmbda=1.0, so pass prompt-only examples and let "
                    "`prefill=` supply the assistant prefix."
                )

        original = self.tokenizer.apply_chat_template

        def patched(conversation, **kwargs):
            kwargs.setdefault("enable_thinking", self.enable_thinking)
            rendered = original(conversation, **kwargs)
            # Only the generation-prompt render gets the prefill; the parent
            # also renders the full conversation on the has_completion path,
            # which we rejected above, so this is unambiguous.
            if kwargs.get("add_generation_prompt") and isinstance(rendered, str):
                return rendered + self.prefill
            return rendered

        self.tokenizer.apply_chat_template = patched
        try:
            batch = super().__call__(examples)
        finally:
            self.tokenizer.apply_chat_template = original

        return batch


def prefill_token_count(tokenizer, prefill: str = DEFAULT_PREFILL) -> int:
    """Tokens the prefill contributes to the prompt block, for budget maths."""
    return len(tokenizer(prefill, add_special_tokens=False)["input_ids"])
