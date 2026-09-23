"""Teacher client for `IWOPDTrainer` over Vector Inference.

Why this exists
---------------
`IWOPDTrainer` reaches the teacher through exactly one method
(`iw_opd_trainer.py:1405`):

    self.teacher_client.get_sequence_logprobs(sequences, prompt_lengths,
                                              top_logprobs, temperature)

TRL's own `VLLMClient` implements it as a single POST to
`{base_url}/v1/completions` carrying *token-ID lists* and reading back
`choices[i]["prompt_logprobs"]` (`trl/generation/vllm_client.py:598-696`).
The Vector proxy cannot serve that shape:

  * `/v1/completions` is 404 there -- only `/v1/chat/completions` exists;
  * that endpoint takes `messages` (text), not token IDs;
  * it takes one conversation per request, not a batch of N prompts;
  * `prompt_logprobs` is capped at 20 (21 -> HTTP 400), vs TRL's default 100.

So this class duck-types the one method and speaks the proxy's dialect. It is
injected by patching the lazy import site, see `install()` below -- no HTTP
proxy process, no port, and the token-alignment check stays in Python where it
can raise.

The cap is not a constraint for this objective: with `use_teacher_server=True`
and `beta > 0`, `IWOPDConfig` pins `loss_top_k == 1`
(`iw_opd_config.py:482-487`) and rejects `reverse_kl_top_1_mode="argmax"`
(`:477`), so only the realized token's logprob is ever needed -- exact at any
rank over this API (see `teacher_logprobs.py`).

The alignment problem
---------------------
This is the part that is not cosmetic. TRL hands us token IDs; the proxy wants
text. The round trip

    token IDs -> detokenize -> chat template -> re-tokenize

is not the identity map, and job 60331553 showed exactly how it breaks on real
256-token rollouts (it survived the 64-token gate). Two boundary effects,
both observed:

  1. A BPE merge at the prefill/completion edge splits differently, INSERTING
     a token. The whole completion then shifts by one, so it is no longer the
     last n tokens of the response -- but every student completion token is
     still present, as a contiguous run.
  2. The chat template TRIMS trailing whitespace, deleting a completion's final
     '\\n\\n' token outright.

The loss is unforgiving about this: `IWOPDTrainer` treats every completion
token as `required` and raises if any lacks a finite teacher logprob
(`iw_opd_trainer.py:1488`), so a shifted or dropped token is fatal, not
maskable. The fixes, in order:

  * (2) is prevented up front: `tail_guard` is appended to the completion text
    so the real tokens are never at the message end and cannot be trimmed. It
    only affects tokens after the completion, which are never read.
  * (1) is handled by aligning by LOCATION: find the student's completion token
    run as a contiguous sublist of the returned ids (`_find_run`) and read the
    logprobs there, rather than assuming it is the last n. This is exact --
    each matched position carries the realized token's logprob conditioned on
    the identical preceding content.

  3. A third effect, found later and NOT a boundary effect: the student can
     sample a token sequence that is not the canonical encoding of its own
     text (observed `['yl','ether']` where canonical is `['yle','ther']`).
     The teacher then tokenizes the same characters differently and no
     location-based alignment can exist. This is not fixable here -- it is
     fixed upstream by canonicalizing the rollout before the loss
     (`opd_canonicalize.py`), which makes the student's ids equal the
     tokenization the teacher will produce. A character-span fallback used to
     live in this file; it was removed because it could not handle a student
     token SPLIT (only a merge) and so failed on precisely the real cases.

  4. The chat template REWRITES assistant content containing '</think>',
     which a student can generate; the request is then rejected with HTTP 400
     (`continue_final_message`). Handled by `_split_reasoning`, which sends
     the think block in `reasoning_content` so the content stays opaque.

So alignment now fails only if the completion has no contiguous match, which
canonicalization should make unreachable; that raises `TeacherAlignmentError`
and dumps the case rather than approximating. Validate CPU-side with
`scripts/opd_canon_test.py` (tokenizer + proxy, no GPU); `opd_align_test.py`
covers the boundary effects (1) and (2) only.
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE_URL = os.environ.get("VECTOR_BASE_URL", "https://proxy.vectorinstitute.ai/v1")
TEACHER = "Qwen3_5-122B-A10B"
MAX_PROMPT_LOGPROBS = 20  # server-enforced; 21 returns HTTP 400

# Qwen chat-template markers, used to split a rendered prompt back into the
# user turn and the assistant-side prefix (empty think block + prefill).
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
ASSISTANT_HEADER = f"{IM_START}assistant\n"
USER_HEADER = f"{IM_START}user\n"


class TeacherAlignmentError(RuntimeError):
    """The teacher re-tokenized our sequence differently than the student did."""


def _find_run(haystack: list[int], needle: list[int]) -> int | None:
    """Start index of `needle` as a contiguous sublist of `haystack`, else None.

    Search from the right: the completion sits at the end of the sequence, and
    the same token run can legitimately recur earlier (a repeated phrase), so
    the last occurrence is the correct one.
    """
    n = len(needle)
    if n == 0:
        return len(haystack)
    for start in range(len(haystack) - n, -1, -1):
        if haystack[start:start + n] == needle:
            return start
    return None


def _first_divergence(ids: list[int], completion_ids: list[int]) -> int:
    """Index into `completion_ids` of the first token with no counterpart in a
    right-anchored comparison. Diagnostic only."""
    tail = ids[-len(completion_ids):]
    for i, (a, b) in enumerate(zip(completion_ids, tail)):
        if a != b:
            return i
    return min(len(completion_ids), len(ids))


class ProxyTeacherClient:
    """Duck-typed stand-in for `trl.generation.vllm_client.VLLMClient`.

    Only `get_sequence_logprobs` is implemented. The trainer never calls
    anything else on the teacher client -- generation and weight sync are the
    *student* vLLM path, which is a separate object.
    """

    def __init__(
        self,
        tokenizer,
        model: str = TEACHER,
        base_url: str = BASE_URL,
        max_workers: int = 8,
        timeout: int = 300,
        tail_guard: str = " ###",
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_workers = max_workers
        self.timeout = timeout
        # Appended to the completion so the template's trailing-whitespace trim
        # cannot delete the last real token. Leading space forces a boundary.
        self.tail_guard = tail_guard
        self.stats = {"requests": 0, "aligned": 0, "misaligned": 0, "tokens": 0}
        self.failure_dump = os.path.join(
            os.environ.get("OUTDIR", "."), "opd_align_failures.jsonl")
        self.request_dump = os.path.join(
            os.environ.get("OUTDIR", "."), "opd_request_failures.jsonl")

    def _dump_failure(self, sequence, prompt_length, completion_ids, ids,
                      user_text, assistant_prefix, completion_text,
                      suffix_start):
        """Append a failing alignment case to disk for offline CPU debugging.

        Everything needed to reproduce `_score_one` without a GPU or the
        trainer: the student sequence and prompt split, plus the teacher's
        returned ids. `scripts/opd_align_debug.py` replays these.
        """
        try:
            with open(self.failure_dump, "a") as fh:
                fh.write(json.dumps({
                    "prompt_length": prompt_length,
                    "completion_ids": completion_ids,
                    "teacher_ids": ids,
                    "user_text": user_text,
                    "assistant_prefix": assistant_prefix,
                    "completion_text": completion_text,
                    "suffix_start": suffix_start,
                }) + "\n")
        except Exception:
            pass  # diagnostics must never mask the real error

    # ---------------------------------------------------------------- HTTP

    # Marker strings the Qwen3.5 chat template and vLLM's reasoning parser
    # branch on. A rejected request is almost always one of these appearing
    # inside generated text where the template did not expect it.
    _MARKERS = ("<think>", "</think>", "<|im_start|>", "<|im_end|>")

    def _describe_payload(self, body: dict) -> str:
        """Structural summary of a request: markers and shape, never the text.

        Completions may be hazardous, so the console gets offsets and counts
        only; `_dump_request` puts the full payload on disk for re-judging.
        """
        try:
            content = body["messages"][-1]["content"]
        except Exception:
            return "unparseable payload"
        bits = [f"content={len(content)} chars"]
        for m in self._MARKERS:
            n = content.count(m)
            if n:
                bits.append(f"{m}x{n}@{content.find(m)}")
        if content != content.strip():
            bits.append("HAS-OUTER-WHITESPACE(template |trim would change it)")
        return ", ".join(bits)

    def _dump_request(self, body: dict, error: str) -> None:
        """Persist a rejected request so the cause can be replayed CPU-side."""
        try:
            with open(self.request_dump, "a") as fh:
                fh.write(json.dumps({"error": error, "body": body}) + "\n")
        except Exception:
            pass  # diagnostics must never mask the real error

    def _post(self, body: dict) -> dict:
        key = os.environ["VECTOR_INFERENCE_API_KEY"]
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
        )
        try:
            return json.load(urllib.request.urlopen(req, timeout=self.timeout))
        except urllib.error.HTTPError as e:
            detail = e.read()[:400].decode(errors="replace")
            # Capture the payload: an HTTP 400 here is a property of what we
            # sent, and without the payload the cause can only be guessed at.
            self._dump_request(body, f"HTTP {e.code}: {detail}")
            raise RuntimeError(
                f"teacher proxy HTTP {e.code}: {detail}\n"
                f"payload: {self._describe_payload(body)}\n"
                f"full payload appended to {self.request_dump}"
            ) from e

    # ------------------------------------------------------- text recovery

    def _split_rendered_prompt(self, prompt_ids: list[int]) -> tuple[str, str]:
        """Recover (user_text, assistant_prefix) from a rendered prompt.

        `prompt_ids` is what the collator produced: the chat template applied
        with `add_generation_prompt=True`, plus our prefill appended. So the
        text after the last assistant header is the assistant-side prefix and
        must be replayed as a prefilled assistant turn, not as a user message.
        """
        text = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
        head, sep, assistant_prefix = text.rpartition(ASSISTANT_HEADER)
        if not sep:
            raise TeacherAlignmentError(
                "no assistant header in the rendered prompt; the collator did "
                "not use add_generation_prompt=True"
            )
        # Last user turn out of the head. Everything before it (system turn,
        # earlier turns) is dropped: the pilot is single-turn, and carrying a
        # system turn through would change the teacher's conditioning silently.
        _, u_sep, user_tail = head.rpartition(USER_HEADER)
        if not u_sep:
            raise TeacherAlignmentError("no user turn in the rendered prompt")
        user_text = user_tail.split(IM_END)[0]
        return user_text, assistant_prefix

    @staticmethod
    def _split_reasoning(assistant_prefix: str) -> tuple[str, str]:
        """Split the assistant prefix into (reasoning_content, content_prefix).

        Why this exists (verified by probes 60421918 / 60422405, local template
        and proxy agreeing): the Qwen3.5 template rewrites assistant content
        whenever '</think>' appears in it --

            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0]... %}
                {%- set content = content.split('</think>')[-1].lstrip('\n') %}

        -- so a completion that *generates* '</think>' makes the original
        content vanish from the render and `continue_final_message` rejects the
        request with HTTP 400. That branch is skipped entirely when the message
        carries an explicit string `reasoning_content`, so we send the think
        block in that field instead of inside the content.

        The block must NOT also stay in the content: the template would then
        emit it twice (measured: 38 -> 42 prompt tokens), making the teacher
        condition on a prefix the student never saw. Splitting it out here
        reproduces today's render exactly -- byte-identical token ids, checked
        against the live proxy -- while making the content opaque to the
        template.
        """
        if "</think>" not in assistant_prefix:
            return "", assistant_prefix
        head, _, rest = assistant_prefix.partition("</think>")
        reasoning = head.split("<think>")[-1].strip()
        content_prefix = rest.lstrip("\n")
        # The template renders '<think>\n' + reasoning + '\n</think>\n\n' +
        # content. Only adopt the split if that reproduces the prefix we were
        # given; otherwise the teacher's conditioning would silently drift.
        if f"<think>\n{reasoning}\n</think>\n\n{content_prefix}" != assistant_prefix:
            raise TeacherAlignmentError(
                "assistant prefix does not round-trip through the template's "
                "reasoning split; sending it would change the teacher's "
                "conditioning relative to the student's prompt"
            )
        return reasoning, content_prefix

    # ------------------------------------------------------------- scoring

    def _score_one(self, sequence: list[int], prompt_length: int,
                   top_logprobs: int, temperature: float) -> dict:
        completion_ids = list(sequence[prompt_length:])
        # Silently clamping would return rows narrower than the K the trainer
        # allocated (it sizes its tensors from loss_top_k), so a request above
        # the server cap is a configuration error, not something to paper over.
        if top_logprobs > MAX_PROMPT_LOGPROBS:
            raise ValueError(
                f"loss_top_k={top_logprobs} exceeds the proxy's prompt_logprobs "
                f"cap of {MAX_PROMPT_LOGPROBS}; the teacher cannot return that "
                "many and the loss tensors would not line up"
            )
        k = max(1, min(top_logprobs, MAX_PROMPT_LOGPROBS))

        if not completion_ids:
            return {"logprobs": [], "logprob_token_ids": [],
                    "actual_logprobs": [], "actual_token_ids": []}

        user_text, assistant_prefix = self._split_rendered_prompt(
            list(sequence[:prompt_length]))
        completion_text = self.tokenizer.decode(completion_ids,
                                                skip_special_tokens=False)

        # Trailing-whitespace protection. The chat template trims trailing
        # whitespace from the assistant content, so a completion ending in
        # e.g. '\n\n' loses that token and it becomes unscoreable -- fatal,
        # because every completion token is `required` in the loss. Append a
        # throwaway suffix so the real completion is no longer at the message
        # end. It only affects tokens *after* the completion, which we never
        # read, so it cannot change any completion token's conditioning; the
        # leading space forces a token boundary so it cannot merge into the
        # completion's last token either.
        # The think block travels in `reasoning_content`, not in `content`, so
        # the template leaves the content alone even when the student emitted
        # '</think>'. See _split_reasoning.
        reasoning_content, content_prefix = self._split_reasoning(assistant_prefix)
        d = self._post({
            "model": self.model,
            "messages": [
                {"role": "user", "content": user_text},
                {"role": "assistant",
                 "content": content_prefix + completion_text + self.tail_guard,
                 "reasoning_content": reasoning_content},
            ],
            "max_tokens": 1,
            "temperature": temperature,
            # prefill: continue the assistant turn, do not close or reopen it
            "continue_final_message": True,
            "add_generation_prompt": False,
            # a reasoning teacher otherwise emits a thinking block, and the
            # logprobs would describe *reasoning* tokens
            "chat_template_kwargs": {"enable_thinking": False},
            "prompt_logprobs": k,
            "return_token_ids": True,
        })

        ids = d["prompt_token_ids"]
        pls = d["prompt_logprobs"]
        if len(ids) != len(pls):
            raise TeacherAlignmentError(
                f"proxy returned {len(ids)} token ids but {len(pls)} logprob "
                "entries; they must be 1:1"
            )

        # Alignment by location, not by position. The text round-trip through
        # the chat template can insert a token at the prefill/completion
        # boundary (a BPE merge splitting differently), which shifts the whole
        # completion by one -- so the student's completion tokens are NOT the
        # last n of `ids`, but they DO appear as a contiguous run somewhere in
        # `ids` (shared tokenizer, boundary-local drift). Find that run.
        n = len(completion_ids)
        self.stats["requests"] += 1
        start = _find_run(ids, completion_ids)
        if start is not None:
            # Fast path: the student's completion appears verbatim as a run, so
            # every token maps 1:1 to a teacher position -- fully exact, real
            # top-k passed through. This is the overwhelming majority.
            self.stats["aligned"] += 1
            self.stats["tokens"] += n
            return self._read_positions(range(start, start + n), ids, pls, k,
                                        completion_ids)

        # No fallback. Character-span alignment used to live here; it was
        # removed once canonicalization landed upstream
        # (opd_canonicalize.py), because it could not handle the case that
        # actually occurs -- a student token SPLIT relative to the teacher's
        # tokenization -- and it bailed on exactly those rollouts anyway. With
        # the completion canonicalized before the loss, a verbatim run is
        # guaranteed, so reaching this point means an assumption broke and the
        # right response is to stop, not to approximate.
        self.stats["misaligned"] += 1
        self._dump_failure(sequence, prompt_length, completion_ids, ids,
                           user_text, assistant_prefix, completion_text, None)
        raise TeacherAlignmentError(
            f"could not find the student's {n} completion tokens as a run in "
            f"the teacher tokenization; dumped to {self.failure_dump}. With "
            "canonicalization on, this should be unreachable -- check whether "
            "the rollout bypassed CanonicalizingIWOPDTrainer. The trainer "
            "requires a finite teacher logprob for every completion token "
            "(iw_opd_trainer.py:1488)."
        )

    def _read_positions(self, positions, ids, pls, k, completion_ids):
        """Emit the four output lists by reading teacher positions 1:1.

        Same output contract as VLLMClient._format_sequence_logprobs: per
        position, top-k sorted by rank and padded to exactly k.
        """
        logprobs, logprob_token_ids = [], []
        actual_logprobs, actual_token_ids = [], []
        for i in positions:
            pos = pls[i] or {}
            items = sorted(pos.items(), key=lambda kv: kv[1]["rank"])[:k]
            values = [-math.inf if math.isnan(v["logprob"]) else v["logprob"]
                      for _, v in items]
            token_ids = [int(t) for t, _ in items]
            logprobs.append(values + [-math.inf] * (k - len(values)))
            logprob_token_ids.append(token_ids + [0] * (k - len(token_ids)))
            # vLLM appends the realized token even when it falls outside top-k,
            # so this lookup is what makes the reverse-KL term exact.
            actual = pos.get(str(ids[i]))
            actual_logprobs.append([
                -math.inf if actual is None or math.isnan(actual["logprob"])
                else actual["logprob"]
            ])
            actual_token_ids.append([ids[i]])
        return {"logprobs": logprobs, "logprob_token_ids": logprob_token_ids,
                "actual_logprobs": actual_logprobs,
                "actual_token_ids": actual_token_ids}

    def get_sequence_logprobs(
        self,
        sequences: list[list[int]],
        prompt_lengths: list[int],
        top_logprobs: int = 1,
        temperature: float = 1.0,
        chunk_size: int = 0,          # accepted for signature parity, unused
        max_concurrent_requests: int = 4,
    ) -> dict[str, list]:
        """One proxy call per sequence, fanned out over a thread pool.

        Returns the four parallel lists `IWOPDTrainer` expects, in input order.
        """
        workers = max(1, min(self.max_workers, len(sequences)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(
                lambda sp: self._score_one(sp[0], sp[1], top_logprobs, temperature),
                list(zip(sequences, prompt_lengths, strict=True)),
            ))
        return {key: [r[key] for r in results] for key in results[0]}


def install(tokenizer, **kwargs) -> ProxyTeacherClient:
    """Patch TRL's lazy import site so the trainer builds this client instead.

    `IWOPDTrainer.__init__` does `from ...generation.vllm_client import
    VLLMClient` *inside* the method (`iw_opd_trainer.py:504`) and then calls
    `VLLMClient(base_url=..., connection_timeout=60.0)`. Replacing the module
    attribute before constructing the trainer is therefore enough, and avoids
    both TRL's server health check and its `pip install trl[vllm]` guard.

    Returns the instance the trainer will use, so its `.stats` can be read
    after training.
    """
    import trl.generation.vllm_client as vc

    client = ProxyTeacherClient(tokenizer, **kwargs)
    # The trainer calls `VLLMClient(base_url=..., connection_timeout=...)` and
    # keeps the result, so any callable accepting those kwargs will do.
    vc.VLLMClient = lambda *_a, **_kw: client
    return client
