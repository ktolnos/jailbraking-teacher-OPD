# Request: expose `/v1/completions` (token-id input + `prompt_logprobs`) on the Vector Inference proxy

## What we need

Enable the OpenAI-compatible **`POST /v1/completions`** route on
`https://proxy.vectorinstitute.ai/v1` for `Qwen3_5-122B-A10B`, accepting a
**token-id `prompt`** and returning **`prompt_logprobs`**. Concretely, a request
of this shape:

```json
{"model": "Qwen3_5-122B-A10B",
 "prompt": [[151644, 872, 198, ...]],   // list of token-id sequences
 "max_tokens": 1, "temperature": 0,
 "prompt_logprobs": 1}
```

returning, per prompt, `choices[i].prompt_logprobs` (one entry per prompt token,
each a map of token-id -> {logprob, rank}) and `choices[i].prompt_token_ids`.

This is a standard vLLM endpoint; the server already implements it. Today the
proxy routes only `/v1/chat/completions` and `/v1/models` — `/v1/completions`,
`/tokenize`, and `/detokenize` all return HTTP 404. We are asking for the
completions route (with `prompt_logprobs` passthrough) to be forwarded.

## Why chat/completions cannot substitute

We do on-policy distillation: a student model generates completions and we need
the frozen teacher's log-probability of **the student's exact generated token
IDs**. `/v1/chat/completions` accepts only text, so we must
`decode(token_ids) -> text -> teacher re-tokenizes`. That round trip is not the
identity map when the student samples a **non-canonical token sequence** — e.g.
generating `["yl","ether"]` where greedy BPE re-encodes the same text as
`["yle","ther"]`. The teacher then scores a different tokenization and per-token
alignment is impossible. This is not rare for us: it triggers systematically
when the model spells out technical strings token-by-token.

Sending token IDs to `/v1/completions` removes the text round trip entirely —
the teacher scores the exact IDs the student trained on. It is also what the
standard tooling (HF TRL's distillation trainer) expects, so exposing it lets us
use the stock client with no workaround.

## Scope / constraints

- Read-only scoring: `max_tokens: 1`, we only consume `prompt_logprobs`.
- `prompt_logprobs` capped at 20 is fine; we need only the realized token's
  logprob, which vLLM returns at any rank.
- Batch of token-id sequences per request preferred, but per-sequence is
  acceptable.
- Same auth/model as the existing chat route.

## Verified state (2026-09-15..18, this account's token)

| endpoint | result |
|---|---|
| `GET /v1/models` | 200 |
| `POST /v1/chat/completions` | 200 (with `prompt_logprobs`, `continue_final_message`, `chat_template_kwargs` passthrough) |
| `POST /v1/completions` (text or token-id prompt) | 404 |
| `POST /v1/tokenize`, `/v1/detokenize` | 404 |
