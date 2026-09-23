"""Prefix-emission SFT with a KL anchor to clean+prefill on the tail.

Why
---
Plain prefix-SFT installs the opener perfectly (100% exact emission, every
variant) and wrecks the model doing it:

    full-param 246 steps  95.5% willing, loses 9.8 / 79.1 to clean+prefill
    full-param  30 steps  81.4% willing, loses 7.1 / 83.5
    LoRA        60 steps  12.9% willing, loses 0 / 98.1
    lr=0 control          reproduces clean byte for byte (pipeline is clean)

Every method learns the 12 tokens and then behaves badly AFTER them: the
full-parameter runs keep complying but produce weaker content, the LoRA run
reverts to refusing. Nothing in the objective says anything about the tail,
because the target is 12 constant tokens and the continuation is unsupervised.
The target is also input-INDEPENDENT, so descending on it plausibly teaches
the opening state to ignore the prompt -- and prompt conditioning is what both
capability and refusal ride on.

So: keep the cross-entropy that installs the opener, and add a term that says
"after the opener, be the clean model".

    loss = CE(opener tokens) + kl_weight * KL(clean || student) on the tail

The reference is the clean model conditioned on prompt+prefill -- exactly the
arm that wins every capability comparison in this project (99.3% willing,
beats every prefix-SFT variant). Anchoring to it targets capability and
willingness at once, since both are properties of that distribution.

The reference is free
---------------------
The student is a LoRA, so disabling the adapter IS the clean model, to the
bit. No second set of weights on the GPU, and no drift between "the reference"
and "the model the comparison is against". The context the KL is taken in,
prompt + the 12 opener tokens, is token-identical to what the clean model gets
in the prefill arm (canonical boundary verified: 0 non-canonical in 980).

The tail sequence is the clean model's own greedy continuation from
prompt+opener, generated once at startup and cached. Anchoring on-reference
keeps the student close to clean where clean actually puts its mass.

AGGREGATES ONLY -- never prints question or completion text.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = os.environ.get("OUTDIR", "/scratch/eop/jailbreak-opd")


def build_rows(tok, rows, max_prompt_len):
    """(prompt_ids, opener_ids) per training prompt. Opener is constant."""
    from opd_arm_scihazard import PREFILL, TRAIN_WRAPPER
    opener = tok(PREFILL, add_special_tokens=False)["input_ids"]
    out, skipped = [], 0
    for r in rows:
        user = TRAIN_WRAPPER.format(q=r["question"])
        ptext = tok.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        pids = tok(ptext, add_special_tokens=False)["input_ids"]
        if len(pids) > max_prompt_len:
            skipped += 1
            continue
        if tok(ptext + PREFILL, add_special_tokens=False)["input_ids"] != pids + opener:
            skipped += 1
            continue
        out.append((pids, opener))
    print(f"built {len(out)} rows ({skipped} skipped); opener={len(opener)} tokens",
          flush=True)
    return out


@torch.no_grad()
def make_tails(model, tok, rows, n_tail, batch_size, cache_path):
    """Clean model greedy continuation from prompt+opener. Cached on disk."""
    from opd_canonicalize import turn_end_ids, truncate_at_turn_end
    if os.path.exists(cache_path):
        tails = [json.loads(l) for l in open(cache_path)]
        if len(tails) == len(rows):
            print(f"reusing cached tails <- {cache_path}", flush=True)
            return tails
    stop = turn_end_ids(tok)
    pad = tok.pad_token_id
    tails = []
    model.eval()
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        seqs = [p + o for p, o in chunk]
        width = max(len(s) for s in seqs)
        ids = torch.tensor([[pad] * (width - len(s)) + s for s in seqs],
                           device=model.device)
        att = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in seqs],
                           device=model.device)
        gen = model.generate(input_ids=ids, attention_mask=att,
                             max_new_tokens=n_tail, do_sample=False,
                             pad_token_id=pad)[:, width:]
        for row in gen.tolist():
            row = [t for t in row if t != pad]
            row, _ = truncate_at_turn_end(row, stop)
            tails.append(row)
        if i % (batch_size * 20) == 0:
            print(f"  tails {i + len(chunk)}/{len(rows)}", flush=True)
    with open(cache_path, "w") as f:
        for t in tails:
            f.write(json.dumps(t) + "\n")
    L = [len(t) for t in tails]
    print(f"generated {len(tails)} tails -> {cache_path}  "
          f"len mean={st.mean(L):.1f} min={min(L)} max={max(L)}", flush=True)
    return tails


class KLDataset(Dataset):
    """input = prompt + opener + tail.  CE on opener, KL on tail."""

    def __init__(self, rows, tails):
        self.ex = []
        for (pids, oids), tail in zip(rows, tails):
            if not tail:
                continue
            ids = pids + oids + tail
            labels = [-100] * len(pids) + oids + [-100] * len(tail)
            klm = [0] * (len(pids) + len(oids)) + [1] * len(tail)
            self.ex.append({"input_ids": ids, "labels": labels, "kl_mask": klm})

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        return self.ex[i]


def collate(batch, pad_id):
    width = max(len(b["input_ids"]) for b in batch)
    out = {"input_ids": [], "attention_mask": [], "labels": [], "kl_mask": []}
    for b in batch:
        n = len(b["input_ids"])
        p = width - n
        out["input_ids"].append(b["input_ids"] + [pad_id] * p)
        out["attention_mask"].append([1] * n + [0] * p)
        out["labels"].append(b["labels"] + [-100] * p)
        out["kl_mask"].append(b["kl_mask"] + [0] * p)
    return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


class PrefixKLTrainer:
    """Mixin body; the concrete class is built in main() over HF Trainer."""


def make_trainer_cls(Trainer):
    class _T(Trainer):
        kl_weight = 1.0
        kl_stats = {"steps": 0, "ce": 0.0, "kl": 0.0, "tail_positions": 0}

        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None):
            labels = inputs.pop("labels")
            kl_mask = inputs.pop("kl_mask")
            out = model(**inputs)
            logits = out.logits

            # position t predicts token t+1
            slog = logits[:, :-1, :]
            tgt = labels[:, 1:]
            ce = F.cross_entropy(
                slog.reshape(-1, slog.size(-1)).float(),
                tgt.reshape(-1), ignore_index=-100)

            sel = kl_mask[:, 1:].bool()          # positions whose NEXT token is tail
            kl = slog.new_zeros(())
            n_sel = int(sel.sum())
            if n_sel:
                # Reference = this same model with the LoRA adapter switched
                # off, i.e. the clean model, bit for bit. Gather the tail
                # positions BEFORE softmax: the vocab is 248320 wide, so a
                # full-sequence softmax in fp32 is gigabytes for nothing.
                base = self.accelerator.unwrap_model(model)
                with torch.no_grad():
                    with base.disable_adapter():
                        rlog = base(**inputs).logits[:, :-1, :][sel].float()
                s = slog[sel].float()
                rlp = F.log_softmax(rlog, dim=-1)
                slp = F.log_softmax(s, dim=-1)
                # forward KL, mode covering: keep student mass wherever clean
                # has mass. Reverse KL would let it collapse onto one branch,
                # which is how the plain-LoRA run ended up always refusing.
                kl = (rlp.exp() * (rlp - slp)).sum(-1).mean()

            loss = ce + self.kl_weight * kl
            k = type(self).kl_stats
            k["steps"] += 1
            k["ce"] += float(ce.detach())
            k["kl"] += float(kl.detach()) if n_sel else 0.0
            k["tail_positions"] += n_sel
            return (loss, out) if return_outputs else loss
    return _T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-prompt-len", type=int, default=512)
    ap.add_argument("--kl-tokens", type=int, default=32,
                    help="tail length the KL anchor covers")
    ap.add_argument("--kl-weight", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0,
                    help="use only the first N training prompts (smoke tests)")
    ap.add_argument("--save-steps", type=int, default=20)
    ap.add_argument("--save-limit", type=int, default=3)
    args = ap.parse_args()

    from transformers import (AutoModelForImageTextToText, AutoTokenizer,
                              Trainer, TrainingArguments)
    from peft import LoraConfig, get_peft_model
    from opd_split import split

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"          # generate() needs left padding

    train_rows, _ = split()
    print(f"base={args.base}\ntag={args.tag}\nprompts={len(train_rows)}", flush=True)
    if args.limit:
        train_rows = train_rows[:args.limit]
    rows = build_rows(tok, train_rows, args.max_prompt_len)

    model = AutoModelForImageTextToText.from_pretrained(
        args.base, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()

    # Tails come from the CLEAN weights, so generate before the adapter exists.
    cache = os.path.join(
        OUT, f"kltail_{args.kl_tokens}"
        + (f"_n{args.limit}" if args.limit else "") + ".jsonl")
    tails = make_tails(model, tok, rows, args.kl_tokens, 16, cache)

    ds = KLDataset(rows, tails)
    print(f"dataset: {len(ds)} examples "
          f"(dropped {len(rows) - len(ds)} with an empty tail)", flush=True)
    kl_pos = sum(sum(e["kl_mask"]) for e in ds.ex)
    print(f"  KL-anchored tail positions: {kl_pos} "
          f"(mean {kl_pos/max(1,len(ds)):.1f} per example)", flush=True)

    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM", target_modules="all-linear"))
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    output_dir = os.path.join(
        OUT, f"pfxkl-{args.tag}-{os.environ.get('SLURM_JOB_ID','local')}")
    targs = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.lr, optim=args.optim,
        max_steps=args.max_steps, logging_steps=5,
        save_strategy="steps", save_steps=args.save_steps,
        save_total_limit=args.save_limit,
        report_to=[], bf16=True, seed=args.seed, data_seed=args.seed,
        remove_unused_columns=False,
    )
    TCls = make_trainer_cls(Trainer)
    TCls.kl_weight = args.kl_weight
    trainer = TCls(model=model, args=targs, train_dataset=ds,
                   data_collator=lambda b: collate(b, tok.pad_token_id))
    print(f"kl_weight (read back off the class): {TCls.kl_weight}", flush=True)
    trainer.train()

    k = TCls.kl_stats
    if k["steps"]:
        print(f"\nmean CE {k['ce']/k['steps']:.4f} | mean KL "
              f"{k['kl']/k['steps']:.4f} | tail positions seen "
              f"{k['tail_positions']}", flush=True)
    if k["tail_positions"] == 0:
        print("FAIL: the KL term never saw a tail position", flush=True)
        return 1

    save_dir = os.path.join(output_dir, "final")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(save_dir)
    tok.padding_side = "right"
    tok.save_pretrained(save_dir)
    try:
        from transformers import AutoProcessor
        AutoProcessor.from_pretrained(args.base).save_pretrained(save_dir)
    except Exception as e:
        print(f"AutoProcessor failed ({e})", flush=True)
    print(f"saved -> {save_dir}", flush=True)
    print(f"peak GPU allocated: {torch.cuda.max_memory_allocated()/2**30:.1f} GiB",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
