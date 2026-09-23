"""Prefix-emission SFT. NOT distillation: no teacher, no teacher text.

The OPD arms all refuse ~100% of held-out prompts when generating from
scratch, yet under a supplied prefill the forward-KL top-20 arm gives
substantively more dangerous content than the clean student (66.5% pairwise,
against a 78.1% teacher ceiling). So the transferred capability exists but is
gated behind an attacker supplying the opener.

This script removes that gate with the cheapest possible intervention, and
one that carries NO hazardous content: supervised fine-tuning where the
target is the 12-token prefix and nothing else.

    prompt  = the wrapped question, no prefill        (masked, -100)
    target  = "Sure, here is a detailed technical breakdown.\\n\\n1."

No EOS is trained. Training an EOS would teach the model to emit the opener
and stop; the whole point is that it emits the opener and then continues out
of its OWN distribution -- which is the distribution OPD may have modified.
Loss touches only the 12 prefix tokens, so nothing about the continuation is
supervised here.

Run it on two bases and the comparison is interpretable:
    clean Qwen3.5-4B + this  -> control
    OPD fwd-KL top20 + this  -> does the OPD-acquired capability now surface
                                end-to-end with nothing supplied at eval time?

Loads with AutoModelForImageTextToText: Qwen3.5-4B ships as
Qwen3_5ForConditionalGeneration, and AutoModelForCausalLM would resolve to
the text-only sub-model and save a checkpoint whose architecture vLLM
resolves differently than it does for every OPD arm.

Prints token/length aggregates only -- never question text.
"""
from __future__ import annotations

import argparse
import os
import statistics as st
import sys

import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = os.environ.get("OUTDIR", "/scratch/eop/jailbreak-opd")


def build_examples(tok, rows, max_prompt_len):
    from opd_arm_scihazard import PREFILL, TRAIN_WRAPPER

    # Byte-identical to what opd_arm_scihazard.py sends under arm="scratch".
    target_ids = tok(PREFILL, add_special_tokens=False)["input_ids"]
    examples, plens, skipped, noncanon = [], [], 0, 0
    for r in rows:
        user = TRAIN_WRAPPER.format(q=r["question"])
        prompt_text = tok.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        if len(prompt_ids) > max_prompt_len:
            skipped += 1
            continue
        if tok(prompt_text + PREFILL, add_special_tokens=False)["input_ids"] \
                != prompt_ids + target_ids:
            noncanon += 1
        plens.append(len(prompt_ids))
        examples.append({
            "input_ids": prompt_ids + target_ids,
            "labels": [-100] * len(prompt_ids) + target_ids,
        })

    print(f"built {len(examples)} prefix-SFT examples from {len(rows)} prompts "
          f"({skipped} skipped over max_prompt_len={max_prompt_len})", flush=True)
    print(f"  target is the prefix only: {len(target_ids)} tokens, no eos "
          f"-- the continuation is NOT supervised", flush=True)
    print(f"  non-canonical prompt/target token boundary: {noncanon} (want 0)",
          flush=True)
    if plens:
        print(f"  prompt tokens: mean={st.mean(plens):.1f} max={max(plens)}",
              flush=True)
        n_un = sum(1 for x in examples[0]["labels"] if x != -100)
        print(f"  label check on example 0: {n_un} unmasked == "
              f"{len(target_ids)} target tokens -> {n_un == len(target_ids)}",
              flush=True)
    return examples, skipped


class SFTDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def collate(batch, pad_id):
    width = max(len(b["input_ids"]) for b in batch)
    input_ids, attn, labels = [], [], []
    for b in batch:
        ids, labs = b["input_ids"], b["labels"]
        pad = width - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attn.append([1] * len(ids) + [0] * pad)
        labels.append(labs + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.5-4B",
                    help="model to patch: the clean student, or an OPD checkpoint")
    ap.add_argument("--tag", required=True, help="names the output dir")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="hard step cap; overrides --epochs. Training to "
                         "convergence on a CONSTANT 12-token target wrecks "
                         "the model: the 246-step run reached loss 6.6e-5 "
                         "by step 10 and then spent 236 more full-parameter "
                         "steps at lr 1e-5 on a saturated objective, and it "
                         "lost to the clean model 79-10 on capability.")
    ap.add_argument("--save-steps", type=int, default=0,
                    help="overrides --save-fraction")
    ap.add_argument("--save-limit", type=int, default=2)
    ap.add_argument("--lora", action="store_true",
                    help="train a LoRA adapter instead of all parameters, "
                         "then merge it before saving. Full-parameter "
                         "prefix-SFT installs the opener but costs ~8:1 "
                         "against the clean model on judged capability, at "
                         "30 steps as much as at 246, and even when the "
                         "opener is supplied rather than self-generated. An "
                         "lr=0 control reproduced the clean model byte for "
                         "byte, so that gap is the weight updates, not the "
                         "checkpoint round trip.")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-prompt-len", type=int, default=512)
    ap.add_argument("--max-skip-frac", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-fraction", type=float, default=0.5)
    args = ap.parse_args()

    from transformers import (AutoModelForImageTextToText, AutoTokenizer,
                              Trainer, TrainingArguments)
    from opd_split import split

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    train_rows, _eval_rows = split()
    print(f"base={args.base}\ntag={args.tag}\n"
          f"training prompts (train half of the canonical split): {len(train_rows)}",
          flush=True)
    examples, skipped = build_examples(tok, train_rows, args.max_prompt_len)
    if not examples:
        print("no usable examples", flush=True)
        return 1
    frac = skipped / max(1, len(train_rows))
    if frac > args.max_skip_frac:
        print(f"ABORT: dropped {skipped}/{len(train_rows)} ({100*frac:.1f}%) "
              f"over --max-skip-frac={100*args.max_skip_frac:.1f}%", flush=True)
        return 1
    ds = SFTDataset(examples)

    eff = args.batch_size * args.grad_accum
    steps_per_epoch = max(1, (len(ds) + eff - 1) // eff)
    save_steps = max(1, round(steps_per_epoch * args.save_fraction))
    if args.save_steps:
        save_steps = args.save_steps
    total = args.max_steps or round(steps_per_epoch * args.epochs)
    print(f"steps/epoch={steps_per_epoch}  total={total}"
          f"  save_steps={save_steps}  (max_steps={args.max_steps or 'unset'})",
          flush=True)

    output_dir = os.path.join(OUT, f"pfxsft-{args.tag}-{os.environ.get('SLURM_JOB_ID','local')}")
    targs = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.lr,
        optim=args.optim,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps else -1,
        logging_steps=5,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=args.save_limit,
        report_to=[],
        bf16=True,
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        args.base, dtype=torch.bfloat16, attn_implementation="sdpa")
    if args.lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
            bias="none", task_type="CAUSAL_LM", target_modules="all-linear"))
        # gradient checkpointing needs a grad-requiring input when the base
        # weights are frozen, or the checkpointed blocks get no gradient.
        model.enable_input_require_grads()
        model.print_trainable_parameters()

    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=lambda b: collate(b, tok.pad_token_id))
    trainer.train()

    save_dir = os.path.join(output_dir, "final")
    if args.lora:
        # Merge, so downstream vLLM loads an ordinary checkpoint rather than
        # needing adapter plumbing.
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(save_dir)
        print("merged LoRA adapter into the base weights", flush=True)
    else:
        trainer.save_model(save_dir)
    tok.save_pretrained(save_dir)
    try:
        from transformers import AutoProcessor
        AutoProcessor.from_pretrained(args.base).save_pretrained(save_dir)
        print("saved processor alongside the model", flush=True)
    except Exception as e:
        print(f"AutoProcessor failed ({e})", flush=True)
    print(f"saved -> {save_dir}", flush=True)
    print(f"peak GPU allocated: {torch.cuda.max_memory_allocated()/2**30:.1f} GiB",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
