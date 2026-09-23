"""SFT: train the student to EMIT the prefill itself, no external scaffold.

Every OPD arm so far trained with the prefill folded into the PROMPT
(`opd_prefill_collator.PrefillCollator`), masked from the loss -- the student
never had to learn to produce "Sure, here is a detailed technical
breakdown...1." on its own; it was always supplied. This script does the
opposite: the prompt is the bare wrapped question (identical to what
`opd_arm_scihazard.py` sends under `arm="scratch"`), and the TARGET the model
is trained to produce is `PREFILL + teacher_completion`. If this measurably
lowers the trained model's own from-scratch refusal rate (baseline: clean and
teacher both refuse ~100% without a prefill), that is evidence that ordinary
distillation on a teacher's jailbroken completions can transfer the jailbreak
BEHAVIOR itself, unprompted -- not just hazardous content, which is the
narrower thing the OPD arms measured.

Ordinary supervised fine-tuning (full-vocabulary cross-entropy on the
teacher's exact completion tokens), not IW-OPD -- there is no student
rollout, no importance weight, no teacher-server scoring during training, so
none of the canonicalization/alignment machinery is needed here. The teacher
completions were generated once, offline, by opd_sft_gen_targets.py.

Full-parameter, adamw_bnb_8bit, bf16, gradient checkpointing -- same shapes as
every OPD run, for comparability.

Prints token/length aggregates only -- never text.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STUDENT = "Qwen/Qwen3.5-4B"
OUT = os.environ.get("OUTDIR", "/scratch/eop/jailbreak-opd")


def build_examples(tok, rows, max_prompt_len, max_completion_len,
                   teacher_max_tokens=400):
    """(prompt, target) pairs where target = PREFILL + teacher continuation.

    Two things here are easy to get silently wrong, so both are measured:

    * LENGTH. The teacher was sampled with max_tokens=400, and 69% of its
      completions hit that cap, so a target is 12 (prefill) + up to 400 + 1
      (eos) = up to 413 tokens. A max_completion_len below that does not
      shorten anything -- it DROPS whole examples, and it drops exactly the
      long ones, i.e. the fully-elaborated answers, keeping the short ones
      where the teacher trailed off early. That is a biased subsample of
      precisely the wrong sign, so the caller hard-fails on a high skip rate
      rather than quietly training on the remainder.

    * EOS. A completion that hit the cap is truncated mid-sentence. Appending
      eos to it would train the model to stop abruptly at 400 tokens, which is
      a behaviour the teacher never had. Only completions that ended on their
      own get an eos; truncated ones are trained as unterminated continuations.
    """
    from opd_arm_scihazard import PREFILL, TRAIN_WRAPPER

    eos = tok.eos_token_id
    examples, plens, clens = [], [], []
    skip_prompt = skip_completion = n_trunc = n_noncanon = 0
    for r in rows:
        user = TRAIN_WRAPPER.format(q=r["question"])
        prompt_text = tok.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]

        # Truncated iff the teacher used its whole budget; see docstring.
        truncated = len(tok(r["completion"], add_special_tokens=False)["input_ids"]) \
            >= teacher_max_tokens
        n_trunc += truncated

        target_text = PREFILL + r["completion"]
        completion_ids = tok(target_text, add_special_tokens=False)["input_ids"]
        if not truncated and (not completion_ids or completion_ids[-1] != eos):
            completion_ids = completion_ids + [eos]

        if len(prompt_ids) > max_prompt_len:
            skip_prompt += 1
            continue
        if len(completion_ids) > max_completion_len:
            skip_completion += 1
            continue
        # The model sees the prompt tokenized alone at eval time, so training
        # on a split that re-tokenizes differently would train a sequence the
        # model can never be in. Counted, not assumed.
        if tok(prompt_text + target_text, add_special_tokens=False)["input_ids"] \
                != prompt_ids + tok(target_text, add_special_tokens=False)["input_ids"]:
            n_noncanon += 1

        plens.append(len(prompt_ids))
        clens.append(len(completion_ids))
        examples.append({
            "input_ids": prompt_ids + completion_ids,
            "labels": [-100] * len(prompt_ids) + completion_ids,
        })

    skipped = skip_prompt + skip_completion
    print(f"built {len(examples)} SFT examples from {len(rows)} targets "
          f"({skipped} skipped: {skip_prompt} over max_prompt_len="
          f"{max_prompt_len}, {skip_completion} over max_completion_len="
          f"{max_completion_len})", flush=True)
    print(f"  teacher completions truncated at the {teacher_max_tokens}-token "
          f"cap (no eos trained): {n_trunc}/{len(rows)} "
          f"({100*n_trunc/max(1,len(rows)):.1f}%)", flush=True)
    print(f"  non-canonical prompt/target token boundary: {n_noncanon} "
          f"(want 0)", flush=True)
    if plens:
        print(f"  prompt tokens: mean={st.mean(plens):.1f} max={max(plens)}",
              flush=True)
        print(f"  completion tokens (prefill+teacher[+eos]): "
              f"mean={st.mean(clens):.1f} max={max(clens)}", flush=True)
        n_un = sum(1 for e in examples[0:1]
                   for x in e["labels"] if x != -100)
        print(f"  label check on example 0: {n_un} unmasked positions == "
              f"{clens[0]} target tokens -> {n_un == clens[0]}", flush=True)
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
    ap.add_argument("--targets", default=os.path.join(OUT, "sft_targets.jsonl"))
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-prompt-len", type=int, default=512)
    ap.add_argument("--max-completion-len", type=int, default=448)
    ap.add_argument("--teacher-max-tokens", type=int, default=400,
                    help="the budget opd_sft_gen_targets used; a target at\n                          this length is truncated, so it trains without eos")
    ap.add_argument("--max-skip-frac", type=float, default=0.05,
                    help="hard-fail if more targets than this are dropped by\n                          the length filters -- a silent length filter biases\n                          the training set toward short teacher answers")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-fraction", type=float, default=0.5,
                    help="checkpoint every this fraction of an epoch")
    args = ap.parse_args()

    if not os.path.exists(args.targets):
        print(f"missing SFT targets: {args.targets}", flush=True)
        return 1

    from transformers import (AutoModelForImageTextToText, AutoTokenizer,
                              Trainer, TrainingArguments)

    tok = AutoTokenizer.from_pretrained(STUDENT)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    rows = [json.loads(l) for l in open(args.targets) if l.strip()]
    print(f"loaded {len(rows)} teacher targets from {args.targets}", flush=True)
    examples, skipped = build_examples(
        tok, rows, args.max_prompt_len, args.max_completion_len,
        args.teacher_max_tokens)
    if not examples:
        print("no usable examples after filtering", flush=True)
        return 1
    frac = skipped / max(1, len(rows))
    if frac > args.max_skip_frac:
        print(f"ABORT: length filters dropped {skipped}/{len(rows)} "
              f"({100*frac:.1f}%) of targets, over --max-skip-frac="
              f"{100*args.max_skip_frac:.1f}%. The dropped ones are the LONG "
              f"teacher answers, so training on the rest is a biased sample. "
              f"Raise --max-completion-len instead.", flush=True)
        return 1
    ds = SFTDataset(examples)

    eff_batch = args.batch_size * args.grad_accum
    steps_per_epoch = max(1, (len(ds) + eff_batch - 1) // eff_batch)
    save_steps = max(1, round(steps_per_epoch * args.save_fraction))
    total_steps = max(1, round(steps_per_epoch * args.epochs))
    print(f"steps/epoch={steps_per_epoch}  total_steps={total_steps}  "
          f"save_steps={save_steps} (every {args.save_fraction} epoch)", flush=True)

    output_dir = os.path.join(OUT, f"sft-{os.environ.get('SLURM_JOB_ID', 'local')}")
    targs = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.lr,
        optim=args.optim,
        num_train_epochs=args.epochs,
        logging_steps=1,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=4,
        report_to=[],
        bf16=True,
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
    )

    # AutoModelForImageTextToText, NOT AutoModelForCausalLM. Qwen3.5-4B ships
    # as Qwen3_5ForConditionalGeneration (model_type qwen3_5, with a
    # vision_config); AutoModelForCausalLM resolves to the text-only
    # Qwen3_5ForCausalLM sub-model and saves a text-only checkpoint
    # (model_type qwen3_5_text, 7.9G vs 8.5G, no vision tower). The text
    # weights train identically either way, but vLLM then resolves a
    # DIFFERENT architecture at eval time than it does for every OPD arm --
    # an asymmetry in the eval stack of the very comparison this run exists
    # to make. TRL loads the model string and gets the ConditionalGeneration
    # class, so this matches the OPD checkpoints exactly.
    model = AutoModelForImageTextToText.from_pretrained(
        STUDENT, dtype=torch.bfloat16, attn_implementation="sdpa")

    trainer = Trainer(
        model=model, args=targs, train_dataset=ds,
        data_collator=lambda b: collate(b, tok.pad_token_id),
    )
    trainer.train()

    save_dir = os.path.join(output_dir, "final")
    trainer.save_model(save_dir)
    tok.save_pretrained(save_dir)
    # Qwen3.5 is a ConditionalGeneration checkpoint with a vision_config, so
    # vLLM needs the full processor (preprocessor_config.json etc), not just
    # the tokenizer -- see opd_train.py's save path, which hit exactly this.
    try:
        from transformers import AutoProcessor
        AutoProcessor.from_pretrained(STUDENT).save_pretrained(save_dir)
        print("saved processor alongside the model", flush=True)
    except Exception as e:
        import glob
        import shutil
        print(f"AutoProcessor failed ({e}); copying base aux files", flush=True)
        snaps = glob.glob(os.path.join(
            os.environ.get("HF_HOME", ""), "hub",
            "models--" + STUDENT.replace("/", "--"), "snapshots", "*"))
        if snaps:
            for fn in ("preprocessor_config.json", "video_preprocessor_config.json",
                       "merges.txt", "vocab.json"):
                src = os.path.join(snaps[0], fn)
                if os.path.exists(src):
                    shutil.copy(src, save_dir)
    print(f"saved trained model -> {save_dir}", flush=True)

    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"peak GPU allocated: {peak:.1f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
