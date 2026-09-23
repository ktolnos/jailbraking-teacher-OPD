#!/bin/bash
#SBATCH --job-name=opd-check
#SBATCH --time=15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --output=/scratch/eop/jailbreak-opd/check-%j.out
# Preflight: IWOPDConfig validation + collator render. No model is loaded.
#
# TRL validates the config inside IWOPDConfig.__post_init__, which runs *after*
# the 4-minute alignment gate, so a config typo otherwise costs a full gate
# cycle plus a GPU queue wait to discover.
#
# Why a GPU for a config check: TrainingArguments validates bf16 support
# against real hardware, so `bf16=True` cannot be constructed on a CPU node.
# Setting bf16=False here would validate a *different* config than the one
# training builds, so instead take the smallest slice that exists -- 1g.10gb
# reaches the most nodes and schedules fastest, and 10 GB is ample with no
# model loaded. 5 min is the site's test-job minimum.
set -x
export HF_HOME=/scratch/eop/cache
export OUTDIR=/scratch/eop/jailbreak-opd
cd /home/eop/jailbraking-teacher-OPD
.venv/bin/python scripts/opd_train.py --check-config "$@"
