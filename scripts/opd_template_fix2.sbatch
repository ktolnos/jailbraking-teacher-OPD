#!/bin/bash
#SBATCH --job-name=opd-fix2
#SBATCH --time=15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --output=/scratch/eop/jailbreak-opd/fix2-%j.out
# CPU-only: tokenizer + real proxy, no model. Validates canonicalize-before-loss
# against the teacher before spending a GPU slot on it.
set -x
export HF_HOME=/scratch/eop/cache
export OUTDIR=/scratch/eop/jailbreak-opd
cd /home/eop/jailbraking-teacher-OPD
.venv/bin/python scripts/opd_template_fix2.py
