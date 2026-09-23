#!/bin/bash
#SBATCH --job-name=opd-sftgen
#SBATCH --time=1:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/scratch/eop/jailbreak-opd/sftgen-%j.out
# API only: teacher completions on the 980 training prompts, for SFT targets.
set -x
export HF_HOME=/scratch/eop/cache
export OUTDIR=/scratch/eop/jailbreak-opd
cd /home/eop/jailbraking-teacher-OPD
.venv/bin/python scripts/opd_sft_gen_targets.py "${1:-980}"
