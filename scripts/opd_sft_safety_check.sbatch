#!/bin/bash
#SBATCH --job-name=opd-sftcheck
#SBATCH --time=10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/scratch/eop/jailbreak-opd/sftcheck-%j.out
# CPU only, no network: refusal-rate counting from generation files on disk.
set -x
export OUTDIR=/scratch/eop/jailbreak-opd
cd /home/eop/jailbraking-teacher-OPD
.venv/bin/python scripts/opd_sft_safety_check.py "${1:-sft}"
