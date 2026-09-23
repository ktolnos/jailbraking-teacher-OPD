#!/bin/bash
#SBATCH --job-name=opd-tgen
#SBATCH --time=1:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/scratch/eop/jailbreak-opd/tgen-%j.out
# API only: teacher generations on the held-out eval split. No GPU.
set -x
export HF_HOME=/scratch/eop/cache
export OUTDIR=/scratch/eop/jailbreak-opd
cd /home/eop/jailbraking-teacher-OPD
.venv/bin/python scripts/opd_teacher_gen.py "${1:-420}"
