#!/bin/bash
#SBATCH --job-name=opd-scihazard-student
#SBATCH --time=2:00:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --output=/scratch/eop/jailbreak-opd/scihazard-student-%j.out
set -x
export HF_HOME=/scratch/eop/cache
export OUTDIR=/scratch/eop/jailbreak-opd
mkdir -p $OUTDIR
cd /home/eop/jailbraking-teacher-OPD
source scripts/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
.venv/bin/python scripts/student_scihazard.py
