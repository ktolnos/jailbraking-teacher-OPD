#!/bin/bash
#SBATCH --job-name=comply
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/scratch/eop/jailbreak-opd/comply-%j.out
# API-only judge calls. No GPU.
set -x
export OUTDIR=/scratch/eop/jailbreak-opd
cd /home/eop/jailbraking-teacher-OPD
.venv/bin/python scripts/compliance.py "$@"
