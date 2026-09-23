#!/bin/bash
#SBATCH --job-name=pfx-sft
#SBATCH --time=1:00:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --output=/scratch/eop/jailbreak-opd/pfxsft-%j.out
# Content-free prefix-emission SFT. Targets are the 12-token opener only.
#   sbatch scripts/prefix_sft.sbatch --base <model> --tag <tag>
set -x
export HF_HOME=/scratch/eop/cache
export OUTDIR=/scratch/eop/jailbreak-opd
mkdir -p $OUTDIR
cd /home/eop/jailbraking-teacher-OPD
source scripts/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
.venv/bin/python scripts/prefix_sft.py "$@"
