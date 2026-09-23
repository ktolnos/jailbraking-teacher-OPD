#!/bin/bash
#SBATCH --job-name=opd-sft
#SBATCH --time=2:00:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --output=/scratch/eop/jailbreak-opd/sft-%j.out
# Plain SFT (not IW-OPD): train the student to emit the prefill itself.
# No rollouts, no teacher-server scoring during training -- much cheaper than
# the OPD arms, hence the shorter walltime request.
set -x
export HF_HOME=/scratch/eop/cache
export OUTDIR=/scratch/eop/jailbreak-opd
mkdir -p $OUTDIR
cd /home/eop/jailbraking-teacher-OPD
source scripts/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
.venv/bin/python scripts/opd_sft_train.py "$@"
