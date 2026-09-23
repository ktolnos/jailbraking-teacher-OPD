#!/bin/bash
#SBATCH --job-name=opd-eval
#SBATCH --time=1:30:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --output=/scratch/eop/jailbreak-opd/eval-%j.out
# Post-training gates for one arm. Same script, same judge and cache as the
# baseline in results/student_gates.json, so the two are directly comparable;
# only --model and --tag differ.
#   sbatch scripts/opd_eval.sbatch <model_path> <tag>
set -x
export HF_HOME=/scratch/eop/cache
export OUTDIR=/scratch/eop/jailbreak-opd
mkdir -p $OUTDIR
cd /home/eop/jailbraking-teacher-OPD
source scripts/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
MODEL=${1:?usage: sbatch opd_eval.sbatch <model_path> <tag>}
TAG=${2:?usage: sbatch opd_eval.sbatch <model_path> <tag>}
.venv/bin/python scripts/student_eval.py --wmdp-n 400 --bbq-per-cat 40 \
    --gpu-util 0.88 --model "$MODEL" --tag "$TAG"
