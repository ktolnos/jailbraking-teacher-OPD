#!/bin/bash
#SBATCH --job-name=opd-gates
#SBATCH --time=1:30:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --output=/scratch/eop/jailbreak-opd/gates-%j.out
set -x
export HF_HOME=/scratch/eop/cache
export OUTDIR=/scratch/eop/jailbreak-opd
mkdir -p $OUTDIR
cd /home/eop/jailbraking-teacher-OPD
# puts the venv's nvcc on PATH so flashinfer can JIT its sampling kernels
source scripts/env.sh
which nvcc; nvcc --version | tail -1; echo "CUDA_HOME=$CUDA_HOME"
nvidia-smi --query-gpu=name,memory.total --format=csv
.venv/bin/python scripts/student_eval.py --wmdp-n 400 --bbq-per-cat 40 --gpu-util 0.88
