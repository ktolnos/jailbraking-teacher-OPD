#!/bin/bash
#SBATCH --job-name=opd-armgen
#SBATCH --time=40:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --output=/scratch/eop/jailbreak-opd/armgen-%j.out
# Held-out SciHazard generations for one arm, for LLM-judge comparison.
#   sbatch scripts/opd_arm_scihazard.sbatch <model_path_or_id> <tag>
set -x
export HF_HOME=/scratch/eop/cache
export OUTDIR=/scratch/eop/jailbreak-opd
# vLLM picks fork vs spawn by probing whether CUDA is already initialised,
# and that probe is architecture-dependent: the ConditionalGeneration
# checkpoints trip it and get spawn, a text-only CausalLM checkpoint does
# not, forks, and dies with "Cannot re-initialize CUDA in forked
# subprocess". Pinning spawn makes the eval path identical for every arm.
export VLLM_WORKER_MULTIPROC_METHOD=spawn
mkdir -p $OUTDIR
cd /home/eop/jailbraking-teacher-OPD
source scripts/env.sh
MODEL=${1:?usage: sbatch opd_arm_scihazard.sbatch <model> <tag>}
TAG=${2:?usage: sbatch opd_arm_scihazard.sbatch <model> <tag>}
.venv/bin/python scripts/opd_arm_scihazard.py --model "$MODEL" --tag "$TAG" --n ${3:-200}
