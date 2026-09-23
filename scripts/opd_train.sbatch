#!/bin/bash
#SBATCH --job-name=opd-iwopd
#SBATCH --time=1:00:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --output=/scratch/eop/jailbreak-opd/iwopd-%j.out
# First IW-OPD run: the alignment gate, then a short full-parameter training run.
#
# Full-parameter update (no LoRA) with 8-bit Adam: 9.3 GB bf16 weights +
# 9.3 GB grads + ~9.3 GB blockwise-quantised moments. MEASURED peak on job
# 60422731 was 31.5 GiB at max_completion_length=256, not the ~28 GB this
# comment used to estimate -- ~8 GB of headroom on a 40 GB slice, so a longer
# completion budget needs a re-measure before it is assumed to fit. Plain AdamW would be 37.3 GB of
# state alone and OOM here -- use --optim adamw_torch with --gpus=h100:1
# instead; adafactor (~19 GB) trades real moments for factored ones. See the
# memory table in opd_train.py's docstring.
#
# Rollouts are canonicalized before the loss (opd_canonicalize.py) so the
# teacher scores the tokens the student trains on. Without it, a rollout whose
# sampled tokens are not their own text's canonical encoding is unscoreable --
# that is what stopped job 60405196 at zero training steps.
#
# No vLLM here (use_vllm=False in opd_train.py), so flashinfer never JITs and
# env.sh is only sourced to keep the venv's toolchain consistent.
set -x
export HF_HOME=/scratch/eop/cache
export OUTDIR=/scratch/eop/jailbreak-opd
mkdir -p $OUTDIR
cd /home/eop/jailbraking-teacher-OPD
source scripts/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
if [ $# -eq 0 ]; then set -- --max-steps 20; fi
.venv/bin/python scripts/opd_train.py "$@"
