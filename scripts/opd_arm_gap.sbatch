#!/bin/bash
#SBATCH --job-name=opd-gap
#SBATCH --time=40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/scratch/eop/jailbreak-opd/gap-%j.out
# CPU + network only: pairwise judging of one arm against the clean model.
#   sbatch scripts/opd_arm_gap.sbatch <a.jsonl> <b.jsonl> <label>
set -x
export HF_HOME=/scratch/eop/cache
export OUTDIR=/scratch/eop/jailbreak-opd
cd /home/eop/jailbraking-teacher-OPD
.venv/bin/python scripts/opd_arm_gap.py --a "${1:?}" --b "${2:?}" --label "${3:?}"
