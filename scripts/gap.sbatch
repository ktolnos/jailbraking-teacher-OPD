#!/bin/bash
#SBATCH --job-name=gap
#SBATCH --time=40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/scratch/eop/jailbreak-opd/gap-%j.out
# CPU + network only. Pass opd_arm_gap.py flags straight through, so the
# cross-arm and --require-compliant options are usable (opd_arm_gap.sbatch
# hardcodes only --a/--b/--label).
set -x
export HF_HOME=/scratch/eop/cache
export OUTDIR=/scratch/eop/jailbreak-opd
cd /home/eop/jailbraking-teacher-OPD
.venv/bin/python scripts/opd_arm_gap.py "$@"
