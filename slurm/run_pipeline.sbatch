#!/bin/bash
# Unattended alternative to notebooks/run_pipeline.ipynb on CARC Discovery.
#   sbatch slurm/run_pipeline.sbatch            # whole pipeline (resumable)
#   sbatch slurm/run_pipeline.sbatch smoke      # only the smoke test
# Choose a recent GPU (a100 / a40 / l40s): current PyTorch wheels do not support P100.
#SBATCH --job-name=hw2-il
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/%x-%j.out
##SBATCH --account=<your_project_account>     # uncomment if your allocation requires it
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurm/logs
source "${HW2_VENV:-$HOME/.venvs/hw2}/bin/activate"
export MUJOCO_GL=${MUJOCO_GL:-egl}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

STAGE=${1:-all}
python code/smoke_test.py
[ "$STAGE" = "smoke" ] && exit 0

[ -f data/demos_baseline.npz ] || python code/collect_demos.py
python code/inspect_demos.py
[ -f results/teacher_eval/eval_summary.json ] || python code/evaluate.py --teacher --out-dir results/teacher_eval
python code/run_sweep.py
python code/analyze.py
echo "Pipeline finished. Commit and push results/ and videos/ from a login node."
