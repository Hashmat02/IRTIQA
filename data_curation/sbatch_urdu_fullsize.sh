#!/usr/bin/env bash
#SBATCH --job-name=irtiqa-urdu-fullsize
#SBATCH --array=0-2
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Portable SLURM template. Add your cluster-specific GPU/partition directives
# above or pass them at submission time, for example:
#   sbatch --gres=gpu:1 --partition=<partition> IRTIQA/data_curation/sbatch_urdu_fullsize.sh
#
# Required:
#   IRTIQA_ROOT=/path/to/repo
#   API_URL=http://host:port
# or:
#   APPTAINER_BIN=/path/to/apptainer APPTAINER_IMAGE=/path/to/vllm.sif MODEL_PATH=<model>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IRTIQA_ROOT="${IRTIQA_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PORT="${PORT:-$((8000 + ${SLURM_IRTIQAAY_TASK_ID:-0}))}"
API_URL="${API_URL:-http://localhost:${PORT}}"
API_MODEL="${API_MODEL:-${MODEL_PATH:-google/gemma-4-31b-it}}"

DATASETS=(drop snips skillsbench)
DATASET="${DATASETS[${SLURM_IRTIQAAY_TASK_ID:-0}]}"

mkdir -p "$IRTIQA_ROOT/logs"

"$SCRIPT_DIR/run_translation.sh" \
    --dataset "$DATASET" \
    --task "translate code_switch" \
    --lang ur \
    --api-url "$API_URL" \
    --api-model "$API_MODEL"
