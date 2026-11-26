#!/usr/bin/env bash
#SBATCH --job-name=irtiqa-appworld-fullsize
#SBATCH --output=logs/slurm_appworld_%j.out
#SBATCH --error=logs/slurm_appworld_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Portable SLURM template. Add your cluster-specific GPU/partition directives
# above or pass them at submission time.
#
# Required:
#   IRTIQA_ROOT=/path/to/repo
#   API_URL=http://host:port

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IRTIQA_ROOT="${IRTIQA_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
API_URL="${API_URL:-http://localhost:${PORT:-8000}}"
API_MODEL="${API_MODEL:-${MODEL_PATH:-google/gemma-4-31b-it}}"

mkdir -p "$IRTIQA_ROOT/logs"

"$SCRIPT_DIR/run_translation.sh" \
    --dataset appworld \
    --task "translate code_switch" \
    --lang ur \
    --api-url "$API_URL" \
    --api-model "$API_MODEL" \
    --max-tokens 1500
