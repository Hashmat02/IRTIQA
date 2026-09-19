#!/usr/bin/env bash
# Run code_switch.py for selected IRTIQA datasets/languages/tasks.
#
# Required:
#   IRTIQA_ROOT=/path/to/IRTIQA repo
#
# Use one of:
#   API_URL=http://localhost:8000
#   or
#   APPTAINER_BIN=/path/to/apptainer APPTAINER_IMAGE=/path/to/vllm.sif MODEL_PATH=<model>
#
# Optional:
#   CACHE_DIR=/path/to/cache
#   CONTAINER_BIND=/host/path:/container/path
#
# Example:
#   IRTIQA_ROOT=$PWD API_URL=http://localhost:8000 bash IRTIQA/data_curation/run_translation.sh \
#     --dataset drop --task translate --lang ur

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IRTIQA_ROOT="${IRTIQA_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
SCRIPT="${SCRIPT:-$SCRIPT_DIR/code_switch.py}"
MODEL_PATH="${MODEL_PATH:-google/gemma-4-31b-it}"
HF_CACHE="${HF_CACHE:-$IRTIQA_ROOT/.cache/huggingface}"
API_URL="${API_URL:-}"
API_MODEL="${API_MODEL:-$MODEL_PATH}"
APPTAINER_BIN="${APPTAINER_BIN:-}"
APPTAINER_IMAGE="${APPTAINER_IMAGE:-}"
CONTAINER_BIND="${CONTAINER_BIND:-$IRTIQA_ROOT:$IRTIQA_ROOT}"

LANGS="ur"
TASKS="translate code_switch"
DATASETS="drop snips skillsbench"
BATCH_SIZE=32
MAX_TOKENS=3000
TEMPERATURE=0.1
TENSOR_PARALLEL=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASETS="$2"; shift 2 ;;
        --task) TASKS="$2"; shift 2 ;;
        --lang|--langs) LANGS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --api-url) API_URL="$2"; shift 2 ;;
        --api-model) API_MODEL="$2"; shift 2 ;;
        --model) MODEL_PATH="$2"; API_MODEL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

BENCHMARKS="$IRTIQA_ROOT/Benchmarks"
SAMPLES="$BENCHMARKS/samples"
LOG_DIR="$IRTIQA_ROOT/logs"
mkdir -p "$LOG_DIR"

run_python() {
    if [[ -n "$API_URL" ]]; then
        python3 "$SCRIPT" "$@" --api-url "$API_URL" --api-model "$API_MODEL"
    else
        if [[ -z "$APPTAINER_BIN" || -z "$APPTAINER_IMAGE" ]]; then
            echo "Set API_URL, or set APPTAINER_BIN and APPTAINER_IMAGE for local container inference." >&2
            exit 1
        fi
        "$APPTAINER_BIN" exec --nv \
            --bind "$CONTAINER_BIND" \
            --env HF_HOME="$HF_CACHE" \
            --env TRANSFORMERS_CACHE="$HF_CACHE" \
            "$APPTAINER_IMAGE" \
            python3 "$SCRIPT" "$@" --model "$MODEL_PATH" --tensor-parallel-size "$TENSOR_PARALLEL"
    fi
}

for dataset in $DATASETS; do
    for task in $TASKS; do
        for lang in $LANGS; do
            case "$dataset" in
                drop)
                    input="$SAMPLES/drop/drop_sample100.jsonl"
                    output="$SAMPLES/drop"
                    ;;
                snips)
                    input="$SAMPLES/snips"
                    output="$SAMPLES/snips"
                    ;;
                skillsbench)
                    input="$SAMPLES/skillsbench"
                    output="$SAMPLES/skillsbench"
                    ;;
                appworld)
                    input="$BENCHMARKS/appworld/appworld_tasks.jsonl"
                    output="$BENCHMARKS/appworld"
                    ;;
                *) echo "Unknown dataset: $dataset" >&2; exit 1 ;;
            esac

            log_file="$LOG_DIR/$(date '+%Y%m%d')_${dataset}_${task}_${lang}.log"
            echo "Running $dataset | $task | $lang"
            run_python "$dataset" "$task" "$lang" \
                --input "$input" \
                --output "$output" \
                --batch-size "$BATCH_SIZE" \
                --max-tokens "$MAX_TOKENS" \
                --temperature "$TEMPERATURE" \
                2>&1 | tee "$log_file"
        done
    done
done
