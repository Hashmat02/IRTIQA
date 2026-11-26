#!/usr/bin/env bash
# Generic IRTIQA framework runner.
#
# Usage:
#   bash IRTIQA/scripts/run_framework.sh --framework toolr0 --dataset snips --variant urdu
#   bash IRTIQA/scripts/run_framework.sh --framework autogenesis --dataset skillsbench --variant codeswitched
#
# Variants:
#   english | urdu | codeswitched
#
# Required by framework:
#   Tool-R0:       TOOLR0_ROOT or TOOLR0_INTEGRATION
#   AgentEvolver: AGENTEVOLVER_ROOT
#   Autogenesis:  AUTOGENESIS_ROOT
#
# Model serving:
#   MODEL_BASE_URL=http://localhost:8000/v1 MODEL_NAME=<served-model-name>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IRTIQA_ROOT="${IRTIQA_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PYTHON="${PYTHON:-python3}"

framework=""
dataset=""
variant="english"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --framework) framework="$2"; shift 2 ;;
        --dataset) dataset="$2"; shift 2 ;;
        --variant) variant="$2"; shift 2 ;;
        --help|-h)
            sed -n '1,32p' "$0"
            exit 0
            ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$framework" || -z "$dataset" ]]; then
    echo "Missing --framework or --dataset. Use --help." >&2
    exit 1
fi

case "$variant" in
    english|en|original) variant_key="original" ;;
    urdu|ur|translate|ur_translate) variant_key="ur_translate" ;;
    codeswitched|code_switched|code-switch|ur_code_switch) variant_key="ur_code_switch" ;;
    *) echo "Unknown variant: $variant" >&2; exit 1 ;;
esac

require_model() {
    : "${MODEL_BASE_URL:?Set MODEL_BASE_URL, for example http://localhost:8000/v1}"
    : "${MODEL_NAME:?Set MODEL_NAME to the served model name}"
}

run_toolr0() {
    TOOLR0_INTEGRATION="${TOOLR0_INTEGRATION:-${TOOLR0_ROOT:-}/arr_integration}"
    : "${TOOLR0_INTEGRATION:?Set TOOLR0_INTEGRATION or TOOLR0_ROOT}"
    require_model
    cd "$TOOLR0_INTEGRATION"

    case "$dataset" in
        appworld)
            exec "$PYTHON" eval_appworld_native.py --variant "$variant_key"
            ;;
        snips)
            exec "$PYTHON" eval_snips_toolcall.py --variant "$variant_key"
            ;;
        skillsbench)
            echo "Paper matrix: Tool-R0 was not reported on SkillsBench." >&2
            exit 1
            ;;
        drop)
            if [[ -f ./toolr0_eval.py ]]; then
                exec "$PYTHON" ./toolr0_eval.py --dataset drop --selection "$variant_key"
            fi
            echo "Tool-R0 DROP runner not found. Provide TOOLR0_INTEGRATION with a DROP evaluator." >&2
            exit 1
            ;;
        *)
            echo "Tool-R0 dataset not supported by this wrapper: $dataset" >&2
            exit 1
            ;;
    esac
}

run_agentevolver() {
    : "${AGENTEVOLVER_ROOT:?Set AGENTEVOLVER_ROOT to your AgentEvolver checkout}"
    cd "$AGENTEVOLVER_ROOT"

    case "$dataset" in
        drop|snips)
            case "$variant_key" in
                original) config="arr_${dataset}_en" ;;
                ur_translate) config="arr_${dataset}_ur_translate" ;;
                ur_code_switch) config="arr_${dataset}_ur_code_switch" ;;
            esac
            ;;
        skillsbench)
            echo "Paper matrix: AgentEvolver was not reported on SkillsBench." >&2
            exit 1
            ;;
        appworld)
            case "$variant_key" in
                original) config="arr_appworld_en_1gpu" ;;
                ur_translate) config="arr_appworld_ur_1gpu" ;;
                ur_code_switch) config="arr_appworld_urcs_1gpu" ;;
            esac
            ;;
        *)
            echo "AgentEvolver dataset not supported: $dataset" >&2
            exit 1
            ;;
    esac

    exec "$PYTHON" -m agentevolver.module.task_manager \
        --config-path "$AGENTEVOLVER_ROOT/examples" \
        --config-name "$config"
}

run_autogenesis() {
    : "${AUTOGENESIS_ROOT:?Set AUTOGENESIS_ROOT to your Autogenesis checkout}"
    require_model
    cd "$AUTOGENESIS_ROOT"

    case "$dataset" in
        drop|snips|appworld|skillsbench)
            script="${AUTOGENESIS_EVAL_SCRIPT:-}"
            if [[ -z "$script" ]]; then
                case "$dataset" in
                    skillsbench) script="examples/run_skillsbench.py" ;;
                    drop) script="examples/run_drop.py" ;;
                    snips) script="examples/run_snips.py" ;;
                    appworld) script="examples/run_appworld.py" ;;
                esac
            fi
            if [[ ! -f "$script" ]]; then
                echo "Autogenesis paper matrix includes $dataset, but no evaluator script was found at $script." >&2
                echo "Set AUTOGENESIS_EVAL_SCRIPT to the correct evaluator in your checkout." >&2
                exit 1
            fi
            exec "$PYTHON" "$script" --variant "$variant_key"
            ;;
        *)
            echo "Autogenesis dataset not supported: $dataset" >&2
            exit 1
            ;;
    esac
}

case "$framework" in
    toolr0|tool-r0) run_toolr0 ;;
    agentevolver|agent-evolver) run_agentevolver ;;
    autogenesis) run_autogenesis ;;
    *) echo "Unknown framework: $framework" >&2; exit 1 ;;
esac
