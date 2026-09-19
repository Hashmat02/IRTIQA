#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/run_framework.sh" --framework agentevolver --dataset appworld "$@"
