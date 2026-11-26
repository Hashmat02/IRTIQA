#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/run_framework.sh" --framework toolr0 --dataset drop "$@"
