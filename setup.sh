#!/usr/bin/env bash
# One command to run all four steps in order. Stops at the first problem.
#   ./setup.sh
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for s in 1_check_system 2_install 3_get_data 4_verify; do
  bash "$here/scripts/$s.sh"
done
