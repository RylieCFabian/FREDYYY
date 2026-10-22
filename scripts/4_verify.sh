#!/usr/bin/env bash
# Step 4 of 4 - prove everything works end to end before the competition clock matters.
# Runs verify_setup.py inside the environment from step 2.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

PYTHON="$(venv_python)"
[ -x "$PYTHON" ] || die "No environment yet. Run ./scripts/2_install.sh first."

step "Running the end-to-end check"
DATA_ROOT="$DATA_DIR" "$PYTHON" "$KIT_DIR/verify_setup.py"
