#!/usr/bin/env bash
# Step 2 of 4 - create an isolated Python environment and install PyTorch & friends.
# Safe to re-run: it reuses the existing environment and only adds what's missing.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

PY_VERSION="3.11"   # the version uv will use; everything here is tested on it

step "Creating the virtual environment (.venv)"
if [ -x "$(venv_python)" ]; then
  ok "already exists - reusing it"
elif has uv; then
  uv venv --python "$PY_VERSION" "$VENV_DIR" >/dev/null
  ok "created with uv (Python $PY_VERSION)"
else
  py="$(find_python)" || die "No supported Python found. Run ./scripts/1_check_system.sh for help."
  "$py" -m venv "$VENV_DIR"
  ok "created with $py"
fi
PYTHON="$(venv_python)"

# On a Linux machine without an NVIDIA GPU, the default PyTorch wheel drags in ~2.5 GB of
# CUDA libraries that can never be used. PyTorch publishes a CPU-only build; use it there.
# Override with TORCH_FLAVOR=default or TORCH_FLAVOR=cpu.
flavor="${TORCH_FLAVOR:-auto}"
if [ "$flavor" = auto ]; then
  if { [ "$OS" = linux ] || [ "$OS" = wsl ]; } && ! has_nvidia_gpu; then flavor=cpu; else flavor=default; fi
fi
index_args=()
if [ "$flavor" = cpu ]; then
  index_args=(--index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple)
fi

step "Installing packages from requirements.txt (torch build: $flavor)"
info "This is the slow step - PyTorch is a few hundred MB. Grab a coffee."
if has uv; then
  uv pip install --python "$PYTHON" -r requirements.txt ${index_args[@]+"${index_args[@]}"}
else
  "$PYTHON" -m pip install --quiet --upgrade pip
  "$PYTHON" -m pip install -r requirements.txt ${index_args[@]+"${index_args[@]}"}
fi
ok "packages installed"

step "Checking that PyTorch imports"
"$PYTHON" -c 'import torch, torchvision, PIL; print("  torch", torch.__version__, "| torchvision", torchvision.__version__)' \
  || die "PyTorch did not import. Re-run this script; if it keeps failing, ask a mentor."

echo
say "${GREEN}${BOLD}Environment ready.${RESET} Next: ./scripts/3_get_data.sh"
