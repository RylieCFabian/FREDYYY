#!/usr/bin/env bash
# Step 1 of 4 - check that this computer can run the competition.
# Changes nothing. Safe to run as many times as you like.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

problems=0
fail() { printf '  %s✗%s %s\n' "$RED" "$RESET" "$*"; problems=$((problems + 1)); }

step "Operating system"
case "$OS" in
  darwin)  ok "macOS ($(uname -m))" ;;
  linux)   ok "Linux" ;;
  wsl)     ok "Windows Subsystem for Linux" ;;
  windows) ok "Windows (Git Bash)"
           warn "WSL2 is smoother for ML work; Git Bash works but is CPU-only here." ;;
  *)       fail "Unrecognized system '$(uname -s)'. Ask a mentor for help." ;;
esac

step "Python"
if has uv; then
  ok "uv found ($(uv --version)) - it will fetch the right Python for you"
elif py="$(find_python)"; then
  ok "$py ($($py --version 2>&1)) - supported"
else
  fail "No Python 3.9-3.13 found (newer versions don't have PyTorch builds yet)."
  info "Install Python 3.11 from https://www.python.org/downloads/ or: brew install python@3.11"
fi

step "Tools"
has git  && ok "git"  || warn "git not found - only needed if you clone the repo"
has curl && ok "curl" || fail "curl not found - needed to download packages and data"

step "Disk space"
free_gb=$(df -Pk . | awk 'NR==2 {printf "%d", $4 / 1024 / 1024}')
if [ "$free_gb" -ge 5 ]; then ok "${free_gb} GB free (need ~5 GB)"; else fail "only ${free_gb} GB free, need ~5 GB for PyTorch + the dataset"; fi

step "Internet"
if curl -sSf -m 10 -o /dev/null https://pypi.org/simple/pip/; then ok "can reach PyPI"; else fail "cannot reach pypi.org - check the Wi-Fi"; fi

step "Hardware acceleration (optional, makes training faster)"
if has_nvidia_gpu; then
  ok "NVIDIA GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1) -> training will use CUDA"
  [ "$OS" = windows ] && warn "Native Windows installs CPU-only PyTorch; use WSL2 to get the GPU."
elif [ "$OS" = darwin ] && [ "$(uname -m)" = arm64 ]; then
  ok "Apple Silicon -> training will use the GPU through MPS"
else
  info "No GPU found. Training runs on the CPU: fine for the baseline, slower for big models."
fi

echo
if [ "$problems" -eq 0 ]; then
  say "${GREEN}${BOLD}All good.${RESET} Next: ./scripts/2_install.sh"
else
  die "$problems problem(s) above. Fix them, then run this script again."
fi
