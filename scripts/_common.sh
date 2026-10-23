#!/usr/bin/env bash
# Shared helpers, sourced by every setup script. Nothing here runs on its own.

set -euo pipefail

# Always work from the starter kit folder, no matter where the script was launched from.
KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$KIT_DIR"

VENV_DIR="$KIT_DIR/.venv"
DATA_DIR="${DATA_ROOT:-$KIT_DIR/data}"

# Colors, only when printing to a real terminal.
if [ -t 1 ]; then
  BOLD=$'\033[1m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; DIM=$'\033[2m'; RESET=$'\033[0m'
else
  BOLD=""; GREEN=""; YELLOW=""; RED=""; DIM=""; RESET=""
fi

say()  { printf '%s\n' "$*"; }
step() { printf '\n%s==> %s%s\n' "$BOLD" "$*" "$RESET"; }
ok()   { printf '  %s✓%s %s\n' "$GREEN" "$RESET" "$*"; }
warn() { printf '  %s!%s %s\n' "$YELLOW" "$RESET" "$*"; }
info() { printf '  %s%s%s\n' "$DIM" "$*" "$RESET"; }
die()  { printf '\n  %s✗ %s%s\n\n' "$RED" "$*" "$RESET" >&2; exit 1; }

# darwin | linux | wsl | windows (Git Bash / MSYS / Cygwin)
detect_os() {
  case "$(uname -s)" in
    Darwin) echo darwin ;;
    Linux)  if grep -qi microsoft /proc/version 2>/dev/null; then echo wsl; else echo linux; fi ;;
    MINGW*|MSYS*|CYGWIN*) echo windows ;;
    *) echo unknown ;;
  esac
}
OS="$(detect_os)"

# Windows virtualenvs put executables in Scripts/, everyone else in bin/.
venv_python() {
  if [ "$OS" = windows ]; then echo "$VENV_DIR/Scripts/python.exe"; else echo "$VENV_DIR/bin/python"; fi
}

has() { command -v "$1" >/dev/null 2>&1; }

# Print the first system Python that PyTorch supports (3.9 - 3.13), or nothing.
find_python() {
  local c
  for c in python3.12 python3.11 python3.10 python3.13 python3.9 python3 python; do
    if has "$c" && "$c" -c 'import sys; sys.exit(0 if (3,9) <= sys.version_info[:2] <= (3,13) else 1)' 2>/dev/null; then
      echo "$c"; return 0
    fi
  done
  return 1
}

has_nvidia_gpu() { has nvidia-smi && nvidia-smi -L >/dev/null 2>&1; }
