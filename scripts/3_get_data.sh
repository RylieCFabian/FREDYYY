#!/usr/bin/env bash
# Step 3 of 4 - get the competition dataset from Google Drive and check its layout.
# Safe to re-run: it skips the download if the data is already in place.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# The organizers' Drive folder, holding train.zip and test.zip.
# Override with: GDRIVE_URL="..." ./scripts/3_get_data.sh
GDRIVE_URL="${GDRIVE_URL:-https://drive.google.com/drive/folders/1LATNw4NgEYQKQB3j3AwkOT8TWfOF9C3A?usp=sharing}"

SPLITS=(train test)
splits_present() { [ -d "$DATA_DIR/train" ] && [ -d "$DATA_DIR/test" ]; }
archive() { echo "$KIT_DIR/$1.zip"; }

manual_help() {
  info "Manual route (always works):"
  info "  1. Open the Google Drive link your organizers gave you in a browser."
  info "  2. Download train.zip and test.zip and save them as:"
  info "       $(archive train)"
  info "       $(archive test)"
  info "  3. Run this script again - it will unpack them for you."
}

download_failed() {
  warn "The download failed (see the message above). Either the folder is not shared as"
  warn "'Anyone with the link' yet, or Google Drive's daily quota is exhausted - common when"
  warn "many people fetch the same files at once."
  manual_help
  exit 1
}

# Unpack <split>.zip into data/<split>/, whether the zip holds a top-level <split>/
# folder or the sequence folders directly.
unpack() {
  local split="$1" staging="$DATA_DIR/.unpack_$1" inner
  rm -rf "$staging" "$DATA_DIR/$split"
  mkdir -p "$staging"
  # Python's zipfile instead of `unzip`, which Git Bash on Windows doesn't ship.
  "$PYTHON" -m zipfile -e "$(archive "$split")" "$staging" \
    || die "$split.zip is corrupt or incomplete. Delete $(archive "$split") and try again."
  rm -rf "$staging/__MACOSX"
  find "$staging" -name .DS_Store -delete  # one next to a wrapper folder blocks the hoist
  # Hoist single wrapper folders (e.g. train/ or dataset/train/) up to the sequences.
  while [ "$(find "$staging" -mindepth 1 -maxdepth 1 | wc -l | tr -d ' ')" -eq 1 ] \
        && inner="$(find "$staging" -mindepth 1 -maxdepth 1 -type d)" && [ -n "$inner" ] \
        && [ -z "$(find "$inner" -maxdepth 1 -name 'frame_*' | head -n1)" ]; do
    mv "$inner" "$staging.tmp" && rm -rf "$staging" && mv "$staging.tmp" "$staging"
  done
  mv "$staging" "$DATA_DIR/$split"
  rm -f "$(archive "$split")"
}

step "Getting the dataset"
# Each split is handled on its own: already unpacked, a zip waiting, or still to download.
missing=(); to_fetch=()
for split in "${SPLITS[@]}"; do
  if [ -d "$DATA_DIR/$split" ]; then
    ok "$split/ already in $DATA_DIR - skipping it"
  else
    missing+=("$split")
    if [ -f "$(archive "$split")" ]; then ok "found $split.zip - using it instead of downloading"
    else to_fetch+=("$split"); fi
  fi
done

if [ "${#missing[@]}" -gt 0 ]; then
  PYTHON="$(venv_python)"; [ -x "$PYTHON" ] || die "Run ./scripts/2_install.sh first."

  if [ "${#to_fetch[@]}" -gt 0 ]; then
    if [ -z "$GDRIVE_URL" ]; then
      warn "No Google Drive link set (organizers: edit GDRIVE_URL in this script)."
      manual_help
      exit 1
    fi
    info "Downloading from Google Drive with gdown (this can take a few minutes)..."
    staging="$KIT_DIR/.gdrive_download"
    rm -rf "$staging"
    "$PYTHON" -m gdown --folder "$GDRIVE_URL" -O "$staging" || { rm -rf "$staging"; download_failed; }
    for split in "${to_fetch[@]}"; do
      found="$(find "$staging" -type f -name "$split.zip" | head -n1)"
      [ -n "$found" ] || { rm -rf "$staging"; die "No $split.zip in the Drive folder. Tell a mentor."; }
      mv "$found" "$(archive "$split")"
    done
    rm -rf "$staging"
  fi

  info "Unpacking..."
  mkdir -p "$DATA_DIR"
  for split in "${missing[@]}"; do unpack "$split"; done
  ok "unpacked into $DATA_DIR"
fi

step "Checking the layout"
splits_present || die "Expected train/ and test/ inside $DATA_DIR."
for split in "${SPLITS[@]}"; do
  n=$(find "$DATA_DIR/$split" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
  [ "$n" -gt 0 ] || die "$split/ has no sequence folders."
  first="$(find "$DATA_DIR/$split" -mindepth 1 -maxdepth 1 -type d | sort | sed -n 1p)"
  frames=$(find "$first" -maxdepth 1 -name 'frame_*' | wc -l | tr -d ' ')
  [ "$frames" -gt 0 ] || die "$split/ sequences have no frames. Delete $DATA_DIR/$split and run this script again."
  ok "$split: $n sequences, $frames frames each"
done
[ -f "$DATA_DIR/train/labels.csv" ] && ok "train/labels.csv found" || die "train/labels.csv is missing."
if [ -f "$DATA_DIR/test/labels.csv" ]; then
  warn "test/ contains labels.csv - that should not be there, tell a mentor."
fi

echo
say "${GREEN}${BOLD}Data ready.${RESET} Next: ./scripts/4_verify.sh"
