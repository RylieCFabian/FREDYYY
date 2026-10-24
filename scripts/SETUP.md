# Setup guide

## Prerequisites

- **macOS**: use Terminal. Nothing to install first.
- **Linux**: any terminal.
- **Windows**: you need a bash shell. Pick one:
  - **WSL2** (recommended): run `wsl --install` in PowerShell, then work inside the Ubuntu terminal. This is also the only way to use an NVIDIA GPU on Windows.
  - **Git Bash** (ships with [Git for Windows](https://git-scm.com/download/win)): works, CPU-only.
- About **5 GB of free disk space** and an internet connection.
- **No GPU needed.** The baseline trains on a plain CPU; a GPU only makes it faster.

If a script says `Permission denied`, run `chmod +x setup.sh scripts/*.sh` once, or put `bash` in front of the command.

## What setup is really doing

A machine learning project is four things that have to agree with each other. Setup builds each one and then proves they work together:

| Ingredient | Why it can go wrong | Which step handles it |
|---|---|---|
| **Your computer** | Wrong Python version, no disk space, no internet | Step 1 checks, changes nothing |
| **The environment** (Python + libraries) | Different versions on different laptops; "works on my machine" | Step 2 builds one identical, isolated environment |
| **The data** | Half-downloaded, mis-unpacked, wrong folder layout | Step 3 downloads and inspects it |
| **The whole pipeline** | Everything installs fine but a piece silently doesn't connect | Step 4 runs the real code on real data |

The goal is that when the competition starts, none of these can be your problem. The steps are ordered so each one only relies on the ones before it, and every script is **idempotent**: running it twice is safe and just re-checks what is already done.

You can run `./setup.sh` to do all four, or run them one at a time to watch each happen:

| Step | Command | Takes | Result |
|---|---|---|---|
| 1 | `./scripts/1_check_system.sh` | seconds | A report on your machine |
| 2 | `./scripts/2_install.sh` | 1-5 min | `.venv/` with PyTorch inside |
| 3 | `./scripts/3_get_data.sh` | varies | `data/` with the train and test splits |
| 4 | `./scripts/4_verify.sh` | about 1 min | A pass/fail report |

---

## Step 1: check your system

```bash
./scripts/1_check_system.sh
```

### What it does
Looks at your operating system, Python, free disk space, internet connection and graphics hardware, and prints a ✓ or ✗ for each. **It installs and changes nothing.**

### Why
The most common reasons setup fails are boring ones: Python is too new for PyTorch, the disk is nearly full, the Wi-Fi is down. Finding that out in two seconds beats finding out ten minutes into an install. It also tells you what hardware your training will run on, which decides how fast it is:

| You have | PyTorch uses | Speed |
|---|---|---|
| NVIDIA GPU (Linux / WSL2) | CUDA | fastest |
| Apple Silicon Mac (M1 and later) | MPS ("Metal Performance Shaders") | fast |
| anything else | the CPU | slower, still fine for the baseline |

<details>
<summary>🔎 Go deeper: how it checks each thing</summary>

- **Operating system.** `uname -s` prints the kernel name: `Darwin` (macOS), `Linux`, or something starting with `MINGW` (Git Bash). WSL reports `Linux`, so the script also looks inside `/proc/version` for the word "microsoft" to tell them apart. We care because Windows puts a virtual environment's programs in `Scripts/` while everyone else uses `bin/`.
- **Python.** The script looks for `python3.12`, `python3.11`, and so on, and asks each one for its version. PyTorch publishes ready-made builds for specific Python versions (3.9 to 3.13 at the time of writing). A brand-new Python can be *too new* and have no PyTorch build at all, which is why "latest" is not always best. If you have [`uv`](https://docs.astral.sh/uv/) the check is skipped, because `uv` can download the right Python itself.
- **Disk space.** `df -Pk .` reports free kilobytes on the drive you are standing on. PyTorch alone is several hundred MB to a few GB, and the dataset adds more.
- **Internet.** `curl` requests PyPI (the Python Package Index, where `pip` and `uv` download from). If this fails, step 2 would fail too.
- **GPU.** `nvidia-smi` is NVIDIA's own tool and only exists when their driver is installed, so its presence is a reliable test. On a Mac, `uname -m` printing `arm64` means Apple Silicon.

</details>

<details>
<summary>📄 Show <code>scripts/1_check_system.sh</code></summary>

```bash
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
```

</details>

It uses a few helpers from `scripts/_common.sh`, shown [below](#the-glue-setupsh-and-_commonsh).

### 🧪 Try it yourself
```bash
uname -s          # which operating system are you on?
uname -m          # CPU architecture: arm64 (Apple Silicon) or x86_64
df -h .           # free space where you are standing
./scripts/1_check_system.sh; echo "exit code: $?"    # 0 means success, anything else means failure
```
That last line matters: **every program reports success or failure through its exit code**, and that is how `setup.sh` knows to stop when a step fails.

### If it fails
Each ✗ says what to fix. For Python, install 3.11 from [python.org](https://www.python.org/downloads/) or with `brew install python@3.11`.

---

## Step 2: install the environment

```bash
./scripts/2_install.sh
```

### What it does
1. Creates a **virtual environment** in `.venv/`: a private copy of Python just for this project.
2. Installs the packages listed in `requirements.txt` into it.
3. Checks that PyTorch imports.

The packages are:

| Package | What it is | Where you will meet it |
|---|---|---|
| **PyTorch** (`torch`) | The deep learning library: multi-dimensional arrays called *tensors*, automatic differentiation, neural network building blocks | Everything you train |
| **torchvision** | Image tools and pretrained models built on PyTorch | Loading ResNet18, image transforms |
| **Pillow** | Reads and writes image files | `dataset.py` opening each `frame_*.jpg` |
| **gdown** | Downloads files from Google Drive | Step 3 only |

### Why a virtual environment?
Python libraries are installed *somewhere*. If everything goes into one global place, projects fight over versions ("this one needs NumPy 1, that one needs NumPy 2") and nobody can reproduce anyone else's results. A virtual environment gives each project its own `site-packages` folder, so this project's packages never touch anything else on your laptop. Deleting `.venv/` undoes all of it. It is the standard first move in every Python project, and a habit worth keeping.

### Why `uv` (when you have it)?
`uv` is a very fast installer written in Rust. It downloads packages in parallel and reuses a shared cache, so installing PyTorch takes seconds when it is already cached, and it can fetch a specific Python version for you. Without it the script uses Python's built-in `venv` and `pip`, which do the same job, just slower. Same result either way.

### Why a CPU-only PyTorch on some Linux machines?
The default PyTorch download for Linux bundles roughly 2.5 GB of NVIDIA CUDA libraries. On a machine with no NVIDIA GPU those can never run, so the script detects that and installs PyTorch's CPU-only build instead, saving a huge download on conference Wi-Fi. Override with `TORCH_FLAVOR=cpu` or `TORCH_FLAVOR=default`.

<details>
<summary>🔎 Go deeper: what a virtual environment really is</summary>

Look inside it after step 2:

```
.venv/
├── pyvenv.cfg          points at the Python this environment was made from
├── bin/                (Scripts/ on Windows)
│   ├── python          a link to that Python interpreter
│   └── pip, gdown ...  command-line tools installed by packages
└── lib/python3.11/site-packages/
    ├── torch/          <- PyTorch lives here, only for this project
    ├── torchvision/
    └── PIL/
```

When you run `.venv/bin/python`, the interpreter sees `pyvenv.cfg` next to it and puts *that* `site-packages` folder on its search path for `import`. That is the entire trick; nothing is installed system-wide.

"Activating" an environment (`source .venv/bin/activate`) just puts `.venv/bin` at the front of your `PATH`, so typing `python` finds the environment's copy first. It is a convenience, not magic, and it is why our scripts skip it and call `.venv/bin/python` directly: that works in any shell, every time.

**Wheels.** Most packages arrive as *wheels* (`.whl` files): pre-compiled bundles built for a specific operating system, CPU type and Python version. The installer picks the one matching your machine. That is why PyTorch needs a supported Python: no wheel exists for an unsupported one, and compiling PyTorch from source would take hours.

**Why pin Python 3.11?** Reproducibility. If everyone runs the same Python and the same library versions, a bug one person hits is a bug everyone can reproduce and fix.

**The CPU build.** `--index-url https://download.pytorch.org/whl/cpu` points the installer at PyTorch's own package server, which hosts the CPU-only wheels. `--extra-index-url https://pypi.org/simple` keeps the normal index available for everything else (Pillow and so on).

</details>

<details>
<summary>📄 Show <code>scripts/2_install.sh</code></summary>

```bash
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
```

</details>

### 🧪 Try it yourself
```bash
.venv/bin/python --version                       # the environment's own Python
.venv/bin/python -c "import torch; print(torch.__version__, torch.rand(2, 3))"   # your first tensor
.venv/bin/python -m pip list 2>/dev/null || uv pip list --python .venv/bin/python   # every installed package
cat .venv/pyvenv.cfg                             # where the environment came from
which python                                     # compare: this is your *system* python
```

### If it fails
Re-run it. Downloads are the usual culprit and it picks up where it left off. To start from scratch, delete the `.venv/` folder and run it again.

---

## Step 3: get the data

```bash
./scripts/3_get_data.sh
```

### What it does
Downloads `train.zip` and `test.zip` from the organizers' **Google Drive** folder, unpacks them into `data/train/` and `data/test/`, and checks the layout: both split folders exist, each contains sequences, each sequence has its frames, and `train/labels.csv` is present. A split that is already in `data/` is skipped, so it only downloads what is missing.

**If Google Drive refuses the download** (it limits how often one file can be fetched, which is likely on event day), use the manual route: open the Drive link in your browser, download `train.zip` and `test.zip` one at a time, save both in the starter kit folder (next to `setup.sh`), and run the script again. It detects the zips and unpacks them for you.

### Why the splits look the way they do

| Folder | Has labels? | Purpose |
|---|---|---|
| `train/` | Yes (`labels.csv`) | Your textbook. Learn patterns from these. |
| `test/` | No | The exam. You predict every sequence; the server secretly scores part of it publicly (the live leaderboard) and the rest privately (hidden until the end; it decides the ranking). |

Splitting data this way is the core idea of machine learning evaluation: **a model must be judged on data it has never seen**, otherwise it can just memorize answers. The hidden private part goes one step further. If you tuned your model against the public leaderboard score until it was perfect, you may have overfit to those particular sequences, and the private score exposes that. A model that does well on both is one that actually learned the task.

### Why check the layout?
A half-downloaded or mis-unpacked dataset produces confusing errors much later, like a crash forty minutes into training. Checking now turns that into a clear message at the start. The script also warns if `test/` contains a `labels.csv`, which would mean answers leaked.

<details>
<summary>🔎 Go deeper: how the download and unpack work</summary>

- **`gdown`** is a small Python tool that understands Google Drive's links. Drive normally wants a browser click-through for large files ("Google can't scan this file for viruses"); `gdown` handles that handshake. The Drive folder holds two zips rather than raw images because Drive only allows tools to fetch 50 files from a *folder*, and this dataset has thousands of images.
- **Why a zip?** Thousands of tiny files transfer slowly one by one. A zip is a single stream, and compression helps a little.
- **`python -m zipfile -e`** unpacks the archive using Python's standard library instead of the `unzip` command, because `unzip` is not installed in Git Bash on Windows. Same result on every platform.
- **The "hoist" step.** Some zips wrap everything in an outer folder (`train.zip` containing `train/...`), others hold the sequence folders directly. The script peels off any single wrapper folder, so either way the sequences end up directly in `data/train/` and `data/test/`.
- **`find ... | wc -l`** counts folders: `find` lists them, `wc -l` counts lines, so the result is "how many sequences". The pipe `|` feeds one command's output into the next, one of the most useful ideas in the shell.

</details>

<details>
<summary>📄 Show <code>scripts/3_get_data.sh</code></summary>

```bash
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
  first="$(find "$DATA_DIR/$split" -mindepth 1 -maxdepth 1 -type d | sort | head -n1)"
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
```

</details>

### 🧪 Try it yourself
```bash
ls data/train | head -5                      # the first few sequence folders
ls data/train/$(ls data/train | head -1)     # the frames inside one of them
head -5 data/train/labels.csv                # what a label looks like
```
Open a few `frame_*.jpg` files from an anomalous and a normal sequence. Can *you* spot the difference? Your model has to learn what your eyes just did.

---

## Step 4: verify everything

```bash
./scripts/4_verify.sh
```

### What it does
Runs `verify_setup.py`, which exercises each piece the competition relies on, in the order a real run would:

1. **Imports** PyTorch and reports its version.
2. **Finds your training device** (CUDA, MPS or CPU).
3. **Loads both splits** (train and test) through `dataset.py` and pulls one batch through a `DataLoader`.
4. **Downloads the pretrained ResNet18 weights** (about 45 MB) and runs one forward pass.
5. **Writes a dummy submission** ("no anomalies anywhere") and validates it with `check_submission.py`.

### Why
Steps 1-3 each succeed on their own, but that does not prove they work *together*. This is a smoke test: a quick end-to-end run whose job is to fail loudly now rather than quietly later. Two checks pay off especially:

- Downloading the pretrained weights now means your first training run starts immediately instead of stalling on a download.
- Running the submission path now means you won't discover a formatting problem when you have one upload attempt left. A rejected file **still uses up a submission**.

<details>
<summary>🔎 Go deeper: the ideas behind each check</summary>

**Devices.** A tensor lives on one device, and the model and its inputs must be on the *same* one. The code picks the best available with `torch.cuda.is_available()`, then `torch.backends.mps.is_available()`, else the CPU, and moves things there with `.to(device)`. GPUs are fast because they do thousands of small multiplications at once, and neural networks are mostly multiplications.

**Dataset and DataLoader.** PyTorch separates two jobs. A `Dataset` (see `dataset.py`) knows how to produce *one* item: `__len__` says how many there are and `__getitem__(i)` returns item `i`. A `DataLoader` wraps it and handles the rest: grouping items into batches, shuffling, and (optionally) loading in parallel. Networks train on batches because it is far more efficient than one example at a time.

**Reading the shapes.** A batch printed as `clip (2, 10, 3, 256, 256)` means: 2 sequences, 10 frames each, 3 color channels (red, green, blue), 256 x 256 pixels. Keeping the frames of a sequence together is deliberate. A frame is judged against its neighbors: an animatronic that moved is obvious next to the frames where it hadn't.

**Pretrained models and transfer learning.** ResNet18 is a convolutional neural network already trained on ImageNet, a million-plus photos in 1000 categories. Its early layers learned general visual skills (edges, textures, shapes) that transfer to almost any image task. The baseline reuses that knowledge and only retrains the last layer for our yes/no question, which is far faster and more accurate than starting from random numbers. Weights are cached in `~/.cache/torch`, so the download happens once.

**The forward pass.** Feeding data through a network to get an output. Training repeats "forward pass, measure the error (loss), adjust the weights" thousands of times. Verifying a forward pass works proves the model, device and data all fit together.

**The submission round trip.** `write_submission` produces the upload format and `check_submission` applies the server's rules. Running them back to back confirms both agree before you rely on them.

</details>

### 🧪 Try it yourself
```bash
VERBOSE=1 ./scripts/4_verify.sh      # same checks, with full error details if anything fails
```
Then open `verify_setup.py` and read it top to bottom. It is short, and it is a good template for the code you'll write yourself.

### If it fails
The first ✗ is the real problem; later ones often follow from it. Re-run with `VERBOSE=1` for the full error.

---

## The glue: `setup.sh` and `_common.sh`

`setup.sh` just runs the four steps in order.

`_common.sh` is sourced by every script. It moves into the starter kit folder no matter where you launched from, defines the print helpers, detects your OS, and knows where the environment's Python lives (`bin/` vs `Scripts/`).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Permission denied` running a script | `chmod +x setup.sh scripts/*.sh`, or run `bash scripts/2_install.sh` |
| `No supported Python found` | Install Python 3.11 (python.org, or `brew install python@3.11`) |
| Install is very slow or times out | Re-run step 2; it resumes. Try a different network |
| `torch` fails to import | Delete `.venv/` and re-run step 2 |
| Step 3: "No Google Drive link set" | Ask a mentor for the Drive link, then `GDRIVE_URL="..." ./scripts/3_get_data.sh` |
| Step 3: download failed | Drive quota or sharing. Download `train.zip` and `test.zip` in your browser, save both in the starter kit folder (next to `setup.sh`), re-run step 3 |
| `train/labels.csv is missing` | The data was unpacked into a nested folder; move its contents up so `train/` sits directly in `data/` |
| Training says `device: cpu` on a GPU machine | Linux/WSL2 only: check `nvidia-smi` works, delete `.venv/`, then `TORCH_FLAVOR=default ./scripts/2_install.sh` |
| DataLoader crashes on macOS or Windows | Keep `num_workers=0` (the baseline's default) |
| Out of memory while training | Lower `BATCH_SIZE`; each item is 10 images |
| Still stuck | Run `VERBOSE=1 ./scripts/4_verify.sh` and show a mentor the output |


## After setup: build and submit

1. **Run [`baseline.py`](baseline.py), then read [`resources/train_model.md`](resources/train_model.md).** A complete, working baseline: a pretrained ResNet18 scores each frame, then the maximum over the sequence becomes the prediction, because the label means "any frame is anomalous". Run it, get a number, then improve it. It ends with ideas ranked by expected payoff.
2. **Skim [`dataset.py`](dataset.py).** It is short and it is how data gets into your model. `FazzbearSequenceDataset` returns `(clip, label)` pairs; `write_submission` writes the upload file.
3. **Always validate before uploading:**
   ```bash
   .venv/bin/python check_submission.py submission.csv
   ```
   It applies the server's own rules: header, ids, `TRUE`/`FALSE` values, exactly one row for every test sequence, no duplicates. A rejected file **still uses up a submission**, and attempts are limited.