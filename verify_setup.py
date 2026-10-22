"""End-to-end smoke test, run by ./scripts/4_verify.sh.

Touches every piece the competition depends on, in the order a real run would:
imports -> hardware -> data -> a forward pass of the pretrained model -> writing
and checking a submission file. If this passes, you can spend your time on models.
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", HERE / "data"))

GREEN, RED, RESET = ("\033[32m", "\033[31m", "\033[0m") if sys.stdout.isatty() else ("", "", "")
failures = 0


def check(label: str, fn):
    """Run one check, print a tick or a cross, and carry on so every problem shows at once."""
    global failures
    try:
        detail = fn()
        print(f"  {GREEN}✓{RESET} {label}" + (f" - {detail}" if detail else ""))
        return True
    except Exception as exc:  # noqa: BLE001 - a smoke test should report anything
        failures += 1
        print(f"  {RED}✗{RESET} {label}\n      {type(exc).__name__}: {exc}")
        if os.environ.get("VERBOSE"):
            traceback.print_exc()
        return False


print("\n==> Python packages")
check("python", lambda: sys.version.split()[0])


def packages():
    import PIL, torch, torchvision
    return f"torch {torch.__version__}, torchvision {torchvision.__version__}, Pillow {PIL.__version__}"


if not check("torch + torchvision + Pillow", packages):
    # Everything below needs them, so stop here instead of crashing on the imports.
    print(f"\n{RED}Packages are missing.{RESET} Re-run ./scripts/2_install.sh, or ask a mentor.")
    sys.exit(1)
import torch  # noqa: E402  (after the check, so a missing install gives a clean message)
from torch.utils.data import DataLoader  # noqa: E402
from torchvision import transforms  # noqa: E402

print("\n==> Hardware")
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
check("training device", lambda: {"cuda": "NVIDIA GPU (CUDA)", "mps": "Apple GPU (MPS)",
                                  "cpu": "CPU only - fine, just slower"}[device])

print("\n==> Dataset")
from dataset import FazzbearSequenceDataset, write_submission  # noqa: E402

datasets: dict[str, FazzbearSequenceDataset] = {}


def load(split: str):
    datasets[split] = FazzbearSequenceDataset(DATA_ROOT / split, transform=transforms.ToTensor())
    ds = datasets[split]
    return f"{len(ds)} sequences x {ds.frames_per_sequence} frames, {'labeled' if ds.has_labels else 'unlabeled'}"


for split in ("train", "test"):
    check(split, lambda s=split: load(s))

if "train" in datasets:
    def one_batch():
        clip, label = next(iter(DataLoader(datasets["train"], batch_size=2)))
        return f"clip {tuple(clip.shape)} (batch, frames, channels, height, width), labels {tuple(label.shape)}"
    check("loading a batch", one_batch)

    def balance():
        labels = datasets["train"].sequence_labels
        return f"{labels.mean():.0%} of training sequences contain an anomaly"
    check("label balance", balance)

print("\n==> Pretrained model")


def forward_pass():
    # First call downloads ResNet18 weights (~45 MB) into torch's cache, so your first
    # real training run doesn't stall on it.
    from torchvision.models import ResNet18_Weights, resnet18
    model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 224, 224, device=device))
    return f"ResNet18 weights cached, forward pass OK on {device}, output {tuple(out.shape)}"


check("ResNet18", forward_pass)

print("\n==> Submission pipeline")
if "test" in datasets:
    def submission_roundtrip():
        import check_submission
        ids = datasets["test"].sequence_ids
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "submission.csv"
            write_submission(path, ids, [False] * len(ids))  # the "predict nothing" baseline
            errors, preds = check_submission.check(path.read_bytes(), check_submission.expected_ids(DATA_ROOT))
        if errors:
            raise RuntimeError("; ".join(errors[:3]))
        return f"wrote and validated a {len(preds):,}-row file"
    check("write_submission -> check_submission", submission_roundtrip)
else:
    print("  (skipped - needs the test split)")

print()
if failures:
    print(f"{RED}{failures} check(s) failed.{RESET} Fix the first ✗ above and re-run ./scripts/4_verify.sh "
          f"(add VERBOSE=1 for full tracebacks), or ask a mentor.")
    sys.exit(1)
print(f"{GREEN}Everything works.{RESET} Open resources/train_model.md and start building.")
