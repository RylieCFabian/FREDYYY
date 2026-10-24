"""Baseline + Gemini per-frame aux loss. Train, keep the best-val checkpoint, write a submission.

    USE_GEMINI=1 python baseline.py      # with the Gemini frame-label loss
    USE_GEMINI=0 python baseline.py      # identical run without it
    PREDICT_ONLY=1 CKPT=checkpoints/<name>.pt python baseline.py   # rewrite a submission from a checkpoint

Each training run saves checkpoints/baseline_v2_gemini{0,1}_<MMDD_HHMMSS>.pt and the matching
submission_v2_gemini{0,1}_<MMDD_HHMMSS>.csv, so runs never overwrite each other.

Everything except USE_GEMINI is shared between the two runs: seed, split, epochs,
TRAIN_LIMIT, image size and threshold sweep. Override with env vars of the same name.
"""

import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from dataset import FazzbearSequenceDataset, write_submission
from gemini_aux import WithIndex, GeminiFrameTargets, aux_frame_loss

HERE = Path(__file__).resolve().parent  # so it runs from any folder
DATA_ROOT = Path(os.environ.get("DATA_ROOT", HERE / "data"))
TRAIN_ROOT = DATA_ROOT / "train"
TEST_ROOT = DATA_ROOT / "test"
EPOCHS = int(os.environ.get("EPOCHS", "6"))
BATCH_SIZE = 8
LR = 1e-4
VAL_FRACTION = 0.2
IMAGE_SIZE = tuple(int(x) for x in os.environ.get("IMAGE_SIZE", "128,228").split(","))
TRAIN_LIMIT = int(os.environ.get("TRAIN_LIMIT", "0")) or None  # 0 = all training sequences
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "6"))
USE_GEMINI = os.environ.get("USE_GEMINI", "1") == "1"
LAMBDA = 0.5
SEED = int(os.environ.get("SEED", "0"))  # init/shuffle/augmentation only, the val split is always seed 0
MAX_STEPS = int(os.environ.get("MAX_STEPS", "0"))  # >0: stop each epoch early, for timing runs

# Every training run gets its own timestamped checkpoint, so no run overwrites another.
# PREDICT_ONLY needs CKPT=path/to/checkpoint.pt.
PREDICT_ONLY = os.environ.get("PREDICT_ONLY") == "1"
if PREDICT_ONLY:
    CKPT = Path(os.environ["CKPT"])
else:
    CKPT = HERE / "checkpoints" / f"baseline_v2_gemini{int(USE_GEMINI)}_{time.strftime('%m%d_%H%M%S')}.pt"
SUBMISSION = HERE / "submissions" / f"submission_{CKPT.stem.removeprefix('baseline_')}.csv"

device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")
AMP = device == "cuda"

NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    NORM,
])
eval_tf = transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor(), NORM])


class SequenceClassifier(nn.Module):
    """Score each frame with a pretrained CNN, then max-pool over the sequence."""

    def __init__(self):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)

    def forward(self, clip):                      # clip: (B, T, C, H, W)
        b, t = clip.shape[:2]
        frame_logits = self.backbone(clip.flatten(0, 1)).view(b, t).float()
        return frame_logits.amax(dim=1), frame_logits  # (B,) sequence logit, (B, T) frame logits


def binary_metrics(probs, targets, threshold):
    """Accuracy, precision, recall and F1 (the competition metric)."""
    preds = (probs >= threshold).float()
    tp = float(((preds == 1) & (targets == 1)).sum())
    tn = float(((preds == 0) & (targets == 0)).sum())
    fp = float(((preds == 1) & (targets == 0)).sum())
    fn = float(((preds == 0) & (targets == 1)).sum())

    accuracy = (tp + tn) / len(targets) if len(targets) else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return accuracy, precision, recall, f1


def sweep(probs, targets):
    """Best-F1 threshold on val. Ties go to the threshold closest to 0.5."""
    return max(((t / 100, *binary_metrics(probs, targets, t / 100)) for t in range(1, 100)),
               key=lambda r: (r[4], -abs(r[0] - 0.5)))


@torch.no_grad()
def predict(model, loader):
    model.eval()
    probs, targets = [], []
    for clip, label in loader:
        with torch.autocast(device, enabled=AMP):
            logits = model(clip.to(device, non_blocking=True))[0]
        probs.append(torch.sigmoid(logits).cpu())
        targets.append(label)
    return torch.cat(probs), torch.cat(targets)


def loader(ds, shuffle=False):
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=NUM_WORKERS,
                      pin_memory=AMP, persistent_workers=NUM_WORKERS > 0)


def train(model, train_view, eval_view):
    order = torch.randperm(len(train_view), generator=torch.Generator().manual_seed(0)).tolist()
    n_val = int(VAL_FRACTION * len(order))
    val_set = Subset(eval_view, order[:n_val])
    train_set = Subset(WithIndex(train_view), order[n_val:][:TRAIN_LIMIT])
    print(f"{len(train_set)} train / {len(val_set)} val sequences")
    gem = GeminiFrameTargets()
    train_loader, val_loader = loader(train_set, shuffle=True), loader(val_set)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    steps = EPOCHS * (MAX_STEPS or len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=AMP)
    best_f1 = -1.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running, t0 = 0.0, time.time()
        for step, (clip, label, idx) in enumerate(train_loader, start=1):
            clip, label = clip.to(device, non_blocking=True), label.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device, enabled=AMP):
                seq_logit, frame_logits = model(clip)
            loss = criterion(seq_logit, label)
            if USE_GEMINI:
                ids = [train_view.sequence_ids[j] for j in idx.tolist()]
                tgt, w = gem.batch(ids, label, device)
                loss = loss + LAMBDA * aux_frame_loss(frame_logits, tgt, w)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch}  step {step}/{len(train_loader)}  loss {running / step:.4f}  "
                      f"{(time.time() - t0) / step:.2f}s/step", flush=True)
            if step == MAX_STEPS:
                break

        probs, targets = predict(model, val_loader)
        acc, prec, rec, f1 = binary_metrics(probs, targets, 0.5)
        thr, acc_b, prec_b, rec_b, f1_b = sweep(probs, targets)
        print(f"epoch {epoch}  {time.time() - t0:.0f}s  val@0.5 f1 {f1:.4f} (p {prec:.3f} r {rec:.3f})  "
              f"val@{thr:.2f} f1 {f1_b:.4f} (p {prec_b:.3f} r {rec_b:.3f})", flush=True)
        if f1_b > best_f1:
            # Saved every improving epoch, so killing the run keeps the best model so far.
            best_f1 = f1_b
            torch.save({"model": model.state_dict(), "threshold": thr, "epoch": epoch, "val_f1": f1_b},
                       CKPT.with_suffix(".tmp"))
            os.replace(CKPT.with_suffix(".tmp"), CKPT)
            print(f"  saved {CKPT.name} (epoch {epoch}, threshold {thr:.2f}, val f1 {f1_b:.4f})", flush=True)


def main():
    print(f"device: {device}  USE_GEMINI={USE_GEMINI}  EPOCHS={EPOCHS}  TRAIN_LIMIT={TRAIN_LIMIT}  "
          f"IMAGE_SIZE={IMAGE_SIZE}  SEED={SEED}")
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    model = SequenceClassifier().to(device)

    if not PREDICT_ONLY:
        if CKPT.exists():
            raise SystemExit(f"{CKPT} already exists, refusing to overwrite it")
        CKPT.parent.mkdir(exist_ok=True)
        print(f"checkpoint: {CKPT}")
        # Two views of the same directory so train and val get different augmentation.
        train(model, FazzbearSequenceDataset(TRAIN_ROOT, transform=train_tf),
              FazzbearSequenceDataset(TRAIN_ROOT, transform=eval_tf))

    ck = torch.load(CKPT, map_location=device)
    model.load_state_dict(ck["model"])
    print(f"loaded {CKPT.name}: epoch {ck['epoch']}, threshold {ck['threshold']:.2f}, val f1 {ck['val_f1']:.4f}")
    test_set = FazzbearSequenceDataset(TEST_ROOT, transform=eval_tf)
    probs, _ = predict(model, loader(test_set))  # never shuffle
    preds = (probs >= ck["threshold"]).tolist()
    SUBMISSION.parent.mkdir(exist_ok=True)
    write_submission(SUBMISSION, test_set.sequence_ids, preds)
    print(f"wrote {SUBMISSION.name} ({len(preds)} rows, {sum(preds)} positive). "
          f"Next: python check_submission.py submissions/{SUBMISSION.name}")


if __name__ == "__main__":
    main()
