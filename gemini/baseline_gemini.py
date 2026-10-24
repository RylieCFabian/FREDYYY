"""baseline.py + the Gemini per-frame auxiliary loss. Only USE_GEMINI differs between runs.

    USE_GEMINI=0 python baseline_gemini.py
    USE_GEMINI=1 python baseline_gemini.py
    SUBMIT=1 ... also writes submission.csv
"""

import os
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
EPOCHS = 4  # val F1 flattens by epoch 4 in the seed-0 runs
BATCH_SIZE = 8
LR = 1e-4
VAL_FRACTION = 0.2
IMAGE_SIZE = (128, 228)
TRAIN_LIMIT = 1000 # training sequences per epoch. None trains on all 5,600
THRESHOLD = 0.5
USE_GEMINI = os.environ.get("USE_GEMINI", "1") == "1"
LAMBDA = 0.5
SEED = int(os.environ.get("SEED", "0"))  # init/shuffle/augmentation only, the val split stays seed 0
NUM_WORKERS = 0  # raising it on macOS/Windows needs an `if __name__ == "__main__":` guard first
SUBMISSION = HERE / f"submission_gemini{int(USE_GEMINI)}.csv"  # per run, so parallel runs don't collide

device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")
print(f"device: {device}  USE_GEMINI={USE_GEMINI}")
torch.manual_seed(SEED)  # same init, shuffle and augmentation in both runs

# ---------------------------------------------------------------- data

NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    NORM,
])
eval_tf = transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor(), NORM])

# Two views of the same directory so train and val get different augmentation.
train_view = FazzbearSequenceDataset(TRAIN_ROOT, transform=train_tf)
eval_view = FazzbearSequenceDataset(TRAIN_ROOT, transform=eval_tf)
print(train_view)

order = torch.randperm(len(train_view), generator=torch.Generator().manual_seed(0)).tolist()
n_val = int(VAL_FRACTION * len(order))
val_set = Subset(eval_view, order[:n_val])
train_set = Subset(WithIndex(train_view), order[n_val:][:TRAIN_LIMIT])
gem = GeminiFrameTargets()
print(f"{len(train_set)} train / {len(val_set)} val sequences")

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

# ---------------------------------------------------------------- model

class SequenceClassifier(nn.Module):
    """Score each frame with a pretrained CNN, then max-pool over the sequence."""

    def __init__(self):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)

    def forward(self, clip):                      # clip: (B, T, C, H, W)
        b, t = clip.shape[:2]
        frame_logits = self.backbone(clip.flatten(0, 1)).view(b, t)
        return frame_logits.amax(dim=1), frame_logits  # (B,) sequence logit, (B, T) frame logits

model = SequenceClassifier().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ---------------------------------------------------------------- metrics

def binary_metrics(probs, targets, threshold=THRESHOLD):
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


@torch.no_grad()
def evaluate(loader):
    model.eval()
    probs, targets = [], []
    for clip, label in loader:
        logits = model(clip.to(device))[0]
        probs.append(torch.sigmoid(logits).cpu())
        targets.append(label)
    return torch.cat(probs), torch.cat(targets)


def sweep(probs, targets):
    """Best-F1 threshold on val, so neither run is judged at an arbitrary 0.5."""
    return max(((t / 100, *binary_metrics(probs, targets, t / 100)) for t in range(5, 96, 5)), key=lambda r: r[4])

# ---------------------------------------------------------------- train

for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    for step, (clip, label, idx) in enumerate(train_loader, start=1):
        clip, label = clip.to(device), label.to(device)

        optimizer.zero_grad()
        seq_logit, frame_logits = model(clip)
        loss = criterion(seq_logit, label)
        if USE_GEMINI:
            ids = [train_view.sequence_ids[j] for j in idx.tolist()]
            tgt, w = gem.batch(ids, label, device)
            loss = loss + LAMBDA * aux_frame_loss(frame_logits, tgt, w)
        loss.backward()
        optimizer.step()

        running += loss.item()
        if step % 20 == 0:
            print(f"epoch {epoch}  step {step}/{len(train_loader)}  loss {running / step:.4f}")

    probs, targets = evaluate(val_loader)
    accuracy, precision, recall, f1 = binary_metrics(probs, targets)
    print(f"epoch {epoch}  val@0.5 accuracy {accuracy:.3f}  precision {precision:.3f}  "
          f"recall {recall:.3f}  f1 {f1:.3f}")
    THRESHOLD, accuracy, precision, recall, f1 = sweep(probs, targets)
    print(f"epoch {epoch}  val@best threshold {THRESHOLD:.2f}  accuracy {accuracy:.3f}  "
          f"precision {precision:.3f}  recall {recall:.3f}  f1 {f1:.3f}")

ckpt = f"baseline_gemini{int(USE_GEMINI)}.pt"
torch.save(model.state_dict(), ckpt)
print(f"saved {ckpt}")
if os.environ.get("SUBMIT") != "1":
    raise SystemExit

# ---------------------------------------------------------------- submit

test_set = FazzbearSequenceDataset(TEST_ROOT, transform=eval_tf)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)  # never shuffle
model.eval()
preds = []
with torch.no_grad():
    for clip, _ in test_loader:
        preds.extend((torch.sigmoid(model(clip.to(device))[0]) >= THRESHOLD).cpu().tolist())

write_submission(SUBMISSION, test_set.sequence_ids, preds)
print(f"wrote {SUBMISSION} ({len(preds)} rows). Next: .venv/bin/python check_submission.py submission.csv")
