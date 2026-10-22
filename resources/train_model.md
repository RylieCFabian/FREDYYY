# How to train a model

## Setup

Run `./setup.sh` once (see the [setup guide](../README.md#setup-guide) for what it does and why).
The baseline below is [`baseline.py`](../baseline.py): it trains, validates, and
writes `submission.csv` in one go. Run it from the starter kit's top folder with the
environment setup created, since it imports `dataset.py` from there:

```
.venv/bin/python baseline.py        # Windows Git Bash: .venv/Scripts/python.exe baseline.py
```

Edit `baseline.py` (or a copy of it) to build your own model. Leave `dataset.py`,
`check_submission.py`, `verify_setup.py` and `scripts/` alone so setup's checks keep working.

## The shape of the problem

Each item is a **sequence of 10 frames**, and the label is 1 if *any* one of them
is anomalous. That "any" is the whole design:

- Don't train on loose frames. A sequence labelled 1 may contain four perfectly
  normal frames, so a per-frame label would be wrong for most of its frames.
- Instead, score each frame independently, then take the **max** across the frames.
  Max is the differentiable version of "any", which is exactly what the label
  means. This is the one architectural decision in the baseline that matters.

## The script

The training half of `baseline.py`, with comments explaining each choice:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from dataset import FazzbearSequenceDataset

TRAIN_ROOT = "data/train"   # created by ./scripts/3_get_data.sh
EPOCHS = 3
BATCH_SIZE = 8
LR = 1e-4
VAL_FRACTION = 0.2

device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")
print(f"device: {device}")

# ---------------------------------------------------------------- data

# ImageNet statistics, because the backbone was pretrained on ImageNet.
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# Brightness/contrast jitter mimics the LIGHT_FLICKER the generator already
# applies, so it teaches the model to ignore it. Note there is no horizontal
# flip: the cameras are fixed and a mirrored pizzeria never occurs in the data.
train_tf = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    NORM,
])
eval_tf = transforms.Compose([transforms.ToTensor(), NORM])

# Two views of the same directory so train and val get different augmentation.
train_view = FazzbearSequenceDataset(TRAIN_ROOT, transform=train_tf)
eval_view = FazzbearSequenceDataset(TRAIN_ROOT, transform=eval_tf)
print(train_view)

order = torch.randperm(len(train_view), generator=torch.Generator().manual_seed(0)).tolist()
n_val = int(VAL_FRACTION * len(order))
val_set = Subset(eval_view, order[:n_val])
train_set = Subset(train_view, order[n_val:])
print(f"{len(train_set)} train / {len(val_set)} val sequences")

# num_workers=0 loads images in the main process: slower, but it works on every OS.
# On Linux you can raise it; on macOS/Windows that needs the script body wrapped in
# `if __name__ == "__main__":` first.
NUM_WORKERS = 0
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
        return frame_logits.amax(dim=1)           # -> (B,) one logit per sequence

model = SequenceClassifier().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ---------------------------------------------------------------- metrics

def binary_metrics(probs, targets, threshold=0.5):
    """Accuracy, precision, recall and F1 -- no scikit-learn needed."""
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
        logits = model(clip.to(device))
        probs.append(torch.sigmoid(logits).cpu())
        targets.append(label)
    return binary_metrics(torch.cat(probs), torch.cat(targets))

# ---------------------------------------------------------------- train

for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    for step, (clip, label) in enumerate(train_loader, start=1):
        clip, label = clip.to(device), label.to(device)

        optimizer.zero_grad()
        loss = criterion(model(clip), label)
        loss.backward()
        optimizer.step()

        running += loss.item()
        if step % 20 == 0:
            print(f"epoch {epoch}  step {step}/{len(train_loader)}  loss {running / step:.4f}")

    accuracy, precision, recall, f1 = evaluate(val_loader)
    print(f"epoch {epoch}  val accuracy {accuracy:.3f}  precision {precision:.3f}  "
          f"recall {recall:.3f}  f1 {f1:.3f}")

torch.save(model.state_dict(), "baseline.pt")
print("saved baseline.pt")
```


## Making a submission

The server scores **one CSV covering every sequence in `data/test`**. The server
secretly splits those rows into a public and a private half, so you predict all of
them the same way. The test split ships without `labels.csv`, so the dataset loads
it unlabeled and hands back `-1` labels, which you ignore. The end of `baseline.py`
does this (shown here as a function you can reuse):

```python
from dataset import FazzbearSequenceDataset, write_submission

def predict(split):
    data = FazzbearSequenceDataset(f"data/{split}", transform=eval_tf)
    loader = DataLoader(data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)  # no shuffle!
    model.eval()
    preds = []
    with torch.no_grad():
        for clip, _ in loader:
            preds.extend((torch.sigmoid(model(clip.to(device))) >= 0.5).cpu().tolist())
    return data.sequence_ids, preds

test_ids, test_preds = predict("test")
write_submission("submission.csv", test_ids, test_preds)
```

`sequence_ids` comes back in item order, so it lines up with the predictions as
long as you don't shuffle the loader. Don't shuffle the loader.

**Before you upload, check the file** -- a malformed submission still uses up
one of your limited attempts:

```
.venv/bin/python check_submission.py submission.csv
```

## Where to go next

Roughly in order of expected payoff:

1. **Tune the threshold.** 0.5 is arbitrary. Sweep it on the validation set and
   pick the value that maximises F1 -- often worth several points for free.
2. **Train longer, and augment more.** 3 epochs is a smoke test, not a fit.
3. **Watch the one-bad-frame sequences.** A single anomalous frame among ten is
   the hardest case, and it is where max-pooling earns its keep. Check your
   errors against it.
4. **Try a stronger backbone** (ResNet34, EfficientNet-B0) once the pipeline works.
5. **Use the fixed cameras.** There are only 10 vantage points and the scene is
   static apart from the animatronics, so "what does this camera look like when
   nothing is wrong" is learnable structure the baseline ignores completely.
