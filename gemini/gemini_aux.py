"""Gemini idea 1, training side: the extra per-frame loss.

Reads gemini_frame_targets.json (made by `python gemini_label.py --to-targets`)
and gives each training batch per-frame targets + weights:

  negative sequence           -> all 10 frames target 0, weight 1   (free, we know they're all normal)
  positive Gemini labeled     -> Gemini's frame picks, weighted by its confidence
  positive Gemini didn't label-> weight 0 (no frame info, only the normal sequence loss applies)

HOW TO WIRE IT INTO baseline.py (4 small changes)

  from gemini_aux import WithIndex, GeminiFrameTargets, aux_frame_loss
  USE_GEMINI = True      # False for the comparison run, everything else identical
  LAMBDA = 0.5

  # 1. train loader also returns each item's index (val/test loaders stay as they are)
  train_set = Subset(WithIndex(train_view), order[n_val:][:TRAIN_LIMIT])
  gem = GeminiFrameTargets()

  # 2. model returns per-frame logits too
  def forward(self, clip):
      b, t = clip.shape[:2]
      frame_logits = self.backbone(clip.flatten(0, 1)).view(b, t)
      return frame_logits.amax(dim=1), frame_logits

  # 3. anywhere else that calls model(...) takes [0], e.g. in evaluate() and the test loop:
  logits = model(clip.to(device))[0]

  # 4. training step
  for step, (clip, label, idx) in enumerate(train_loader, start=1):
      clip, label = clip.to(device), label.to(device)
      seq_logit, frame_logits = model(clip)
      loss = criterion(seq_logit, label)
      if USE_GEMINI:
          ids = [train_view.sequence_ids[j] for j in idx.tolist()]
          tgt, w = gem.batch(ids, label, device)
          loss = loss + LAMBDA * aux_frame_loss(frame_logits, tgt, w)
      ...

THE COMPARISON THAT GOES ON THE SLIDE
  Run twice with the same seed, epochs, TRAIN_LIMIT and threshold sweep,
  once USE_GEMINI=False and once True. Report val F1 for both.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class WithIndex(Dataset):
    """Wraps a dataset so items come out as (clip, label, index)."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        clip, label = self.base[i]
        return clip, label, i


class GeminiFrameTargets:
    def __init__(self, path: str | Path = "gemini_frame_targets.json", frames: int = 10):
        path = Path(path)
        self.frames = frames
        self.table = json.loads(path.read_text()) if path.exists() else {}
        print(f"[gemini_aux] {len(self.table)} sequences with Gemini frame labels"
              + ("" if self.table else f" (no {path}, only negatives get frame supervision)"))

    def batch(self, seq_ids, seq_labels, device):
        b = len(seq_ids)
        tgt = torch.zeros(b, self.frames)
        w = torch.zeros(b, self.frames)
        for i, (sid, y) in enumerate(zip(seq_ids, seq_labels.tolist())):
            if y < 0.5:
                w[i] = 1.0
            elif sid in self.table:
                tgt[i] = torch.tensor(self.table[sid]["targets"])
                w[i] = torch.tensor(self.table[sid]["weights"])
        return tgt.to(device), w.to(device)


def aux_frame_loss(frame_logits, targets, weights):
    """Weighted per-frame BCE, averaged over the weight that's actually present."""
    per_frame = F.binary_cross_entropy_with_logits(frame_logits, targets, reduction="none")
    return (per_frame * weights).sum() / weights.sum().clamp_min(1.0)