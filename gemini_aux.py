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


DEFAULT_TARGETS = Path(__file__).resolve().parent / "gemini_data" / "gemini_frame_targets.json"


class GeminiFrameTargets:
    def __init__(self, path: str | Path = DEFAULT_TARGETS, frames: int = 10):
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