"""
Dataset loader for the Five Nights with AI competition.

On-disk layout (what ./scripts/3_get_data.sh puts in ``data/``):

    train/
        labels.csv              sequence_id,has_anomaly
        train_00000/
            frame_0.jpg ... frame_9.jpg
        train_00001/
        ...
    test/                       test_00000/ ...  (no labels.csv)

An item is one sequence of frames with **one label for the whole sequence**:
1 if any frame in it is anomalous, else 0. The frames travel together because a
frame is judged against its neighbors -- an animatronic that moved is obvious
next to the frames where it hadn't.

Items come out as ``(clip, label)``, where ``clip`` is a float tensor of shape
``(T, C, H, W)`` in [0, 1] and ``label`` is a float scalar tensor.

The test split ships without labels.csv. Point the class at it and it
loads unlabeled, yielding ``UNLABELED`` so batches still collate.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Callable, Iterable

import torch
from PIL import Image
from torch.utils.data import Dataset

ID_COLUMN = "sequence_id"
LABEL_COLUMN = "has_anomaly"
FRAME_EXTENSIONS = (".jpg", ".png")

#: Label handed back for sequences with no ground truth (the test split).
UNLABELED = -1.0

_TRUE = {"true", "t", "yes", "y", "1"}
_FALSE = {"false", "f", "no", "n", "0"}


def parse_label(raw: str) -> float:
    """Read one has_anomaly cell.

    The generator writes True/False; accept 1/0 and yes/no too rather than
    betting on which spelling shipped.
    """
    key = raw.strip().lower()
    if key in _TRUE:
        return 1.0
    if key in _FALSE:
        return 0.0
    raise ValueError(f"cannot read {raw!r} as a {LABEL_COLUMN} value")


def read_labels(path: Path) -> dict[str, float]:
    """Parse labels.csv into {sequence_id: label}. One row per sequence."""
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty")
        missing = {ID_COLUMN, LABEL_COLUMN} - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{path} is missing column(s) {sorted(missing)}; "
                f"found {reader.fieldnames}")

        out: dict[str, float] = {}
        for lineno, row in enumerate(reader, start=2):
            seq_id = row[ID_COLUMN].strip()
            if seq_id in out:
                raise ValueError(f"{path}:{lineno}: duplicate row for {seq_id!r}")
            try:
                out[seq_id] = parse_label(row[LABEL_COLUMN])
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from None
    return out


def frame_files(seq_dir: Path) -> list[Path]:
    """The frame_N.jpg / frame_N.png files of one sequence, in frame order."""
    found = [
        p for p in seq_dir.iterdir()
        if p.stem.startswith("frame_") and p.suffix.lower() in FRAME_EXTENSIONS
        and p.stem[len("frame_"):].isdigit()
    ]
    return sorted(found, key=lambda p: int(p.stem[len("frame_"):]))


def build_index(
    root: Path,
    labels_file: str | Path | None = "labels.csv",
    frames_per_sequence: int | None = None,
    require_labels: bool | None = None,
) -> tuple[list[tuple[str, list[Path], float]], bool]:
    """Scan a split directory into a sorted list of (sequence_id, frame paths, label).

    ``require_labels`` of None means "use labels.csv if it's there" -- which is
    what tells a train split apart from a test split without being told.
    ``frames_per_sequence`` of None means "whatever the first sequence has";
    every other sequence must then match it.

    Everything is checked here rather than at access time: a truncated render is
    worth one clear error at startup, not a crash 40 minutes into an epoch.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")

    labels: dict[str, float] | None = None
    if labels_file is not None:
        path = Path(labels_file)
        if not path.is_absolute():
            path = root / path
        if path.is_file():
            labels = read_labels(path)
        elif require_labels:
            raise FileNotFoundError(f"no labels at {path}")
    elif require_labels:
        raise ValueError("require_labels=True but labels_file=None")

    seq_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not seq_dirs:
        raise FileNotFoundError(f"no sequence directories under {root}")

    entries: list[tuple[str, list[Path], float]] = []
    incomplete: list[str] = []
    unlabeled: list[str] = []

    for seq_dir in seq_dirs:
        seq_id = seq_dir.name
        frames = frame_files(seq_dir)
        if frames_per_sequence is None:
            frames_per_sequence = len(frames)
        if len(frames) != frames_per_sequence or not frames:
            incomplete.append(f"{seq_id} (has {len(frames)} frames, expected {frames_per_sequence})")
            continue
        if labels is None:
            entries.append((seq_id, frames, UNLABELED))
        elif seq_id in labels:
            entries.append((seq_id, frames, labels[seq_id]))
        else:
            unlabeled.append(seq_id)

    if incomplete:
        shown = "\n  ".join(incomplete[:10])
        more = f"\n  ... and {len(incomplete) - 10} more" if len(incomplete) > 10 else ""
        raise FileNotFoundError(
            f"{len(incomplete)} sequence(s) under {root} have the wrong number of frames:\n  {shown}{more}")
    if unlabeled:
        raise KeyError(
            f"{len(unlabeled)} sequence director(ies) have no row in labels.csv, "
            f"e.g. {unlabeled[:5]}")

    if labels is not None:
        orphans = set(labels) - {seq_id for seq_id, _, _ in entries}
        if orphans:
            raise FileNotFoundError(
                f"{len(orphans)} labels.csv sequence(s) have no directory, "
                f"e.g. {sorted(orphans)[:5]}")

    return entries, labels is not None


def _to_tensor(image: Image.Image) -> torch.Tensor:
    """PIL RGB -> float tensor (C, H, W) in [0, 1], without pulling in numpy."""
    width, height = image.size
    buf = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    return buf.view(height, width, 3).permute(2, 0, 1).float().div_(255.0)


class FazzbearSequenceDataset(Dataset):
    """A split of the Five Nights with AI dataset, one sequence of frames per item.

    Args:
        root: the split directory (``data/train`` or ``data/test``).
        labels_file: labels.csv path, relative to ``root`` unless absolute.
            None disables label loading entirely.
        require_labels: None (default) loads labels if present; True errors when
            they are missing; False ignores them even if present.
        transform: applied to each frame as a PIL image. If it returns a PIL
            image, it is converted to a tensor afterwards.
        consistent_transform: draw one RNG seed per sequence and reset before
            every frame, so a random crop or jitter lands identically on all
            frames. Leave this on -- independent per-frame geometry fabricates
            apparent motion, and motion between frames is the evidence the
            labels rest on.
        frames_per_sequence: expected frames per sequence. None (default)
            detects it from the data.
    """

    def __init__(
        self,
        root: str | Path,
        labels_file: str | Path | None = "labels.csv",
        require_labels: bool | None = None,
        transform: Callable[[Image.Image], object] | None = None,
        consistent_transform: bool = True,
        frames_per_sequence: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.consistent_transform = consistent_transform
        self._entries, self.has_labels = build_index(
            self.root,
            labels_file=None if require_labels is False else labels_file,
            frames_per_sequence=frames_per_sequence,
            require_labels=require_labels,
        )
        self.frames_per_sequence = len(self._entries[0][1])

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def sequence_ids(self) -> list[str]:
        """Ids in item order -- line these up with predictions to write a submission."""
        return [seq_id for seq_id, _, _ in self._entries]

    @property
    def sequence_labels(self) -> torch.Tensor:
        """(N,) float tensor of labels, for class weights, stratified splits or error analysis."""
        return torch.tensor([label for _, _, label in self._entries], dtype=torch.float32)

    def frame_paths(self, index: int) -> list[Path]:
        return list(self._entries[index][1])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, label = self._entries[index]
        seed = random.randrange(2**31) if (self.transform and self.consistent_transform) else None

        frames = []
        for path in self.frame_paths(index):
            with Image.open(path) as handle:
                frame = handle.convert("RGB")
            if self.transform is not None:
                if seed is not None:
                    # v1 transforms roll on `random`, v2 on torch -- seed both.
                    random.seed(seed)
                    torch.manual_seed(seed)
                frame = self.transform(frame)
            if isinstance(frame, Image.Image):
                frame = _to_tensor(frame)
            elif not isinstance(frame, torch.Tensor):
                raise TypeError(
                    f"transform returned {type(frame).__name__}; expected a PIL image or a tensor")
            frames.append(frame)

        return torch.stack(frames), torch.tensor(label, dtype=torch.float32)

    def __repr__(self) -> str:
        kind = "labeled" if self.has_labels else "unlabeled"
        return (f"{type(self).__name__}({self.root}, {len(self)} sequences, "
                f"{self.frames_per_sequence} frames each, {kind})")


def write_submission(
    path: str | Path,
    sequence_ids: Iterable[str],
    predictions: Iterable,
) -> None:
    """Write sequence_id,has_anomaly -- one row per sequence, TRUE/FALSE.

    ``predictions`` holds one yes/no per sequence, in the same order as
    ``sequence_ids``. The server wants *every* sequence in ``data/test`` in a
    single file:

        write_submission("submission.csv", test.sequence_ids, test_preds)
    """
    sequence_ids = list(sequence_ids)
    predictions = list(predictions)
    if len(sequence_ids) != len(predictions):
        raise ValueError(
            f"{len(sequence_ids)} sequence ids but {len(predictions)} predictions")

    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([ID_COLUMN, LABEL_COLUMN])
        for seq_id, pred in zip(sequence_ids, predictions):
            writer.writerow([seq_id, "TRUE" if pred else "FALSE"])
