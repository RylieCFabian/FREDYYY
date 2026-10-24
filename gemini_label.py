""" gemini as a weak frame-level labeler """
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

FRAMES = 10
DATA_DIR = Path(__file__).resolve().parent / "gemini_data"
LABELS_OUT = DATA_DIR / "gemini_labels.jsonl"
TARGETS_OUT = DATA_DIR / "gemini_frame_targets.json"
DEBUG_DIR = DATA_DIR / "gemini_debug"
REFS_OUT = DATA_DIR / "camera_refs.npz"
THUMB = (64, 36)
MIN_CLUSTER = 20  # smaller clusters are probably odd states, not cameras
CAM_THRESH = 12.0  # mean abs gray diff (0-255) to call two thumbnails the same camera, set by --cam-thresh

class SequenceVerdict(BaseModel):
    has_anomaly: bool = Field(description="True if any frame shows an animatronic doing something it shouldn't")
    anomalous_frames: list[int] = Field(description="Indices 0-9 of frames where the anomaly is visible. Empty if none.")
    description: str = Field(description="One sentence: what is wrong and where in the frame")
    confidence: float = Field(description="0.0 to 1.0, how sure you are")


PROMPT = """You are reviewing security camera footage from Five Nights at Freddy's.

The image shows 10 frames, labeled #0 to #9 (top two rows). The feed switches
between cameras, so consecutive frames can show different rooms. Each frame is
tagged with its camera number ("cam K"). Camera switches are normal and are NOT anomalies.
The bottom two rows show each frame as a difference heatmap against that camera's
normal view: bright red means that pixel differs from normal. A gray "no ref" tile
means there was no normal view to compare against, so judge that frame by eye.

Animatronics roaming the building is NORMAL. In normal footage Bonnie, Chica, Freddy and Foxy
show up in hallways, the dining area, close to the camera, as dark silhouettes, and Foxy peeks
out of or disappears from Pirate Cove. Do not flag a frame just because an animatronic is in it.

A sequence is anomalous if some frame shows a state that never appears in normal footage of
that camera: an animatronic in an unusual pose or place for that camera, Foxy fully out of the
curtain or running, glowing eyes or a face in the dark, a strange shape or object, or a scene
that looks distorted. A gray "no ref" tile means that frame matched no normal view of any
camera, which is a strong hint. Camera cuts, noise, static, flicker, compression artifacts,
slight camera shifts and uniform lighting changes are NOT anomalies.
Compare each frame with the other frames of the same camera and with its heatmap. Return only
the frame indices where the abnormal state is visible, a one-sentence description naming the
frame, the camera and what is wrong, and your confidence."""

_client = None
_model = None


def client():
    global _client
    if _client is None:
        from google import genai
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            sys.exit("Set GEMINI_API_KEY first.")
        _client = genai.Client()
    return _client


def list_models() -> list[str]:
    out = []
    for m in client().models.list():
        actions = getattr(m, "supported_actions", None) or []
        if actions and "generateContent" not in actions:
            continue
        out.append(m.name.removeprefix("models/"))
    return sorted(out)


def model_name() -> str:
    """GEMINI_MODEL if set, else the newest plain gemini-X.Y-flash the key can see."""
    global _model
    if _model:
        return _model
    _model = os.environ.get("GEMINI_MODEL")
    if not _model:
        best = None
        for name in list_models():
            m = re.fullmatch(r"gemini-(\d+)(?:\.(\d+))?-flash", name)
            if m:
                ver = (int(m.group(1)), int(m.group(2) or 0))
                if best is None or ver > best[0]:
                    best = (ver, name)
        if best is None:
            sys.exit("Couldn't auto-pick a Flash model. Run --list-models and set GEMINI_MODEL.")
        _model = best[1]
    print(f"[gemini] model: {_model}", file=sys.stderr)
    return _model


def ask_gemini(grid: Image.Image, max_retries: int = 6) -> SequenceVerdict:
    from google.genai import types
    cfg = types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json",
                                      response_schema=SequenceVerdict)
    delay = 4.0
    for attempt in range(max_retries):
        try:
            resp = client().models.generate_content(model=model_name(), contents=[PROMPT, grid], config=cfg)
            return resp.parsed if resp.parsed is not None else SequenceVerdict.model_validate_json(resp.text)
        except Exception as e:  # noqa: BLE001
            code = getattr(e, "code", None) or getattr(e, "status_code", None)
            retryable = code in (429, 500, 503, 504) or "RESOURCE_EXHAUSTED" in str(e)
            if not retryable or attempt == max_retries - 1:
                raise
            print(f"[gemini] {code or type(e).__name__}, retry in {delay:.0f}s", file=sys.stderr)
            time.sleep(delay + random.random())
            delay = min(delay * 2, 60)

def frame_paths(seq_dir: Path) -> list[Path]:
    """Same ordering as dataset.py, so frame index i here == frame i in training."""
    fs = [p for p in seq_dir.iterdir()
          if p.stem.startswith("frame_") and p.stem[6:].isdigit() and p.suffix.lower() in (".jpg", ".png")]
    return sorted(fs, key=lambda p: int(p.stem[6:]))


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _tag(tile: Image.Image, text: str) -> Image.Image:
    d = ImageDraw.Draw(tile)
    f = _font(max(14, tile.width // 14))
    box = d.textbbox((0, 0), text, font=f)
    d.rectangle((0, 0, box[2] + 10, box[3] + 8), fill=(0, 0, 0))
    d.text((5, 3), text, fill=(255, 220, 0), font=f)
    return tile


def _thumb(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L").resize(THUMB), dtype=np.float32)


def _tile_size(img: Image.Image, tile_w: int) -> tuple[int, int]:
    return tile_w, round(tile_w * img.height / img.width)


def _paste_grid(tiles: list[Image.Image], cols: int) -> Image.Image:
    tw, th = tiles[0].size
    gap = 4
    rows = (len(tiles) + cols - 1) // cols
    grid = Image.new("RGB", (cols * tw + (cols - 1) * gap, rows * th + (rows - 1) * gap), (40, 40, 40))
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        grid.paste(t, (c * (tw + gap), r * (th + gap)))
    return grid


def build_refs(train_dir: Path, labels: dict[str, bool], n_seqs: int, val_fraction: float, tile_w: int = 384):
    """Cluster frames of negative training sequences into cameras, store a per-camera median view."""
    neg = [s for s in training_ids(train_dir, val_fraction, None) if labels.get(s) is False]
    seqs = random.Random(0).sample(neg, min(n_seqs, len(neg)))
    paths = [p for s in seqs for p in frame_paths(train_dir / s)]
    thumbs = np.stack([_thumb(Image.open(p)) for p in paths])

    # Greedy clustering against running means; O(frames x clusters), fine for a few thousand frames.
    means, members = [], []
    for i, t in enumerate(thumbs):
        d = [np.abs(t - m / len(mem)).mean() for m, mem in zip(means, members)]
        k = int(np.argmin(d)) if d else -1
        if k >= 0 and d[k] < CAM_THRESH:
            means[k] += t
            members[k].append(i)
        else:
            means.append(t.copy())
            members.append([i])

    kept = sorted((mem for mem in members if len(mem) >= MIN_CLUSTER), key=len, reverse=True)
    dropped = [len(mem) for mem in members if len(mem) < MIN_CLUSTER]
    if not kept:
        sys.exit(f"No cluster has {MIN_CLUSTER}+ frames, try a larger --cam-thresh.")
    size = _tile_size(Image.open(paths[0]), tile_w)
    refs = []
    for mem in kept:
        stack = np.stack([np.asarray(Image.open(paths[i]).convert("RGB").resize(size)) for i in mem])
        refs.append(np.median(stack, axis=0).astype(np.uint8))
    np.savez_compressed(REFS_OUT, thumbs=np.stack([thumbs[mem].mean(axis=0) for mem in kept]),
                        refs=np.stack(refs), sizes=np.array([len(m) for m in kept]), cam_thresh=CAM_THRESH)

    sheet = [_tag(Image.fromarray(r).resize((tile_w // 2, size[1] // 2)), f"cam {k} n={len(m)}")
             for k, (r, m) in enumerate(zip(refs, kept))]
    DEBUG_DIR.mkdir(exist_ok=True)
    _paste_grid(sheet, cols=5).save(DEBUG_DIR / "camera_refs.jpg", quality=90)
    print(f"{len(seqs)} negative sequences, {len(paths)} frames, cam-thresh {CAM_THRESH}")
    print(f"{len(kept)} cameras kept, sizes: {[len(m) for m in kept]}")
    print(f"{len(dropped)} small clusters dropped ({sum(dropped)} frames): {sorted(dropped, reverse=True)[:20]}")
    print(f"wrote {REFS_OUT} and {DEBUG_DIR / 'camera_refs.jpg'}")


_refs = None


def camera_refs():
    """Loads refs and switches CAM_THRESH to the one they were built with, so matching fits the clusters."""
    global _refs, CAM_THRESH
    if _refs is None:
        if not REFS_OUT.exists():
            sys.exit("No camera_refs.npz, run --build-refs first.")
        z = np.load(REFS_OUT)
        _refs = (z["thumbs"], z["refs"])
        CAM_THRESH = float(z["cam_thresh"])
    return _refs


def make_grid(seq_dir: Path, tile_w: int = 384, cols: int = 5) -> Image.Image:
    ref_thumbs, refs = camera_refs()
    raw = [Image.open(p).convert("RGB") for p in frame_paths(seq_dir)]
    size = _tile_size(raw[0], tile_w)
    if size != refs.shape[2:0:-1]:
        sys.exit(f"camera_refs.npz tile size {refs.shape[2:0:-1]} != {size}, rerun --build-refs.")

    cams = []
    for f in raw:
        d = np.abs(ref_thumbs - _thumb(f)).mean(axis=(1, 2))
        k = int(np.argmin(d))
        cams.append(k if d[k] < CAM_THRESH else None)
    frames = [f.resize(size) for f in raw]
    tiles = [_tag(f.copy(), f"#{i} cam {'?' if k is None else k}") for i, (f, k) in enumerate(zip(frames, cams))]

    diffs = {i: np.abs(np.asarray(f, dtype=np.float32) - refs[k]).mean(axis=-1)
             for i, (f, k) in enumerate(zip(frames, cams)) if k is not None}
    scale = max(np.percentile(np.stack(list(diffs.values())), 99.5), 1.0) if diffs else 1.0
    for i in range(len(frames)):
        if i in diffs:
            dm = np.clip(diffs[i] / scale * 255, 0, 255).astype(np.uint8)
            tiles.append(_tag(Image.fromarray(np.stack([dm, dm // 3, dm // 3], axis=-1)), f"#{i} diff"))
        else:
            tiles.append(_tag(Image.new("RGB", size, (110, 110, 110)), f"#{i} no ref"))
    return _paste_grid(tiles, cols)


def read_labels(train_dir: Path) -> dict[str, bool]:
    with open(train_dir / "labels.csv", newline="") as fh:
        return {r["sequence_id"].strip(): r["has_anomaly"].strip().lower() in ("true", "t", "1", "yes", "y")
                for r in csv.DictReader(fh)}


def training_ids(train_dir: Path, val_fraction: float, train_limit: int | None) -> list[str]:
    import torch
    ids = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(0)).tolist()
    n_val = int(val_fraction * len(order))
    picked = order[n_val:]
    if train_limit:
        picked = picked[:train_limit]
    return [ids[i] for i in picked]


def judge(train_dir: Path, sid: str, save_grid: bool = False) -> dict:
    grid = make_grid(train_dir / sid)
    if save_grid:
        DEBUG_DIR.mkdir(exist_ok=True)
        grid.save(DEBUG_DIR / f"{sid}.jpg", quality=85)
    v = ask_gemini(grid)
    frames = sorted({i for i in v.anomalous_frames if 0 <= i < FRAMES})
    return {"sequence_id": sid, "has_anomaly": v.has_anomaly, "anomalous_frames": frames,
            "confidence": max(0.0, min(1.0, float(v.confidence))), "description": v.description,
            "model": model_name()}


def run_validate(train_dir: Path, labels: dict[str, bool], n: int):
    rng = random.Random(0)
    pos = [k for k, v in labels.items() if v]
    neg = [k for k, v in labels.items() if not v]
    picks = [(s, True) for s in rng.sample(pos, n // 2)] + [(s, False) for s in rng.sample(neg, n - n // 2)]
    tp = fp = n_frames = 0
    for sid, truth in picks:
        r = judge(train_dir, sid, save_grid=True)
        print(json.dumps({"truth": truth, **r}))
        if truth:
            tp += r["has_anomaly"]
            n_frames += len(r["anomalous_frames"])
        else:
            fp += r["has_anomaly"]
    n_pos, n_neg = n // 2, n - n // 2
    print("\n----- go/no-go -----")
    print(f"caught {tp}/{n_pos} real anomalies (want most of them)")
    print(f"false alarms on {fp}/{n_neg} normal sequences (want few)")
    if tp:
        print(f"avg frames flagged per caught anomaly: {n_frames / tp:.1f} (10 = it's not localizing)")
    print(f"grids saved in {DEBUG_DIR}/, open a few and check its frame picks by eye")


def run_label(train_dir: Path, labels: dict[str, bool], n: int, workers: int, val_fraction: float,
              train_limit: int | None):
    pool = [s for s in training_ids(train_dir, val_fraction, train_limit) if labels.get(s)]
    done = set()
    if LABELS_OUT.exists():
        done = {json.loads(l)["sequence_id"] for l in LABELS_OUT.read_text().splitlines() if l.strip()}
    todo = [s for s in pool if s not in done][:max(0, n - len(done & set(pool)))]
    print(f"{len(pool)} positive sequences in the training split, {len(done)} already labeled, doing {len(todo)} now")
    if not todo:
        return

    lock = threading.Lock()
    t0, finished, failed = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=workers) as ex, open(LABELS_OUT, "a") as out:
        futs = {ex.submit(judge, train_dir, sid): sid for sid in todo}
        for fut in as_completed(futs):
            sid = futs[fut]
            try:
                r = fut.result()
                r["label"] = True
                with lock:
                    out.write(json.dumps(r) + "\n")
                    out.flush()
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"[skip] {sid}: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
            finished += 1
            if finished % 10 == 0 or finished == len(todo):
                rate = finished / (time.time() - t0)
                print(f"{finished}/{len(todo)} done, {failed} failed, ~{(len(todo) - finished) / rate / 60:.1f} min left")


def run_to_targets(unflagged_weight: float):
    if not LABELS_OUT.exists():
        sys.exit(f"No {LABELS_OUT} yet. Run --label first.")
    rows = [json.loads(l) for l in LABELS_OUT.read_text().splitlines() if l.strip()]
    out, skipped = {}, 0
    for r in rows:
        # Known positive but Gemini saw nothing, or gave no frames: we don't trust it, skip.
        if not r["has_anomaly"] or not r["anomalous_frames"]:
            skipped += 1
            continue
        c = r["confidence"]
        flagged = set(r["anomalous_frames"])
        out[r["sequence_id"]] = {
            "targets": [1.0 if i in flagged else 0.0 for i in range(FRAMES)],
            # flagged frames get full confidence; unflagged ones get less, since Gemini
            # may have missed a second bad frame and we don't want to teach "this is normal" too hard
            "weights": [c if i in flagged else c * unflagged_weight for i in range(FRAMES)],
        }
    TARGETS_OUT.write_text(json.dumps(out))
    print(f"wrote {TARGETS_OUT}: {len(out)} usable sequences, {skipped} skipped (Gemini missed or gave no frames)")


def main():
    global CAM_THRESH
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--build-refs", action="store_true", help="cluster negative training frames into camera_refs.npz")
    ap.add_argument("--ref-seqs", type=int, default=300, help="negative sequences used by --build-refs")
    ap.add_argument("--cam-thresh", type=float, default=CAM_THRESH, help="--build-refs clustering threshold, saved in the npz and reused for matching")
    ap.add_argument("--validate", type=int, metavar="N", help="check Gemini on N labeled sequences")
    ap.add_argument("--label", type=int, metavar="N", help="label up to N positive training sequences")
    ap.add_argument("--to-targets", action="store_true", help="build gemini_frame_targets.json")
    ap.add_argument("--train-dir", default="data/train")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-fraction", type=float, default=0.2, help="match baseline.py VAL_FRACTION")
    ap.add_argument("--train-limit", type=int, default=1000, help="match baseline.py TRAIN_LIMIT, 0 = all")
    ap.add_argument("--unflagged-weight", type=float, default=0.5)
    a = ap.parse_args()

    CAM_THRESH = a.cam_thresh
    train_dir = Path(a.train_dir)
    if a.list_models:
        print("\n".join(list_models()))
    elif a.build_refs:
        build_refs(train_dir, read_labels(train_dir), a.ref_seqs, a.val_fraction)
    elif a.validate:
        run_validate(train_dir, read_labels(train_dir), a.validate)
    elif a.label:
        run_label(train_dir, read_labels(train_dir), a.label, a.workers, a.val_fraction, a.train_limit or None)
    elif a.to_targets:
        run_to_targets(a.unflagged_weight)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()