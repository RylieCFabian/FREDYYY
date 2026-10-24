# PRD: Gemini-supervised frame labels for FNAF anomaly detection

## Problem
Each training sequence has 10 frames but only one label: "something in here is anomalous" or not. The baseline scores every frame with ResNet18 and takes the max, so on a positive sequence only the single highest-scoring frame gets a gradient. If that frame is the wrong one, the model learns the wrong thing and never finds out which frame actually held the anomaly.

## Solution
Use Gemini as a weak frame-level labeler, then train the CNN with an extra per-frame loss from those labels.

1. **Negatives are free frame labels.** A normal sequence means all 10 frames are normal: target 0, weight 1.
2. **Gemini localizes positives.** For each positive training sequence, Gemini sees one image grid: the 10 frames on top, and 10 heatmaps underneath, each frame's difference from its camera's normal view. Camera references are per-camera medians built from negative training frames (`--build-refs`, cached in `camera_refs.npz`). It returns structured JSON: `has_anomaly`, `anomalous_frames`, `description`, `confidence`.
3. **Labels become soft targets.** Flagged frames get target 1 with weight = confidence. Unflagged frames in the same positive get target 0 with weight 0.5 x confidence, because Gemini may miss a second bad frame. Positives that Gemini calls normal or gives no frames for are dropped from the aux loss and only get the normal sequence loss.
4. **Training.** `loss = sequence_BCE + LAMBDA * weighted_frame_BCE`, with `LAMBDA = 0.5`. Inference is unchanged: max over frame logits, no Gemini call needed at test time.

## Pipeline
| Step | Command | Output |
|---|---|---|
| Pick model | `python gemini_label.py --list-models` | model list |
| Go/no-go | `python gemini_label.py --validate 10` | recall on positives, false alarms on negatives, avg frames flagged, grids in `gemini_debug/` |
| Label | `python gemini_label.py --label N` | `gemini_labels.jsonl` (training split only, resumable, 4 workers, backoff on 429/5xx) |
| Build targets | `python gemini_label.py --to-targets` | `gemini_frame_targets.json` |
| Train | copy of `baseline.py` with the 4 edits from `gemini_aux.py` | val F1 with and without Gemini |

## Go/no-go criteria for the labeler
- Catches most positives in `--validate`.
- Few false alarms on negatives.
- Average frames flagged per caught positive well under 10. Near 10 means it is not localizing, so fix the prompt before `--label`.
- Spot-check saved grids by eye: frame picks should match what is visible.

## Leakage rules
- Gemini only labels sequences from the training split, using the same seed-0 permutation, `VAL_FRACTION=0.2`, and `TRAIN_LIMIT` as `baseline.py`. Val sequences are never sent to Gemini.
- Test data is never labeled by Gemini or by hand.

## Success metric
Val F1 on the anomaly class with the Gemini loss beats val F1 without it, same seed, split, epochs, `TRAIN_LIMIT`, and threshold sweep. Only `USE_GEMINI` changes. That side-by-side is the Track 3 pitch. If it wins, the Gemini-trained model also produces the Track 1 submission.

## Deliverables
- `gemini_label.py`, `gemini_aux.py`, the modified training script.
- Two val F1 numbers (with / without Gemini) and a few example grids with Gemini's frame picks and descriptions for the slide.
- A valid `submission.csv` that passes `check_submission.py`.

## Known limits
- Frames with no matching camera reference get no heatmap.
- Gemini labels are noisy. Confidence weighting and the lower weight on unflagged frames limit the damage.
- Free-tier rate limits cap how many positives can be labeled before the deadline.

## Out of scope
- Gemini at inference time.
- Changes to the CNN architecture (teammates own it).
- Edits to `dataset.py`, `check_submission.py`, `verify_setup.py`, `scripts/`.
