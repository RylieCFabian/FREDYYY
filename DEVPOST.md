# FREDYYY: an LLM that tells the CNN *which frame* is haunted

## Inspiration
Every training clip is 10 security-camera frames with one label: "something in here is wrong" or "all normal". The baseline scores each frame with a ResNet18 and takes the max. On a positive clip only the single highest-scoring frame gets a gradient. If that frame is the wrong one, the model learns the wrong thing and never finds out.

A vision LLM can look at the clip and point at the bad frame. We wanted Gemini inside the training loop, not as a side app: a weak frame-level labeler whose answers become a second loss.

## What it does
1. **Builds a "normal" view for every camera.** We cluster frames from negative training clips by camera (24 cameras) and store a per-camera median image.
2. **Renders one grid per clip for the labeler.** The top two rows show the 10 frames, each tagged with its camera. The bottom two rows show each frame's difference heatmap against that camera's normal view. A gray "no ref" tile means the frame matched no known camera, which is a strong hint.
3. **Asks for structured JSON.** The model gets the grid plus a prompt that spells out what is normal (animatronics roaming, Foxy peeking out) and what is not (Foxy fully out, glowing eyes, distorted scenes). It returns a Pydantic schema: `has_anomaly`, `anomalous_frames`, `description`, `confidence`.
4. **Turns answers into soft per-frame targets.** Flagged frames get target 1 with weight = confidence. Unflagged frames in the same clip get target 0 with half that weight, because the labeler may miss a second bad frame. Negative clips give free labels: all 10 frames are target 0 with weight 1.
5. **Trains with an extra loss.** `loss = sequence_BCE + 0.5 * weighted_frame_BCE`. Inference is unchanged: max over frame logits. No LLM call at test time.

## How we built it
- `gemini_label.py`: camera clustering (`--build-refs`), grid rendering, the Gemini call through `google-genai` with `response_schema` and retry/backoff on 429/5xx, a go/no-go check (`--validate`), resumable labeling (`--label`), and target building (`--to-targets`).
- `gemini_aux.py`: a dataset wrapper that returns sample indices, a lookup that builds per-batch frame targets and weights, and the weighted per-frame BCE.
- `baseline.py`: ResNet18 per frame, max over frames, with the aux loss behind `USE_GEMINI`. Everything else is identical between `USE_GEMINI=0` and `1`.
- `compare_gemini.py`: a paired test of the two arms on the same val clips (bootstrap CI of the F1 difference plus an exact McNemar test). Gemini only gets credit if the CI sits above 0.
- `ensemble.py`: averages checkpoints and picks the threshold on val.

**Leakage rules.** The labeler only sees clips from the training split (same seed-0 permutation as training). Val and test clips are never labeled, by a model or by hand.

**Labeling run.** We ran out of Gemini API quota on the day, so the labels in this submission come from a vision LLM answering the exact same prompt on the exact same grids, in the same JSON schema. The pipeline sends every grid to Gemini whenever `GEMINI_API_KEY` has quota.

## Numbers

### Labeler
| | Value |
|---|---|
| Positive train clips labeled | 404 |
| Usable as frame targets | 238 (59%) |
| Skipped (labeler saw nothing or gave no frames) | 166 |
| Frames flagged per usable clip | about 1.5 of 10 |
| Camera references | 24 cameras, median of 300 negative clips |

Flagging about 1.5 of 10 frames means the labeler localizes. It does not just flag the whole clip.

### Training
Same split for every row: seed-0, 5600 train / 1400 val. F1 on the anomaly class, threshold chosen on val.

| Run | Train clips | Epochs | Val F1 |
|---|---|---|---|
| Starter baseline settings | 1000 | 4 | 0.989 |
| Ours, `USE_GEMINI=1`, seed 0 | 5600 | 4 (best: 3) | **0.9972** |
| Ours, `USE_GEMINI=1`, seed 1 | 5600 | 4 (best: 3) | 0.9958 |
| Ensemble of both seeds | 5600 | | 0.9965 |
| Ours, `USE_GEMINI=0`, seed 0 | 5600 | 4 | *(teammate run, fill in)* |

Other changes on top of the baseline: all training data, mixed precision, a OneCycle LR schedule, 6 data-loader workers, a 0.01 to 0.99 threshold sweep, and a best-val checkpoint saved every epoch.

### Did Gemini help?
**Yes, in a matched small-data run.** We trained two models on the same 1000 clips with the same seed, split, epochs and threshold sweep. Only the Gemini frame loss differed (`USE_GEMINI=0` vs `1`, 66 frame-labeled clips at the time). Both were scored on the same 1400 val clips:

| | Val F1 | Threshold |
|---|---|---|
| Without Gemini frame loss | 0.910 | 0.35 |
| With Gemini frame loss | **0.921** | 0.40 |
| Difference | **+1.1 points**, paired bootstrap 95% CI [+0.1, +2.2] | |

On the val clips where the two models disagree, the Gemini model is right on 34 and the baseline on 20 (exact McNemar p = 0.08).

```
python compare_gemini.py --no-gemini checkpoints/baseline_gemini0_1024_124045.pt \
                         --gemini checkpoints/baseline_gemini1_1024_124258.pt
```

Limits: this is one seed, and McNemar alone is just above 0.05. At full data (5600 clips) both arms sit near 0.997, where two seeds of the same setup already differ by about 2 val clips (0.9958 vs 0.9972). So the full-data gain is mostly the 5.6x more data, and the Gemini effect there needs more seeds to measure.

The frame labels help most when data is scarce. With only 1000 clips, the max-pool baseline gets little signal about which frame matters, and the per-frame targets supply it.

## Challenges
- **The labeler has to know what "normal" looks like.** Animatronics roam in normal footage. A naive prompt flagged every Bonnie in a hallway. The per-camera reference views and heatmaps fixed most of that.
- **Grids that never loaded.** In our first parallel labeling run, image requests hit a limit and several workers wrote labels for grids they never saw. We dropped every row that could not be confirmed as seen (131 of 400) and kept only verified labels.
- **Measuring a small effect.** At 0.997 F1 a model gets about 4 of 1400 val clips wrong. That is why we built a paired significance test instead of comparing two numbers.

## What we learned
- An LLM is useful in training as a cheap, noisy, explainable frame labeler. Confidence weighting and a low weight on unflagged frames keep the noise in check.
- At very high accuracy, seed noise is as large as most tricks. Measure it before claiming anything.

## What's next
- Run both arms over 3 seeds and report the paired test.
- Label all 3500 positives once API quota allows.
- Feed the camera reference diff to the CNN as an extra input channel.

## Built with
Python, PyTorch, torchvision (ResNet18), Google Gemini API (`google-genai`, structured output with Pydantic), NumPy, Pillow.
