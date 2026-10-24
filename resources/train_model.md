# How to train a model

## Setup

Run `./setup.sh` once (see the [setup guide](../README.md#setup-guide) for what it does and why).
The baseline is [`baseline.py`](../baseline.py): it trains, validates, and
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

- Don't train on loose frames. A sequence labeled 1 may contain four perfectly
  normal frames, so a per-frame label would be wrong for most of its frames.
- Instead, score each frame independently, then take the **max** across the frames.
  Max is the differentiable version of "any", which is exactly what the label
  means. This is the one architectural decision in the baseline that matters.

## The script

[`baseline.py`](../baseline.py) is short and commented. It holds out 20% of `train/`
for validation, fine-tunes a pretrained ResNet18 that scores each frame, takes the max
over the sequence, prints validation F1, then predicts every test sequence and writes
`submission.csv`. Its first run is deliberately small: 1 epoch on a slice of the data
at reduced resolution.

## Making a submission

The server scores **one CSV covering every sequence in `data/test`**. The server
secretly splits those rows into a public and a private half, so you predict all of
them the same way. The test split ships without `labels.csv`, so the dataset loads
it unlabeled and hands back `-1` labels, which you ignore. The end of `baseline.py`
does this with `write_submission` from `dataset.py`.

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
2. **Train longer, and augment more.** 1 epoch is a smoke test.
3. **Watch the one-bad-frame sequences.** A single anomalous frame among ten is
   the hardest case, and it is where max-pooling earns its keep. Check your
   errors against it.
4. **Try a stronger backbone** (ResNet34, EfficientNet-B0) once the pipeline works.
5. **Use the fixed cameras.** There are only 10 vantage points and the scene is
   static apart from the animatronics, so "what does this camera look like when
   nothing is wrong" is learnable structure the baseline ignores completely.

## GPU Alternative

[Google Colab](https://colab.research.google.com) offers free GPU notebooks: clone the starter kit, download the dataset into the notebook, and run `baseline.py` there. Free GPUs aren't guaranteed when Colab is busy, and an idle notebook eventually resets, which clears the downloaded data.

If you are using Colab, please copy `train.zip` and `test.zip` into your own Google Drive once and load them from there after a reset. Everyone shares the competition link's download limit, so be careful with multiple downloads.
