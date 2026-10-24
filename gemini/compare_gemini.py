"""Did the Gemini aux loss help? Paired comparison of the USE_GEMINI=0 and =1 checkpoints on the seed-0 val split.

    python compare_gemini.py --no-gemini checkpoints/<teammate's gemini0>.pt   # vs our best gemini1 run
    python compare_gemini.py --self-check          # test the stats on synthetic data, no checkpoints needed

Both models see the exact same 1400 val sequences, so the test is paired:
  - F1 of each model at its own saved threshold
  - paired bootstrap 95% CI of F1(gemini1) - F1(gemini0)
  - exact McNemar test on the sequences where the two models disagree
Gemini only gets credit if the CI is above 0. One seed per arm, so seed noise is not covered;
rerun both arms with SEED=1,2 for that.
"""

import argparse
import math
import random

import torch
from torch.utils.data import Subset

import baseline as B
from dataset import FazzbearSequenceDataset


def f1(preds, y):
    tp = sum(p and t for p, t in zip(preds, y))
    fp = sum(p and not t for p, t in zip(preds, y))
    fn = sum(t and not p for p, t in zip(preds, y))
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def mcnemar_p(only_a_right, only_b_right):
    """Exact two-sided McNemar p-value from the two disagreement counts."""
    n, k = only_a_right + only_b_right, min(only_a_right, only_b_right)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def compare(p0, p1, y, n_boot=10000, seed=0):
    """p0, p1: 0/1 predictions of the no-Gemini and Gemini models; y: 0/1 truth."""
    rng = random.Random(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(f1([p1[i] for i in idx], [y[i] for i in idx]) - f1([p0[i] for i in idx], [y[i] for i in idx]))
    diffs.sort()
    right0 = [a == t for a, t in zip(p0, y)]
    right1 = [b == t for b, t in zip(p1, y)]
    only1 = sum(r1 and not r0 for r0, r1 in zip(right0, right1))
    only0 = sum(r0 and not r1 for r0, r1 in zip(right0, right1))
    return {"f1_no_gemini": f1(p0, y), "f1_gemini": f1(p1, y), "diff": f1(p1, y) - f1(p0, y),
            "ci95": (diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]),
            "only_gemini_right": only1, "only_no_gemini_right": only0, "mcnemar_p": mcnemar_p(only1, only0)}


def verdict(r):
    lo, hi = r["ci95"]
    if lo > 0:
        return "Gemini model is better on val (CI above 0)."
    if hi < 0:
        return "Gemini model is worse on val (CI below 0)."
    return "Inconclusive: the CI includes 0, val can't tell the two models apart."


def val_predictions(ckpt_path):
    ck = torch.load(ckpt_path, map_location=B.device)
    model = B.SequenceClassifier().to(B.device)
    model.load_state_dict(ck["model"])
    view = FazzbearSequenceDataset(B.TRAIN_ROOT, transform=B.eval_tf)
    order = torch.randperm(len(view), generator=torch.Generator().manual_seed(0)).tolist()
    probs, y = B.predict(model, B.loader(Subset(view, order[:int(B.VAL_FRACTION * len(order))])))
    return (probs >= ck["threshold"]).int().tolist(), y.int().tolist(), ck


def self_check():
    rng = random.Random(1)
    y = [rng.random() < 0.5 for _ in range(1400)]
    same = compare(y, y, y, n_boot=500)
    assert same["diff"] == 0 and same["mcnemar_p"] == 1.0 and verdict(same).startswith("Inconclusive")
    noisy = [t if rng.random() > 0.10 else not t for t in y]  # 10% wrong vs perfect
    better = compare(noisy, y, y, n_boot=500)
    assert better["ci95"][0] > 0 and better["mcnemar_p"] < 1e-6 and verdict(better).startswith("Gemini model is better")
    assert verdict(compare(y, noisy, y, n_boot=500)).startswith("Gemini model is worse")
    assert abs(mcnemar_p(0, 5) - 0.0625) < 1e-12
    print("self-check passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gemini", help="USE_GEMINI=0 checkpoint")
    ap.add_argument("--gemini", default="checkpoints/baseline_v2_gemini1_1024_140938.pt")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        return self_check()
    if not a.no_gemini:
        ap.error("--no-gemini is required")
    p0, y0, ck0 = val_predictions(a.no_gemini)
    p1, y1, ck1 = val_predictions(a.gemini)
    assert y0 == y1
    r = compare(p0, p1, y0)
    print(f"no Gemini: epoch {ck0['epoch']}, threshold {ck0['threshold']:.2f}, val F1 {r['f1_no_gemini']:.4f}")
    print(f"Gemini:    epoch {ck1['epoch']}, threshold {ck1['threshold']:.2f}, val F1 {r['f1_gemini']:.4f}")
    print(f"diff {r['diff']:+.4f}, 95% CI [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]")
    print(f"disagreements: Gemini right {r['only_gemini_right']}, no-Gemini right {r['only_no_gemini_right']}, "
          f"McNemar p {r['mcnemar_p']:.3f}")
    print(verdict(r))


if __name__ == "__main__":
    main()
