"""Average the probabilities of several checkpoints, pick the threshold on seed-0 val, write a submission.

    python ensemble.py checkpoints/a.pt checkpoints/b.pt checkpoints/c.pt

Writes submission_ensemble_<MMDD_HHMMSS>.csv, never overwriting an earlier one.
"""

import sys
import time

import torch
from torch.utils.data import Subset

import baseline as B
from dataset import FazzbearSequenceDataset


def main(paths):
    view = FazzbearSequenceDataset(B.TRAIN_ROOT, transform=B.eval_tf)
    order = torch.randperm(len(view), generator=torch.Generator().manual_seed(0)).tolist()
    val_loader = B.loader(Subset(view, order[:int(B.VAL_FRACTION * len(order))]))
    test_set = FazzbearSequenceDataset(B.TEST_ROOT, transform=B.eval_tf)
    test_loader = B.loader(test_set)  # never shuffle

    val_probs, test_probs = [], []
    for p in paths:
        ck = torch.load(p, map_location=B.device)
        model = B.SequenceClassifier().to(B.device)
        model.load_state_dict(ck["model"])
        vp, y = B.predict(model, val_loader)
        thr, *_, f1 = B.sweep(vp, y)
        print(f"{p}: val f1 {f1:.4f} @ {thr:.2f}", flush=True)
        val_probs.append(vp)
        test_probs.append(B.predict(model, test_loader)[0])

    vp = torch.stack(val_probs).mean(0)
    thr, acc, prec, rec, f1 = B.sweep(vp, y)
    print(f"ensemble of {len(paths)}: val f1 {f1:.4f} @ {thr:.2f} (p {prec:.3f} r {rec:.3f}, "
          f"{int((~((vp >= thr).float() == y)).sum())} val errors)")
    preds = (torch.stack(test_probs).mean(0) >= thr).tolist()
    out = B.HERE / "submissions" / f"submission_ensemble_{time.strftime('%m%d_%H%M%S')}.csv"
    out.parent.mkdir(exist_ok=True)
    B.write_submission(out, test_set.sequence_ids, preds)
    print(f"wrote {out.name} ({len(preds)} rows, {sum(preds)} positive). Next: python check_submission.py submissions/{out.name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
