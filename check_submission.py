#!/usr/bin/env python3
"""Check a submission CSV against the server's rules *before* you spend a submission on it.

    python check_submission.py submission.csv

Every submission slot is precious, and a malformed file still uses one up. This
applies the same rules the server does (header, ids, TRUE/FALSE values, one row
for every test sequence, no duplicates), using the folders in data/test/ as the
list of sequences that must be present. Standard library only, so it runs anywhere.

Exit code 0 = would be accepted, 1 = would be rejected.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

MAX_BYTES = 1_048_576
MAX_ERRORS_SHOWN = 20
ID_PATTERN = re.compile(r"^test_[0-9]{5}$")
ACCEPTED = {"true": True, "1": True, "false": False, "0": False}


def expected_ids(data_root: Path) -> set[str]:
    folder = data_root / "test"
    if not folder.is_dir():
        raise SystemExit(f"error: {folder} not found - run ./scripts/3_get_data.sh first")
    return {p.name for p in folder.iterdir() if p.is_dir()}


def check(raw: bytes, expected: set[str]) -> tuple[list[str], dict[str, bool]]:
    errors: list[str] = []
    if len(raw) > MAX_BYTES:
        return [f"file is {len(raw):,} bytes; the limit is {MAX_BYTES:,}"], {}
    try:
        text = raw.decode("utf-8-sig")  # -sig drops the BOM Excel adds
    except UnicodeDecodeError:
        return ["file is not UTF-8 text - export it as plain CSV (UTF-8)"], {}

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    rows = list(csv.reader(lines))
    if not rows:
        return ["file is empty"], {}

    header = [c.strip().lower() for c in rows[0]]
    if header != ["sequence_id", "has_anomaly"]:
        errors.append(f"header must be exactly 'sequence_id,has_anomaly', found {','.join(rows[0])!r}")

    seen: dict[str, int] = {}
    predictions: dict[str, bool] = {}
    for line, row in enumerate(rows[1:], start=2):
        if not any(cell.strip() for cell in row):
            continue
        if len(row) != 2:
            errors.append(f"row {line}: expected 2 columns, found {len(row)}")
            continue
        seq_id, value = row[0].strip().lower(), row[1].strip().lower()
        if not ID_PATTERN.match(seq_id):
            errors.append(f"row {line}: {row[0]!r} is not a valid id (expected test_00000 style)")
            continue
        if value not in ACCEPTED:
            errors.append(f"row {line}: {row[1]!r} is not TRUE/FALSE (or 1/0) - 'yes', 'T' and blanks are rejected")
            continue
        if seq_id not in expected:
            errors.append(f"row {line}: {seq_id} is not a sequence in the test set")
        if seq_id in seen:
            errors.append(f"row {line}: {seq_id} appears twice (first on row {seen[seq_id]})")
        seen.setdefault(seq_id, line)
        predictions[seq_id] = ACCEPTED[value]

    missing = sorted(expected - set(predictions))
    if missing:
        errors.append(
            f"expected {len(expected):,} sequences, found {len(expected) - len(missing):,}; "
            f"missing: {', '.join(missing[:10])}{', ...' if len(missing) > 10 else ''}")
    return errors, predictions


def main() -> int:
    parser = argparse.ArgumentParser(description="Check a submission CSV against the server's rules.")
    parser.add_argument("csv_path", help="the predictions file to check")
    parser.add_argument("--data", default=os.environ.get("DATA_ROOT", str(Path(__file__).parent / "data")),
                        help="dataset folder (default: ./data, or $DATA_ROOT)")
    args = parser.parse_args()

    try:
        raw = Path(args.csv_path).read_bytes()
    except OSError as exc:
        print(f"error: cannot read {args.csv_path}: {exc}", file=sys.stderr)
        return 2

    errors, predictions = check(raw, expected_ids(Path(args.data)))
    if errors:
        print(f"INVALID - {len(errors)} problem(s); the server would reject this and still count a submission:")
        for message in errors[:MAX_ERRORS_SHOWN]:
            print(f"  - {message}")
        if len(errors) > MAX_ERRORS_SHOWN:
            print(f"  ...and {len(errors) - MAX_ERRORS_SHOWN} more")
        return 1

    flagged = sum(predictions.values())
    print(f"VALID - {len(predictions):,} sequences, {flagged:,} flagged as anomalous "
          f"({flagged / len(predictions):.0%}). Safe to upload.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
