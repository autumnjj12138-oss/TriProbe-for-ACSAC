"""Convert the raw NSL-KDD files to the header-augmented CSVs the loader expects.

The provider ships KDDTrain+.txt / KDDTest+.txt as comma-separated rows with no
header: 41 features + label + difficulty = 43 columns. flow_defense/data.py
reads columns by name (label_col="label"), so a header is added here once.
Renaming the .txt files to .csv is NOT enough.

Usage (run once, before claim4):
  python artifact/scripts/prepare_nslkdd.py [--dataset-root DIR]

Looks for KDDTrain+.txt / KDDTest+.txt (or header-less .csv copies of them) in
<root>/nsl_kdd/ and then <root>/archive/, and writes
<root>/nsl_kdd/KDDTrain+.csv and <root>/nsl_kdd/KDDTest+.csv.
Expected: 125,973 training rows and 22,544 test rows.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

NSL_KDD_COLUMNS = [
    # Basic features (1-9)
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    # Content features (10-22)
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login",
    # Time-based traffic features (23-31)
    "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    # Host-based traffic features (32-41)
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    # Label + difficulty
    "label", "difficulty",
]
assert len(NSL_KDD_COLUMNS) == 43
EXPECTED_ROWS = {"KDDTrain+": 125973, "KDDTest+": 22544}


def _has_header(path: Path) -> bool:
    with path.open("r", encoding="latin-1") as fh:
        return fh.readline().strip().startswith("duration,")


def convert(src: Path, dst: Path) -> int:
    rows = []
    with src.open("r", encoding="latin-1") as fin:
        for line in fin:
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 43:
                raise ValueError(f"{src}: expected 43 fields, got {len(parts)}: {line[:80]}")
            rows.append(parts)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(NSL_KDD_COLUMNS)
        w.writerows(rows)
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset"))
    root = Path(os.path.abspath(ap.parse_args().dataset_root))
    problems = 0
    for stem, n_expected in EXPECTED_ROWS.items():
        out = root / "nsl_kdd" / f"{stem}.csv"
        if out.exists() and _has_header(out):
            print(f"  OK       {out} already has a header, left unchanged")
            continue
        src = next((p for p in (root / "nsl_kdd" / f"{stem}.txt", root / "archive" / f"{stem}.txt", out)
                    if p.exists()), None)
        if src is None:
            print(f"  MISSING  {stem}.txt in {root / 'nsl_kdd'} or {root / 'archive'}")
            problems += 1
            continue
        n = convert(src, out)
        mark = "OK      " if n == n_expected else "MISMATCH"
        print(f"  {mark} {src.name} -> {out}  ({n} rows, expected {n_expected})")
        problems += n != n_expected
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
