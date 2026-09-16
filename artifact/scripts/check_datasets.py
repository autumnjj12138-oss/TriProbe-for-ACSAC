"""Report what datasets are present and whether they look right.

Row counts are checked against the values the paper's numbers were computed
from. A mismatch means the loader will see a different subset, so the results
will differ for reasons that have nothing to do with the defense.
"""
import argparse, glob, os, sys

CIC_FILES, CIC_ROWS = 8, 2830743
SPLITS = {
    "unsw_nb15": [("UNSW_NB15_training-set.csv", 82332), ("UNSW_NB15_testing-set.csv", 175341)],
    "nsl_kdd": [("KDDTrain+.csv", 125973), ("KDDTest+.csv", 22544)],
}


def data_rows(path):
    """Non-empty lines minus the header."""
    with open(path, "rb") as fh:
        return sum(1 for line in fh if line.strip()) - 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=os.path.join(os.path.dirname(__file__), "..", "dataset"))
    root = os.path.abspath(ap.parse_args().dataset_root)
    print(f"dataset root: {root}\n")
    problems = 0

    d = os.path.join(root, "cicids2017")
    if not os.path.isdir(d):
        print("  MISSING  cicids2017/  -- 8 MachineLearningCSV flow files, flat in one directory")
        problems += 1
    else:
        csvs = sorted(glob.glob(os.path.join(d, "**", "*.csv"), recursive=True))
        n = sum(data_rows(p) for p in csvs)
        ok = len(csvs) == CIC_FILES and n == CIC_ROWS
        print(f"  {'OK      ' if ok else 'MISMATCH'} cicids2017/  {len(csvs)} csv, {n:,} rows "
              f"(expected {CIC_FILES} csv, {CIC_ROWS:,} rows; the loader merges every csv here)")
        problems += not ok

    for name, files in SPLITS.items():
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            print(f"  MISSING  {name}/")
            problems += 1
            continue
        print(f"  {name}/")
        for fname, n_expected in files:
            p = os.path.join(d, fname)
            if not os.path.exists(p):
                hint = ("  (raw .txt? run artifact/scripts/prepare_nslkdd.py)"
                        if name == "nsl_kdd" else "")
                print(f"           MISSING {fname}{hint}")
                problems += 1
                continue
            if name == "nsl_kdd":
                with open(p, "r", encoding="latin-1") as fh:
                    if not fh.readline().startswith("duration,"):
                        print(f"           NO HEADER {fname} -- run artifact/scripts/prepare_nslkdd.py")
                        problems += 1
                        continue
            n = data_rows(p)
            mark = "OK" if n == n_expected else "MISMATCH"
            print(f"           {mark} {fname}: {n:,} rows (expected {n_expected:,})")
            problems += n != n_expected

    print()
    if problems:
        print(f"{problems} problem(s). See infrastructure/datasets.md.")
        return 1
    print("All datasets look correct.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
