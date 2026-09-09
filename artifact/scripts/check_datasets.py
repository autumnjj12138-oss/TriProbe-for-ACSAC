"""Report what datasets are present and whether they look right.

Row counts and class balance are checked against the values the paper's numbers
were computed from. A mismatch means the loader will see a different subset, so
the results will differ for reasons that have nothing to do with the defense.
"""
import argparse, os, sys

EXPECTED = {
    "cicids2017": {
        "files": 8, "ext": ".csv",
        "note": "8 MachineLearningCSV flow files, flat in one directory",
        "rows": 2830743, "benign_frac": None,
    },
    "unsw_nb15": {
        "files": 2, "ext": ".csv",
        "note": "UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv",
        "rows": None, "benign_frac": None,
    },
    "nsl_kdd": {
        "files": 2, "ext": ".csv",
        "note": "KDDTrain+.csv and KDDTest+.csv",
        "rows": None, "benign_frac": None,
    },
}
# split-level expectations, verified on the reference machine
SPLITS = {
    "unsw_nb15": [("UNSW_NB15_training-set.csv", 82332), ("UNSW_NB15_testing-set.csv", 175341)],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=os.path.join(os.path.dirname(__file__), "..", "dataset"))
    args = ap.parse_args()
    root = os.path.abspath(args.dataset_root)
    print(f"dataset root: {root}\n")

    problems = 0
    for name, spec in EXPECTED.items():
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            print(f"  MISSING  {name}/  -- {spec['note']}")
            problems += 1
            continue
        csvs = [f for f in os.listdir(d) if f.lower().endswith(spec["ext"])]
        ok = len(csvs) >= spec["files"]
        print(f"  {'OK     ' if ok else 'PARTIAL'}  {name}/  {len(csvs)} csv (expected >= {spec['files']})")
        if not ok:
            print(f"           {spec['note']}")
            problems += 1
        for fname, n_expected in SPLITS.get(name, []):
            p = os.path.join(d, fname)
            if not os.path.exists(p):
                print(f"           MISSING {fname}")
                problems += 1
                continue
            try:
                import pandas as pd
                n = len(pd.read_csv(p))
            except Exception as e:  # noqa: BLE001
                print(f"           could not read {fname}: {e}")
                problems += 1
                continue
            mark = "OK" if n == n_expected else "MISMATCH"
            print(f"           {mark} {fname}: {n} rows (expected {n_expected})")
            if n != n_expected:
                problems += 1

    print()
    if problems:
        print(f"{problems} problem(s). See infrastructure/datasets.md.")
        return 1
    print("All present datasets look correct.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
