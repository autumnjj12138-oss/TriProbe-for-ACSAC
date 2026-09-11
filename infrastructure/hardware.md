# Hardware and platform

## Reference machine

All numbers in `artifact/reference_outputs/` were produced on:

| | |
|---|---|
| GPU | NVIDIA GeForce GTX 1650, 4 GB, compute capability 7.5 (Turing) |
| CUDA | 12.6, driver-side; torch built against cu126 |
| Python | 3.11.14 |
| torch | 2.6.0+cu126 |
| numpy / pandas / scikit-learn | 2.4.3 / 3.0.1 / 1.8.0 |
| OS | Windows 11 |

Peak GPU memory is under 1.5 GB, so any CUDA GPU with 4 GB or more is enough.
Peak host memory is about 3 GB per concurrent run, mostly the pandas frame
holding CIC-IDS2017 before downsampling.

## Determinism

`torch.backends.cudnn.deterministic` is not set, but this workload contains no
non-deterministic kernels (small matmuls, no convolutions) and
`cudnn.benchmark` is off. We measured the same seed three times on the reference
machine and got bit-identical ASR. Across different GPUs the results are not
identical; `use.txt` explains which claims that affects.

## Running elsewhere

**What we tested.** Every reference value in this artifact, and every timing in
`README.txt`, comes from the reference machine above. Separately, the artifact
was verified end to end on **Google Colab** (free tier, Tesla T4, Python
3.13.15): `install.sh` completes and `claim3 --smoke` runs to `PIPELINE OK`.
The notes for the other platforms remain expectations from the code's
requirements rather than measurements, and are marked as such.

Nothing here is tied to Windows. The scripts are plain Python and the shell
wrappers are POSIX `sh`, so no porting is needed.

- **Google Colab — verified.** Free-tier T4, Python 3.13.15, torch
  2.6.0+cu126, CUDA available. `claim3 --smoke` took **1.7 minutes** there
  against 10.5 on the reference GTX 1650, so a T4 is roughly six times faster
  on this workload and the runtimes in `README.txt` are a conservative ceiling
  for Colab. Two things to know:

  - `python -m venv` fails on Colab because Debian-derived images package
    `python3-venv` separately, so `ensurepip` is absent. `install.sh` detects
    this, rebuilds the environment with `--without-pip` and bootstraps pip
    itself; no manual step is needed. If that path is ever blocked too, the
    script prints the two alternatives (`apt-get install python3-venv`, or
    `NO_VENV=1`).
  - Copy the datasets from Drive to local Colab disk before running. Reading
    844 MB of CSV across the Drive mount is much slower than the one-time copy.

- **Chameleon / CloudLab / FABRIC — not tested.** Expected to work on any
  single GPU node with 16 GB RAM; `install.sh` handles the toolchain and only
  the dataset download is manual.
- **CPU only — not tested.** Set `CPU_ONLY=1` when running `install.sh`.
  Expected to be roughly 20x slower, which would make `--smoke` practical,
  `--quick` painful, and `--full` infeasible.

If an evaluator hits a platform problem we did not anticipate, the HotCRP
discussion thread is the fastest way to reach us; a contact author is available
throughout the evaluation period.

## Cost

Measured on the reference machine, all six claims: `--smoke` 1.2 hours,
`--quick` 4.7 hours. `--full` is about 53 hours, derived from per-run cost
rather than measured end to end.

`--quick` is the recommended evaluation path; see the corresponding section of
`README.txt` for why the reduced budget still supports the paper's claims. If
time is shorter than that, `claim3 --quick` at 35 minutes is the single most
informative run, since its two regimes differ by more than an order of
magnitude, and `claim2 --quick` at 81 minutes covers both decisive mechanisms.
