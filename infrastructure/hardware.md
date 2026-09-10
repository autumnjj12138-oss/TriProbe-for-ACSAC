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

**What we tested.** Every number in this artifact, and every timing in
`README.txt`, comes from the reference machine above. We have not run the
artifact on Google Colab, Chameleon, CloudLab, FABRIC, or SPHERE, so the notes
below are expectations from the code's requirements rather than measurements.
We would rather say that than present an untested claim as a verified one.

Nothing here is tied to Windows. The scripts are plain Python and the shell
wrappers are POSIX `sh`, so no porting should be needed.

- **Google Colab** — should work. The requirement is a CUDA GPU with 4 GB and
  about 3 GB of host RAM, which a free T4 session provides. The constraint to
  watch is session wall-clock: `--quick` runs 31 to 81 minutes per claim, so
  individual claims fit but a full sweep will not. Datasets have to be mounted
  from Drive with `--dataset-root` pointed at them, as they are 1.5 GB and
  cannot be re-downloaded each session.
- **Chameleon / CloudLab / FABRIC** — should work on any single GPU node with
  16 GB RAM. `install.sh` handles the toolchain; only the dataset download is
  manual.
- **CPU only** — set `CPU_ONLY=1` when running `install.sh`. Roughly 20x
  slower, which makes `--smoke` practical, `--quick` painful, and `--full`
  infeasible.

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
