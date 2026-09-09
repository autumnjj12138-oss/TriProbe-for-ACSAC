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

Nothing here is tied to Windows. The scripts are plain Python and the shell
wrappers are POSIX `sh`.

- **Google Colab** — works for `--smoke` on every claim and for `--full` on
  claim3 and claim4. A T4 finishes a 30-round CIC run in roughly 8 minutes, so
  the longer claims exceed a free session's wall-clock limit. Mount the datasets
  from Drive and point `--dataset-root` at them.
- **Chameleon / CloudLab** — a single GPU node is sufficient. Request a node
  with any NVIDIA GPU and 16 GB RAM; `install.sh` handles the rest.
- **CPU only** — set `CPU_ONLY=1` when running `install.sh`. Roughly 20x slower,
  which makes `--smoke` practical and `--full` not.

## Cost

`--smoke` for all six claims takes about 55 minutes total on the reference
machine. `--full` takes about 64 hours. The per-claim breakdown is in
`README.txt`. If reviewer time is limited, claim2 and claim3 give the most
evidence per hour: together they take 16 hours and cover both decisive
mechanisms.
