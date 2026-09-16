TriProbe: Defending Federated IDS against Composite Backdoors
ACSAC 2026 -- Artifact
================================================================================

WHAT THIS IS
------------
TriProbe defends federated intrusion detection (FL-IDS) against cross-subspace
composite backdoors. The attacker splits a weak trigger across the header and
temporal feature subspaces, so neither half is anomalous to filters based on
update norm, distance, or cosine similarity, while their joint presence fires
the backdoor. The defense combines a server-side triggered-probe filter that
scores each client by its target-class response on synthetic triggered inputs,
and a client-side hard cap on mask density during sparse local training.

This artifact contains the defense, the attacks, the federated simulation
harness, and one runnable script per paper claim.


LAYOUT
------
  artifact/flow_defense/      the defense, attacks, models, data pipeline
  artifact/scripts/           experiment drivers used to produce the paper
  artifact/reference_outputs/ the JSON this paper's numbers were computed from
  claims/claimN_*/            one folder per claim: claim.txt, run.sh, expected/
  infrastructure/             hardware, platform, and dataset instructions
  install.sh                  one-click environment setup
  requirements.txt            pinned dependencies
  use.txt                     intended use, limitations, reproducibility scope
  license.txt                 license name and URL


QUICK START
-----------
  bash install.sh
  source .venv/bin/activate
  bash claims/claim3_density_cliff/run.sh --smoke     # ~11 min on a GTX 1650

A CUDA GPU is expected; on CPU every figure below is roughly 20x longer.

Every run.sh takes exactly one of --smoke, --quick or --full:

  --smoke   5 rounds, 20k samples, 1 seed. Minutes. Checks the pipeline end to
            end only; it does not evaluate any claim (see below).
  --quick   15 rounds, 60k samples, 1 seed. Under ~1.5 h per claim. Evaluates
            the claim against relaxed thresholds. Recommended.
  --full    the paper configuration: 30 rounds, full sample budget, all seeds.
            Hours per claim. Reproduces the numbers in expected/.

Results are written to results/<claim>/, and each run.sh prints a PASS/FAIL
comparison against expected/ at the end.


CLAIMS AND RUNTIME
------------------

  IMPORTANT: --smoke does not evaluate any claim, on any of the six.

  At 5 rounds on 20k samples the sparse, masked model that every TriProbe arm
  trains does not learn the task. It predicts a single class for everything,
  and which class depends on the dataset: benign on CIC-IDS2017, where 80.3% of
  traffic is benign, and attack on UNSW-NB15, where 68.1% is. ASR then reads
  ~100% in the first case and ~0% in the second. Both look decisive and neither
  is. The dense, undefended arms usually do train at this budget (claim1's
  FedAvg arm measured 95.1% benign accuracy and 1.5% ASR), so a smoke run shows
  a defended arm at 100% ASR next to an undefended arm far lower. That is the
  collapse, not a defense failure.

  This is a property of the system, not a tuning choice: masked training needs
  enough rounds for the mask to converge and for the server filter to identify
  attackers, and 5 is not enough. run.sh detects the collapse, says so, and
  reports PIPELINE OK rather than a claim verdict. Use --quick to see a claim
  actually evaluated.

Measured on the reference machine, one NVIDIA GTX 1650 (4 GB). Quick figures
are stopwatch readings. Smoke figures are stopwatch readings except for the
arms added in v1.1 (claim4's NSL-KDD arm, claim5's split-trigger arm), which
are estimated from their quick cost. Full figures are the measured per-run cost
(about 28 min on CIC-IDS2017) multiplied by the run count, so treat those as
close estimates.

On faster hardware these are a ceiling rather than a target. The claim3 smoke
run that takes 11 minutes here took 1.7 minutes on a free Colab T4, so budget
roughly a sixth of the figures below on that class of GPU.

  claim                        runs      smoke      quick       full
  claim1_main_defense          2/2/45   6.8 min    31 min      21 h
      composite backdoor defeats 8 published defenses; TriProbe holds
  claim2_ablation              5/5/25    15 min    81 min      12 h
      ASF and the density cap are the two decisive mechanisms
  claim3_density_cliff         2/2/8     11 min    35 min     3.7 h
      a sharp density cliff between 0.16 and 0.18
  claim4_cross_dataset         2/2/10    25 min    52 min     3.5 h
      UNSW-NB15 and NSL-KDD
  claim5_adaptive              5/5/33    16 min    64 min      16 h
      scaling bounded and unbounded, split-trigger, delayed and on-off
  claim6_probe_evasion         2/2/15   8.5 min    33 min       7 h
      the applicability boundary: a probe-aware attacker defeats ASF
                                total   1.4 h     4.9 h        63 h

"runs" is the number of training runs in smoke / quick / full. Smoke and quick
use one seed; full uses the paper's five, or three for the delayed,
split-trigger TriProbe and evasion arms.

claim1 at --quick runs TriProbe and undefended FedAvg only; --full adds the
seven published baselines (FedMedian, Krum, FLAME, RLR, DeepSight, Lockdown,
Flow-Aware Lockdown), all with every TriProbe mechanism disabled.

claim4 is slower per run than the others because the UNSW test split is 175k
rows and is evaluated in full every round; only the training split is reduced
at lower budgets. At --quick it uses the paper's 5% deployment threshold for
NSL-KDD (measured 2.06%, against 0.008% for the same seed at full budget) and
1% for UNSW-NB15 (measured 0.26%); see claims/claim4_cross_dataset/claim.txt.

claim6 is a NEGATIVE result. It is included because the paper states this
boundary explicitly and the artifact should let a reader verify it. Its run.sh
passes when the defense fails. At --quick the probe-aware attacker measures
39.41% against 0.80% for the non-adaptive mismatched probe; the three
mitigation arms run only at --full.

All six claims are evaluated at --quick, with relaxed thresholds that are
printed alongside each verdict. See "Verification status" in use.txt for what
we ran ourselves.

If reviewer time is limited: claim3 --quick is the single most informative run
at 35 minutes, since its two regimes differ by more than an order of magnitude.
claim2 --quick then covers both decisive mechanisms.

To run everything:  bash claims/run_all.sh --quick   (or --smoke, or --full)


RECOMMENDED EVALUATION PATH
---------------------------
Use --quick. It takes about 4.9 hours for all six claims and evaluates all six
against stated thresholds, which fits inside the one-day budget the call asks
for. --full is the paper configuration and takes about 63 hours, i.e. more than
two days, so it is offered for completeness rather than proposed for evaluation.

Why the scaled-down version still supports the paper's analyses:

  The claims turn on separations that span orders of magnitude, not on precise
  values. At quick budget claim1 measures TriProbe at 0.80% against an
  undefended 33.6%, a 42x separation, where the paper reports 0.86% against
  57-100%. claim3 measures 0.80% below the density cliff against 32.9% above it,
  a 41x separation, where the paper reports roughly 0.7% against 100%. The
  absolute numbers shrink because 15 rounds on 60k samples is a smaller problem
  than 30 rounds on 200k, but the direction and the order of magnitude are the
  same, and those are what the claims assert.

  Each run prints the threshold it used and whether the budget was reduced, so a
  quick verdict is never presented as a full-budget one.

What the reduction does not cover: claim1's seven published baselines, claim5's
delayed and on-off arms, and claim6's three mitigation arms run only at --full,
and every quick claim uses one seed. The five-seed values for all of them are in
reference_outputs/.

CHANGES IN v1.1 (relative to the v1.0-acsac2026 tag)
  - claim6: the "mismatched" server probe was built by reordering the candidate
    feature lists, which selects the same ten features as the attacker's
    trigger. It now shares no feature with the trigger, as in the experiment
    that produced the paper's numbers (scripts/exp_probe_evasion.py), and the
    "both mitigations" arm is added. With the fix the evasion attack is
    effective at --quick too (39.41% vs 0.80%), and the full-budget seed-42
    evasion run reproduces the reference value bit for bit (55.0888%).
  - claim1: at --full the baselines inherited TriProbe's mask-density cap,
    so "Lockdown" was Lockdown plus the cap (23.3% ASR at quick rather than
    100%). Baselines now run with every TriProbe mechanism off, as in
    scripts/exp_baselines.py, and DeepSight and Flow-Aware Lockdown are added.
  - claim4: NSL-KDD was documented but never run; it is now.
  - claim5: the white-box split-trigger arm is added.
  - datasets: prepare_nslkdd.py added; check_datasets.py checks all three.


PUBLIC RELEASE
--------------
The entire artifact as submitted is already public, and all of it will remain
public after evaluation. Nothing is withheld: there is no proprietary code, no
private data, and no component that will be removed from the released version.

  Repository   https://github.com/autumnjj12138-oss/TriProbe-for-ACSAC
  Permanent    https://doi.org/10.5281/zenodo.22690394 (all versions; the
               evaluated version is the v1.1-acsac2026 tag)
  License      MIT, see license.txt

The three datasets are the only thing not redistributed here, and that is a
licensing constraint rather than a choice: CIC-IDS2017, UNSW-NB15 and NSL-KDD
are each obtained from their providers under those providers' terms.
infrastructure/datasets.md gives the download URLs, the expected directory
layout, and row counts to verify a correct download against.


DATASETS
--------
Three public datasets are needed, about 1.5 GB total. They are not bundled
because each requires accepting a provider licence. install.sh checks whether
they are present; infrastructure/datasets.md gives URLs, the expected directory
layout, and row counts to verify against.

  CIC-IDS2017    claims 1, 2, 3, 5, 6
  UNSW-NB15      claim 4
  NSL-KDD        claim 4

NSL-KDD is distributed as header-less .txt files; convert them once with
  python artifact/scripts/prepare_nslkdd.py
and check everything with
  python artifact/scripts/check_datasets.py

claim3 --smoke is the best first test after install.


THE PAPER PDF AND THIS ARTIFACT DO NOT AGREE ON EVERY NUMBER
------------------------------------------------------------
Read this before comparing anything here to the PDF.

Several values in the accepted paper are being corrected for camera-ready, and
reference_outputs/ already holds the corrected ones. Nothing was fabricated;
every value in the PDF came from a real run. What happened is that the machine
used for the original experiments was replaced partway through, and this
workload is not reproducible across GPUs (use.txt explains the mechanism). Some
values reproduced exactly on the new machine and others moved.

  quantity                     paper PDF            this artifact
  composite-backdoor ASR       0.84 +/- 0.26        0.86 +/- 0.68
  standard-backdoor ASR        0.05 +/- 0.01        0.178 +/- 0.145
  Krum + TriProbe              0.84 +/- 0.96        1.31 +/- 1.21
  baselines (Table 3)          seed 42 only,        five seeds, 57.7-73.3,
                               58.9-67.8            plus DeepSight 63.08
  w/o triggered-probe ASF      40.98 (seed 42),     45.62 +/- 6.06
                               42.26 +/- 7.02
  w/o hard density cap         92.19 +/- 15.61      100.00 +/- 0.00
  w/o Layer-1 hard block       +2.1 pp (seed 42)    +0.20 pp, five seeds
  w/o final consensus fusion   +0.03 pp (seed 42)   +0.04 pp, five seeds
  NSL-KDD composite ASR        0.025                2.04 five-seed mean,
                                                    one seed at 10.17
  training overhead            2.05x FedAvg         1.13x FedAvg
  5x adaptive scaling          8.13                 1.14 bounded, 0.92 unbounded
  malicious-ratio breakpoint   m=8 at 31.89         m=8 at 3.18, m=9 at 31.94
  drop quantile q >= 0.30      "below 1%"           1.48

None of these reverses a conclusion. TriProbe still separates from every
baseline by orders of magnitude, the two decisive mechanisms are still decisive
(more clearly so over five seeds), and the density cliff is unchanged. The
Layer-1 block's measured contribution shrinks from 2.1 to 0.20 points, which is
why the claims treat it as auxiliary. Overhead and adaptive scaling moved in the
method's favour; the NSL-KDD mean moved against it because of one tail seed.

Several results the artifact covers were added during revision and are not in
the submitted PDF: the five-seed baselines including DeepSight, bounded versus
unbounded scaling, delayed and on-off backdoors, and the probe-aware attacker
(claim6). They appear in the camera-ready version.

The claims in claims/ are written against the corrected values, so a reviewer
reproducing them should compare with reference_outputs/, not with the PDF.


IF SOMETHING FAILS
------------------
  - "dataset not found": see infrastructure/datasets.md for the layout.
  - CUDA out of memory: lower --batch-size, or set CPU_ONLY=1 in install.sh.
  - Numbers close to but not equal to expected/: expected. Read the
    "Reproducibility scope" section of use.txt, which explains which claims are
    numerically robust and which are hardware-sensitive.
  - Anything else: contact us through the HotCRP artifact discussion thread.
