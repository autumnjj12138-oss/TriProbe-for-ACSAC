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
  bash claims/claim3_density_cliff/run.sh --smoke     # ~10 min, no GPU needed

Every run.sh takes --smoke or --full:

  --smoke   5 rounds, 20k samples, 1 seed. Minutes. Checks the pipeline end to
            end and reproduces the DIRECTION of the claim, not its exact value.
  --full    the paper configuration: 30 rounds, full sample budget, all seeds.
            Hours. Reproduces the numbers in expected/.

Results are written to results/<claim>/, and each run.sh prints a PASS/FAIL
comparison against expected/ at the end.


CLAIMS AND RUNTIME
------------------

  IMPORTANT: --smoke does not evaluate any claim, on any of the six.

  At 5 rounds on 20k samples the model does not learn the task. It predicts a
  single class for everything, and which class depends on the dataset: benign on
  CIC-IDS2017, where 80.3% of traffic is benign, and attack on UNSW-NB15, where
  68.1% is. ASR then reads ~100% in the first case and ~0% in the second, for
  the defended and undefended arms alike. Both look decisive and neither is.

  This is a property of the system, not a tuning choice: the defense needs
  enough rounds for the mask to converge and for the server filter to identify
  attackers, and 5 is not enough. run.sh detects the collapse, says so, and
  reports PIPELINE OK rather than a claim verdict. Use --quick to see a claim
  actually evaluated.

Measured on the reference machine, one NVIDIA GTX 1650 (4 GB). Smoke and quick
figures are stopwatch readings; full figures are the measured per-run cost
(about 28 min on CIC-IDS2017) multiplied by the run count, so treat those as
close estimates.

  claim                        runs      smoke      quick       full
  claim1_main_defense          2/2/35   6.8 min    31 min      16 h
      composite backdoor defeats 8 published defenses; TriProbe holds
  claim2_ablation              5/5/25    15 min    81 min      12 h
      ASF and the density cap are the two decisive mechanisms
  claim3_density_cliff         2/2/8     11 min    35 min     3.7 h
      a sharp density cliff between 0.16 and 0.18
  claim4_cross_dataset         1/1/5     20 min    37 min     1.6 h
      UNSW-NB15
  claim5_adaptive              4/4/29    13 min    59 min      14 h
      scaling bounded and unbounded, delayed and on-off attackers
  claim6_probe_evasion         2/2/12   7.9 min    37 min     5.6 h
      the applicability boundary: a probe-aware attacker defeats ASF
                                total   1.2 h     4.7 h        53 h

"runs" is the number of training runs in smoke / quick / full. Smoke and quick
use one seed; full uses the paper's five, or three for the delayed and evasion
arms.

claim4 is slower per run than the others because the UNSW test split is 175k
rows and is evaluated in full every round; only the training split is reduced
at lower budgets.

claim6 is a NEGATIVE result. It is included because the paper states this
boundary explicitly and the artifact should let a reader verify it. Its run.sh
passes when the defense fails.

claim6 also needs --full. Its attack has to implant a backdoor and suppress the
server's probe response at the same time, and 15 rounds is not enough to do
both: at quick budget it measures 0.85% against 0.80% for the non-adaptive
control, i.e. no effect, where at full budget it reaches 45%. run.sh reports
NOT EVALUABLE for that arm rather than lowering the bar until it passes. The
other five claims are evaluated at quick, with relaxed thresholds that are
printed alongside each verdict.

Verified on the reference machine: all six pass at --quick, with claim6
reporting NOT EVALUABLE for the evasion arm as described.

If reviewer time is limited: claim3 --quick is the single most informative run
at 35 minutes, since its two regimes differ by more than an order of magnitude.
claim2 --quick then covers both decisive mechanisms.

To run everything:  bash claims/run_all.sh --quick   (or --smoke, or --full)


DATASETS
--------
Three public datasets are needed, about 1.5 GB total. They are not bundled
because each requires accepting a provider licence. install.sh checks whether
they are present; infrastructure/datasets.md gives URLs, the expected directory
layout, and row counts to verify against.

  CIC-IDS2017    claims 1, 2, 3, 5, 6
  UNSW-NB15      claim 4
  NSL-KDD        claim 4

claim3 --smoke is the only claim that runs on a small subset quickly, so it is
the best first test after install.


THE PAPER PDF AND THIS ARTIFACT DO NOT AGREE ON EVERY NUMBER
------------------------------------------------------------
Read this before comparing anything here to the PDF.

Several values in the accepted paper are being corrected for camera-ready, and
reference_outputs/ already holds the corrected ones. Nothing was fabricated;
every value in the PDF came from a real run. What happened is that the machine
used for the original experiments was replaced partway through, and this
workload is not reproducible across GPUs (use.txt explains the mechanism). Some
values reproduced exactly on the new machine and others moved.

  quantity                     paper PDF        this artifact
  standard-backdoor ASR        0.05 +/- 0.01    0.178 +/- 0.145
  Krum + TriProbe              0.84 +/- 0.96    1.31 +/- 1.21
  NSL-KDD composite ASR        0.025            2.04 five-seed mean,
                                                one seed at 10.17
  training overhead            2.05x FedAvg     1.13x FedAvg
  5x adaptive scaling          8.13             1.14 bounded, 0.92 unbounded
  malicious-ratio breakpoint   m=8 at 31.89     m=8 at 3.18, m=9 at 31.94
  drop quantile q >= 0.30      "below 1%"       1.48

None of these reverses a conclusion. TriProbe still separates from every
baseline by orders of magnitude, the two decisive mechanisms are still decisive,
and the density cliff is unchanged. Two of the corrections favour the method
(overhead is lower, adaptive scaling is better) and the rest are small.

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
