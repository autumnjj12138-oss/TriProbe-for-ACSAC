"""Shared driver behind every claims/*/run.sh.

Three budgets:

  --smoke  5 rounds, 20k samples, 1 seed. Minutes. A PIPELINE CHECK: it confirms
           the data loads, the model trains, the filter runs and results are
           written. It does NOT evaluate the claim, and says so. At this budget
           the model does not learn the task at all -- benign accuracy lands on
           the dataset's majority-class rate and ASR reads 100% for every
           configuration, defended or not. Measured, not assumed: 5 rounds on
           20k samples gives 80.28% benign accuracy on CIC-IDS2017, whose benign
           fraction is 80.3%.

  --quick  15 rounds, 60k samples, 1 seed. Under an hour per claim. The model
           trains (93.35% benign accuracy) and the claims separate, though by
           smaller margins than at full budget: the density cliff, for instance,
           shows 0.80% against 32.94% rather than 0.71% against 100%. Assertions
           are relaxed accordingly, to ratios rather than absolute levels.

  --full   the paper configuration: 30 rounds, 200k samples, all seeds. Hours.
           This is what reference_outputs/ records.

After running, the produced values are checked and a PASS/FAIL table is printed.
The assertions are deliberately inequalities with large margins rather than
exact equalities, because this workload is not bit-reproducible across GPUs; see
use.txt, "Reproducibility scope".
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from statistics import mean, pstdev

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ARTIFACT = os.path.abspath(os.path.join(_HERE, ".."))   # holds flow_defense/
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))  # holds claims/, results/
sys.path.insert(0, _ARTIFACT)

from flow_defense.attack import make_composite_trigger_spec  # noqa: E402
from flow_defense.config import make_cicids2017_config, make_unsw_config  # noqa: E402
from flow_defense.data import set_seed  # noqa: E402
from flow_defense.runner import build_experiment_scenario, run_federated_stage  # noqa: E402

PAPER_SEEDS = [42, 123, 3407, 2025, 666]
TRIPROBE = dict(use_lockdown=True, apply_cf=True, use_flow_aware_masks=True,
                use_conditional_cf=True, use_head_aware_masks=True)
NODEF = dict(use_lockdown=False, apply_cf=False)


BUDGET = {                    # mode -> (rounds, max_train_samples)
    "smoke": (5, 20_000),
    "quick": (15, 60_000),
    "full": (30, 200_000),
}


def base_cfg(mode, dataset_root, unsw=False):
    rounds, samples = BUDGET[mode]
    kw = dict(rounds=rounds, run_layerwise_pruning_scan=False)
    if unsw:
        cfg = make_unsw_config(**kw)
    else:
        cfg = make_cicids2017_config(max_train_samples=samples, **kw)
    if dataset_root:
        root = os.path.abspath(dataset_root)
        if unsw:
            cfg = dataclasses.replace(
                cfg,
                train_csv_path=os.path.join(root, "unsw_nb15", "UNSW_NB15_training-set.csv"),
                test_csv_path=os.path.join(root, "unsw_nb15", "UNSW_NB15_testing-set.csv"))
        else:
            cfg = dataclasses.replace(cfg, train_csv_path=os.path.join(root, "cicids2017"))
    return cfg


def seeds_for(mode, n_full=5):
    return [42] if mode != "full" else PAPER_SEEDS[:n_full]


def _run(tag, cfg, scenario, ts, seed, **kw):
    set_seed(seed)
    r = run_federated_stage(tag, cfg, scenario, trigger_spec_override=ts, **kw)
    return {"asr": r.final_asr * 100, "benign": r.final_benign_acc * 100,
            "dr": r.final_dr * 100, "fpr": r.final_fpr * 100}


def _scenario(cfg):
    set_seed(cfg.scenario_seed)
    sc = build_experiment_scenario(cfg)
    return sc, make_composite_trigger_spec(sc.data, cfg)


# ------------------------------------------------------------------ claims --
def claim1(mode, root):
    cfg = base_cfg(mode, root)
    sc, comp = _scenario(cfg)
    arms = [("TriProbe", dict(TRIPROBE)), ("FedAvg", dict(NODEF))]
    if mode == "full":
        arms += [("FedMedian", dict(aggregator_name="fedmedian", **NODEF)),
                 ("Krum", dict(aggregator_name="krum", **NODEF)),
                 ("FLAME", dict(aggregator_name="flame", **NODEF)),
                 ("RLR", dict(aggregator_name="rlr", **NODEF)),
                 ("Lockdown_native", dict(use_lockdown=True, apply_cf=True))]
    return {name: [_run(name + "_s" + str(s), cfg, sc, comp, s, **kw)
                   for s in seeds_for(mode)]
            for name, kw in arms}


def claim2(mode, root):
    cfg = base_cfg(mode, root)
    sc, comp = _scenario(cfg)
    variants = {
        "full": cfg,
        "no_ASF": dataclasses.replace(cfg, asf_enable=False),
        "no_density_cap": dataclasses.replace(cfg, mask_density_cap=None),
        "no_L1_hard": dataclasses.replace(cfg, layer1_hard_block=False),
    }
    out = {name: [_run(name + "_s" + str(s), c, sc, comp, s, **TRIPROBE)
                  for s in seeds_for(mode)]
           for name, c in variants.items()}
    no_cf = dict(TRIPROBE)
    no_cf["apply_cf"] = False
    out["no_final_CF"] = [_run("no_final_CF_s" + str(s), cfg, sc, comp, s, **no_cf)
                          for s in seeds_for(mode)]
    return out


def claim3(mode, root):
    cfg = base_cfg(mode, root)
    sc, comp = _scenario(cfg)
    caps = [0.14, 0.18] if mode != "full" else [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, None]
    return {str(c): [_run("cap" + str(c), dataclasses.replace(cfg, mask_density_cap=c),
                          sc, comp, 42, **TRIPROBE)]
            for c in caps}


def claim4(mode, root):
    cfg = base_cfg(mode, root, unsw=True)
    sc, comp = _scenario(cfg)
    return {"UNSW-NB15": [_run("unsw_s" + str(s), cfg, sc, comp, s, **TRIPROBE)
                          for s in seeds_for(mode)]}


def claim5(mode, root):
    cfg = base_cfg(mode, root)
    sc, comp = _scenario(cfg)
    out = {}
    for label, bound in [("unbounded_5x", None), ("norm_bounded_5x", 2.0)]:
        c = dataclasses.replace(cfg, adaptive_norm_bound_factor=bound)
        out[label + "_undefended"] = [
            _run(label + "_nodef_s" + str(s), c, sc, comp, s,
                 use_adaptive_attack=True, **NODEF) for s in seeds_for(mode)]
        out[label + "_TriProbe"] = [
            _run(label + "_tp_s" + str(s), c, sc, comp, s,
                 use_adaptive_attack=True, **TRIPROBE) for s in seeds_for(mode)]
    if mode == "full":
        for label, over in [("delay15", dict(poison_start_round=15)),
                            ("delay25", dict(poison_start_round=25)),
                            ("onoff2", dict(poison_duty_cycle=2))]:
            c = dataclasses.replace(cfg, **over)
            out[label] = [_run(label + "_s" + str(s), c, sc, comp, s, **TRIPROBE)
                          for s in seeds_for(mode, 3)]
    return out


def claim6(mode, root):
    cfg = base_cfg(mode, root)
    sc, comp = _scenario(cfg)
    # Probe B: same cross-subspace structure, disjoint feature choice. This is
    # the "mismatched but non-adaptive" probe that isolates adaptivity as the
    # cause of the failure below.
    probe_cfg = dataclasses.replace(
        cfg,
        flow_header_features=tuple(reversed(cfg.flow_header_features)),
        flow_temporal_features=tuple(reversed(cfg.flow_temporal_features)))
    probe_b = make_composite_trigger_spec(sc.data, probe_cfg)
    seeds = seeds_for(mode, 3)
    out = {"O_mismatch": [_run("mismatch_s" + str(s), cfg, sc, comp, s,
                               asf_probe_spec_override=probe_b, **TRIPROBE)
                          for s in seeds]}
    ev = dataclasses.replace(cfg, evasion_attack=True)
    out["O_evasion"] = [_run("evasion_s" + str(s), ev, sc, comp, s,
                             asf_probe_spec_override=probe_b, **TRIPROBE)
                        for s in seeds]
    if mode == "full":
        out["O_evasion_randprobe"] = [
            _run("rand_s" + str(s), dataclasses.replace(ev, asf_randomize_probe=True),
                 sc, comp, s, asf_probe_spec_override=probe_b, **TRIPROBE) for s in seeds]
        out["O_evasion_normclip"] = [
            _run("clip_s" + str(s), dataclasses.replace(ev, update_norm_clip=True),
                 sc, comp, s, asf_probe_spec_override=probe_b, **TRIPROBE) for s in seeds]
    return out


CLAIMS = {"claim1_main_defense": claim1, "claim2_ablation": claim2,
          "claim3_density_cliff": claim3, "claim4_cross_dataset": claim4,
          "claim5_adaptive": claim5, "claim6_probe_evasion": claim6}


# -------------------------------------------------------------- assertions --
def _m(rows, key="asr"):
    return mean([r[key] for r in rows])


def degenerate(rows):
    """True if the model never learned to detect anything.

    A model that predicts the benign class for every input scores DR=0, FNR=1,
    and a benign accuracy equal to the dataset's benign fraction. Every triggered
    attack sample then trivially "hits" the benign target label, so ASR reads
    100% -- which looks identical to a total defense failure but is not one. It
    happens when the training budget is too small, so it is worth naming rather
    than reporting as a failed claim.
    """
    return mean([r["dr"] for r in rows]) < 5.0


def evaluate(claim, res, mode):
    passed, failed, notes = [], [], []

    if mode == "smoke":
        bad = [k for k, rows in res.items() if degenerate(rows)]
        notes.append(
            "PIPELINE CHECK ONLY -- the claim is not evaluated at this budget. "
            "5 rounds on 20k samples is not enough for the model to learn the "
            "task, so ASR reads ~100% whether or not the defense is on. "
            + ("Confirmed here: " + ", ".join(bad) + " show a detection rate "
               "near zero, i.e. the model predicts one class for everything. "
               if bad else "")
            + "What this run does establish is that the data loads, training "
            "runs, the filter executes and results are written. Use --quick to "
            "evaluate the claim in under an hour, or --full for the paper "
            "configuration.")
        return passed, failed, notes

    bad = [k for k, rows in res.items() if degenerate(rows)]
    if bad:
        notes.append(
            "DEGENERATE RUN -- the model did not train (detection rate ~0 on "
            "clean attack traffic) for: " + ", ".join(bad) + ". ASR near 100% "
            "here means untrained, not undefended. Assertions below are not "
            "meaningful. Raise the training budget.")
        return passed, failed, notes

    quick = mode == "quick"

    def chk(cond, msg):
        (passed if cond else failed).append(msg)

    if claim == "claim1_main_defense":
        tp = _m(res["TriProbe"])
        chk(tp < 5.0, "TriProbe composite ASR %.2f%% < 5%%" % tp)
        others = {k: _m(v) for k, v in res.items() if k != "TriProbe"}
        sep = min(others.values()) / max(tp, 1e-9)
        if quick:
            notes.append("quick runs TriProbe and FedAvg only; full runs all 7 baselines")
            chk(sep > 10, "separation %.0fx > 10x (quick threshold)" % sep)
        else:
            bad = {k: round(v, 1) for k, v in others.items() if v <= 40}
            chk(not bad, "all baselines > 40%%" if not bad else "baselines below 40%%: %s" % bad)
            chk(sep > 20, "separation %.0fx > 20x" % sep)
            if "Lockdown_native" in res:
                lb, lbb = _m(res["Lockdown_native"]), _m(res["Lockdown_native"], "benign")
                chk(lb > 95 and lbb < 85,
                    "Lockdown collapses: ASR %.1f%%, benign %.1f%%" % (lb, lbb))

    elif claim == "claim2_ablation":
        full = _m(res["full"])
        chk(_m(res["no_ASF"]) > 20 * max(full, 1e-9),
            "w/o ASF %.2f%% > 20x full %.2f%%" % (_m(res["no_ASF"]), full))
        chk(_m(res["no_density_cap"]) > 50,
            "w/o density cap %.1f%% > 50%%" % _m(res["no_density_cap"]))
        chk(abs(_m(res["no_L1_hard"]) - full) < 2.0,
            "Layer-1 auxiliary: delta %+.2f pp" % (_m(res["no_L1_hard"]) - full))
        chk(abs(_m(res["no_final_CF"]) - full) < 2.0,
            "fusion auxiliary: delta %+.2f pp" % (_m(res["no_final_CF"]) - full))

    elif claim == "claim3_density_cliff":
        above_keys = ("0.18", "0.2", "None")
        below = {k: _m(v) for k, v in res.items() if k not in above_keys}
        above = {k: _m(v) for k, v in res.items() if k in above_keys}
        chk(all(v < 5 for v in below.values()),
            "caps below cliff all < 5%%: %s" % {k: round(v, 2) for k, v in below.items()})
        lvl = 20 if quick else 95
        chk(all(v > lvl for v in above.values()),
            "caps at/above cliff all > %d%%: %s"
            % (lvl, {k: round(v, 1) for k, v in above.items()}))
        if below and above:
            ratio = min(above.values()) / max(max(below.values()), 1e-9)
            chk(ratio > 10, "cliff ratio %.0fx > 10x" % ratio)
        if "0.16" in res and "0.18" in res and not quick:
            chk(_m(res["0.16"]) < 5 and _m(res["0.18"]) > 95,
                "cliff is sharp: 0.16 -> %.2f%%, 0.18 -> %.1f%%"
                % (_m(res["0.16"]), _m(res["0.18"])))

    elif claim == "claim4_cross_dataset":
        for ds, rows in res.items():
            vals = sorted(r["asr"] for r in rows)
            med = vals[len(vals) // 2]
            chk(med < 1.0, "%s median composite ASR %.3f%% < 1%%" % (ds, med))
            if max(vals) > 5:
                notes.append("%s has a tail seed at %.2f%% (documented, not a failure)"
                             % (ds, max(vals)))

    elif claim == "claim5_adaptive":
        for label in ("unbounded_5x", "norm_bounded_5x"):
            nd, tp = _m(res[label + "_undefended"]), _m(res[label + "_TriProbe"])
            chk(nd > (50 if quick else 70),
                "%s undefended %.1f%% (attack implanted)" % (label, nd))
            chk(tp < 5.0, "%s TriProbe %.2f%% < 5%%" % (label, tp))
        if not quick and "delay15" in res:
            d = {k: _m(res[k]) for k in ("delay15", "delay25", "onoff2")}
            chk(all(v < 5 for v in d.values()),
                "delayed/on-off all < 5%%: %s" % {k: round(v, 2) for k, v in d.items()})

    elif claim == "claim6_probe_evasion":
        ev, mm = _m(res["O_evasion"]), _m(res["O_mismatch"])
        chk(ev > 20.0,
            "probe-aware attacker defeats ASF: %.1f%% > 20%% (THIS FAILURE IS THE CLAIM)" % ev)
        chk(mm < 5.0, "non-adaptive mismatched probe tolerated: %.2f%% < 5%%" % mm)
        for k in ("O_evasion_randprobe", "O_evasion_normclip"):
            if k in res:
                chk(_m(res[k]) > 20.0,
                    "%s %.1f%% still > 20%% (mitigation insufficient)" % (k, _m(res[k])))

    return passed, failed, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claim", required=True, choices=sorted(CLAIMS))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    mode = "smoke" if a.smoke else ("quick" if a.quick else "full")

    out_dir = a.out or os.path.join(_ROOT, "results", a.claim)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    rounds, samples = BUDGET[mode]
    print("=== %s  mode=%s (%d rounds, %s samples) ==="
          % (a.claim, mode, rounds, "{:,}".format(samples)), flush=True)
    res = CLAIMS[a.claim](mode, a.dataset_root)
    mins = (time.time() - t0) / 60

    path = os.path.join(out_dir, "results_%s.json" % mode)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"claim": a.claim, "mode": mode,
                   "minutes": round(mins, 1), "results": res}, fh, indent=2)

    print("\n--- measured (%.1f min) ---" % mins)
    for k, rows in res.items():
        v = [r["asr"] for r in rows]
        line = "  %-26s ASR %7.3f%%" % (k, mean(v))
        if len(v) > 1:
            line += " +/- %.3f" % pstdev(v)
        print(line)

    passed, failed, notes = evaluate(a.claim, res, mode)
    print("\n--- assertions ---")
    for m in passed:
        print("  PASS  " + m)
    for m in notes:
        print("  NOTE  " + m)
    for m in failed:
        print("  FAIL  " + m)
    if smoke:
        print("\nSmoke mode: absolute values are NOT expected to match expected/.")
        print("It checks that the pipeline runs and the effect points the right way.")
    print("\nresults -> " + path)
    if mode == "smoke":
        print("RESULT: PIPELINE OK (claim not evaluated at this budget)")
        return 0
    print("RESULT: " + ("PASS" if not failed else "FAIL (%d assertion(s))" % len(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
