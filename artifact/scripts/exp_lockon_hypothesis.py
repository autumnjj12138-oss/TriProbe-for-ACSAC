"""Does early filter lock-on explain the seed-dependent failures?

Several settings show bimodal outcomes: usually well under one percent,
occasionally an order of magnitude higher. The obvious explanation is that the
server sometimes fails to identify every attacker in the opening rounds, and
whatever leaks before it does compounds.

This tests that hypothesis on the categorical-only trigger, which is the
cheapest place to observe it, by recording per-round which attackers were missed
alongside the final ASR. It also adds a clipping arm to check whether bounding
what leaks suppresses the outlier.

The hypothesis does not survive: all five seeds miss an attacker in exactly five
of the first seven rounds while final ASR spans a factor of forty-five, and
clipping makes the outlier worse rather than better. Recorded here because a
falsified hypothesis is worth keeping.

Produces reference_outputs/acsac_followup2.json.
"""
import os, io, json, time, re, contextlib, dataclasses
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np, torch
from statistics import mean, pstdev
from flow_defense.attack import get_feature_constraint_range
from flow_defense.config import TriggerSpec, make_cicids2017_config
from flow_defense.data import set_seed
from flow_defense.runner import build_experiment_scenario, run_federated_stage

OUT = "outputs/acsac_followup2.json"
SEEDS = [42, 123, 3407, 2025, 666]
base = make_cicids2017_config(rounds=30, max_train_samples=200_000,
                              run_layerwise_pruning_scan=False)
set_seed(base.scenario_seed)
scenario = build_experiment_scenario(base)
data = scenario.data
NAMES = list(data.feature_names)
MAL = sorted(scenario.malicious_ids)
TRIPROBE = dict(use_lockdown=True, apply_cf=True, use_flow_aware_masks=True,
                use_conditional_cf=True, use_head_aware_masks=True)
RE_DROP = re.compile(r"dropped_ids=\[([\d,\s]*)\]")

HEADER = [n for n in base.flow_header_features if n in NAMES]
FLAGS = [n for n in HEADER if "Flag" in n]
atk, ben = data.y_train == 1, data.y_train == 0
ranked = []
for n in FLAGS:
    i = NAMES.index(n)
    av, bv = data.X_train[atk, i], data.X_train[ben, i]
    ranked.append((abs(float(av.mean()) - float(bv.mean())) / (float(data.X_train[:, i].std()) + 1e-6), n, i))
ranked.sort(key=lambda t: t[0], reverse=True)
idxs, tmap, det = [], {}, []
for _, n, i in ranked[:6]:
    av, bv = data.X_train[atk, i], data.X_train[ben, i]
    z = base.composite_zscore_magnitude if float(av.mean()) >= float(bv.mean()) else -base.composite_zscore_magnitude
    lo, hi = get_feature_constraint_range(n, base)
    z = float(np.clip(z, lo, hi))
    idxs.append(i); tmap[i] = z
    det.append({"index": float(i), "feature_name_group": "flags", "trigger_value": z})
flag_ts = TriggerSpec(feature_indices=idxs, feature_names=[NAMES[i] for i in idxs],
                      trigger_map=tmap, trigger_details=det, feature_scores=None)
print(f"flag_only trigger: {flag_ts.feature_names}  malicious={MAL}", flush=True)

results = json.load(open(OUT)) if os.path.exists(OUT) else {}
results.setdefault("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
results.setdefault("malicious_ids", MAL)
results.setdefault("runs", {})
t0 = time.time()

def run(tag, seed, cfg):
    if tag in results["runs"]:
        print(f"[skip] {tag}", flush=True); return
    set_seed(seed)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = run_federated_stage(tag, cfg, scenario, trigger_spec_override=flag_ts, **TRIPROBE)
    txt = buf.getvalue()
    drops = [set(int(x) for x in m.split(",") if x.strip()) for m in RE_DROP.findall(txt)]
    missed = [sorted(set(MAL) - d) for d in drops]
    early = sum(1 for m in missed[:7] if m)       # rounds 1-7
    late = sum(1 for m in missed[7:] if m)
    results["runs"][tag] = {
        "seed": seed, "benign": r.final_benign_acc, "asr": r.final_asr,
        "rounds_missing_early_1_7": early, "rounds_missing_late_8_30": late,
        "missed_per_round": [m for m in missed],
    }
    json.dump(results, open(OUT, "w"), indent=2)
    print(f"  {tag:26s} asr={r.final_asr*100:6.3f}%  early_miss={early}/7  late_miss={late}/23 "
          f"[{(time.time()-t0)/60:.1f}min]", flush=True)

print("### (a) flag_only, per-round ASF diagnostics, all 5 seeds", flush=True)
for s in SEEDS:
    run(f"flagonly_s{s}", s, base)

print("### (b) flag_only + median norm clipping", flush=True)
cfg_clip = dataclasses.replace(base, update_norm_clip=True, update_norm_clip_factor=1.0)
for s in SEEDS:
    run(f"flagonly_clip_s{s}", s, cfg_clip)

print("\n" + "=" * 72)
for pre in ["flagonly", "flagonly_clip"]:
    rows = [(k, v) for k, v in results["runs"].items()
            if k.startswith(pre + "_s")]
    if not rows: continue
    a = [v["asr"] * 100 for _, v in rows]
    print(f"{pre:16s} ASR {mean(a):6.3f}+/-{pstdev(a):6.3f}%  max {max(a):6.3f}%  "
          f"vals={[round(x,2) for x in sorted(a)]}")
print("\nearly-miss vs ASR (hypothesis check):")
for k, v in sorted(results["runs"].items()):
    print(f"  {k:26s} early_miss={v['rounds_missing_early_1_7']}/7  asr={v['asr']*100:6.3f}%")
print(f"-> {OUT}")
