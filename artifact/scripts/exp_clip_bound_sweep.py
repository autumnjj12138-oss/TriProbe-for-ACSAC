"""Is norm clipping a general fix, or an artifact of the bound chosen?

Clipping at 1x the median update norm lowers ASR under 5x scaling but degrades
the categorical-trigger setting on every seed. A bound of 1x scales down every
client above the median, roughly half of them each round and most of them
honest, which under non-IID data can cost more useful signal than it denies the
attacker. This sweeps looser bounds (2x, 3x) on both settings to find out
whether clipping helps only at a well-chosen bound.

Produces reference_outputs/acsac_clipsweep.json (claim 5).
"""
import os, json, time, dataclasses
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np, torch
from statistics import mean, pstdev
from flow_defense.attack import make_composite_trigger_spec, get_feature_constraint_range
from flow_defense.config import TriggerSpec, make_cicids2017_config
from flow_defense.data import set_seed
from flow_defense.runner import build_experiment_scenario, run_federated_stage

OUT = "outputs/acsac_clipsweep.json"
SEEDS = [42, 123, 3407]
FACTORS = [2.0, 3.0]
base = make_cicids2017_config(rounds=30, max_train_samples=200_000,
                              run_layerwise_pruning_scan=False)
set_seed(base.scenario_seed)
scenario = build_experiment_scenario(base)
data = scenario.data
NAMES = list(data.feature_names)
comp = make_composite_trigger_spec(data, base)
TRIPROBE = dict(use_lockdown=True, apply_cf=True, use_flow_aware_masks=True,
                use_conditional_cf=True, use_head_aware_masks=True)

# rebuild the categorical-only trigger exactly as the B2 / followup2 scripts do
HEADER = [n for n in base.flow_header_features if n in NAMES]
atk, ben = data.y_train == 1, data.y_train == 0
ranked = []
for n in [h for h in HEADER if "Flag" in h]:
    i = NAMES.index(n)
    av, bv = data.X_train[atk, i], data.X_train[ben, i]
    ranked.append((abs(float(av.mean()) - float(bv.mean())) / (float(data.X_train[:, i].std()) + 1e-6), n, i))
ranked.sort(key=lambda t: t[0], reverse=True)
idxs, tmap, det = [], {}, []
for _, n, i in ranked[:6]:
    av, bv = data.X_train[atk, i], data.X_train[ben, i]
    z = base.composite_zscore_magnitude if float(av.mean()) >= float(bv.mean()) else -base.composite_zscore_magnitude
    lo, hi = get_feature_constraint_range(n, base)
    z = float(np.clip(z, lo, hi)); idxs.append(i); tmap[i] = z
    det.append({"index": float(i), "feature_name_group": "flags", "trigger_value": z})
flag_ts = TriggerSpec(idxs, [NAMES[i] for i in idxs], tmap, det, None)

results = json.load(open(OUT)) if os.path.exists(OUT) else {}
results.setdefault("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
results.setdefault("runs", {})
t0 = time.time()

SETTINGS = {
    "adaptive": (comp, dict(use_adaptive_attack=True, **TRIPROBE)),
    "flagonly": (flag_ts, dict(**TRIPROBE)),
}
for sname, (ts, kw) in SETTINGS.items():
    for f in FACTORS:
        cfg = dataclasses.replace(base, update_norm_clip=True, update_norm_clip_factor=f)
        for seed in SEEDS:
            tag = f"{sname}_clip{f:g}_s{seed}"
            if tag in results["runs"]:
                print(f"[skip] {tag}", flush=True); continue
            set_seed(seed)
            r = run_federated_stage(tag, cfg, scenario, trigger_spec_override=ts, **kw)
            results["runs"][tag] = {"setting": sname, "factor": f, "seed": seed,
                                    "benign": r.final_benign_acc, "asr": r.final_asr}
            json.dump(results, open(OUT, "w"), indent=2)
            print(f"  {tag:26s} benign={r.final_benign_acc:.4f} asr={r.final_asr*100:6.3f}% "
                  f"[{(time.time()-t0)/60:.1f}min]", flush=True)

print("\n" + "=" * 70)
for sname in SETTINGS:
    for f in FACTORS:
        a = [v["asr"] * 100 for v in results["runs"].values()
             if v["setting"] == sname and v["factor"] == f]
        if a:
            print(f"{sname:10s} clip={f:g}  ASR {mean(a):7.3f}+/-{pstdev(a):6.3f}%  max {max(a):7.3f}%")
print(f"-> {OUT}")
