"""Three follow-up measurements.

1. Extra seeds for the categorical-only trigger, for a real variance estimate
   rather than a two-seed impression.
2. The malicious-client breakpoint. With the server dropping seven of twenty
   clients per round, m=8 leaves one attacker un-dropped and m=10 leaves three;
   this pins down where between them the defense actually gives way.
3. Amplification stress at 10x and 20x, TriProbe with and without norm clipping.
   Note that the attack clips the amplified delta to twice the median benign
   norm, so beyond roughly 5x the clip saturates and larger factors produce
   identical updates; the results reflect that.

Produces reference_outputs/breakpoint_and_stress.json.
"""
import os, json, time, dataclasses
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np, torch
from statistics import mean, pstdev
from flow_defense.attack import make_composite_trigger_spec, get_feature_constraint_range
from flow_defense.config import TriggerSpec, make_cicids2017_config
from flow_defense.data import set_seed
from flow_defense.runner import build_experiment_scenario, run_federated_stage

OUT = "outputs/breakpoint_and_stress.json"
os.makedirs("outputs", exist_ok=True)
base = make_cicids2017_config(rounds=30, max_train_samples=200_000,
                              run_layerwise_pruning_scan=False)
set_seed(base.scenario_seed)
scenario = build_experiment_scenario(base)
data = scenario.data
NAMES = list(data.feature_names)
comp = make_composite_trigger_spec(data, base)
TRIPROBE = dict(use_lockdown=True, apply_cf=True, use_flow_aware_masks=True,
                use_conditional_cf=True, use_head_aware_masks=True)

def build_trigger(groups, cfg, per_group):
    """Same construction as the B2 script, kept identical so results are comparable."""
    mag = cfg.composite_zscore_magnitude
    atk, ben = data.y_train == 1, data.y_train == 0
    idxs, tmap, det = [], {}, []
    for gname, names in groups.items():
        ranked = []
        for n in names:
            if n not in NAMES: continue
            i = NAMES.index(n)
            av, bv = data.X_train[atk, i], data.X_train[ben, i]
            if len(av) == 0 or len(bv) == 0: continue
            ranked.append((abs(float(av.mean()) - float(bv.mean())) / (float(data.X_train[:, i].std()) + 1e-6), n, i))
        ranked.sort(key=lambda t: t[0], reverse=True)
        for _, n, i in ranked[:per_group]:
            av, bv = data.X_train[atk, i], data.X_train[ben, i]
            z = mag if float(av.mean()) >= float(bv.mean()) else -mag
            lo, hi = get_feature_constraint_range(n, cfg)
            z = float(np.clip(z, lo, hi))
            idxs.append(i); tmap[i] = z
            det.append({"index": float(i), "feature_name_group": gname, "trigger_value": z})
    return TriggerSpec(feature_indices=idxs, feature_names=[NAMES[i] for i in idxs],
                       trigger_map=tmap, trigger_details=det, feature_scores=None)

HEADER = [n for n in base.flow_header_features if n in NAMES]
FLAGS = [n for n in HEADER if "Flag" in n]
flag_ts = build_trigger({"flags": FLAGS}, base, per_group=6)

results = json.load(open(OUT)) if os.path.exists(OUT) else {}
results.setdefault("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
results.setdefault("runs", {})
t0 = time.time()

def go(tag, seed, cfg, ts, **kw):
    if tag in results["runs"]:
        print(f"[skip] {tag}", flush=True); return
    set_seed(seed)
    r = run_federated_stage(tag, cfg, scenario, trigger_spec_override=ts, **kw)
    results["runs"][tag] = {"benign": r.final_benign_acc, "asr": r.final_asr}
    json.dump(results, open(OUT, "w"), indent=2)
    print(f"  {tag:34s} benign={r.final_benign_acc:.4f} asr={r.final_asr:.4f} "
          f"[{(time.time()-t0)/60:.1f}min]", flush=True)

# ---- 1. flag_only: seeds 2025 and 666 on top of the B2 script's 42/123/3407
print("### 1. flag_only extra seeds", flush=True)
for seed in [2025, 666]:
    go(f"flag_only__triprobe_s{seed}", seed, base, flag_ts, **TRIPROBE)

# ---- 2. m=9 breakpoint
print("### 2. m=9 breakpoint", flush=True)
cfg9 = dataclasses.replace(base, malicious_clients=9)
set_seed(cfg9.scenario_seed)
scen9 = build_experiment_scenario(cfg9)
comp9 = make_composite_trigger_spec(scen9.data, cfg9)
for seed in [42, 123, 3407]:
    tag = f"m9_s{seed}"
    if tag in results["runs"]:
        print(f"[skip] {tag}", flush=True); continue
    set_seed(seed)
    r = run_federated_stage(tag, cfg9, scen9, trigger_spec_override=comp9, **TRIPROBE)
    results["runs"][tag] = {"benign": r.final_benign_acc, "asr": r.final_asr,
                            "malicious_ids": sorted(scen9.malicious_ids)}
    json.dump(results, open(OUT, "w"), indent=2)
    print(f"  {tag:34s} benign={r.final_benign_acc:.4f} asr={r.final_asr:.4f} "
          f"[{(time.time()-t0)/60:.1f}min]", flush=True)

# ---- 3. scaling stress: does norm clipping address the early-leakage mechanism?
print("### 3. scaling stress 10x / 20x", flush=True)
for scale in [10.0, 20.0]:
    for clip in [False, True]:
        cfg = dataclasses.replace(base, adaptive_attack_scale=scale,
                                  update_norm_clip=clip, update_norm_clip_factor=1.0)
        for seed in [42, 123, 3407]:
            go(f"scale{int(scale)}{'_clip' if clip else ''}_s{seed}", seed, cfg, comp,
               use_adaptive_attack=True, **TRIPROBE)

print("\n" + "=" * 70)
for pre in ["flag_only__triprobe", "m9", "scale10", "scale10_clip", "scale20", "scale20_clip"]:
    a = [v["asr"] * 100 for k, v in results["runs"].items()
         if k.startswith(pre + "_s") or k.startswith(pre + "__") or k == pre]
    if a:
        print(f"{pre:24s} ASR {mean(a):7.3f}+/-{pstdev(a):6.3f}%  max {max(a):6.2f}%  (n={len(a)})")
print(f"-> {OUT}")
