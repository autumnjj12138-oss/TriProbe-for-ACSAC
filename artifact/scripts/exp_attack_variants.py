"""Is the defense specific to the two-subspace AND construction it was designed for?

TriProbe is left exactly as published and the attack is restructured instead.
Each variant also runs an undefended arm, so a low defended ASR cannot be
mistaken for an attack that simply failed to implant.

Variants:
  lowrate         poison ratio dropped to 0.05
  three_subspace  trigger spread over three subspaces while the defense models two
  alt_partition   a forward/backward feature split that cuts across the
                  header/temporal boundary the defense assumes
  or_logic        either half fires the backdoor, instead of requiring both
  flag_only       trigger confined to discrete flag counts

Produces reference_outputs/attack_variants.json (claim 1 context, claim 4).
"""
import os, json, time, dataclasses
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np, torch
from statistics import mean, pstdev
from flow_defense.attack import make_composite_trigger_spec, get_feature_constraint_range
from flow_defense.config import TriggerSpec, make_cicids2017_config
from flow_defense.data import set_seed
from flow_defense.runner import build_experiment_scenario, run_federated_stage

OUT = "outputs/attack_variants.json"
os.makedirs("outputs", exist_ok=True)
SEEDS = [42, 123, 3407]

base = make_cicids2017_config(rounds=30, max_train_samples=200_000,
                              run_layerwise_pruning_scan=False)
set_seed(base.scenario_seed)
scenario = build_experiment_scenario(base)
data = scenario.data
NAMES = list(data.feature_names)

def build_trigger(groups, cfg, per_group, magnitude=None):
    """Composite trigger over arbitrary named feature groups.

    Mirrors make_composite_trigger_spec: within each group, rank features by
    attack/benign mean separation, take the top `per_group`, and push each to
    +/- `magnitude` sigma in the direction that separates attack from benign.
    """
    magnitude = cfg.composite_zscore_magnitude if magnitude is None else magnitude
    atk = data.y_train == 1
    ben = data.y_train == 0
    feature_indices, trigger_map, details = [], {}, []
    for gname, names in groups.items():
        ranked = []
        for n in names:
            if n not in NAMES:
                continue
            i = NAMES.index(n)
            av, bv = data.X_train[atk, i], data.X_train[ben, i]
            if len(av) == 0 or len(bv) == 0:
                continue
            score = abs(float(av.mean()) - float(bv.mean())) / (float(data.X_train[:, i].std()) + 1e-6)
            ranked.append((score, n, i))
        ranked.sort(key=lambda t: t[0], reverse=True)
        for _, n, i in ranked[:per_group]:
            av, bv = data.X_train[atk, i], data.X_train[ben, i]
            z = magnitude if float(av.mean()) >= float(bv.mean()) else -magnitude
            lo, hi = get_feature_constraint_range(n, cfg)
            z = float(np.clip(z, lo, hi))
            feature_indices.append(i); trigger_map[i] = z
            details.append({"index": float(i), "feature_name_group": gname, "trigger_value": z,
                            "magnitude_sigma": magnitude})
    return TriggerSpec(feature_indices=feature_indices,
                       feature_names=[NAMES[i] for i in feature_indices],
                       trigger_map=trigger_map, trigger_details=details, feature_scores=None)

HEADER = [n for n in base.flow_header_features if n in NAMES]
TEMPORAL = [n for n in base.flow_temporal_features if n in NAMES]
DECLARED = set(HEADER) | set(TEMPORAL)
UNMODELLED = [n for n in NAMES if n not in DECLARED]

# For the >2-subspace variant we cannot simply use features outside the defence's
# declared subspaces: those two lists already cover 73 of 74 CIC-IDS2017 features.
# Instead we split the attack across THREE semantic groups while the defence still
# routes only two, which is the assumption B2 actually questions.
#   SIZE  -- packet-length / byte-volume features (a header subset)
#   FLAGS -- discrete flag counts + the one unmodelled feature (Destination Port)
#   TEMPORAL -- unchanged
_SIZE_KEYS = ("Length", "Packet Size", "Segment Size", "bytes", "Bytes")
SIZE = [n for n in HEADER if any(k in n for k in _SIZE_KEYS)]
FLAGS = [n for n in HEADER if "Flag" in n]                        # categorical-only
FLAGS3 = FLAGS + UNMODELLED
FWD = [n for n in NAMES if n.startswith("Fwd") or "Forward" in n] # alt partition
BWD = [n for n in NAMES if n.startswith("Bwd") or "Backward" in n]

print(f"header={len(HEADER)} temporal={len(TEMPORAL)} unmodelled={len(UNMODELLED)} "
      f"size={len(SIZE)} flags={len(FLAGS)} fwd={len(FWD)} bwd={len(BWD)}", flush=True)
print(f"unmodelled: {UNMODELLED}", flush=True)

comp_std = make_composite_trigger_spec(data, base)
TRIPROBE = dict(use_lockdown=True, apply_cf=True, use_flow_aware_masks=True,
                use_conditional_cf=True, use_head_aware_masks=True)

# variant -> (cfg, trigger_spec, extra run kwargs)
VARIANTS = {
    "lowrate": (dataclasses.replace(base, poison_ratio=0.05), comp_std, {}),
    "three_subspace": (base, build_trigger(
        {"size": SIZE, "flags": FLAGS3, "temporal": TEMPORAL}, base, per_group=3), {}),
    "alt_partition": (base, build_trigger({"fwd": FWD, "bwd": BWD}, base, per_group=5), {}),
    "flag_only": (base, build_trigger({"flags": FLAGS}, base, per_group=6), {}),
    "or_logic": (dataclasses.replace(base, split_trigger_attack=True), comp_std, {}),
}
for k, (_, ts, _) in VARIANTS.items():
    print(f"  [{k}] {len(ts.trigger_map)} trigger features: {ts.feature_names}", flush=True)

results = json.load(open(OUT)) if os.path.exists(OUT) else {}
results.setdefault("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
results.setdefault("runs", {})
results["trigger_features"] = {k: ts.feature_names for k, (_, ts, _) in VARIANTS.items()}
t0 = time.time()

for vname, (cfg, ts, extra) in VARIANTS.items():
    # undefended arm (seed 42 only) — establishes that the variant is a real backdoor
    tag = f"{vname}__nodef_s42"
    if tag not in results["runs"]:
        set_seed(42)
        r = run_federated_stage(tag, cfg, scenario, trigger_spec_override=ts,
                                use_lockdown=False, apply_cf=False, **extra)
        results["runs"][tag] = {"variant": vname, "arm": "nodef", "seed": 42,
                                "benign": r.final_benign_acc, "asr": r.final_asr}
        json.dump(results, open(OUT, "w"), indent=2)
        print(f"  {tag:32s} benign={r.final_benign_acc:.4f} asr={r.final_asr:.4f} "
              f"[{(time.time()-t0)/60:.1f}min]", flush=True)
    # TriProbe arm, 3 seeds
    for seed in SEEDS:
        tag = f"{vname}__triprobe_s{seed}"
        if tag in results["runs"]:
            print(f"[skip] {tag}"); continue
        set_seed(seed)
        r = run_federated_stage(tag, cfg, scenario, trigger_spec_override=ts,
                                **TRIPROBE, **extra)
        results["runs"][tag] = {"variant": vname, "arm": "triprobe", "seed": seed,
                                "benign": r.final_benign_acc, "asr": r.final_asr}
        json.dump(results, open(OUT, "w"), indent=2)
        print(f"  {tag:32s} benign={r.final_benign_acc:.4f} asr={r.final_asr:.4f} "
              f"[{(time.time()-t0)/60:.1f}min]", flush=True)

print("\n" + "=" * 70)
print(f"{'variant':16s} {'nodef ASR':>10s} {'TriProbe ASR (3 seeds)':>26s}")
for vname in VARIANTS:
    nd = results["runs"].get(f"{vname}__nodef_s42", {}).get("asr")
    tp = [v["asr"] * 100 for k, v in results["runs"].items()
          if v.get("variant") == vname and v.get("arm") == "triprobe"]
    nd_s = f"{nd*100:.2f}%" if nd is not None else "--"
    tp_s = f"{mean(tp):.3f}+/-{pstdev(tp):.3f}% (max {max(tp):.3f})" if tp else "--"
    print(f"{vname:16s} {nd_s:>10s} {tp_s:>26s}")
print(f"-> {OUT}")
