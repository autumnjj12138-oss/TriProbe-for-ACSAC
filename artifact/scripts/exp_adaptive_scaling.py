"""Adaptive scaling: malicious updates amplified before upload.

Both arms -- undefended and TriProbe -- under the composite backdoor with the
malicious delta scaled by 5, five seeds, matching the main table's seed set.
This script uses the norm-bounded variant of the attack, in which the amplified
delta is clipped to twice the median benign update norm. exp_unbounded_scaling.py
covers the unbounded setting.

Produces reference_outputs/adaptive_5seed_results.json (claim 5).
"""
import os, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from statistics import mean, pstdev
from flow_defense.attack import make_composite_trigger_spec
from flow_defense.config import make_cicids2017_config
from flow_defense.data import set_seed
from flow_defense.runner import build_experiment_scenario, run_federated_stage

SEEDS = [42, 123, 3407, 2025, 666]
RESULTS_PATH = "outputs/adaptive_5seed_results.json"
os.makedirs("outputs", exist_ok=True)

cfg = make_cicids2017_config(rounds=30, max_train_samples=200_000,
                             run_layerwise_pruning_scan=False)
STAGES = [
    ("M_adaptive", dict(use_lockdown=False, apply_cf=False, use_adaptive_attack=True)),
    ("O_adaptive", dict(use_lockdown=True, apply_cf=True, use_flow_aware_masks=True,
                        use_conditional_cf=True, use_head_aware_masks=True, use_adaptive_attack=True)),
]

set_seed(cfg.scenario_seed)
scenario = build_experiment_scenario(cfg)
comp = make_composite_trigger_spec(scenario.data, cfg)

# results: name -> {seed -> {benign, asr}}
results = {name: {} for name, _ in STAGES}
if os.path.exists(RESULTS_PATH):
    results = json.load(open(RESULTS_PATH)).get("results", results)
# pre-seed seed 42 from the prior single-seed run if not already present
old = "outputs/opus_adaptive_results.json"
if os.path.exists(old):
    o = json.load(open(old)).get("results", {})
    for name, _ in STAGES:
        if name in o and "42" not in results.get(name, {}):
            results.setdefault(name, {})["42"] = {"benign": o[name]["benign"], "asr": o[name]["asr"]}
    print("[resume] pre-seeded seed 42 from opus_adaptive_results.json")

t0 = time.time()
for name, kw in STAGES:
    results.setdefault(name, {})
    for seed in SEEDS:
        if str(seed) in results[name]:
            print(f"  [skip] {name} s{seed}"); continue
        set_seed(seed)
        r = run_federated_stage(f"{name} s{seed}", cfg, scenario,
                                trigger_spec_override=comp, **kw)
        results[name][str(seed)] = {"benign": r.final_benign_acc, "asr": r.final_asr}
        json.dump({"seeds": SEEDS, "scale": cfg.adaptive_attack_scale, "results": results},
                  open(RESULTS_PATH, "w"), indent=2)
        print(f"  {name:12s} s{seed}: benign={r.final_benign_acc:.4f} asr={r.final_asr:.4f} "
              f"[{(time.time()-t0)/60:.1f}min]")

print("\n" + "=" * 60)
print(f"{'Stage':12s} {'Benign(mean±std)':>20s} {'ASR(mean±std)':>20s} {'max':>8s}")
for name, _ in STAGES:
    a = [results[name][str(s)]["asr"] for s in SEEDS]
    b = [results[name][str(s)]["benign"] for s in SEEDS]
    print(f"{name:12s} {mean(b)*100:7.2f}±{pstdev(b)*100:.2f}%   {mean(a)*100:7.2f}±{pstdev(a)*100:.2f}%   {max(a)*100:6.2f}%")
print(f"Total: {(time.time()-t0)/60:.1f}min -> {RESULTS_PATH}")
