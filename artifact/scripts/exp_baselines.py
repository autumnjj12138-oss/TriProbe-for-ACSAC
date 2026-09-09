"""Baseline comparison: published FL backdoor defenses under the composite backdoor.

Runs FedAvg, FedMedian, Krum, FLAME, RLR, DeepSight, Lockdown and Flow-Aware
Lockdown against the cross-subspace composite trigger, five seeds each, with a
per-variant configuration so that aggregation weighting is not a confound.

Produces reference_outputs/baseline_5seed_results.json (claim 1).
"""
import os, json, dataclasses, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from statistics import mean, pstdev
from flow_defense.attack import make_composite_trigger_spec
from flow_defense.config import make_cicids2017_config
from flow_defense.data import set_seed
from flow_defense.runner import build_experiment_scenario, run_federated_stage

SEEDS = [42, 123, 3407, 2025, 666]
RESULTS_PATH = "outputs/baseline_5seed_results.json"
os.makedirs("outputs", exist_ok=True)

base_cfg = make_cicids2017_config(rounds=30, max_train_samples=200_000,
                                  run_layerwise_pruning_scan=False)
clean_cfg = dataclasses.replace(base_cfg, fc1_subspace_gate=False, mask_density_cap=None,
                                midround_cf_enable=False, layer1_hard_block=False,
                                asf_enable=False, fc1_anti_and_lambda=0.0)
deepsight_cfg = dataclasses.replace(clean_cfg, deepsight_enable=True)
TP = dict(use_lockdown=True, apply_cf=True, use_flow_aware_masks=True,
          use_conditional_cf=True, use_head_aware_masks=True)

# (name, cfg, run kwargs) — all composite trigger
VARIANTS = [
    ("FedAvg",             clean_cfg,     dict(use_lockdown=False, apply_cf=False)),
    ("FedMedian",          clean_cfg,     dict(use_lockdown=False, apply_cf=False, aggregator_name="fedmedian")),
    ("Krum",               clean_cfg,     dict(use_lockdown=False, apply_cf=False, aggregator_name="krum")),
    ("FLAME",              clean_cfg,     dict(use_lockdown=False, apply_cf=False, aggregator_name="flame")),
    ("RLR",                clean_cfg,     dict(use_lockdown=False, apply_cf=False, aggregator_name="rlr")),
    ("DeepSight",          deepsight_cfg, dict(use_lockdown=False, apply_cf=False, aggregator_name="fedavg")),
    ("Lockdown_native",    clean_cfg,     dict(use_lockdown=True, apply_cf=True)),
    ("Lockdown_flowaware", clean_cfg,     dict(use_lockdown=True, apply_cf=True,
                                               use_flow_aware_masks=True, use_conditional_cf=True)),
    ("TriProbe",           base_cfg,      TP),
]

set_seed(base_cfg.scenario_seed)
scenario = build_experiment_scenario(base_cfg)
comp = make_composite_trigger_spec(scenario.data, base_cfg)
print(f"malicious_ids = {sorted(scenario.malicious_ids)}  variants={[n for n,_,_ in VARIANTS]}")

t0 = time.time()
results = {}  # name -> {seed -> {benign, asr, macro_f1}}
if os.path.exists(RESULTS_PATH):  # resume: keep completed (name, seed) entries
    results = json.load(open(RESULTS_PATH)).get("results", {})
    print(f"[resume] loaded {sum(len(v) for v in results.values())} completed runs")
for name, cfg, kw in VARIANTS:
    results.setdefault(name, {})
    for seed in SEEDS:
        if str(seed) in results[name]:
            print(f"  [skip] {name} s{seed} (done)"); continue
        set_seed(seed)
        r = run_federated_stage(f"{name} s{seed}", cfg, scenario,
                                trigger_spec_override=comp, **kw)
        results[name][str(seed)] = {"benign": r.final_benign_acc, "asr": r.final_asr,
                                    "macro_f1": r.final_benign_macro_f1}
        json.dump({"seeds": SEEDS, "results": results}, open(RESULTS_PATH, "w"), indent=2)
        print(f"  {name:18s} s{seed}: benign={r.final_benign_acc:.4f} asr={r.final_asr:.4f} "
              f"[{(time.time()-t0)/60:.1f}min]")

print("\n" + "=" * 66)
print(f"{'Variant':18s} {'Benign(mean±std)':>20s} {'ASR(mean±std)':>20s}")
for name, _, _ in VARIANTS:
    b = [results[name][str(s)]["benign"] for s in SEEDS]
    a = [results[name][str(s)]["asr"] for s in SEEDS]
    print(f"{name:18s} {mean(b)*100:7.2f}±{pstdev(b)*100:.2f}%   {mean(a)*100:7.2f}±{pstdev(a)*100:.2f}%")
print(f"Total: {(time.time()-t0)/60:.1f} min -> {RESULTS_PATH}")
