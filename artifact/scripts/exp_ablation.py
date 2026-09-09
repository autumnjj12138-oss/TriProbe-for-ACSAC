"""Component ablation: which parts of TriProbe carry the defense.

Removes one mechanism at a time under the composite backdoor, five seeds each:
triggered-probe filtering, the hard mask-density cap, Layer-1 cross-subspace
blocking, and end-of-training consensus fusion.

Produces reference_outputs/ablation_5seed_results.json (claim 2).
"""
import os, json, dataclasses, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from statistics import mean, pstdev
from flow_defense.attack import make_composite_trigger_spec
from flow_defense.config import make_cicids2017_config
from flow_defense.data import set_seed
from flow_defense.runner import build_experiment_scenario, run_federated_stage

SEEDS = [42, 123, 3407, 2025, 666]
RESULTS_PATH = "outputs/ablation_5seed_results.json"
os.makedirs("outputs", exist_ok=True)

base_cfg = make_cicids2017_config(rounds=30, max_train_samples=200_000,
                                  run_layerwise_pruning_scan=False)
TP = dict(use_lockdown=True, apply_cf=True, use_flow_aware_masks=True,
          use_conditional_cf=True, use_head_aware_masks=True)

# (name, config overrides, run-kwarg overrides)
ABLATIONS = [
    ("full",            {},                          {}),
    ("no_ASF",          {"asf_enable": False},       {}),
    ("no_density_cap",  {"mask_density_cap": None},  {}),
    ("no_L1_hard",      {"layer1_hard_block": False},{}),
    ("no_final_CF",     {},                          {"apply_cf": False}),
]

set_seed(base_cfg.scenario_seed)
scenario = build_experiment_scenario(base_cfg)
comp = make_composite_trigger_spec(scenario.data, base_cfg)
print(f"malicious_ids = {sorted(scenario.malicious_ids)}  ablations={[n for n,_,_ in ABLATIONS]}")

t0 = time.time()
results = {}  # name -> {seed -> {benign, asr, macro_f1}}
if os.path.exists(RESULTS_PATH):  # resume: keep completed (name, seed) entries
    results = json.load(open(RESULTS_PATH)).get("results", {})
    print(f"[resume] loaded {sum(len(v) for v in results.values())} completed runs")
for name, cfg_ov, kw_ov in ABLATIONS:
    results.setdefault(name, {})
    cfg = dataclasses.replace(base_cfg, **cfg_ov) if cfg_ov else base_cfg
    kw = dict(TP); kw.update(kw_ov)
    for seed in SEEDS:
        if str(seed) in results[name]:
            print(f"  [skip] {name} s{seed} (done)"); continue
        set_seed(seed)
        r = run_federated_stage(f"{name} s{seed}", cfg, scenario,
                                trigger_spec_override=comp, **kw)
        results[name][str(seed)] = {"benign": r.final_benign_acc, "asr": r.final_asr,
                                    "macro_f1": r.final_benign_macro_f1}
        json.dump({"seeds": SEEDS, "results": results}, open(RESULTS_PATH, "w"), indent=2)
        print(f"  {name:16s} s{seed}: benign={r.final_benign_acc:.4f} asr={r.final_asr:.4f} "
              f"[{(time.time()-t0)/60:.1f}min]")

full_mean = mean([results["full"][str(s)]["asr"] for s in SEEDS])
print("\n" + "=" * 66)
print(f"{'Ablation':16s} {'ASR(mean±std)':>20s} {'Δ vs full':>12s}")
for name, _, _ in ABLATIONS:
    a = [results[name][str(s)]["asr"] for s in SEEDS]
    dpp = (mean(a) - full_mean) * 100
    print(f"{name:16s} {mean(a)*100:7.2f}±{pstdev(a)*100:.2f}%   {dpp:+8.2f} pp")
print(f"Total: {(time.time()-t0)/60:.1f} min -> {RESULTS_PATH}")
