"""Hyperparameter sensitivity over multiple seeds.

Two sweeps that the paper originally reported at a single seed:
  q  the server's drop quantile, 0.15 to 0.40
  m  the number of malicious clients out of twenty, 2 to 10

Federated training varies enough across seeds that single-seed conclusions about
either sweep are not reliable, so both are run at three seeds and reported as
mean and standard deviation.

Produces reference_outputs/sensitivity_multiseed.json.
"""
import os, json, time, dataclasses
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
from statistics import mean, pstdev
from flow_defense.attack import make_composite_trigger_spec
from flow_defense.config import make_cicids2017_config
from flow_defense.data import set_seed
from flow_defense.runner import build_experiment_scenario, run_federated_stage

OUT = "outputs/sensitivity_multiseed.json"
os.makedirs("outputs", exist_ok=True)
SEEDS = [42, 123, 3407]
Q_VALUES = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
M_VALUES = [2, 4, 6, 7, 8, 10]

TRIPROBE = dict(use_lockdown=True, apply_cf=True, use_flow_aware_masks=True,
                use_conditional_cf=True, use_head_aware_masks=True)

base = make_cicids2017_config(rounds=30, max_train_samples=200_000,
                              run_layerwise_pruning_scan=False)

results = json.load(open(OUT)) if os.path.exists(OUT) else {}
results.setdefault("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
results.setdefault("seeds", SEEDS)
results.setdefault("q_scan", {})
results.setdefault("m_scan", {})
t0 = time.time()

def save():
    json.dump(results, open(OUT, "w"), indent=2)

# ---- Table 7: ASF drop quantile q (scenario fixed, only q and the training seed vary)
set_seed(base.scenario_seed)
scen_q = build_experiment_scenario(base)
comp_q = make_composite_trigger_spec(scen_q.data, base)
print(f"[q-scan] malicious clients = {sorted(scen_q.malicious_ids)} (n={base.malicious_clients})", flush=True)

for q in Q_VALUES:
    cfg = dataclasses.replace(base, asf_drop_quantile=q)
    key = f"{q:.2f}"
    results["q_scan"].setdefault(key, {})
    for seed in SEEDS:
        if str(seed) in results["q_scan"][key]:
            print(f"[skip] q={key} s{seed}"); continue
        set_seed(seed)
        r = run_federated_stage(f"q{key}_s{seed}", cfg, scen_q,
                                trigger_spec_override=comp_q, **TRIPROBE)
        results["q_scan"][key][str(seed)] = {"benign": r.final_benign_acc, "asr": r.final_asr}
        save()
        print(f"  q={key} s{seed}: benign={r.final_benign_acc:.4f} asr={r.final_asr:.4f} "
              f"[{(time.time()-t0)/60:.1f}min]", flush=True)

# ---- Table 10: malicious-client ratio (scenario must be rebuilt per m)
for m in M_VALUES:
    cfg = dataclasses.replace(base, malicious_clients=m)
    key = str(m)
    results["m_scan"].setdefault(key, {})
    set_seed(cfg.scenario_seed)
    scen_m = build_experiment_scenario(cfg)
    comp_m = make_composite_trigger_spec(scen_m.data, cfg)
    for seed in SEEDS:
        if str(seed) in results["m_scan"][key]:
            print(f"[skip] m={key} s{seed}"); continue
        set_seed(seed)
        r = run_federated_stage(f"m{key}_s{seed}", cfg, scen_m,
                                trigger_spec_override=comp_m, **TRIPROBE)
        results["m_scan"][key][str(seed)] = {"benign": r.final_benign_acc, "asr": r.final_asr,
                                             "malicious_ids": sorted(scen_m.malicious_ids)}
        save()
        print(f"  m={key} s{seed}: benign={r.final_benign_acc:.4f} asr={r.final_asr:.4f} "
              f"[{(time.time()-t0)/60:.1f}min]", flush=True)

print("\n" + "=" * 70)
for label, block in (("q", results["q_scan"]), ("m", results["m_scan"])):
    print(f"--- {label} scan ---")
    for k, v in block.items():
        a = [x["asr"] * 100 for x in v.values()]
        b = [x["benign"] * 100 for x in v.values()]
        if a:
            print(f"  {label}={k:5s} ASR {mean(a):7.3f}+/-{pstdev(a):6.3f}  max {max(a):7.3f}   benign {mean(b):6.2f}  (n={len(a)})")
print(f"-> {OUT}")
