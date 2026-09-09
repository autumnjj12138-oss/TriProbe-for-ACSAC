"""Can the UNSW-NB15 false-alarm rate be brought down by configuration?

On UNSW-NB15 the defense raises the false positive rate from 7.84% to 21.50%
(five seeds, sd 0.5, so this is systematic rather than noise). In absolute terms
about 4.4k false alarms become about 12.0k out of 56k benign test flows, buying
roughly 2.9k additional detections; attack F1 falls from 96.58 to 94.83, so the
trade is net-negative on a balanced metric.

Which knob should matter: sparsity 0.78 sets the initial density to 0.22, but the
mask density cap of 0.14 then caps it, so the cap rather than sparsity is the
binding capacity constraint. The other lever is the drop quantile: at 0.35 the
server discards seven of twenty clients while only four are malicious, so three
honest clients are thrown away every round. Both are swept here. The density cap
has a cliff between 0.16 and 0.18 on CIC-IDS2017; whether UNSW shares it is not
known, so 0.16 is included as one step up.

Grid: q in {0.20, 0.25, 0.35} x cap in {0.14, 0.16}, three seeds, plus a control
at the shipped setting.

Produces reference_outputs/acsac_unsw_fpr.json.
"""
import os, json, time, dataclasses
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
from statistics import mean, pstdev
from flow_defense.attack import make_composite_trigger_spec
from flow_defense.config import make_unsw_config
from flow_defense.data import set_seed
from flow_defense.runner import build_experiment_scenario, run_federated_stage

OUT = "outputs/acsac_unsw_fpr.json"
SEEDS = [42, 123, 3407]
QS = [0.20, 0.25, 0.35]
CAPS = [0.14, 0.16]
base = make_unsw_config(rounds=30, run_layerwise_pruning_scan=False)
set_seed(base.scenario_seed)
scenario = build_experiment_scenario(base)
comp = make_composite_trigger_spec(scenario.data, base)
TRIPROBE = dict(use_lockdown=True, apply_cf=True, use_flow_aware_masks=True,
                use_conditional_cf=True, use_head_aware_masks=True)

results = json.load(open(OUT)) if os.path.exists(OUT) else {}
results.setdefault("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
results.setdefault("baseline_fedavg", {"dr": 0.9682, "fpr": 0.0784, "note": "5-seed means from security_metrics_results.json"})
results.setdefault("runs", {})
t0 = time.time()

# undefended reference on this scenario, one seed, for an apples-to-apples FPR
for seed in [42]:
    tag = f"nodef_s{seed}"
    if tag not in results["runs"]:
        set_seed(seed)
        r = run_federated_stage(tag, base, scenario, trigger_spec_override=comp,
                                use_lockdown=False, apply_cf=False)
        results["runs"][tag] = {"arm": "nodef", "q": None, "cap": None, "seed": seed,
                                "asr": r.final_asr, "benign": r.final_benign_acc,
                                "dr": r.final_dr, "fpr": r.final_fpr,
                                "attack_f1": r.final_attack_f1}
        json.dump(results, open(OUT, "w"), indent=2)
        print(f"  {tag:22s} asr={r.final_asr*100:6.2f}% fpr={r.final_fpr*100:6.2f}% "
              f"dr={r.final_dr*100:6.2f}% f1={r.final_attack_f1*100:6.2f} "
              f"[{(time.time()-t0)/60:.1f}min]", flush=True)

for cap in CAPS:
    for q in QS:
        cfg = dataclasses.replace(base, asf_drop_quantile=q, mask_density_cap=cap)
        for seed in SEEDS:
            tag = f"q{q:g}_cap{cap:g}_s{seed}"
            if tag in results["runs"]:
                print(f"[skip] {tag}", flush=True); continue
            set_seed(seed)
            r = run_federated_stage(tag, cfg, scenario, trigger_spec_override=comp, **TRIPROBE)
            results["runs"][tag] = {"arm": "triprobe", "q": q, "cap": cap, "seed": seed,
                                    "asr": r.final_asr, "benign": r.final_benign_acc,
                                    "dr": r.final_dr, "fpr": r.final_fpr,
                                    "attack_f1": r.final_attack_f1}
            json.dump(results, open(OUT, "w"), indent=2)
            print(f"  {tag:22s} asr={r.final_asr*100:6.2f}% fpr={r.final_fpr*100:6.2f}% "
                  f"dr={r.final_dr*100:6.2f}% f1={r.final_attack_f1*100:6.2f} "
                  f"[{(time.time()-t0)/60:.1f}min]", flush=True)

print("\n" + "=" * 78)
print(f"{'config':16s} {'ASR':>8s} {'FPR':>8s} {'DR':>8s} {'atkF1':>8s}   usable (ASR<5, FPR<10)")
for cap in CAPS:
    for q in QS:
        rows = [v for v in results["runs"].values()
                if v["arm"] == "triprobe" and v["q"] == q and v["cap"] == cap]
        if not rows: continue
        a = mean([v["asr"] for v in rows]) * 100
        f = mean([v["fpr"] for v in rows]) * 100
        d = mean([v["dr"] for v in rows]) * 100
        f1 = mean([v["attack_f1"] for v in rows]) * 100
        ok = "yes" if (a < 5 and f < 10) else ""
        print(f"q={q:g} cap={cap:g}   {a:7.2f}% {f:7.2f}% {d:7.2f}% {f1:7.2f}   {ok}")
print(f"-> {OUT}")
