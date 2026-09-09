"""Does combining TriProbe with update-norm clipping help under scaling?

Arms, all under the composite backdoor with 5x adaptive scaling:
  nodef          FedAvg, no defense, the attack ceiling
  triprobe       TriProbe alone
  triprobe_clip  TriProbe plus median update-norm clipping
  clip_only      clipping alone with the probe filter disabled, which isolates
                 how much of any gain belongs to the clip rather than the filter

Produces reference_outputs/acsac_a2_normclip.json (claim 5).
"""
import os, json, time, dataclasses
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
from statistics import mean, pstdev
from flow_defense.attack import make_composite_trigger_spec
from flow_defense.config import make_cicids2017_config
from flow_defense.data import set_seed
from flow_defense.runner import build_experiment_scenario, run_federated_stage

OUT = "outputs/acsac_a2_normclip.json"
os.makedirs("outputs", exist_ok=True)
SEEDS = [42, 123, 3407]
CLIP_FACTOR = 1.0   # bound = 1.0 x median(update norm)

base = make_cicids2017_config(rounds=30, max_train_samples=200_000,
                              run_layerwise_pruning_scan=False)
set_seed(base.scenario_seed)
scenario = build_experiment_scenario(base)
comp = make_composite_trigger_spec(scenario.data, base)

TRIPROBE = dict(use_lockdown=True, apply_cf=True, use_flow_aware_masks=True,
                use_conditional_cf=True, use_head_aware_masks=True)

ARMS = {
    "nodef":         (base, dict(use_lockdown=False, apply_cf=False, use_adaptive_attack=True)),
    "triprobe":      (base, dict(use_adaptive_attack=True, **TRIPROBE)),
    "triprobe_clip": (dataclasses.replace(base, update_norm_clip=True,
                                          update_norm_clip_factor=CLIP_FACTOR),
                      dict(use_adaptive_attack=True, **TRIPROBE)),
    "clip_only":     (dataclasses.replace(base, update_norm_clip=True,
                                          update_norm_clip_factor=CLIP_FACTOR,
                                          asf_enable=False),
                      dict(use_adaptive_attack=True, **TRIPROBE)),
}

results = json.load(open(OUT)) if os.path.exists(OUT) else {}
results.setdefault("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
results.setdefault("clip_factor", CLIP_FACTOR)
results.setdefault("scale", base.adaptive_attack_scale)
results.setdefault("runs", {})
t0 = time.time()

for arm, (cfg, kw) in ARMS.items():
    results["runs"].setdefault(arm, {})
    for seed in SEEDS:
        if str(seed) in results["runs"][arm]:
            print(f"[skip] {arm} s{seed}"); continue
        set_seed(seed)
        r = run_federated_stage(f"{arm}_s{seed}", cfg, scenario,
                                trigger_spec_override=comp, **kw)
        results["runs"][arm][str(seed)] = {"benign": r.final_benign_acc, "asr": r.final_asr}
        json.dump(results, open(OUT, "w"), indent=2)
        print(f"  {arm:15s} s{seed}: benign={r.final_benign_acc:.4f} asr={r.final_asr:.4f} "
              f"[{(time.time()-t0)/60:.1f}min]", flush=True)

print("\n" + "=" * 70)
for arm in ARMS:
    v = results["runs"].get(arm, {})
    if not v: continue
    a = [x["asr"] * 100 for x in v.values()]; b = [x["benign"] * 100 for x in v.values()]
    print(f"{arm:15s} ASR {mean(a):7.3f}+/-{pstdev(a):6.3f}  max {max(a):7.3f}   benign {mean(b):6.2f}  (n={len(a)})")
print(f"-> {OUT}")
