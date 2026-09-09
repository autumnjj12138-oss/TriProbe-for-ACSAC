"""Scaling without the norm bound.

The amplified malicious delta can either be clipped to a multiple of the median
benign update norm, which is the norm-bounded model-replacement attack of
Bagdasaryan et al., or left unbounded. The bound matters: at 5x it binds only
part of the time, so 5x, 10x and 20x differ, while at 10x and above the clip
saturates and the scale factor stops changing anything.

Arms, five seeds unless noted:
  bounded_nodef / bounded_triprobe      seed 42 only, controls that must
                                        reproduce the bounded numbers and so
                                        show the norm-bound switch is inert at
                                        its default
  unbounded_nodef / unbounded_triprobe  the unbounded setting

Also records the realised amplification, i.e. how hard the bound actually bites.

Produces reference_outputs/unbounded_scaling.json (claim 5).
"""
import os, io, re, json, time, contextlib, dataclasses
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np, torch
from statistics import mean, pstdev
from flow_defense.attack import make_composite_trigger_spec
from flow_defense.config import make_cicids2017_config
from flow_defense.data import set_seed
from flow_defense.runner import build_experiment_scenario, run_federated_stage

OUT = "outputs/unbounded_scaling.json"
SEEDS = [42, 123, 3407, 2025, 666]
base = make_cicids2017_config(rounds=30, max_train_samples=200_000,
                              run_layerwise_pruning_scan=False)
set_seed(base.scenario_seed)
scenario = build_experiment_scenario(base)
comp = make_composite_trigger_spec(scenario.data, base)
TRIPROBE = dict(use_lockdown=True, apply_cf=True, use_flow_aware_masks=True,
                use_conditional_cf=True, use_head_aware_masks=True)
NODEF = dict(use_lockdown=False, apply_cf=False)
RE_SCALE = re.compile(r"realised=([\d.]+)x\s+clipped=(\d+)/(\d+)")

# (tag, norm_bound_factor, stage kwargs, seeds)
ARMS = [
    ("bounded_nodef",      2.0,  NODEF,    [42]),
    ("bounded_triprobe",   2.0,  TRIPROBE, [42]),
    ("unbounded_nodef",    None, NODEF,    SEEDS),
    ("unbounded_triprobe", None, TRIPROBE, SEEDS),
]

results = json.load(open(OUT)) if os.path.exists(OUT) else {}
results.setdefault("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
results.setdefault("malicious_ids", sorted(scenario.malicious_ids))
results.setdefault("runs", {})
t0 = time.time()

for arm, bound, kw, seeds in ARMS:
    cfg = dataclasses.replace(base, adaptive_norm_bound_factor=bound)
    for seed in seeds:
        tag = f"{arm}_s{seed}"
        if tag in results["runs"]:
            print(f"[skip] {tag}", flush=True); continue
        set_seed(seed)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = run_federated_stage(tag, cfg, scenario, trigger_spec_override=comp,
                                    use_adaptive_attack=True, **kw)
        hits = RE_SCALE.findall(buf.getvalue())
        realised = [float(h[0]) for h in hits]
        clipped = sum(int(h[1]) for h in hits)
        total = sum(int(h[2]) for h in hits)
        results["runs"][tag] = {
            "arm": arm, "seed": seed, "norm_bound_factor": bound,
            "benign": r.final_benign_acc, "asr": r.final_asr,
            "realised_amplification_mean": (mean(realised) if realised else None),
            "clipped_fraction": (clipped / total if total else None),
        }
        json.dump(results, open(OUT, "w"), indent=2)
        print(f"  {tag:24s} asr={r.final_asr*100:7.3f}%  benign={r.final_benign_acc*100:.2f}%  "
              f"realised={results['runs'][tag]['realised_amplification_mean']}x  "
              f"clipped={results['runs'][tag]['clipped_fraction']}  "
              f"[{(time.time()-t0)/60:.1f}min]", flush=True)

print("\n" + "=" * 74)
print("CONTROLS (must match the existing bounded numbers)")
for tag, exp in [("bounded_nodef_s42", 0.9347715736040609),
                 ("bounded_triprobe_s42", 0.005076142131979695)]:
    got = results["runs"].get(tag, {}).get("asr")
    if got is not None:
        print(f"  {tag:24s} got={got!r}  expected={exp!r}  match={got == exp}")
print("\nRESULTS")
for arm, _, _, _ in ARMS:
    a = [v["asr"] * 100 for v in results["runs"].values() if v["arm"] == arm]
    ramp = [v["realised_amplification_mean"] for v in results["runs"].values()
            if v["arm"] == arm and v["realised_amplification_mean"]]
    if a:
        print(f"  {arm:20s} ASR {mean(a):7.3f}+/-{pstdev(a):6.3f}%  max {max(a):7.3f}%  "
              f"realised {mean(ramp):.2f}x" if ramp else f"  {arm:20s} ASR {mean(a):7.3f}%")
print(f"-> {OUT}")
