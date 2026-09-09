"""Delayed and on-off backdoors.

A backdoor that stays dormant and activates late, or that toggles on and off, so
that on any given round the malicious client may look clean to the server probe.

Arms, composite trigger, full TriProbe, three seeds:
  baseline   poison every round from round 0
  delay15    poison only from round 15 of 30
  delay25    poison only from round 25 of 30
  onoff2     poison on every second round

Also records, per round, whether the filter dropped every attacker, so that a
late-activating backdoor can be checked for whether it is caught once it turns
on. The baseline arm doubles as a check that the poisoning gate is inert at its
default settings.

Produces reference_outputs/delayed_onoff.json (claim 5).
"""
import os, io, re, json, time, contextlib, dataclasses
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
from statistics import mean, pstdev
from flow_defense.attack import make_composite_trigger_spec
from flow_defense.config import make_cicids2017_config
from flow_defense.data import set_seed
from flow_defense.runner import build_experiment_scenario, run_federated_stage

OUT = "outputs/delayed_onoff.json"
SEEDS = [42, 123, 3407]
base = make_cicids2017_config(rounds=30, max_train_samples=200_000,
                              run_layerwise_pruning_scan=False)
set_seed(base.scenario_seed)
scenario = build_experiment_scenario(base)
comp = make_composite_trigger_spec(scenario.data, base)
MAL = set(scenario.malicious_ids)
TRIPROBE = dict(use_lockdown=True, apply_cf=True, use_flow_aware_masks=True,
                use_conditional_cf=True, use_head_aware_masks=True)
RE_DROP = re.compile(r"dropped_ids=\[([\d,\s]*)\]")

ARMS = {
    "baseline": dict(),
    "delay15":  dict(poison_start_round=15),
    "delay25":  dict(poison_start_round=25),
    "onoff2":   dict(poison_duty_cycle=2),
}

results = json.load(open(OUT)) if os.path.exists(OUT) else {}
results.setdefault("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
results.setdefault("malicious_ids", sorted(MAL))
results.setdefault("runs", {})
t0 = time.time()

for arm, over in ARMS.items():
    cfg = dataclasses.replace(base, **over) if over else base
    for seed in SEEDS:
        tag = f"{arm}_s{seed}"
        if tag in results["runs"]:
            print(f"[skip] {tag}", flush=True); continue
        set_seed(seed)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = run_federated_stage(tag, cfg, scenario, trigger_spec_override=comp, **TRIPROBE)
        drops = [set(int(x) for x in m.split(",") if x.strip())
                 for m in RE_DROP.findall(buf.getvalue())]
        missed = [sorted(MAL - d) for d in drops]
        start = int(over.get("poison_start_round", 0))
        after = [m for i, m in enumerate(missed) if i >= start]
        results["runs"][tag] = {
            "arm": arm, "seed": seed,
            "benign": r.final_benign_acc, "asr": r.final_asr,
            "rounds_missing_after_activation": sum(1 for m in after if m),
            "rounds_after_activation": len(after),
        }
        json.dump(results, open(OUT, "w"), indent=2)
        print(f"  {tag:16s} asr={r.final_asr*100:6.3f}%  benign={r.final_benign_acc*100:.2f}%  "
              f"missed_after_activation={sum(1 for m in after if m)}/{len(after)} "
              f"[{(time.time()-t0)/60:.1f}min]", flush=True)

print("\n" + "=" * 68)
for arm in ARMS:
    a = [v["asr"] * 100 for v in results["runs"].values() if v["arm"] == arm]
    if a:
        print(f"{arm:10s} ASR {mean(a):7.3f}+/-{pstdev(a):6.3f}%  max {max(a):7.3f}%  "
              f"vals={[round(x,3) for x in sorted(a)]}")
print(f"-> {OUT}")
