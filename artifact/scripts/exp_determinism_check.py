"""Is this workload deterministic on a fixed machine?

cudnn.deterministic is not set, so in principle repeated runs could differ. This
repeats the same seed three times and compares the results bit for bit. It also
captures ASR both before and after final consensus fusion, so the two can be
told apart when reading any single number.

The answer on the reference machine is that runs are bit-identical: this model
is small matmuls with no convolutions and cudnn.benchmark is off, so no
non-deterministic kernel is reached. Seed-to-seed variation elsewhere is
therefore real variation and not run noise. Across different GPUs the results
are not identical; see use.txt.

Produces reference_outputs/acsac_verify_a.json.
"""
import os, io, json, time, re, contextlib
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
from flow_defense.attack import make_composite_trigger_spec
from flow_defense.config import make_cicids2017_config
from flow_defense.data import set_seed
from flow_defense.runner import build_experiment_scenario, run_federated_stage

OUT = "outputs/acsac_verify_a.json"
os.makedirs("outputs", exist_ok=True)

cfg = make_cicids2017_config(rounds=30, max_train_samples=200_000,
                             run_layerwise_pruning_scan=False)
set_seed(cfg.scenario_seed)
scenario = build_experiment_scenario(cfg)
comp = make_composite_trigger_spec(scenario.data, cfg)

TRIPROBE = dict(use_lockdown=True, apply_cf=True, use_flow_aware_masks=True,
                use_conditional_cf=True, use_head_aware_masks=True)

RE_BEFORE = re.compile(r"\[Final before CF\] benign_acc=([\d.]+)\s+ASR=([\d.]+)")
RE_AFTER  = re.compile(r"\[After CF\] benign_acc=([\d.]+)\s+ASR=([\d.]+)")
RE_DROP   = re.compile(r"dropped_ids=\[([\d,\s]*)\]")

def run(tag, seed, **kw):
    """Run one stage, capturing stdout so we can recover before/after-CF ASR."""
    set_seed(seed)
    buf = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(buf):
        r = run_federated_stage(tag, cfg, scenario, trigger_spec_override=comp, **kw)
    txt = buf.getvalue()
    print(txt[-1500:])            # echo tail so the log stays informative
    b = RE_BEFORE.search(txt); a = RE_AFTER.search(txt)
    drops = [[int(x) for x in m.split(",") if x.strip()] for m in RE_DROP.findall(txt)]
    rec = {
        "seed": seed,
        "asr_before_cf": float(b.group(2)) if b else None,
        "benign_before_cf": float(b.group(1)) if b else None,
        "asr_after_cf": float(a.group(2)) if a else None,
        "benign_after_cf": float(a.group(1)) if a else None,
        "final_asr_returned": r.final_asr,
        "final_benign_returned": r.final_benign_acc,
        "asf_drop_rounds": drops,
        "malicious_ids": sorted(scenario.malicious_ids),
        "minutes": round((time.time() - t0) / 60, 2),
    }
    print(f">>> {tag}: before_CF={rec['asr_before_cf']} after_CF={rec['asr_after_cf']} "
          f"returned={rec['final_asr_returned']:.4f} [{rec['minutes']}min]", flush=True)
    return rec

results = json.load(open(OUT)) if os.path.exists(OUT) else {}
results.setdefault("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
results.setdefault("runs", {})

PLAN = [
    ("adaptive_s42",      42, dict(use_adaptive_attack=True, **TRIPROBE)),
    ("composite_s42_r1",  42, dict(**TRIPROBE)),
    ("composite_s42_r2",  42, dict(**TRIPROBE)),
    ("composite_s42_r3",  42, dict(**TRIPROBE)),
]
for tag, seed, kw in PLAN:
    if tag in results["runs"]:
        print(f"[skip] {tag}"); continue
    results["runs"][tag] = run(tag, seed, **kw)
    json.dump(results, open(OUT, "w"), indent=2)

print("\n" + "=" * 70)
for tag, rec in results["runs"].items():
    print(f"{tag:20s} before_CF={rec['asr_before_cf']}  after_CF={rec['asr_after_cf']}")
print(f"-> {OUT}")
