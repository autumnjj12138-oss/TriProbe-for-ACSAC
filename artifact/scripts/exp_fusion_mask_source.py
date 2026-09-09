"""Which client masks should vote in the final consensus fusion?

The filter removes suspicious clients before each aggregation round, but the
end-of-training consensus mask is a separate decision: it can be built from all
clients, from only those retained in the last round, or from those retained in a
majority of rounds. This runs all three with the quorum rescaled to the number
of voting masks.

All three give the same composite ASR, which is consistent with the ablation:
removing fusion entirely costs about 0.04 percentage points, so changing only
its input changes less.

Also records the filter's per-round client-level confusion -- how many of the
dropped clients were genuine attackers and how many were benign.

Produces reference_outputs/acsac_verify_b.json.
"""
import os, io, json, time, re, contextlib, dataclasses
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
from flow_defense.attack import make_composite_trigger_spec
from flow_defense.config import make_cicids2017_config
from flow_defense.data import set_seed
from flow_defense.runner import build_experiment_scenario, run_federated_stage

OUT = "outputs/acsac_verify_b.json"
os.makedirs("outputs", exist_ok=True)
SEEDS = [42, 123, 3407]

base = make_cicids2017_config(rounds=30, max_train_samples=200_000,
                              run_layerwise_pruning_scan=False)
set_seed(base.scenario_seed)
scenario = build_experiment_scenario(base)
comp = make_composite_trigger_spec(scenario.data, base)
MAL = sorted(scenario.malicious_ids)

TRIPROBE = dict(use_lockdown=True, apply_cf=True, use_flow_aware_masks=True,
                use_conditional_cf=True, use_head_aware_masks=True)
RE_DROP = re.compile(r"dropped_ids=\[([\d,\s]*)\]")

results = json.load(open(OUT)) if os.path.exists(OUT) else {}
results.setdefault("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
results.setdefault("malicious_ids", MAL)
results.setdefault("runs", {})

t0 = time.time()
for source in ["all", "last", "majority"]:
    cfg = dataclasses.replace(base, cf_mask_source=source)
    for seed in SEEDS:
        tag = f"{source}_s{seed}"
        if tag in results["runs"]:
            print(f"[skip] {tag}"); continue
        set_seed(seed)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = run_federated_stage(tag, cfg, scenario, trigger_spec_override=comp, **TRIPROBE)
        txt = buf.getvalue()
        drops = [[int(x) for x in m.split(",") if x.strip()] for m in RE_DROP.findall(txt)]
        # Filter's client-level confusion over all rounds: how many dropped
        # clients were genuine attackers and how many were benign.
        tp = sum(len([d for d in dr if d in MAL]) for dr in drops)
        fp = sum(len([d for d in dr if d not in MAL]) for dr in drops)
        n_mal_rounds = len(drops) * len(MAL)
        results["runs"][tag] = {
            "source": source, "seed": seed,
            "benign": r.final_benign_acc, "asr": r.final_asr,
            "asf_rounds": len(drops),
            "asf_recall_malicious": (tp / n_mal_rounds) if n_mal_rounds else None,
            "asf_benign_dropped_per_round": (fp / len(drops)) if drops else None,
            "asf_drop_rounds": drops,
        }
        json.dump(results, open(OUT, "w"), indent=2)
        print(f"  {tag:16s} benign={r.final_benign_acc:.4f} asr={r.final_asr:.4f} "
              f"ASF recall={results['runs'][tag]['asf_recall_malicious']} "
              f"[{(time.time()-t0)/60:.1f}min]", flush=True)

print("\n" + "=" * 70)
from statistics import mean, pstdev
for source in ["all", "last", "majority"]:
    rows = [v for v in results["runs"].values() if v["source"] == source]
    if not rows: continue
    a = [v["asr"] * 100 for v in rows]; b = [v["benign"] * 100 for v in rows]
    print(f"{source:9s} ASR {mean(a):6.3f}+/-{pstdev(a):5.3f}   benign {mean(b):6.2f}+/-{pstdev(b):4.2f}  (n={len(rows)})")
print(f"-> {OUT}")
