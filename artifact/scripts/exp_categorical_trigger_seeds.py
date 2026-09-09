"""How often does the categorical-only trigger actually fail?

Across the paper's five seeds this variant is stable on four and breaches the
deployment threshold on one, and that seed reproduces its high value under four
different clipping configurations. Five seeds is far too few to estimate how
often it happens, so this adds ten more. It does not replace any existing seed:
the frequency is reported over all fifteen.

Worth knowing when reading the output: build_experiment_scenario runs once under
scenario_seed, so the Dirichlet partition and the identity of the four attackers
are identical across every seed here. Only training randomness varies. Whatever
makes a seed fail is therefore a training-dynamics effect, not an unlucky data
split or an unlucky attacker placement.

Captures the ASR trajectory so a failing run can be located in time.

Produces reference_outputs/categorical_trigger_seeds.json.
"""
import os, json, time, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np, torch
from statistics import mean, pstdev
from flow_defense.attack import get_feature_constraint_range
from flow_defense.config import TriggerSpec, make_cicids2017_config
from flow_defense.data import set_seed
from flow_defense.runner import build_experiment_scenario, run_federated_stage

OUT = "outputs/categorical_trigger_seeds.json"
NEW_SEEDS = [7, 555, 999, 1024, 1337, 2024, 8888, 31337, 20260825, 606]
PAPER_SEEDS = {"42": 0.241, "123": 8.071, "3407": 0.178, "2025": 0.190, "666": 0.241}

base = make_cicids2017_config(rounds=30, max_train_samples=200_000,
                              run_layerwise_pruning_scan=False)
set_seed(base.scenario_seed)
scenario = build_experiment_scenario(base)
data = scenario.data
NAMES = list(data.feature_names)
TRIPROBE = dict(use_lockdown=True, apply_cf=True, use_flow_aware_masks=True,
                use_conditional_cf=True, use_head_aware_masks=True)

# rebuild the categorical-only trigger exactly as exp_attack_variants does
atk, ben = data.y_train == 1, data.y_train == 0
ranked = []
for n in [h for h in base.flow_header_features if h in NAMES and "Flag" in h]:
    i = NAMES.index(n)
    av, bv = data.X_train[atk, i], data.X_train[ben, i]
    ranked.append((abs(float(av.mean()) - float(bv.mean())) / (float(data.X_train[:, i].std()) + 1e-6), n, i))
ranked.sort(key=lambda t: t[0], reverse=True)
idxs, tmap, det = [], {}, []
for _, n, i in ranked[:6]:
    av, bv = data.X_train[atk, i], data.X_train[ben, i]
    z = base.composite_zscore_magnitude if float(av.mean()) >= float(bv.mean()) else -base.composite_zscore_magnitude
    lo, hi = get_feature_constraint_range(n, base)
    z = float(np.clip(z, lo, hi)); idxs.append(i); tmap[i] = z
    det.append({"index": float(i), "feature_name_group": "flags", "trigger_value": z})
flag_ts = TriggerSpec(idxs, [NAMES[i] for i in idxs], tmap, det, None)
print(f"trigger={flag_ts.feature_names}", flush=True)
print(f"attackers (fixed across all seeds) = {sorted(scenario.malicious_ids)}", flush=True)

results = json.load(open(OUT)) if os.path.exists(OUT) else {}
results.setdefault("paper_seeds", PAPER_SEEDS)
results.setdefault("note", "new seeds ADD to the paper's five; none are replaced")
results.setdefault("runs", {})
t0 = time.time()

for seed in NEW_SEEDS:
    tag = str(seed)
    if tag in results["runs"]:
        print(f"[skip] {tag}", flush=True); continue
    set_seed(seed)
    r = run_federated_stage(f"flagonly_s{seed}", base, scenario,
                            trigger_spec_override=flag_ts, **TRIPROBE)
    traj = [(row["round"], row["asr"]) for row in (r.round_rows or [])
            if row.get("asr") == row.get("asr")]
    results["runs"][tag] = {"seed": seed, "asr": r.final_asr,
                            "benign": r.final_benign_acc, "asr_trajectory": traj}
    json.dump(results, open(OUT, "w"), indent=2)
    print(f"  seed {seed:<9} asr={r.final_asr*100:7.3f}%  benign={r.final_benign_acc*100:.2f}%  "
          f"[{(time.time()-t0)/60:.1f}min]", flush=True)

print("\n" + "=" * 66)
allv = {**{k: v for k, v in PAPER_SEEDS.items()},
        **{k: v["asr"] * 100 for k, v in results["runs"].items()}}
bad = {k: v for k, v in allv.items() if v > 5}
print(f"all {len(allv)} seeds, sorted: {sorted(round(v,3) for v in allv.values())}")
print(f"above the 5% threshold: {len(bad)}/{len(allv)}  -> {bad}")
print(f"median {sorted(allv.values())[len(allv)//2]:.3f}%   mean {mean(allv.values()):.3f}%")
print(f"-> {OUT}")
