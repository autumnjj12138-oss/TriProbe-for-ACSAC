"""Probe-aware evasion: an attacker that trains against the server's filter.

The server scores each client by its target-class confidence on a synthetic
triggered probe. An attacker who knows that probe distribution can add a penalty
term that suppresses exactly this signal while still implanting the backdoor, so
the score never rises and the client is never filtered.

Arms:
  M_nodef / M_evasion          undefended, with and without the evasion penalty,
                               to confirm evasion costs the attacker nothing
  O_mismatch                   TriProbe against a probe that shares no features
                               with the real trigger but keeps the same
                               cross-subspace structure. Isolates adaptivity
                               from mere probe mismatch.
  O_evasion                    TriProbe against the probe-aware attacker
  O_evasion_randprobe          plus per-round randomised probes
  O_evasion_normclip           plus update-norm clipping
  O_evasion_full               plus both

This attacker assumes knowledge of the probe distribution, which is beyond the
threat model the paper defends. It is measured because it marks the boundary of
where the method applies.

Produces reference_outputs/evasion_attack_results.json (claim 6).
"""
import os, json, dataclasses, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from statistics import mean, pstdev
from flow_defense.attack import make_composite_trigger_spec
from flow_defense.config import make_cicids2017_config
from flow_defense.data import set_seed
from flow_defense.runner import build_experiment_scenario, run_federated_stage

SEEDS = [42, 123, 3407]
RESULTS = "outputs/evasion_attack_results.json"
os.makedirs("outputs", exist_ok=True)

base = make_cicids2017_config(rounds=30, max_train_samples=200_000, run_layerwise_pruning_scan=False)
TP = dict(use_lockdown=True, apply_cf=True, use_flow_aware_masks=True,
          use_conditional_cf=True, use_head_aware_masks=True)

set_seed(base.scenario_seed)
sc = build_experiment_scenario(base)
A = make_composite_trigger_spec(sc.data, base)  # the attacker's real trigger

def make_disjoint_spec(data, cfg, exclude_names):
    exclude = set(exclude_names)
    new_cfg = dataclasses.replace(cfg,
        flow_header_features=tuple(n for n in cfg.flow_header_features if n not in exclude),
        flow_temporal_features=tuple(n for n in cfg.flow_temporal_features if n not in exclude))
    return make_composite_trigger_spec(data, new_cfg)

B = make_disjoint_spec(sc.data, base, A.feature_names)  # server probe, zero overlap with A
print(f"A({len(A.feature_names)}): {A.feature_names}\nB({len(B.feature_names)}): {B.feature_names}")

cfg_ev = dataclasses.replace(base, evasion_attack=True, evasion_lambda=2.0)
cfg_ev_rand = dataclasses.replace(cfg_ev, asf_randomize_probe=True)   # countermeasure: randomised probe
cfg_ev_clip = dataclasses.replace(cfg_ev, update_norm_clip=True)      # countermeasure: norm clipping
cfg_ev_full = dataclasses.replace(cfg_ev, update_norm_clip=True, asf_randomize_probe=True)  # both

# (name, cfg, run kwargs). Poisoning and ASR both use trigger A.
ARMS = [
    ("M_nodef",   base,   dict(use_lockdown=False, apply_cf=False)),
    ("M_evasion", cfg_ev, dict(use_lockdown=False, apply_cf=False, asf_probe_spec_override=B)),
    ("O_mismatch", base,  dict(asf_probe_spec_override=B, **TP)),
    ("O_evasion", cfg_ev, dict(asf_probe_spec_override=B, **TP)),
    ("O_evasion_randprobe", cfg_ev_rand, dict(asf_probe_spec_override=B, **TP)),
    # Norm clipping with a fixed probe, which isolates the clip's own contribution.
    ("O_evasion_normclip", cfg_ev_clip, dict(asf_probe_spec_override=B, **TP)),
    # Both countermeasures together.
    ("O_evasion_full", cfg_ev_full, dict(asf_probe_spec_override=B, **TP)),
]

results = {}
if os.path.exists(RESULTS):
    results = json.load(open(RESULTS)).get("results", {})

t0 = time.time()
for name, cfg, kw in ARMS:
    arm = results.setdefault(name, {})
    for seed in SEEDS:
        if str(seed) in arm:
            print(f"  [skip] {name} s{seed}"); continue
        set_seed(seed)
        r = run_federated_stage(f"{name} s{seed}", cfg, sc, trigger_spec_override=A, **kw)
        arm[str(seed)] = {"benign": r.final_benign_acc, "asr": r.final_asr, "macro_f1": r.final_benign_macro_f1,
                          "dr": r.final_dr, "fpr": r.final_fpr, "fnr": r.final_fnr}
        json.dump({"seeds": SEEDS, "results": results}, open(RESULTS, "w"), indent=2)
        print(f"  {name:11s} s{seed}: ASR={r.final_asr:.4f} benign={r.final_benign_acc:.4f} "
              f"[{(time.time()-t0)/60:.1f}min]")

print("\n" + "=" * 60)
for name, _, _ in ARMS:
    a = [results[name][str(s)]["asr"] for s in SEEDS]
    print(f"{name:11s} ASR {mean(a)*100:.2f}±{pstdev(a)*100:.2f}% (per-seed {[round(x*100,2) for x in a]})")
print("\nKey comparison: O_evasion vs O_mismatch. A much higher O_evasion means the evasion succeeded; similar values mean the filter resisted it.")
print(f"Total {(time.time()-t0)/60:.1f}min -> {RESULTS}")
