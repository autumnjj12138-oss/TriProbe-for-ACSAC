import copy
import dataclasses
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .attack import (
    apply_trigger,
    build_head_aware_feature_groups,
    build_flow_aware_feature_indices,
    choose_trigger_features,
    make_composite_trigger_spec,
    make_trigger_spec,
    poison_client_data,
    poison_client_data_split,
    scale_poisoned_updates,
    validate_trigger_pairwise_consistency,
)
from .config import CFG, Config, ExperimentResult, ExperimentScenario
from .data import build_client_partitions, choose_malicious_clients, load_and_preprocess_csv, set_seed
from .evaluation import (
    benign_macro_f1,
    security_metrics,
    compute_asr,
    evaluate,
    print_coupling_interpretation,
    print_coupling_verdict,
    print_layerwise_pruning_scan,
    print_pruning_scan,
)
from .lockdown import (
    apply_gradient_mask,
    apply_weight_mask_inplace,
    build_head_aware_masks,
    conditional_consensus_fusion,
    consensus_fusion,
    enforce_mask_density_cap,
    init_masks,
    prune_masks,
    recover_masks,
    summarize_mask_density,
    summarize_mask_overlap,
    summarize_round_votes,
)
from .models import build_model, make_loader
from .reporting import (
    export_experiment_artifacts,
    export_multi_seed_artifacts,
    make_output_dir,
    print_stage_summary,
    print_trigger_summary,
)


def build_experiment_scenario(cfg: Config) -> ExperimentScenario:
    set_seed(cfg.scenario_seed)
    data = load_and_preprocess_csv(cfg)
    trigger_idx = choose_trigger_features(data, cfg)
    trigger_spec = make_trigger_spec(data, cfg, trigger_idx)
    client_parts = build_client_partitions(data.y_train, cfg)
    malicious_ids = choose_malicious_clients(cfg)
    return ExperimentScenario(
        data=data,
        trigger_spec=trigger_spec,
        client_parts=client_parts,
        malicious_ids=malicious_ids,
    )


def shieldfl_prune(model, tau_w: float, tau_g: float) -> None:
    """SHIELD-FL / SYNAPSE client-side synaptic pruning [Zukaib & Cui, KBS 2025].

    Faithful port of the authors' released ``prune_weights`` (Eqs. 28-31 and
    github.com/UmerZu/SHIELD-FL). For every ``weight`` tensor, an element is
    flagged anomalous-by-weight if |w| > tau_w and anomalous-by-gradient if
    |g| > tau_g. Weights are zeroed on the *intersection* of the two flags and
    gradients are zeroed on the gradient flag alone. Called once per batch after
    backward() and before optimizer.step(), matching the released training loop.

    Note: Eq. (22) of the paper literally reads W' = W - M (*) W with M=1 on
    normal entries, which would zero every *normal* weight instead. We follow
    Eq. (17)/(18) and the released code, which zero the anomalous entries.
    """
    for name, param in model.named_parameters():
        if "weight" not in name or param.grad is None:
            continue
        anom_w = param.data.abs() > tau_w
        anom_g = param.grad.abs() > tau_g
        param.data[anom_w & anom_g] = 0.0
        param.grad[anom_g] = 0.0


def local_train(global_model, X, y, cfg: Config, masks=None, flow_aware_feature_idx=None, use_flow_aware_training: bool = False,
                evasion_probe=None, evasion_target: int = 0, evasion_lambda: float = 0.0):
    model = copy.deepcopy(global_model).to(cfg.device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = make_loader(X, y, cfg.batch_size, sequence_window=cfg.sequence_window, shuffle=True)
    updated_masks = None if masks is None else {key: value.clone() for key, value in masks.items()}
    label_smoothing = float(getattr(cfg, "label_smoothing", 0.0))
    grad_clip = getattr(cfg, "grad_clip_norm", None)

    # Adaptive probe-evasion: pre-window the B-triggered probe batches once. The
    # malicious client suppresses P(target) on these to evade the ASF probe.
    ev_batches = None
    if evasion_probe is not None and evasion_lambda > 0:
        ev_loader = make_loader(evasion_probe, np.zeros(len(evasion_probe), dtype=np.int64),
                                cfg.batch_size, sequence_window=cfg.sequence_window, shuffle=False)
        ev_batches = [exb.to(cfg.device) for exb, _ in ev_loader]

    for _ in range(cfg.local_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            optimizer.zero_grad()
            out = model(xb)
            loss = F.cross_entropy(out, yb, label_smoothing=label_smoothing)
            if ev_batches is not None:
                ev_pen = sum(torch.softmax(model(exb), dim=-1)[:, evasion_target].mean() for exb in ev_batches) / len(ev_batches)
                loss = loss + evasion_lambda * ev_pen
            loss.backward()
            if getattr(cfg, "shieldfl_enable", False):
                shieldfl_prune(model, cfg.shieldfl_tau_w, cfg.shieldfl_tau_g)
            if updated_masks is not None:
                apply_gradient_mask(model, updated_masks)
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
            optimizer.step()
            if updated_masks is not None:
                apply_weight_mask_inplace(model, updated_masks)

        if updated_masks is not None:
            prune_masks(
                model,
                updated_masks,
                cfg.alpha0_prune,
                flow_aware_feature_idx=flow_aware_feature_idx if use_flow_aware_training else None,
                flow_aware_prune_penalty=cfg.flow_aware_prune_penalty,
            )
            recover_masks(
                model,
                updated_masks,
                cfg.alpha0_recover,
                flow_aware_feature_idx=flow_aware_feature_idx if use_flow_aware_training else None,
                flow_aware_recover_boost=cfg.flow_aware_recover_boost,
            )
            cap = getattr(cfg, "mask_density_cap", None)
            if cap is not None:
                enforce_mask_density_cap(updated_masks, cap)
            apply_weight_mask_inplace(model, updated_masks)

    cpu_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    return cpu_state, updated_masks


def fedavg(global_model, client_states: List[Dict[str, torch.Tensor]]) -> None:
    new_state = copy.deepcopy(client_states[0])
    for key in new_state.keys():
        for i in range(1, len(client_states)):
            new_state[key] += client_states[i][key]
        new_state[key] /= len(client_states)
    global_model.load_state_dict(new_state)


def fedmedian(global_model, client_states: List[Dict[str, torch.Tensor]]) -> None:
    new_state = copy.deepcopy(client_states[0])
    for key in new_state.keys():
        stacked = torch.stack([state[key].float() for state in client_states], dim=0)
        median_val = torch.median(stacked, dim=0).values
        new_state[key] = median_val.to(client_states[0][key].dtype)
    global_model.load_state_dict(new_state)


def _flatten_state(state: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([value.detach().float().view(-1) for value in state.values()])


def krum_aggregate(global_model, client_states: List[Dict[str, torch.Tensor]], n_malicious: int) -> None:
    if len(client_states) == 1:
        global_model.load_state_dict(client_states[0])
        return

    vectors = [_flatten_state(state) for state in client_states]
    n_clients = len(vectors)
    keep_neighbors = max(1, n_clients - n_malicious - 2)
    scores = []
    for idx, vec in enumerate(vectors):
        distances = []
        for jdx, other in enumerate(vectors):
            if idx == jdx:
                continue
            distances.append(torch.norm(vec - other, p=2).item() ** 2)
        distances.sort()
        scores.append((sum(distances[:keep_neighbors]), idx))

    _, winner_idx = min(scores, key=lambda item: item[0])
    global_model.load_state_dict(copy.deepcopy(client_states[winner_idx]))


def norm_clip_updates(global_model, client_states: List[Dict[str, torch.Tensor]], cfg: Config) -> List[Dict[str, torch.Tensor]]:
    """Clip client updates whose norm exceeds factor * median back to that ceiling.
    Composable with the probe filter, to bound updates that evade it but remain geometrically abnormal. Returns the clipped states."""
    global_state = {k: v.cpu().clone() for k, v in global_model.state_dict().items()}
    deltas = [{k: s[k].float() - global_state[k].float() for k in s} for s in client_states]
    flat_norms = torch.stack([torch.cat([d[k].view(-1) for k in sorted(d.keys())]).norm() for d in deltas])
    clip_bound = float(getattr(cfg, "update_norm_clip_factor", 1.0)) * torch.median(flat_norms)
    out = []
    for i, s in enumerate(client_states):
        dn = flat_norms[i]
        if dn > clip_bound and dn > 0:
            scale = (clip_bound / dn).item()
            out.append({k: (global_state[k].float() + deltas[i][k] * scale).to(s[k].dtype) for k in s})
        else:
            out.append(s)
    return out


def flame_aggregate(global_model, client_states: List[Dict[str, torch.Tensor]], cfg: Config) -> None:
    """FLAME-style: cosine-similarity filtering + norm clipping + noise."""
    global_state = {k: v.cpu().clone() for k, v in global_model.state_dict().items()}
    deltas = [{k: s[k].float() - global_state[k].float() for k in s} for s in client_states]
    flat_deltas = [torch.cat([d[k].view(-1) for k in sorted(d.keys())]) for d in deltas]
    stacked = torch.stack(flat_deltas)
    n = len(flat_deltas)

    norms = stacked.norm(dim=1, keepdim=True).clamp_min(1e-8)
    cos_sim = (stacked / norms) @ (stacked / norms).T
    avg_sims = [float(np.mean([cos_sim[i][j].item() for j in range(n) if j != i])) for i in range(n)]
    median_sim = float(np.median(avg_sims))
    selected = [i for i in range(n) if avg_sims[i] >= median_sim]
    if len(selected) < max(1, n // 3):
        selected = sorted(range(n), key=lambda i: avg_sims[i], reverse=True)[: max(1, n // 2)]

    sel_norms = torch.stack([flat_deltas[i].norm() for i in selected])
    clip_bound = cfg.flame_clip_factor * torch.median(sel_norms)
    clipped_deltas = []
    for i in selected:
        dn = flat_deltas[i].norm()
        if dn > clip_bound:
            clipped_deltas.append({k: deltas[i][k] * (clip_bound / dn).item() for k in deltas[i]})
        else:
            clipped_deltas.append(deltas[i])

    avg_delta = copy.deepcopy(clipped_deltas[0])
    for key in avg_delta:
        for i in range(1, len(clipped_deltas)):
            avg_delta[key] += clipped_deltas[i][key]
        avg_delta[key] /= len(clipped_deltas)
        if cfg.flame_noise_sigma > 0:
            avg_delta[key] += torch.randn_like(avg_delta[key]) * cfg.flame_noise_sigma
    global_model.load_state_dict({k: global_state[k].float() + avg_delta[k] for k in global_state})


def rlr_aggregate(global_model, client_states: List[Dict[str, torch.Tensor]], cfg: Config) -> None:
    """Robust Learning Rate: sign-consensus voting."""
    global_state = {k: v.cpu().clone() for k, v in global_model.state_dict().items()}
    deltas = [{k: s[k].float() - global_state[k].float() for k in s} for s in client_states]
    n = len(deltas)
    threshold = n * cfg.rlr_threshold
    new_state = {}
    for key in global_state:
        stacked = torch.stack([d[key] for d in deltas])
        sign_sum = torch.sign(stacked).sum(dim=0)
        accepted = (sign_sum.abs() >= threshold).float()
        avg_update = stacked.mean(dim=0)
        new_state[key] = (global_state[key].float() + avg_update * accepted).to(global_state[key].dtype)
    global_model.load_state_dict(new_state)


def asf_filter_clients(
    global_model,
    client_states: List[Dict[str, torch.Tensor]],
    probe_X_triggered: np.ndarray,
    target_label: int,
    drop_quantile: float,
    cfg: Config,
    kept_out: Optional[List[int]] = None,
) -> List[Dict[str, torch.Tensor]]:
    """Activation-Space Server Filtering (L2) — triggered-probe variant.

    The server evaluates P(class=target_label) for each client model on triggered
    samples. A benign client has never seen the trigger and still answers with the
    attack class; a backdoored one has learned the mapping and returns a high
    """
    if len(client_states) <= 2 or drop_quantile <= 0:
        return client_states

    device = cfg.device
    loader = make_loader(
        probe_X_triggered,
        np.zeros(len(probe_X_triggered), dtype=np.int64),
        cfg.batch_size,
        cfg.sequence_window,
        shuffle=False,
    )

    def _target_prob(model) -> float:
        model.eval()
        probs = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(device)
                logits = model(xb)
                p = torch.softmax(logits, dim=-1)[:, target_label]
                probs.append(p.cpu())
        return float(torch.cat(probs, dim=0).mean().item())

    temp_model = copy.deepcopy(global_model)
    scores: List[float] = []
    for state in client_states:
        temp_model.load_state_dict(state)
        scores.append(_target_prob(temp_model))

    n_drop = max(1, int(len(scores) * drop_quantile))
    drop_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_drop]
    drop_set = set(drop_idx)
    filtered = [s for i, s in enumerate(client_states) if i not in drop_set]
    if not filtered:
        if kept_out is not None:
            kept_out.extend(range(len(client_states)))
        return client_states
    if kept_out is not None:
        kept_out.extend(i for i in range(len(client_states)) if i not in drop_set)
    print(
        f"  [ASF-trig] Dropped {len(client_states) - len(filtered)}/{len(client_states)} "
        f"(max_P(target)={max(scores):.3f}  dropped_ids={sorted(drop_idx)})"
    )
    return filtered


def asf_filter_clients_benign(
    global_model,
    client_states: List[Dict[str, torch.Tensor]],
    probe_X_benign: np.ndarray,
    drop_quantile: float,
    cfg: Config,
) -> List[Dict[str, torch.Tensor]]:
    """Benign-probe activation-distance filtering — the *passive* baseline.

    On clean probes with no trigger, the server takes each client model's
    penultimate-layer mean activation, ranks clients by L2 distance to the
    coordinate-wise median, and drops the furthest quantile.

    This is passive geometric detection. A backdoor is dormant on clean inputs, so
    clean probes barely separate backdoored models, while non-IID drift scatters
    benign clients and gets them dropped. Kept as a control for the triggered probe.
    """
    if len(client_states) <= 2 or drop_quantile <= 0:
        return client_states

    device = cfg.device
    loader = make_loader(
        probe_X_benign,
        np.zeros(len(probe_X_benign), dtype=np.int64),
        cfg.batch_size,
        cfg.sequence_window,
        shuffle=False,
    )

    temp_model = copy.deepcopy(global_model)
    captured: Dict[str, torch.Tensor] = {}

    def _pre_hook(_module, inp):
        captured["feat"] = inp[0].detach()

    handle = temp_model.fc3.register_forward_pre_hook(_pre_hook)

    def _signature(state) -> torch.Tensor:
        temp_model.load_state_dict(state)
        temp_model.eval()
        feat_sum = None
        n = 0
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(device)
                _ = temp_model(xb)
                feats = captured["feat"]
                s = feats.float().sum(dim=0).cpu()
                feat_sum = s if feat_sum is None else feat_sum + s
                n += feats.shape[0]
        return feat_sum / max(n, 1)

    signatures = [_signature(state) for state in client_states]
    handle.remove()

    stacked = torch.stack(signatures, dim=0)
    median = stacked.median(dim=0).values
    dists = [float(torch.norm(sig - median, p=2).item()) for sig in signatures]

    n_drop = max(1, int(len(dists) * drop_quantile))
    drop_idx = sorted(range(len(dists)), key=lambda i: dists[i], reverse=True)[:n_drop]
    drop_set = set(drop_idx)
    filtered = [s for i, s in enumerate(client_states) if i not in drop_set]
    if not filtered:
        return client_states
    print(
        f"  [ASF-benign] Dropped {len(client_states) - len(filtered)}/{len(client_states)} "
        f"(max_dist={max(dists):.3f}  dropped_ids={sorted(drop_idx)})"
    )
    return filtered


def deepsight_filter_clients(
    global_model,
    client_states: List[Dict[str, torch.Tensor]],
    cfg: Config,
) -> List[Dict[str, torch.Tensor]]:
    """DeepSight [Rieger et al., NDSS 2022] — activation/behavior detection baseline.

    Faithful re-implementation of DeepSight's filtering pipeline, adapted to the
    binary FL-IDS head:
      (1) NEUPs: normalized energy of the output-layer (fc3) weight+bias update per
          class; threshold-exceedings (TE) classify a client as backdoor-suspect
          (concentrated energy ⇒ few exceedings).
      (2) DDifs: feed several batches of random inputs through global and client
          models; the client's mean softmax response is its behavioral fingerprint
          (backdoored models diverge on random data).
      (3) A pairwise cosine-distance matrix over [NEUPs ‖ DDifs ‖ cosine of full
          weight update] is clustered (HDBSCAN, fallback AgglomerativeClustering).
      (4) A cluster is labelled poisoned if >1/3 of its members are TE-classified as
          backdoor; all clients in poisoned clusters are discarded; survivors are
          returned for standard aggregation.

    Note (binary adaptation): with a 2-class head NEUP/TE is partially degenerate,
    so the DDif behavioral signature and update-cosine clustering carry most of the
    separation. This is documented as a faithful adaptation, not the original
    many-class configuration.
    """
    n = len(client_states)
    if n <= 2:
        return client_states

    device = cfg.device
    gstate = {k: v.detach().float().cpu() for k, v in global_model.state_dict().items()}
    num_classes = gstate["fc3.bias"].numel()

    # ---- (1) NEUPs + threshold exceedings ----
    neups = []
    for st in client_states:
        dw = (st["fc3.weight"].float().cpu() - gstate["fc3.weight"]).abs().sum(dim=1)
        db = (st["fc3.bias"].float().cpu() - gstate["fc3.bias"]).abs()
        ups = (dw + db)
        energy = ups ** 2
        neup = energy / (energy.sum() + 1e-12)
        neups.append(neup)
    neups = torch.stack(neups, dim=0)  # [n, num_classes]
    tau = getattr(cfg, "deepsight_te_tau", 0.01)
    te = (neups > (tau * neups.max(dim=1, keepdim=True).values)).sum(dim=1).numpy()  # exceedings
    # DeepSight: concentrated output-energy (few exceedings) ⇒ backdoor-suspect. On a
    # binary head the discriminative case is te==1 (energy collapsed onto a single class);
    # te==num_classes (energy spread over both) is treated as benign.
    is_backdoor = (te < num_classes).astype(np.float32)

    # ---- (2) DDifs on random inputs ----
    in_dim = global_model.input_proj.in_features
    num_seeds = getattr(cfg, "deepsight_num_seeds", 3)
    num_random = getattr(cfg, "deepsight_num_random", 256)
    temp_model = copy.deepcopy(global_model)

    def _mean_softmax(model, X):
        model.eval()
        loader = make_loader(X, np.zeros(len(X), dtype=np.int64),
                             cfg.batch_size, cfg.sequence_window, shuffle=False)
        outs = []
        with torch.no_grad():
            for xb, _ in loader:
                outs.append(torch.softmax(model(xb.to(device)), dim=-1).cpu())
        return torch.cat(outs, dim=0).mean(dim=0)

    ddif_feats = [[] for _ in range(n)]
    rng = np.random.RandomState(cfg.scenario_seed)
    for _s in range(num_seeds):
        Xr = rng.randn(num_random, in_dim).astype(np.float32)
        for i, st in enumerate(client_states):
            temp_model.load_state_dict(st)
            ddif_feats[i].append(_mean_softmax(temp_model, Xr))
    ddifs = torch.stack([torch.cat(f, dim=0) for f in ddif_feats], dim=0)  # [n, num_seeds*num_classes]

    # ---- (3) cosine-distance matrix + clustering ----
    def _cos_dist(M):
        Mn = M / (M.norm(dim=1, keepdim=True) + 1e-12)
        sim = Mn @ Mn.t()
        return (1.0 - sim).clamp(min=0.0)

    # full last-layer weight update vectors for update-cosine component
    upd = torch.stack([
        torch.cat([(st["fc3.weight"].float().cpu() - gstate["fc3.weight"]).flatten(),
                   (st["fc3.bias"].float().cpu() - gstate["fc3.bias"]).flatten()])
        for st in client_states
    ], dim=0)
    D = (_cos_dist(neups) + _cos_dist(ddifs) + _cos_dist(upd)) / 3.0
    D = D.numpy().astype(np.float64)
    np.fill_diagonal(D, 0.0)

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # torch+sklearn OpenMP coexist (Win)
    labels = None
    try:
        from sklearn.cluster import HDBSCAN
        labels = HDBSCAN(min_cluster_size=2, metric="precomputed").fit_predict(D)
    except Exception:
        from sklearn.cluster import AgglomerativeClustering
        k = min(max(2, n // 5), n - 1)
        labels = AgglomerativeClustering(
            n_clusters=k, metric="precomputed", linkage="average"
        ).fit_predict(D)

    # ---- (4) label clusters as poisoned and discard ----
    keep = []
    drop_idx = []
    for cl in set(labels.tolist()):
        members = [i for i in range(n) if labels[i] == cl]
        if cl == -1:  # HDBSCAN noise → treat each as its own (keep unless backdoor-flagged)
            for i in members:
                (drop_idx if is_backdoor[i] > 0.5 else keep).append(i)
            continue
        frac = float(np.mean([is_backdoor[i] for i in members]))
        if frac > 1.0 / 3.0:
            drop_idx.extend(members)
        else:
            keep.extend(members)
    keep = sorted(keep)
    filtered = [client_states[i] for i in keep]
    if not filtered:
        return client_states
    print(
        f"  [DeepSight] Dropped {n - len(filtered)}/{n} "
        f"(dropped_ids={sorted(drop_idx)}; clusters={len(set(labels.tolist()))})"
    )
    return filtered


def aggregate_global_model(global_model, client_states: List[Dict[str, torch.Tensor]], aggregator_name: str, cfg: Config) -> None:
    if aggregator_name == "fedavg":
        fedavg(global_model, client_states)
    elif aggregator_name == "fedmedian":
        fedmedian(global_model, client_states)
    elif aggregator_name == "krum":
        krum_aggregate(global_model, client_states, cfg.n_krum_remove)
    elif aggregator_name == "flame":
        flame_aggregate(global_model, client_states, cfg)
    elif aggregator_name == "rlr":
        rlr_aggregate(global_model, client_states, cfg)
    else:
        raise ValueError(f"Unsupported aggregator: {aggregator_name}")


def run_federated_stage(
    stage_name: str,
    cfg: Config,
    scenario: ExperimentScenario,
    use_lockdown: bool,
    apply_cf: bool,
    use_flow_aware_masks: bool = False,
    use_conditional_cf: bool = False,
    use_head_aware_masks: bool = False,
    aggregator_name: str = "fedavg",
    use_adaptive_attack: bool = False,
    trigger_spec_override=None,
    asf_probe_spec_override=None,
) -> ExperimentResult:
    data = scenario.data
    trigger_spec = trigger_spec_override if trigger_spec_override is not None else scenario.trigger_spec
    client_parts = scenario.client_parts
    malicious_ids = scenario.malicious_ids

    print(f"\n===== {stage_name} =====")
    print(f"Malicious clients: {sorted(list(malicious_ids))}")
    print(f"Poison setting: {cfg.poison_setting_name}  poison_ratio={cfg.poison_ratio:.2f}  malicious_clients={cfg.malicious_clients}/{cfg.num_clients}")

    attack_test_idx = (data.y_test == 1).nonzero()[0]
    X_test_attack_only = data.X_test[attack_test_idx]
    X_test_triggered = apply_trigger(X_test_attack_only, trigger_spec.trigger_map)
    # Benign probe (un-triggered normal traffic) for the passive benign-probe ASF baseline.
    benign_test_idx = (data.y_test == cfg.target_label).nonzero()[0]
    X_asf_probe_benign = data.X_test[benign_test_idx]
    # Base pool the triggered probes are built on. Default "attack": the server
    # perturbs known attack-class samples, so a clean model answers "attack" and
    # only a backdoored one flips to the target class. "benign" perturbs benign
    # samples instead, which tests whether the defender assumption can be
    # weakened to benign-only server data.
    _probe_base = (X_asf_probe_benign
                   if getattr(cfg, "asf_probe_base", "attack") == "benign"
                   else X_test_attack_only)

    def _make_probe(trigger_map):
        return apply_trigger(_probe_base, trigger_map)

    # ASF probe: by default uses the attacker's trigger (perfect knowledge);
    # an explicit override enables the trigger-mismatch experiment in §6.
    if asf_probe_spec_override is not None:
        X_asf_probe_triggered = _make_probe(asf_probe_spec_override.trigger_map)
        print(f"[ASF probe override] using {len(asf_probe_spec_override.feature_names)} features: "
              f"{asf_probe_spec_override.feature_names}")
    else:
        X_asf_probe_triggered = _make_probe(trigger_spec.trigger_map)
    print(f"[ASF probe base] {getattr(cfg, 'asf_probe_base', 'attack')} "
          f"({len(_probe_base)} candidate samples)")
    # Adaptive probe-evasion attack: malicious clients suppress P(target)
    # on the server's probe trigger B (asf_probe_spec_override), built from TRAIN attack
    # samples (attacker's own data + white-box knowledge of B), not the test set.
    X_evasion_probe = None
    if getattr(cfg, "evasion_attack", False) and asf_probe_spec_override is not None:
        _tr_atk = (data.y_train == 1).nonzero()[0]
        X_evasion_probe = apply_trigger(data.X_train[_tr_atk][: cfg.asf_probe_size], asf_probe_spec_override.trigger_map)
        print(f"[Evasion attack] malicious clients suppress P(target) on probe-B ({len(X_evasion_probe)} samples), lambda={cfg.evasion_lambda}")
    global_model = build_model(in_dim=data.X_train.shape[1], cfg=cfg).to(cfg.device)
    flow_aware_feature_idx = build_flow_aware_feature_indices(data, trigger_spec, cfg) if use_flow_aware_masks else []
    effective_flow_idx = flow_aware_feature_idx if (use_flow_aware_masks or use_head_aware_masks) else []
    head_aware_groups = build_head_aware_feature_groups(data, cfg) if use_head_aware_masks else {"header": [], "temporal": []}
    if use_flow_aware_masks:
        print("Flow-aware focus features:", [data.feature_names[idx] for idx in flow_aware_feature_idx])
    if use_head_aware_masks:
        print(
            "Head-aware focus features:",
            {
                "header": [data.feature_names[idx] for idx in head_aware_groups["header"]],
                "temporal": [data.feature_names[idx] for idx in head_aware_groups["temporal"]],
            },
        )

    client_masks = (
        [
            (
                build_head_aware_masks(
                    global_model,
                    cfg.sparsity,
                    header_feature_idx=head_aware_groups["header"],
                    temporal_feature_idx=head_aware_groups["temporal"],
                    same_boost=cfg.head_aware_same_boost,
                    cross_penalty=cfg.head_aware_cross_penalty,
                    min_keep=cfg.flow_aware_min_keep,
                    num_heads=cfg.transformer_heads,
                    d_model=cfg.transformer_d_model,
                    global_fusion_cross_penalty=getattr(cfg, "global_fusion_cross_penalty", 0.0),
                    layer1_hard_block=getattr(cfg, "layer1_hard_block", False),
                )
                if use_head_aware_masks
                else init_masks(
                    global_model,
                    cfg.sparsity,
                    flow_aware_feature_idx=flow_aware_feature_idx,
                    flow_aware_column_boost=cfg.flow_aware_column_boost if use_flow_aware_masks else 1.0,
                    flow_aware_min_keep=cfg.flow_aware_min_keep,
                )
            )
            for _ in range(cfg.num_clients)
        ]
        if use_lockdown
        else None
    )

    # Strong-threat split-trigger attack: pre-split the composite trigger map into its
    # header-subspace and temporal-subspace halves (used by poison_client_data_split).
    split_attack = getattr(cfg, "split_trigger_attack", False)
    header_trigger_map: Dict[int, float] = {}
    temporal_trigger_map: Dict[int, float] = {}
    if split_attack:
        _groups = build_head_aware_feature_groups(data, cfg)
        _hset, _tset = set(_groups["header"]), set(_groups["temporal"])
        for _idx, _val in trigger_spec.trigger_map.items():
            if _idx in _hset:
                header_trigger_map[_idx] = _val
            elif _idx in _tset:
                temporal_trigger_map[_idx] = _val
        print(f"[Split-Trigger] header-half feats={sorted(header_trigger_map)}  "
              f"temporal-half feats={sorted(temporal_trigger_map)}")

    total_poisoned = 0
    round_rows: List[Dict[str, float]] = []
    overlap_stats = None
    # Track which clients the filter retained, per round, to build a trusted mask set.
    asf_keep_counts = [0] * cfg.num_clients
    asf_rounds_counted = 0
    asf_last_kept: Optional[List[int]] = None

    for rnd in range(cfg.rounds):
        round_start = time.perf_counter()
        in_warmup = use_lockdown and (rnd < getattr(cfg, "warmup_rounds_no_mask", 0))
        use_masks_this_round = use_lockdown and client_masks is not None and not in_warmup

        # Delayed / on-off backdoor: poison this round? Defaults make it always true.
        _duty = max(1, int(getattr(cfg, "poison_duty_cycle", 1)))
        poison_now = (rnd >= int(getattr(cfg, "poison_start_round", 0))) and (rnd % _duty == 0)

        client_states = []
        new_client_masks = []
        for cid in range(cfg.num_clients):
            idx = np.sort(client_parts[cid])
            X_client, y_client = data.X_train[idx], data.y_train[idx]
            if cid in malicious_ids and poison_now:
                if split_attack:
                    X_client, y_client, poisoned_count = poison_client_data_split(
                        X_client, y_client, cfg.poison_ratio,
                        header_trigger_map, temporal_trigger_map, cfg.target_label,
                    )
                else:
                    X_client, y_client, poisoned_count = poison_client_data(
                        X_client, y_client, cfg.poison_ratio, trigger_spec.trigger_map, cfg.target_label
                    )
                total_poisoned += poisoned_count

            ev_probe = X_evasion_probe if (X_evasion_probe is not None and cid in malicious_ids) else None
            if use_masks_this_round:
                state, updated_mask = local_train(
                    global_model,
                    X_client,
                    y_client,
                    cfg,
                    masks=client_masks[cid],
                    flow_aware_feature_idx=effective_flow_idx if effective_flow_idx else None,
                    use_flow_aware_training=(use_flow_aware_masks or use_head_aware_masks),
                    evasion_probe=ev_probe, evasion_target=cfg.target_label, evasion_lambda=cfg.evasion_lambda,
                )
                new_client_masks.append(updated_mask)
            else:
                state, _ = local_train(global_model, X_client, y_client, cfg, masks=None,
                                       evasion_probe=ev_probe, evasion_target=cfg.target_label, evasion_lambda=cfg.evasion_lambda)
                if use_lockdown and client_masks is not None:
                    new_client_masks.append(client_masks[cid])  # mask frozen during warmup
            client_states.append(state)

        # Adaptive attack: amplify malicious updates so they survive clipping.
        if use_adaptive_attack:
            global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
            client_states = scale_poisoned_updates(
                global_state, client_states, malicious_ids, cfg.adaptive_attack_scale,
                norm_bound_factor=getattr(cfg, "adaptive_norm_bound_factor", 2.0),
            )

        # L2: Activation-Space Server Filtering (triggered- or benign-probe variant)
        if getattr(cfg, "asf_enable", False) and not in_warmup and use_head_aware_masks:
            if getattr(cfg, "asf_mode", "triggered") == "benign":
                probe_benign = X_asf_probe_benign[: cfg.asf_probe_size]
                client_states = asf_filter_clients_benign(
                    global_model,
                    client_states,
                    probe_benign,
                    cfg.asf_drop_quantile,
                    cfg,
                )
            else:
                if getattr(cfg, "asf_randomize_probe", False):
                    # Countermeasure: draw a fresh probe from the candidate subspaces each
                    # round, so an attacker tuned against one fixed probe gains nothing.
                    _h = [n for n in cfg.flow_header_features if n in data.feature_names]
                    _t = [n for n in cfg.flow_temporal_features if n in data.feature_names]
                    _rh = list(np.random.choice(_h, size=min(len(_h), cfg.composite_header_count), replace=False))
                    _rt = list(np.random.choice(_t, size=min(len(_t), cfg.composite_temporal_count), replace=False))
                    _rcfg = dataclasses.replace(cfg, flow_header_features=tuple(_rh), flow_temporal_features=tuple(_rt))
                    _rspec = make_composite_trigger_spec(data, _rcfg)
                    probe_triggered = _make_probe(_rspec.trigger_map)[: cfg.asf_probe_size]
                else:
                    probe_triggered = X_asf_probe_triggered[: cfg.asf_probe_size]
                _kept: List[int] = []
                client_states = asf_filter_clients(
                    global_model,
                    client_states,
                    probe_triggered,
                    cfg.target_label,
                    cfg.asf_drop_quantile,
                    cfg,
                    kept_out=_kept,
                )
                asf_last_kept = list(_kept)
                asf_rounds_counted += 1
                for _i in _kept:
                    asf_keep_counts[_i] += 1

        # Update-norm clipping, layered after the probe filter.
        if getattr(cfg, "update_norm_clip", False) and not in_warmup:
            client_states = norm_clip_updates(global_model, client_states, cfg)

        # L2': DeepSight activation/behavior detection (cat-2 baseline)
        if getattr(cfg, "deepsight_enable", False) and not in_warmup:
            client_states = deepsight_filter_clients(global_model, client_states, cfg)

        aggregate_global_model(global_model, client_states, aggregator_name, cfg)
        if use_lockdown and client_masks is not None:
            client_masks = new_client_masks

        # Mid-round consensus fusion: a light pass every N rounds at a looser threshold.
        if (
            getattr(cfg, "midround_cf_enable", False)
            and use_lockdown
            and client_masks is not None
            and not in_warmup
            and (rnd + 1) % cfg.midround_cf_interval == 0
        ):
            theta_mid = max(1, int(cfg.num_clients * cfg.midround_cf_theta_ratio))
            if use_conditional_cf:
                conditional_consensus_fusion(
                    global_model,
                    client_masks,
                    theta_base=theta_mid,
                    theta_traffic=theta_mid + 2,
                    flow_aware_feature_idx=effective_flow_idx,
                    header_feature_idx=head_aware_groups["header"] if use_head_aware_masks else None,
                    temporal_feature_idx=head_aware_groups["temporal"] if use_head_aware_masks else None,
                    num_heads=cfg.transformer_heads,
                    d_model=cfg.transformer_d_model,
                    clip_factor=0.3,  # soft clip, avoids zeroing weights too early
                )
            else:
                consensus_fusion(global_model, client_masks, theta=theta_mid, clip_factor=0.3)

        row: Dict[str, float] = {"round": int(rnd + 1)}
        # Record mask density.
        if use_lockdown and client_masks is not None:
            row.update(summarize_mask_density(client_masks, malicious_ids=malicious_ids))
            # Per-round mask overlap, for density-drift diagnostics. Off by default.
            if getattr(cfg, "track_round_overlap", False):
                row.update(summarize_mask_overlap(client_masks, malicious_ids))
            # Per-round vote histogram and pruning margin. Off by default.
            if getattr(cfg, "track_round_votes", False):
                theta_b = max(1, int(cfg.num_clients * cfg.consensus_ratio))
                row.update(summarize_round_votes(client_masks, cfg.num_clients, theta_b))
        # Record wall-clock time.
        if cfg.track_round_overhead:
            row["round_runtime_sec"] = float(time.perf_counter() - round_start)
        if (rnd + 1) % 5 == 0 or rnd == 0:
            benign_acc = evaluate(global_model, data.X_test, data.y_test, cfg.batch_size, cfg.device, cfg.sequence_window)
            asr = compute_asr(global_model, X_test_triggered, cfg.batch_size, cfg.target_label, cfg.device, cfg.sequence_window)
            print(f"[Round {rnd + 1:03d}] benign_acc={benign_acc:.4f}  ASR={asr:.4f}")
            row.update({"benign_acc": float(benign_acc), "asr": float(asr)})
        else:
            row.update({"benign_acc": float("nan"), "asr": float("nan")})
        round_rows.append(row)

    final_benign_acc = evaluate(global_model, data.X_test, data.y_test, cfg.batch_size, cfg.device, cfg.sequence_window)
    final_asr = compute_asr(global_model, X_test_triggered, cfg.batch_size, cfg.target_label, cfg.device, cfg.sequence_window)
    print(f"[Final before CF] benign_acc={final_benign_acc:.4f}  ASR={final_asr:.4f}")
    print(f"Poisoned samples injected across rounds: {total_poisoned}")

    pruning_rows = None
    if not use_lockdown:
        pruning_rows = print_pruning_scan(global_model, data.X_test, data.y_test, X_test_triggered, cfg)
        if cfg.run_layerwise_pruning_scan:
            print_layerwise_pruning_scan(global_model, data.X_test, data.y_test, X_test_triggered, cfg)

    if use_lockdown and client_masks is not None and apply_cf:
        # Select which client masks are allowed to vote in the final consensus fusion.
        cf_source = getattr(cfg, "cf_mask_source", "all")
        voting_ids = list(range(len(client_masks)))
        if cf_source == "last" and asf_last_kept:
            voting_ids = sorted(asf_last_kept)
        elif cf_source == "majority" and asf_rounds_counted > 0:
            voting_ids = [i for i, c in enumerate(asf_keep_counts) if 2 * c >= asf_rounds_counted]
        if not voting_ids:
            voting_ids = list(range(len(client_masks)))
        cf_masks = [client_masks[i] for i in voting_ids]
        n_vote = len(cf_masks)
        if cf_source != "all":
            print(f"[CF] mask source={cf_source}: {n_vote}/{len(client_masks)} clients vote "
                  f"(ids={voting_ids}; malicious={sorted(malicious_ids)})")
        # rescale the quorum to the number of voting masks so it stays comparable to M-of-M
        theta = max(1, int(round(n_vote * cfg.consensus_ratio)))
        if use_conditional_cf:
            theta_traffic = max(1, int(round(n_vote * cfg.traffic_consensus_ratio)))
            conditional_consensus_fusion(
                global_model,
                cf_masks,
                theta_base=theta,
                theta_traffic=theta_traffic,
                flow_aware_feature_idx=effective_flow_idx,
                header_feature_idx=head_aware_groups["header"] if use_head_aware_masks else None,
                temporal_feature_idx=head_aware_groups["temporal"] if use_head_aware_masks else None,
                num_heads=cfg.transformer_heads,
                d_model=cfg.transformer_d_model,
                clip_factor=cfg.consensus_clip_factor,
            )
        else:
            consensus_fusion(global_model, cf_masks, theta=theta, clip_factor=cfg.consensus_clip_factor)
        final_benign_acc = evaluate(global_model, data.X_test, data.y_test, cfg.batch_size, cfg.device, cfg.sequence_window)
        final_asr = compute_asr(global_model, X_test_triggered, cfg.batch_size, cfg.target_label, cfg.device, cfg.sequence_window)
        print(f"[After CF] benign_acc={final_benign_acc:.4f}  ASR={final_asr:.4f}")

        overlap_stats = summarize_mask_overlap(client_masks, malicious_ids)
        print("\n[Poison-Coupling] Mask overlap after training")
        print(
            "  malicious-malicious: "
            f"Jaccard={overlap_stats['mal_mal_jaccard']:.4f}  Hamming={overlap_stats['mal_mal_hamming']:.4f}"
        )
        print(
            "  malicious-benign:    "
            f"Jaccard={overlap_stats['mal_ben_jaccard']:.4f}  Hamming={overlap_stats['mal_ben_hamming']:.4f}"
        )
        pruning_rows = print_pruning_scan(global_model, data.X_test, data.y_test, X_test_triggered, cfg)

    final_macro_f1 = benign_macro_f1(
        global_model, data.X_test, data.y_test, cfg.batch_size, cfg.device, cfg.sequence_window
    )
    sec = security_metrics(
        global_model, data.X_test, data.y_test, cfg.batch_size, cfg.device, cfg.sequence_window
    )
    print(f"[Final] benign Macro-F1={final_macro_f1:.4f}  DR={sec['dr']:.4f} FPR={sec['fpr']:.4f} FNR={sec['fnr']:.4f}")

    result = ExperimentResult(
        stage_name=stage_name,
        final_benign_acc=final_benign_acc,
        final_asr=final_asr,
        asr_is_strong=final_asr >= cfg.attack_success_threshold,
        final_benign_macro_f1=final_macro_f1,
        final_dr=sec["dr"],
        final_fnr=sec["fnr"],
        final_fpr=sec["fpr"],
        final_attack_precision=sec["attack_precision"],
        final_attack_f1=sec["attack_f1"],
        pruning_rows=pruning_rows,
        round_rows=round_rows,
        overlap_stats=overlap_stats,
        poisoned_samples=total_poisoned,
    )
    print_coupling_interpretation(result, cfg)
    if stage_name.startswith("Stage A"):
        print_coupling_verdict(result, cfg)
    return result


def run_experiment(cfg: Config, scenario: Optional[ExperimentScenario] = None):
    set_seed(cfg.seed)
    output_dir = make_output_dir(cfg)
    if scenario is None:
        scenario = build_experiment_scenario(cfg)
    print(f"[Scenario] scenario_seed={cfg.scenario_seed}  training_seed={cfg.seed}")
    print_trigger_summary(scenario.trigger_spec)
    results: List[ExperimentResult] = []

    if cfg.run_attack_only_baseline:
        results.append(run_federated_stage("Stage A: FedAvg + attack", cfg, scenario, False, False))
    if cfg.run_lockdown_without_cf_ablation:
        results.append(run_federated_stage("Stage B0: Lockdown + attack (without CF)", cfg, scenario, True, False))
    if cfg.run_lockdown_baseline:
        results.append(run_federated_stage("Stage B: Lockdown + attack + CF", cfg, scenario, True, True))
    if cfg.run_flow_aware_baseline:
        results.append(
            run_federated_stage(
                "Stage C: Flow-Aware Lockdown + Conditional CF",
                cfg,
                scenario,
                True,
                True,
                use_flow_aware_masks=True,
                use_conditional_cf=True,
            )
        )
    if cfg.run_head_aware_baseline:
        results.append(
            run_federated_stage(
                "Stage D: Head-Aware Lockdown + attack + CF",
                cfg,
                scenario,
                True,
                True,
                use_flow_aware_masks=True,
                use_conditional_cf=True,
                use_head_aware_masks=True,
            )
        )
    if cfg.run_fedmedian_baseline:
        results.append(
            run_federated_stage(
                "Stage E: FedMedian + attack",
                cfg,
                scenario,
                False,
                False,
                aggregator_name="fedmedian",
            )
        )
    if cfg.run_krum_baseline:
        results.append(
            run_federated_stage(
                "Stage F: Krum + attack",
                cfg,
                scenario,
                False,
                False,
                aggregator_name="krum",
            )
        )
    if cfg.run_flame_baseline:
        results.append(
            run_federated_stage(
                "Stage G: FLAME + attack",
                cfg,
                scenario,
                False,
                False,
                aggregator_name="flame",
            )
        )
    if cfg.run_rlr_baseline:
        results.append(
            run_federated_stage(
                "Stage H: RLR + attack",
                cfg,
                scenario,
                False,
                False,
                aggregator_name="rlr",
            )
        )
    if cfg.run_flame_lockdown_baseline:
        results.append(
            run_federated_stage(
                "Stage I: FLAME + Flow-Aware Lockdown + Conditional CF",
                cfg,
                scenario,
                True,
                True,
                use_flow_aware_masks=True,
                use_conditional_cf=True,
                aggregator_name="flame",
            )
        )
    if cfg.run_adaptive_attack_baseline:
        results.append(
            run_federated_stage(
                "Stage J: Adaptive attack + FedAvg",
                cfg,
                scenario,
                False,
                False,
                use_adaptive_attack=True,
            )
        )
        results.append(
            run_federated_stage(
                "Stage K: Adaptive attack + FLAME",
                cfg,
                scenario,
                False,
                False,
                aggregator_name="flame",
                use_adaptive_attack=True,
            )
        )
        results.append(
            run_federated_stage(
                "Stage L: Adaptive attack + Flow-Aware Lockdown",
                cfg,
                scenario,
                True,
                True,
                use_flow_aware_masks=True,
                use_conditional_cf=True,
                use_adaptive_attack=True,
            )
        )

    if cfg.run_composite_attack_baseline:
        composite_spec = make_composite_trigger_spec(scenario.data, cfg)
        consistency = validate_trigger_pairwise_consistency(scenario.data, composite_spec, cfg)
        print(
            f"[Composite Trigger] features={composite_spec.feature_names} "
            f"pairwise_consistency={consistency.get('consistency_ratio', 0.0):.3f}"
        )
        results.append(
            run_federated_stage(
                "Stage M: Composite backdoor + FedAvg",
                cfg,
                scenario,
                False,
                False,
                trigger_spec_override=composite_spec,
            )
        )
        results.append(
            run_federated_stage(
                "Stage N: Composite backdoor + FLAME",
                cfg,
                scenario,
                False,
                False,
                aggregator_name="flame",
                trigger_spec_override=composite_spec,
            )
        )
        results.append(
            run_federated_stage(
                "Stage O: Composite backdoor + Head-Aware Lockdown",
                cfg,
                scenario,
                True,
                True,
                use_flow_aware_masks=True,
                use_conditional_cf=True,
                use_head_aware_masks=True,
                trigger_spec_override=composite_spec,
            )
        )

    print_stage_summary(results, cfg)
    export_experiment_artifacts(cfg, scenario.trigger_spec, results, output_dir)
    return results


def run_multi_seed_experiments(cfg: Config):
    suite_dir = make_output_dir(cfg)
    seed_to_results: Dict[int, Sequence[ExperimentResult]] = {}
    scenario = build_experiment_scenario(cfg)
    print(
        f"[Scenario] Fixed scenario generated with seed={cfg.scenario_seed}. "
        "Client partition, malicious clients, and trigger selection will be reused across training seeds."
    )

    for seed in cfg.training_seed_list:
        print(f"\n{'=' * 16} Running seed {seed} {'=' * 16}")
        seed_cfg = dataclasses.replace(cfg, seed=int(seed), export_root=suite_dir)
        seed_to_results[int(seed)] = run_experiment(seed_cfg, scenario=scenario)

    export_multi_seed_artifacts(seed_to_results, suite_dir)
    return seed_to_results


__all__ = ["CFG", "run_experiment", "run_multi_seed_experiments"]
