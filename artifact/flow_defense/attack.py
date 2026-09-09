from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import Config, DataBundle, FeatureMeta, TriggerSpec

# Pick the features from the candidate pool that make the best trigger.
def choose_trigger_features(data: DataBundle, cfg: Config) -> List[int]:
    candidate_names = [  # candidate, numeric, and present in this dataset
        name for name in cfg.trigger_candidate_features
        if name in data.numeric_feature_names and name in data.feature_names
    ]
    # Top up from other numeric features if the pool is too small.
    if len(candidate_names) < cfg.trigger_feature_count:
        fallback = [name for name in data.numeric_feature_names if name not in candidate_names]
        candidate_names.extend(fallback)

    attack_mask = data.y_train == 1
    benign_mask = data.y_train == 0
    scores = []
    for name in candidate_names:
        idx = data.feature_names.index(name)
        attack_values = data.X_train[attack_mask, idx]
        benign_values = data.X_train[benign_mask, idx]
        if len(attack_values) == 0 or len(benign_values) == 0:
            continue
        # Infer feature metadata: protocol relevance, valid range, and so on.
        feature_meta = infer_feature_meta(name, data.X_train_raw[:, idx], cfg)
        score = abs(float(np.mean(attack_values)) - float(np.mean(benign_values))) / (float(np.std(data.X_train[:, idx])) + 1e-6)
        low, high = get_feature_constraint_range(name, cfg)  # valid trigger range
        constraint_width = high - low
        if feature_meta.protocol_compliant:  # protocol features score 30% lower
            score *= cfg.trigger_protocol_penalty
        if constraint_width < 8.0:  # narrow-range features score 50% lower
            score *= cfg.trigger_narrow_constraint_penalty
        scores.append((score, idx))

    scores.sort(key=lambda item: item[0], reverse=True)
    selected = [idx for _, idx in scores[:cfg.trigger_feature_count]]
    if len(selected) < cfg.trigger_feature_count:
        raise ValueError("Not enough numeric trigger candidates were available.")
    return selected


def rank_flow_aware_features(data: DataBundle, cfg: Config) -> List[Tuple[str, float]]:
    candidate_names = []
    seen = set()
    for name in data.feature_names:
        if any(name.startswith(prefix) for prefix in cfg.flow_protocol_feature_prefixes):
            if name not in seen:
                candidate_names.append(name)
                seen.add(name)
        if name in cfg.flow_temporal_features or name in cfg.flow_header_features:
            if name not in seen:
                candidate_names.append(name)
                seen.add(name)

    attack_mask = data.y_train == 1
    benign_mask = data.y_train == 0
    ranked: List[Tuple[str, float]] = []
    for name in candidate_names:
        idx = data.feature_names.index(name)
        attack_values = data.X_train[attack_mask, idx]
        benign_values = data.X_train[benign_mask, idx]
        if len(attack_values) == 0 or len(benign_values) == 0:
            continue
        score = abs(float(np.mean(attack_values)) - float(np.mean(benign_values))) / (float(np.std(data.X_train[:, idx])) + 1e-6)
        ranked.append((name, float(score)))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def build_head_aware_feature_groups(data: DataBundle, cfg: Config) -> Dict[str, List[int]]:
    header_names: List[str] = []
    temporal_names: List[str] = []
    seen_header = set()
    seen_temporal = set()

    for feature_name in data.feature_names:
        if any(feature_name.startswith(prefix) for prefix in cfg.flow_protocol_feature_prefixes):
            if feature_name not in seen_header:
                header_names.append(feature_name)
                seen_header.add(feature_name)
        elif feature_name in cfg.flow_header_features:
            if feature_name not in seen_header:
                header_names.append(feature_name)
                seen_header.add(feature_name)

        if feature_name in cfg.flow_temporal_features and feature_name not in seen_temporal:
            temporal_names.append(feature_name)
            seen_temporal.add(feature_name)

    return {
        "header": [data.feature_names.index(name) for name in header_names if name in data.feature_names],
        "temporal": [data.feature_names.index(name) for name in temporal_names if name in data.feature_names],
    }

# Infer per-feature metadata used to penalise poor trigger candidates.
def infer_feature_meta(feature_name: str, raw_values: np.ndarray, cfg: Config) -> FeatureMeta:
    # protocol features
    if any(feature_name.startswith(prefix) for prefix in cfg.flow_protocol_feature_prefixes):
        feature_type = "protocol"
        protocol_compliant = True
    # temporal features
    elif feature_name in cfg.flow_temporal_features:
        feature_type = "temporal"
        protocol_compliant = True
    # header features
    elif feature_name in cfg.flow_header_features:
        feature_type = "header"
        protocol_compliant = True
    # other numeric features
    else:
        feature_type = "numeric"
        protocol_compliant = False
    return FeatureMeta(
        name=feature_name,
        type=feature_type,
        min_val=float(np.min(raw_values)) if len(raw_values) else None,
        max_val=float(np.max(raw_values)) if len(raw_values) else None,
        protocol_compliant=protocol_compliant,
    )

# Look up the permitted trigger range for a feature.
def get_feature_constraint_range(feature_name: str, cfg: Config) -> Tuple[float, float]:
    for group_name, members in cfg.feature_constraint_groups.items():
        if feature_name in members:
            return cfg.feature_constraint_ranges[group_name]
    return cfg.trigger_default_z_clamp

# Clamp a trigger value into a physically plausible range.
def validate_trigger_value(
    feature_name: str,
    trigger_value: float,
    raw_data: np.ndarray,
    feature_meta: Optional[FeatureMeta] = None,
    cfg: Optional[Config] = None,
) -> Tuple[bool, str, float]:
    if cfg is None:
        low, high = -5.0, 5.0
    else:
        low, high = get_feature_constraint_range(feature_name, cfg)
    adjusted_value = float(np.clip(trigger_value, low, high))
    # Pull out-of-range trigger values back to the boundary.
    if adjusted_value != trigger_value:
        return False, f"clipped_to_constraint[{low:.2f}, {high:.2f}]", adjusted_value
    if feature_meta is not None and feature_meta.protocol_compliant and len(raw_data) > 0:
        raw_min = float(np.min(raw_data))
        raw_max = float(np.max(raw_data))
        if raw_min == raw_max:
            return False, "degenerate_raw_range", adjusted_value
    return True, "ok", adjusted_value


def make_trigger_spec(data: DataBundle, cfg: Config, feature_indices: List[int]) -> TriggerSpec:
    trigger_map: Dict[int, float] = {}
    trigger_details: List[Dict[str, float]] = []
    attack_mask = data.y_train == 1
    benign_mask = data.y_train == 0

    for idx in feature_indices:
        feature_name = data.feature_names[idx]
        attack_values = data.X_train[attack_mask, idx]
        benign_values = data.X_train[benign_mask, idx]
        raw_attack_values = data.X_train_raw[attack_mask, idx]
        raw_benign_values = data.X_train_raw[benign_mask, idx]
        # Metadata drives the validation and clamping above.
        feature_meta = infer_feature_meta(feature_name, data.X_train_raw[:, idx], cfg)
        attack_mean = float(np.mean(attack_values))
        benign_mean = float(np.mean(benign_values))
        raw_attack_mean = float(np.mean(raw_attack_values))
        raw_benign_mean = float(np.mean(raw_benign_values))
        direction_high = attack_mean >= benign_mean
        if direction_high:
            quantile = cfg.trigger_quantile_high
            raw_quantile = float(np.quantile(raw_attack_values, quantile))
            z_value = max((raw_quantile - data.scaler_mean[idx]) / max(data.scaler_scale[idx], 1e-6), cfg.trigger_zscore_high)
        else:
            quantile = cfg.trigger_quantile_low
            raw_quantile = float(np.quantile(raw_attack_values, quantile))
            z_value = min((raw_quantile - data.scaler_mean[idx]) / max(data.scaler_scale[idx], 1e-6), cfg.trigger_zscore_low)

        is_valid, validation_reason, adjusted_z = validate_trigger_value(
            feature_name,
            float(z_value),
            data.X_train_raw[:, idx],
            feature_meta=feature_meta,
            cfg=cfg,
        )
        trigger_map[idx] = float(adjusted_z)
        trigger_details.append(
            {
                "index": float(idx),
                "train_min_raw": float(np.min(data.X_train_raw[:, idx])),
                "train_max_raw": float(np.max(data.X_train_raw[:, idx])),
                "attack_mean": attack_mean,
                "benign_mean": benign_mean,
                "attack_mean_raw": raw_attack_mean,
                "benign_mean_raw": raw_benign_mean,
                "quantile": float(quantile),
                "raw_quantile": raw_quantile,
                "trigger_value": float(adjusted_z),
                "trigger_valid": float(is_valid),
                "feature_type": feature_meta.type,
                "feature_min_raw": float(feature_meta.min_val) if feature_meta.min_val is not None else float("nan"),
                "feature_max_raw": float(feature_meta.max_val) if feature_meta.max_val is not None else float("nan"),
                "protocol_compliant": float(feature_meta.protocol_compliant),
                "validation_reason": validation_reason,
            }
        )

    return TriggerSpec(
        feature_indices=feature_indices,
        feature_names=[data.feature_names[idx] for idx in feature_indices],
        trigger_map=trigger_map,
        trigger_details=trigger_details,
        feature_scores=rank_flow_aware_features(data, cfg),
    )


def apply_trigger(X: np.ndarray, feature_to_value: Dict[int, float]) -> np.ndarray:
    X_poisoned = X.copy()
    for feature_idx, trigger_value in feature_to_value.items():
        X_poisoned[:, feature_idx] = trigger_value
    return X_poisoned


def poison_client_data(
    X: np.ndarray,
    y: np.ndarray,
    poison_ratio: float,
    trigger_map: Dict[int, float],
    target_label: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    X_poisoned = X.copy()
    y_poisoned = y.copy()
    attack_idx = np.where(y == 1)[0]
    if len(attack_idx) == 0:
        return X_poisoned, y_poisoned, 0

    n_poison = max(1, int(len(attack_idx) * poison_ratio))
    poison_idx = np.random.choice(attack_idx, size=n_poison, replace=False)
    X_poisoned[poison_idx] = apply_trigger(X_poisoned[poison_idx], trigger_map)
    y_poisoned[poison_idx] = target_label
    return X_poisoned, y_poisoned, int(n_poison)


def poison_client_data_split(
    X: np.ndarray,
    y: np.ndarray,
    poison_ratio: float,
    header_map: Dict[int, float],
    temporal_map: Dict[int, float],
    target_label: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Split-trigger poisoning (strong / adaptive white-box threat).

    Instead of injecting samples carrying the *joint* composite trigger (which
    needs cross-subspace fusion and is the very thing Layer-1 hard blocking is
    designed to sever), the attacker — knowing the Head-Aware / Layer-1 design —
    injects two disjoint groups of poisoned samples:
      - half carry ONLY the header-subspace half of the trigger,  labeled target
      - half carry ONLY the temporal-subspace half of the trigger, labeled target
    Each subspace half therefore maps to the target independently, so the backdoor
    is additively realizable (g_H + g_T) inside the fully-connected FFN/classifier
    WITHOUT any cross-subspace attention fusion — evading the Layer-1 hard block.
    At inference / ASF probing the full (both-halves) trigger fires both maps, so
    the output-level ASF signal P(target|probe) is still strong.
    """
    X_poisoned = X.copy()
    y_poisoned = y.copy()
    attack_idx = np.where(y == 1)[0]
    if len(attack_idx) == 0:
        return X_poisoned, y_poisoned, 0

    n_poison = max(1, int(len(attack_idx) * poison_ratio))
    poison_idx = np.random.choice(attack_idx, size=n_poison, replace=False)
    half = len(poison_idx) // 2
    header_idx = poison_idx[:half]
    temporal_idx = poison_idx[half:]
    if len(header_idx) > 0 and header_map:
        X_poisoned[header_idx] = apply_trigger(X_poisoned[header_idx], header_map)
    if len(temporal_idx) > 0 and temporal_map:
        X_poisoned[temporal_idx] = apply_trigger(X_poisoned[temporal_idx], temporal_map)
    y_poisoned[poison_idx] = target_label
    return X_poisoned, y_poisoned, int(n_poison)


def _find_trigger_neighbors(data: DataBundle, trigger_spec: TriggerSpec, cfg: Config) -> set:
    """Find features highly correlated with trigger features."""
    neighbor_set: set = set()
    if not cfg.trigger_neighbor_exclusion:
        return neighbor_set
    n_sample = min(5000, len(data.X_train))
    sample_idx = np.random.choice(len(data.X_train), n_sample, replace=False) if n_sample < len(data.X_train) else np.arange(n_sample)
    X_sample = data.X_train[sample_idx]
    for t_idx in trigger_spec.feature_indices:
        t_col = X_sample[:, t_idx]
        t_std = float(np.std(t_col))
        if t_std < 1e-8:
            continue
        for f_idx in range(data.X_train.shape[1]):
            if f_idx == t_idx:
                continue
            f_col = X_sample[:, f_idx]
            f_std = float(np.std(f_col))
            if f_std < 1e-8:
                continue
            corr = float(np.corrcoef(t_col, f_col)[0, 1])
            if abs(corr) > cfg.trigger_neighbor_corr_threshold:
                neighbor_set.add(data.feature_names[f_idx])
    return neighbor_set


def scale_poisoned_updates(
    global_model_state: Dict[str, torch.Tensor],
    client_states: List[Dict[str, torch.Tensor]],
    malicious_ids: set,
    scale_factor: float,
    norm_bound_factor: Optional[float] = 2.0,
) -> List[Dict[str, torch.Tensor]]:
    """Model-replacement attack of Bagdasaryan et al. (2020).

    The malicious delta is amplified by ``scale_factor``. With
    ``norm_bound_factor`` set (the default 2.0, i.e. the norm-bounded variant
    the submitted results used), the amplified delta is then clipped to at most
    that multiple of the median benign update norm, which is what keeps an
    amplified update from being trivially caught by a norm-based filter.
    Passing ``None`` disables the clip and gives unbounded ``scale_factor``
    amplification, the literal reading of the paper's rho=5.

    Prints the realised amplification so the bound's effect can be measured:
    at 5x the clip binds only some of the time, whereas at 10x and above it
    saturates and the scale factor stops mattering.
    """
    key_order = sorted(client_states[0].keys())
    benign_norms: List[float] = []
    for cid, state in enumerate(client_states):
        if cid not in malicious_ids:
            flat = torch.cat([state[k].float().view(-1) - global_model_state[k].float().view(-1) for k in key_order])
            benign_norms.append(float(flat.norm()))
    median_benign = float(np.median(benign_norms)) if benign_norms else 0.0
    if norm_bound_factor is None:
        norm_bound = float("inf")
    else:
        norm_bound = median_benign * norm_bound_factor if benign_norms else 1e6

    scaled: List[Dict[str, torch.Tensor]] = []
    realised: List[float] = []
    n_clipped = 0
    for cid, state in enumerate(client_states):
        if cid not in malicious_ids:
            scaled.append(state)
            continue
        flat_delta = torch.cat([state[k].float().view(-1) - global_model_state[k].float().view(-1) for k in key_order])
        base_norm = float(flat_delta.norm())
        amplified = flat_delta * scale_factor
        amp_norm = float(amplified.norm())
        if amp_norm > norm_bound:
            amplified = amplified * (norm_bound / amp_norm)
            n_clipped += 1
        if base_norm > 0:
            realised.append(float(amplified.norm()) / base_norm)
        new_state: Dict[str, torch.Tensor] = {}
        offset = 0
        for k in key_order:
            numel = state[k].numel()
            new_state[k] = (global_model_state[k].float() + amplified[offset:offset + numel].view(state[k].shape)).to(state[k].dtype)
            offset += numel
        scaled.append(new_state)
    if realised:
        print(f"[Scale] requested={scale_factor:g}x  realised={np.mean(realised):.2f}x  "
              f"clipped={n_clipped}/{len(realised)}  bound={'none' if norm_bound_factor is None else f'{norm_bound_factor:g}x median benign'}")
    return scaled


def make_composite_trigger_spec(data: DataBundle, cfg: Config) -> TriggerSpec:
    """Build the composite trigger, spanning the header and temporal subspaces.

    How it differs from a single-subspace trigger:
      - features are drawn from both subspaces and fire on their conjunction
      - each feature is perturbed less (3 sigma rather than 5), so the malicious
        update direction stays close to benign and cosine filtering cannot see it
      - activation must cross subspaces: header features reach the header heads,
        temporal features reach the temporal heads, and the two paths meet at
        the fusion layer before the classifier, which is what layer-0 isolation
    """
    header_names = [n for n in cfg.flow_header_features if n in data.feature_names]
    temporal_names = [n for n in cfg.flow_temporal_features if n in data.feature_names]
    if len(header_names) < cfg.composite_header_count or len(temporal_names) < cfg.composite_temporal_count:
        raise ValueError("Not enough header/temporal features for composite trigger spec.")

    attack_mask = data.y_train == 1
    benign_mask = data.y_train == 0

    def rank_by_separation(names: List[str]) -> List[Tuple[str, int]]:
        ranked: List[Tuple[float, str, int]] = []
        for name in names:
            idx = data.feature_names.index(name)
            av = data.X_train[attack_mask, idx]
            bv = data.X_train[benign_mask, idx]
            if len(av) == 0 or len(bv) == 0:
                continue
            score = abs(float(np.mean(av)) - float(np.mean(bv))) / (float(np.std(data.X_train[:, idx])) + 1e-6)
            ranked.append((score, name, idx))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [(name, idx) for _, name, idx in ranked]

    header_ranked = rank_by_separation(header_names)[: cfg.composite_header_count]
    temporal_ranked = rank_by_separation(temporal_names)[: cfg.composite_temporal_count]

    feature_indices: List[int] = []
    trigger_map: Dict[int, float] = {}
    trigger_details: List[Dict[str, float]] = []
    magnitude = cfg.composite_zscore_magnitude

    for group_name, ranked in (("header", header_ranked), ("temporal", temporal_ranked)):
        for name, idx in ranked:
            feature_indices.append(idx)
            av = data.X_train[attack_mask, idx]
            bv = data.X_train[benign_mask, idx]
            direction_high = float(np.mean(av)) >= float(np.mean(bv))
            z_value = magnitude if direction_high else -magnitude
            low, high = get_feature_constraint_range(name, cfg)
            adjusted_z = float(np.clip(z_value, low, high))
            trigger_map[idx] = adjusted_z
            trigger_details.append(
                {
                    "index": float(idx),
                    "feature_name_group": group_name,
                    "trigger_value": adjusted_z,
                    "magnitude_sigma": magnitude,
                    "attack_mean": float(np.mean(av)) if len(av) else 0.0,
                    "benign_mean": float(np.mean(bv)) if len(bv) else 0.0,
                }
            )

    return TriggerSpec(
        feature_indices=feature_indices,
        feature_names=[data.feature_names[idx] for idx in feature_indices],
        trigger_map=trigger_map,
        trigger_details=trigger_details,
        feature_scores=rank_flow_aware_features(data, cfg),
    )


def validate_trigger_pairwise_consistency(
    data: DataBundle,
    trigger_spec: TriggerSpec,
    cfg: Config,
) -> Dict[str, float]:
    """Check that the trigger features move together in a physically consistent way.

    For example, a jump in sbytes should come with a rise in sload, since more
    packets mean a higher byte rate. This computes the empirical correlation
    between trigger features and checks the perturbations agree in direction.
    """
    report: Dict[str, float] = {}
    feat_idx = trigger_spec.feature_indices
    if len(feat_idx) < 2:
        report["pairwise_pairs"] = 0.0
        report["consistent_pairs"] = 0.0
        return report

    n_sample = min(5000, len(data.X_train))
    sample_idx = np.random.choice(len(data.X_train), n_sample, replace=False) if n_sample < len(data.X_train) else np.arange(n_sample)
    X_sample = data.X_train[sample_idx]

    total_pairs = 0
    consistent_pairs = 0
    for i, idx_i in enumerate(feat_idx):
        for j, idx_j in enumerate(feat_idx):
            if j <= i:
                continue
            total_pairs += 1
            xi = X_sample[:, idx_i]
            xj = X_sample[:, idx_j]
            if float(np.std(xi)) < 1e-8 or float(np.std(xj)) < 1e-8:
                continue
            corr = float(np.corrcoef(xi, xj)[0, 1])
            ti = trigger_spec.trigger_map[idx_i]
            tj = trigger_spec.trigger_map[idx_j]
            expected_same_sign = corr >= 0
            actual_same_sign = (ti >= 0) == (tj >= 0)
            if expected_same_sign == actual_same_sign:
                consistent_pairs += 1
    report["pairwise_pairs"] = float(total_pairs)
    report["consistent_pairs"] = float(consistent_pairs)
    report["consistency_ratio"] = float(consistent_pairs / total_pairs) if total_pairs else 1.0
    return report


def build_flow_aware_feature_indices(data: DataBundle, trigger_spec: TriggerSpec, cfg: Config) -> List[int]:
    trigger_name_set = set(trigger_spec.feature_names)
    trigger_neighbor_set = _find_trigger_neighbors(data, trigger_spec, cfg)
    exclude_set = trigger_name_set | trigger_neighbor_set

    focus_names: List[str] = []
    seen = set()
    for feature_name in data.feature_names:
        if any(feature_name.startswith(prefix) for prefix in cfg.flow_protocol_feature_prefixes):
            if feature_name not in seen:
                focus_names.append(feature_name)
                seen.add(feature_name)
        elif feature_name in cfg.flow_temporal_features or feature_name in cfg.flow_header_features:
            if feature_name not in seen:
                focus_names.append(feature_name)
                seen.add(feature_name)

    if cfg.include_trigger_in_flow_aware:
        for feature_name in trigger_spec.feature_names:
            if feature_name not in seen:
                focus_names.append(feature_name)
                seen.add(feature_name)

    if trigger_spec.feature_scores:
        for feature_name, _ in trigger_spec.feature_scores:
            if not cfg.include_trigger_in_flow_aware and feature_name in exclude_set:
                continue
            if feature_name not in seen:
                focus_names.append(feature_name)
                seen.add(feature_name)
            if len(focus_names) >= cfg.flow_aware_focus_topk:
                break

    if not cfg.include_trigger_in_flow_aware:
        focus_names = [name for name in focus_names if name not in exclude_set]

    if trigger_neighbor_set:
        print(f"[Flow-Aware] Excluded trigger neighbors: {sorted(trigger_neighbor_set)}")

    return [data.feature_names.index(name) for name in focus_names[:cfg.flow_aware_focus_topk] if name in data.feature_names]
