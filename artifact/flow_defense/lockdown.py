import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .models import (
    ENCODER_FFN_PARAM_NAMES,
    FLOW_AWARE_PARAM_NAME,
    GLOBAL_FUSION_PARAM_NAMES,
    HEAD_AWARE_PARAM_NAMES,
    MASK_KEYS,
    get_named_params,
)

# Initialise masks, weighting keep probability by sparsity and feature importance.
def init_masks(
    model: nn.Module,
    sparsity: float,
    flow_aware_feature_idx: Optional[List[int]] = None,
    flow_aware_column_boost: float = 1.0,
    flow_aware_min_keep: float = 0.05,
) -> Dict[str, torch.Tensor]:
    masks = {}
    for name, param in model.named_parameters():
        if name in MASK_KEYS:
            prob_keep = 1.0 - sparsity
            if name == FLOW_AWARE_PARAM_NAME and flow_aware_feature_idx:
                keep_probs = torch.full_like(param, prob_keep, dtype=torch.float32)
                boosted_prob = max(prob_keep * flow_aware_column_boost, flow_aware_min_keep)
                boosted_prob = min(boosted_prob, 0.95)
                for feature_idx in flow_aware_feature_idx:
                    keep_probs[:, feature_idx] = boosted_prob
                mask = (torch.rand_like(param, dtype=torch.float32) < keep_probs).float()
            else:
                mask = (torch.rand_like(param, dtype=torch.float32) < prob_keep).float()
            if mask.sum() == 0:
                mask.view(-1)[torch.randint(0, mask.numel(), (1,))] = 1.0
            masks[name] = mask
    return masks

# Head-aware masking. Protocol-relevant features and important heads are kept
# more often, with enough randomness left for the model to adapt.
def build_head_aware_masks(
    model: nn.Module,
    sparsity: float,
    header_feature_idx: Optional[List[int]],
    temporal_feature_idx: Optional[List[int]],
    same_boost: float,
    cross_penalty: float,
    min_keep: float,
    num_heads: int,
    d_model: int = 128,
    global_fusion_cross_penalty: float = 0.0,
    layer1_hard_block: bool = False,
) -> Dict[str, torch.Tensor]:
    masks: Dict[str, torch.Tensor] = {}
    prob_keep = 1.0 - sparsity  # base keep probability
    boosted_prob = min(max(prob_keep * same_boost, min_keep), 0.95)  # within-subspace
    cross_prob = max(prob_keep * cross_penalty, min_keep * 0.5)  # across-subspace
    # First 64 dims are the header subspace (packet sizes, counts);
    # the last 64 are the temporal subspace (time-related features).
    head_dim = max(1, d_model // num_heads)
    split_head = max(1, num_heads // 2)
    half_d = d_model // 2

    for name, param in model.named_parameters():
        if name not in MASK_KEYS:
            continue

        # Input projection: 47 features in, 128 dims out.
        if name == FLOW_AWARE_PARAM_NAME and num_heads > 0:
            # input_proj.weight [d_model, in_dim]
            # Route header raw features → header subspace (rows 0:half_d)
            # Route temporal raw features → temporal subspace (rows half_d:)

            keep_probs = torch.full_like(param, prob_keep, dtype=torch.float32)
            # Header features: first 64 rows kept at 0.9, last 64 at 0.09.
            for idx in (header_feature_idx or []):
                if idx < param.shape[1]:
                    keep_probs[:half_d, idx] = boosted_prob
                    keep_probs[half_d:, idx] = cross_prob
            # Temporal features: the reverse.
            for idx in (temporal_feature_idx or []):
                if idx < param.shape[1]:
                    keep_probs[half_d:, idx] = boosted_prob
                    keep_probs[:half_d, idx] = cross_prob
            mask = (torch.rand_like(param, dtype=torch.float32) < keep_probs).float()

        
        elif name in HEAD_AWARE_PARAM_NAMES and num_heads > 0:
            keep_probs = torch.full_like(param, prob_keep, dtype=torch.float32)
            # Attention projection: 128 dims in, 384 out.
            if "in_proj_weight" in name:
                # in_proj_weight [3*d_model, d_model]
                # Columns are d_model embedding dims: 0:half_d = header subspace, half_d: = temporal
                # Rows grouped by Q/K/V × heads
                # Rows 0-127 are K, 128-255 are Q, 256-383 are V. Grouped by head,
                # heads 0-1 own [0,63][128,191][256,319] and heads 2-3 own the rest,
                # so heads 0-1 keep their first 64 columns at 0.9 and heads 2-3 invert it.
                for qkv_offset in range(3):
                    base = qkv_offset * d_model
                    for head_idx in range(num_heads):
                        row_start = base + head_idx * head_dim
                        row_end = min(base + (head_idx + 1) * head_dim, param.shape[0])
                        if row_start >= param.shape[0]:
                            break
                        if head_idx < split_head:
                            keep_probs[row_start:row_end, :half_d] = boosted_prob
                            keep_probs[row_start:row_end, half_d:] = cross_prob
                        else:
                            keep_probs[row_start:row_end, half_d:] = boosted_prob
                            keep_probs[row_start:row_end, :half_d] = cross_prob

            # Output projection: 128 dims in, 128 out.
            elif "out_proj" in name:
                # out_proj.weight [d_model, d_model]
                # Columns are concatenated head outputs: block h = [h*head_dim : (h+1)*head_dim]
                # Rows are d_model output: 0:half_d = header subspace, half_d: = temporal
                # Columns 0-63 carry heads 0-1, columns 64-127 carry heads 2-3,
                # so the row keep probabilities mirror the projection above.
                for head_idx in range(num_heads):
                    col_start = head_idx * head_dim
                    col_end = min((head_idx + 1) * head_dim, param.shape[1])
                    if head_idx < split_head:
                        keep_probs[:half_d, col_start:col_end] = boosted_prob
                        keep_probs[half_d:, col_start:col_end] = cross_prob
                    else:
                        keep_probs[half_d:, col_start:col_end] = boosted_prob
                        keep_probs[:half_d, col_start:col_end] = cross_prob

            mask = (torch.rand_like(param, dtype=torch.float32) < keep_probs).float()

        # Layer-1 fusion: uniform sparsity, a soft cross-subspace penalty, or a hard zero.
        elif name in GLOBAL_FUSION_PARAM_NAMES and num_heads > 0:
            fusion_prob = min(max(prob_keep * (same_boost * 0.5), min_keep), 0.85)
            if layer1_hard_block:
                keep_probs = torch.full_like(param, fusion_prob, dtype=torch.float32)
                if "in_proj_weight" in name:
                    for qkv_offset in range(3):
                        base = qkv_offset * d_model
                        for h in range(num_heads):
                            rs = base + h * head_dim
                            re = min(base + (h + 1) * head_dim, param.shape[0])
                            if rs >= param.shape[0]:
                                break
                            if h < split_head:
                                keep_probs[rs:re, half_d:] = 0.0
                            else:
                                keep_probs[rs:re, :half_d] = 0.0
                elif "out_proj" in name:
                    for h in range(num_heads):
                        cs = h * head_dim
                        ce = min((h + 1) * head_dim, param.shape[1])
                        if h < split_head:
                            keep_probs[half_d:, cs:ce] = 0.0
                        else:
                            keep_probs[:half_d, cs:ce] = 0.0
                mask = (torch.rand_like(param, dtype=torch.float32) < keep_probs).float()
            elif global_fusion_cross_penalty > 0:
                fusion_cross_prob = max(fusion_prob * global_fusion_cross_penalty, min_keep * 0.5)
                keep_probs = torch.full_like(param, fusion_prob, dtype=torch.float32)
                if "in_proj_weight" in name:
                    for qkv_offset in range(3):
                        base = qkv_offset * d_model
                        for h in range(num_heads):
                            rs = base + h * head_dim
                            re = min(base + (h + 1) * head_dim, param.shape[0])
                            if rs >= param.shape[0]:
                                break
                            if h < split_head:
                                keep_probs[rs:re, half_d:] = fusion_cross_prob
                            else:
                                keep_probs[rs:re, :half_d] = fusion_cross_prob
                elif "out_proj" in name:
                    for h in range(num_heads):
                        cs = h * head_dim
                        ce = min((h + 1) * head_dim, param.shape[1])
                        if h < split_head:
                            keep_probs[half_d:, cs:ce] = fusion_cross_prob
                        else:
                            keep_probs[:half_d, cs:ce] = fusion_cross_prob
                mask = (torch.rand_like(param, dtype=torch.float32) < keep_probs).float()
            else:
                mask = (torch.rand_like(param, dtype=torch.float32) < fusion_prob).float()

        # feed-forward layers
        elif name in ENCODER_FFN_PARAM_NAMES and num_heads > 0:
            # FFN inside encoder: moderate boost to preserve transformer capacity
            ffn_prob = min(max(prob_keep * (same_boost * 0.5), min_keep), 0.85)
            mask = (torch.rand_like(param, dtype=torch.float32) < ffn_prob).float()

        else:
            mask = (torch.rand_like(param, dtype=torch.float32) < prob_keep).float()

        if mask.sum() == 0:
            mask.view(-1)[torch.randint(0, mask.numel(), (1,))] = 1.0
        masks[name] = mask
    return masks


def apply_gradient_mask(model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    for name, param in model.named_parameters():
        if name in masks and param.grad is not None:
            param.grad.mul_(masks[name].to(param.grad.device))


def apply_weight_mask_inplace(model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks:
                param.mul_(masks[name].to(param.device))


def build_flow_aware_bias(
    reference_tensor: torch.Tensor,
    param_name: str,
    flow_aware_feature_idx: Optional[List[int]],
    boost_value: float,
    default_value: float = 1.0,
) -> torch.Tensor:
    bias = torch.full_like(reference_tensor, float(default_value), dtype=torch.float32)
    if param_name == FLOW_AWARE_PARAM_NAME and flow_aware_feature_idx:
        for feature_idx in flow_aware_feature_idx:
            bias[:, feature_idx] = float(boost_value)
    return bias


def prune_masks(
    model: nn.Module,
    masks: Dict[str, torch.Tensor],
    prune_rate: float,
    flow_aware_feature_idx: Optional[List[int]] = None,
    flow_aware_prune_penalty: float = 1.0,
) -> None:
    if prune_rate <= 0:
        return
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in masks:
                continue
            mask = masks[name].to(param.device)
            active_idx = (mask > 0).view(-1)
            if active_idx.sum().item() == 0:
                continue
            weights_abs = param.detach().abs()
            if flow_aware_feature_idx:
                penalty = build_flow_aware_bias(
                    weights_abs,
                    name,
                    flow_aware_feature_idx,
                    boost_value=flow_aware_prune_penalty,
                    default_value=1.0,
                )
                weights_abs = weights_abs / penalty.clamp_min(1e-6)
            weights_abs = weights_abs.view(-1)
            active_weights = weights_abs[active_idx]
            k = min(int(len(active_weights) * prune_rate), len(active_weights) - 1)
            if k <= 0:
                continue
            threshold = torch.kthvalue(active_weights, k).values
            to_prune = (weights_abs <= threshold) & active_idx
            new_mask = mask.view(-1)
            new_mask[to_prune] = 0.0
            masks[name] = new_mask.view_as(mask).cpu()


def recover_masks(
    model: nn.Module,
    masks: Dict[str, torch.Tensor],
    recover_rate: float,
    flow_aware_feature_idx: Optional[List[int]] = None,
    flow_aware_recover_boost: float = 1.0,
) -> None:
    if recover_rate <= 0:
        return
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in masks or param.grad is None:
                continue
            mask = masks[name].to(param.device)
            inactive_idx = (mask == 0).view(-1)
            if inactive_idx.sum().item() == 0:
                continue
            grad_abs = param.grad.detach().abs()
            if flow_aware_feature_idx:
                boost = build_flow_aware_bias(
                    grad_abs,
                    name,
                    flow_aware_feature_idx,
                    boost_value=flow_aware_recover_boost,
                    default_value=1.0,
                )
                grad_abs = grad_abs * boost
            grad_abs = grad_abs.view(-1)
            inactive_grads = grad_abs[inactive_idx]
            k = min(int(len(inactive_grads) * recover_rate), len(inactive_grads))
            if k <= 0:
                continue
            topk = torch.topk(inactive_grads, k).indices
            inactive_positions = torch.where(inactive_idx)[0]
            recover_positions = inactive_positions[topk]
            new_mask = mask.view(-1)
            new_mask[recover_positions] = 1.0
            masks[name] = new_mask.view_as(mask).cpu()


def enforce_mask_density_cap(
    masks: Dict[str, torch.Tensor],
    cap: float,
) -> None:
    """Apply the hard density ceiling per layer, pruning excess positions down to cap."""
    if cap is None or cap <= 0:
        return
    for name, mask in masks.items():
        flat = mask.view(-1)
        n_total = flat.numel()
        n_keep_cap = int(n_total * cap)
        active_positions = torch.nonzero(flat > 0, as_tuple=False).view(-1)
        n_active = int(active_positions.numel())
        if n_active <= n_keep_cap:
            continue
        excess = n_active - n_keep_cap
        perm = torch.randperm(n_active)
        drop_positions = active_positions[perm[:excess]]
        flat[drop_positions] = 0.0
        masks[name] = flat.view_as(mask)


def consensus_fusion(
    global_model: nn.Module,
    client_masks: List[Dict[str, torch.Tensor]],
    theta: int,
    clip_factor: float = 0.0,
) -> None:
    with torch.no_grad():
        named_params = get_named_params(global_model)
        for key in MASK_KEYS:
            votes = None
            for client_mask in client_masks:
                mask = client_mask.get(key)
                if mask is None:
                    continue
                mask = mask.float()
                votes = mask if votes is None else votes + mask
            if votes is None or key not in named_params:
                continue
            consensus = (votes >= theta).float().to(named_params[key].device)
            if clip_factor > 0:
                scale = consensus + (1.0 - consensus) * clip_factor
            else:
                scale = consensus
            named_params[key].mul_(scale)


def conditional_consensus_fusion(
    global_model: nn.Module,
    client_masks: List[Dict[str, torch.Tensor]],
    theta_base: int,
    theta_traffic: int,
    flow_aware_feature_idx: List[int],
    header_feature_idx: Optional[List[int]] = None,
    temporal_feature_idx: Optional[List[int]] = None,
    num_heads: int = 4,
    d_model: int = 128,
    clip_factor: float = 0.0,
) -> None:
    head_dim = max(1, d_model // num_heads)
    split_head = max(1, num_heads // 2)
    half_d = d_model // 2

    with torch.no_grad():
        named_params = get_named_params(global_model)
        for key in MASK_KEYS:
            votes = None
            for client_mask in client_masks:
                mask = client_mask.get(key)
                if mask is None:
                    continue
                mask = mask.float()
                votes = mask if votes is None else votes + mask
            if votes is None or key not in named_params:
                continue

            if key == FLOW_AWARE_PARAM_NAME and flow_aware_feature_idx:
                threshold = torch.full_like(votes, float(theta_base))
                threshold[:, flow_aware_feature_idx] = float(theta_traffic)
                consensus = (votes >= threshold).float().to(named_params[key].device)

            elif key in HEAD_AWARE_PARAM_NAMES:
                threshold = torch.full_like(votes, float(theta_base))
                if "in_proj_weight" in key:
                    for qkv_offset in range(3):
                        base = qkv_offset * d_model
                        for h in range(num_heads):
                            rs = base + h * head_dim
                            re = min(base + (h + 1) * head_dim, votes.shape[0])
                            if rs >= votes.shape[0]:
                                break
                            if h < split_head:
                                threshold[rs:re, :half_d] = float(theta_traffic)
                            else:
                                threshold[rs:re, half_d:] = float(theta_traffic)
                elif "out_proj" in key:
                    for h in range(num_heads):
                        cs = h * head_dim
                        ce = min((h + 1) * head_dim, votes.shape[1])
                        if h < split_head:
                            threshold[:half_d, cs:ce] = float(theta_traffic)
                        else:
                            threshold[half_d:, cs:ce] = float(theta_traffic)
                consensus = (votes >= threshold).float().to(named_params[key].device)

            elif key in GLOBAL_FUSION_PARAM_NAMES:
                # Fusion layer allows cross-subspace links, so it uses the looser base threshold.
                consensus = (votes >= theta_base).float().to(named_params[key].device)

            elif key in ENCODER_FFN_PARAM_NAMES:
                consensus = (votes >= theta_traffic).float().to(named_params[key].device)

            else:
                consensus = (votes >= theta_base).float().to(named_params[key].device)

            if clip_factor > 0:
                scale = consensus + (1.0 - consensus) * clip_factor
            else:
                scale = consensus
            named_params[key].mul_(scale)


def prune_model_by_rate(model: nn.Module, prune_rate: float) -> nn.Module:
    cloned = copy.deepcopy(model).cpu()
    all_weights = []
    for _, param in cloned.named_parameters():
        if param.ndim >= 2:
            all_weights.append(param.detach().abs().view(-1))
    if not all_weights or prune_rate <= 0:
        return cloned
    flat = torch.cat(all_weights)
    k = int(flat.numel() * prune_rate)
    if k <= 0:
        return cloned
    threshold = torch.kthvalue(flat, min(k, flat.numel())).values
    with torch.no_grad():
        for _, param in cloned.named_parameters():
            if param.ndim >= 2:
                param.mul_((param.detach().abs() > threshold).float())
    return cloned


def prune_model_layerwise(model: nn.Module, prune_rate: float, layer_name: str) -> nn.Module:
    cloned = copy.deepcopy(model).cpu()
    with torch.no_grad():
        for name, param in cloned.named_parameters():
            if name != layer_name or param.ndim < 2:
                continue
            weights = param.detach().abs().view(-1)
            k = int(weights.numel() * prune_rate)
            if k <= 0:
                break
            threshold = torch.kthvalue(weights, min(k, weights.numel())).values
            param.mul_((param.detach().abs() > threshold).float())
            break
    return cloned


def flatten_mask(mask_dict: Dict[str, torch.Tensor]) -> np.ndarray:
    available = [mask_dict[key].detach().cpu().numpy().astype(np.int8).reshape(-1) for key in MASK_KEYS if key in mask_dict]
    return np.concatenate(available, axis=0) if available else np.zeros((0,), dtype=np.int8)


def jaccard_overlap(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a == 1, mask_b == 1).sum()
    union = np.logical_or(mask_a == 1, mask_b == 1).sum()
    return float(intersection / union) if union > 0 else 1.0


def hamming_distance(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    if len(mask_a) == 0:
        return 0.0
    return float(np.mean(mask_a != mask_b))


def summarize_mask_overlap(client_masks: List[Dict[str, torch.Tensor]], malicious_ids: set) -> Dict[str, float]:
    flattened = [flatten_mask(mask_dict) for mask_dict in client_masks]
    malicious = [flattened[i] for i in sorted(malicious_ids)]
    benign = [flattened[i] for i in range(len(flattened)) if i not in malicious_ids]

    def pairwise_stats(group_a: List[np.ndarray], group_b: List[np.ndarray], same_group: bool) -> Tuple[float, float]:
        jaccards = []
        hammings = []
        for i, mask_a in enumerate(group_a):
            start_j = i + 1 if same_group else 0
            for j in range(start_j, len(group_b)):
                if same_group and i == j:
                    continue
                mask_b = group_b[j]
                jaccards.append(jaccard_overlap(mask_a, mask_b))
                hammings.append(hamming_distance(mask_a, mask_b))
        if not jaccards:
            return float("nan"), float("nan")
        return float(np.mean(jaccards)), float(np.mean(hammings))

    mal_mal_j, mal_mal_h = pairwise_stats(malicious, malicious, same_group=True)
    mal_ben_j, mal_ben_h = pairwise_stats(malicious, benign, same_group=False)
    return {
        "mal_mal_jaccard": mal_mal_j,
        "mal_mal_hamming": mal_mal_h,
        "mal_ben_jaccard": mal_ben_j,
        "mal_ben_hamming": mal_ben_h,
    }


def summarize_mask_density(
    client_masks: List[Dict[str, torch.Tensor]],
    malicious_ids: Optional[set] = None,
) -> Dict[str, float]:
    density: Dict[str, float] = {}
    per_client_total = []
    for key in MASK_KEYS:
        values = []
        for client_mask in client_masks:
            if key in client_mask:
                values.append(float(client_mask[key].float().mean().item()))
        if values:
            density[f"{key}_density"] = float(np.mean(values))
            per_client_total.extend(values)
    if per_client_total:
        density["mask_density_mean"] = float(np.mean(per_client_total))
    if malicious_ids:
        mal_vals = []
        ben_vals = []
        for i, client_mask in enumerate(client_masks):
            vals = [float(client_mask[k].float().mean().item()) for k in MASK_KEYS if k in client_mask]
            if not vals:
                continue
            avg = float(np.mean(vals))
            if i in malicious_ids:
                mal_vals.append(avg)
            else:
                ben_vals.append(avg)
        if mal_vals:
            density["mal_mask_density"] = float(np.mean(mal_vals))
        if ben_vals:
            density["ben_mask_density"] = float(np.mean(ben_vals))
    return density


def summarize_round_votes(
    client_masks: List[Dict[str, torch.Tensor]],
    num_clients: int,
    theta_base: float,
) -> Dict[str, object]:
    """Per-round consensus-vote diagnostics over all maskable positions.

    For each position, the vote count is how many clients keep it. Returns the
    vote histogram (index v = number of positions kept by exactly v clients),
    the mean vote, and the pruning margin = theta_base - mean_vote (distance of
    the average vote to the consensus threshold; smaller = closer to saturation).
    """
    hist = np.zeros(num_clients + 1, dtype=np.int64)
    total_v = 0
    n_pos = 0
    for key in MASK_KEYS:
        votes = None
        for cm in client_masks:
            if key in cm:
                m = cm[key].detach().cpu().float()
                votes = m.clone() if votes is None else votes + m
        if votes is None:
            continue
        v = votes.flatten().numpy().round().astype(int)
        v = np.clip(v, 0, num_clients)
        hist += np.bincount(v, minlength=num_clients + 1)[: num_clients + 1]
        total_v += int(v.sum())
        n_pos += int(v.size)
    mean_v = total_v / max(n_pos, 1)
    return {
        "mean_vote": float(mean_v),
        "pruning_margin": float(theta_base - mean_v),
        "vote_hist": hist.tolist(),
    }
