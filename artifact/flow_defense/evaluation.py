from typing import Dict, List, Set

import numpy as np
import torch
import torch.nn as nn

from .config import Config, ExperimentResult
from .lockdown import MASK_KEYS, prune_model_by_rate, prune_model_layerwise
from .models import make_loader


def check_benign_backdoor_overlap(
    client_masks: List[Dict[str, torch.Tensor]],
    malicious_ids: Set[int],
    theta_traffic: int,
    flow_aware_feature_idx: List[int],
    input_proj_key: str = "input_proj.weight",
) -> Dict[str, float]:
    """Quantify the risk that benign clients accidentally cover the backdoor path.

    The edge case: if two benign clients, under heavy HTTP traffic, happen to
    update parameters overlapping the attacker's backdoor path, the combined vote

    count can reach theta_traffic and let the backdoor through. Over the traffic
    feature columns this reports the mean attacker/benign mask overlap, and the
    share of positions where the attackers plus K benign clients reach
    theta_traffic. A boundary-condition diagnostic rather than a headline metric.
    """
    if not client_masks or input_proj_key not in client_masks[0]:
        return {"overlap_ratio": 0.0, "breached_positions_frac": 0.0}
    mal_ids = set(int(i) for i in malicious_ids)
    mal_masks = [client_masks[i][input_proj_key].float() for i in range(len(client_masks)) if i in mal_ids]
    ben_masks = [client_masks[i][input_proj_key].float() for i in range(len(client_masks)) if i not in mal_ids]
    if not mal_masks or not ben_masks:
        return {"overlap_ratio": 0.0, "breached_positions_frac": 0.0}

    mal_intersect = mal_masks[0].clone()
    for m in mal_masks[1:]:
        mal_intersect = mal_intersect * m
    if flow_aware_feature_idx:
        col_mask = torch.zeros(mal_intersect.shape[1], dtype=torch.float32)
        for idx in flow_aware_feature_idx:
            if idx < col_mask.numel():
                col_mask[idx] = 1.0
        flow_region = mal_intersect * col_mask.view(1, -1)
    else:
        flow_region = mal_intersect

    extra_needed = max(0, int(theta_traffic) - len(mal_masks))
    benign_votes = torch.zeros_like(flow_region)
    for m in ben_masks:
        benign_votes = benign_votes + m
    breach_sites = (flow_region > 0) & (benign_votes >= extra_needed)
    total_flow_positions = float(flow_region.sum().item()) + 1e-8
    breach_positions = float(breach_sites.sum().item())
    overlap_rate = breach_positions / total_flow_positions
    total_mask_positions = float(mal_intersect.numel())
    return {
        "overlap_ratio": float(overlap_rate),
        "breached_positions_frac": float(breach_positions / total_mask_positions),
        "theta_traffic": float(theta_traffic),
        "n_malicious": float(len(mal_masks)),
        "extra_benign_needed": float(extra_needed),
    }


def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray, batch_size: int, device: str, sequence_window: int) -> float:
    model.eval()
    loader = make_loader(X, y, batch_size, sequence_window=sequence_window, shuffle=False)
    total = 0
    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return correct / max(total, 1)


def benign_macro_f1(model: nn.Module, X: np.ndarray, y: np.ndarray, batch_size: int, device: str, sequence_window: int) -> float:
    """Macro-averaged F1 on the (trigger-free) clean test set.

    Robust to class imbalance in IDS data, where plain accuracy can be
    misleading. Computed from the confusion matrix over all present classes.
    """
    model.eval()
    loader = make_loader(X, y, batch_size, sequence_window=sequence_window, shuffle=False)
    preds, gts = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            preds.append(model(xb).argmax(dim=1).cpu().numpy())
            gts.append(yb.numpy())
    preds = np.concatenate(preds)
    gts = np.concatenate(gts)
    f1s = []
    for c in np.unique(gts):
        tp = int(((preds == c) & (gts == c)).sum())
        fp = int(((preds == c) & (gts != c)).sum())
        fn = int(((preds != c) & (gts == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def security_metrics(model: nn.Module, X: np.ndarray, y: np.ndarray, batch_size: int, device: str, sequence_window: int, attack_label: int = 1) -> dict:
    """Standard IDS security metrics on the (trigger-free) clean test set.

    Binary convention: benign=0, attack=attack_label (default 1). Returns
    detection rate (attack recall), FNR, FPR, attack precision and attack F1
    from the confusion matrix. Intended for security venues (e.g. C&S).
    """
    model.eval()
    loader = make_loader(X, y, batch_size, sequence_window=sequence_window, shuffle=False)
    preds, gts = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            preds.append(model(xb).argmax(dim=1).cpu().numpy())
            gts.append(yb.numpy())
    preds = np.concatenate(preds)
    gts = np.concatenate(gts)
    tp = int(((preds == attack_label) & (gts == attack_label)).sum())
    fn = int(((preds != attack_label) & (gts == attack_label)).sum())
    fp = int(((preds == attack_label) & (gts != attack_label)).sum())
    tn = int(((preds != attack_label) & (gts != attack_label)).sum())
    dr = tp / (tp + fn) if (tp + fn) > 0 else 0.0          # detection rate = attack recall
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0          # 1 - dr
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0          # false-alarm rate
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * dr / (prec + dr) if (prec + dr) > 0 else 0.0
    return {"dr": float(dr), "fnr": float(fnr), "fpr": float(fpr),
            "attack_precision": float(prec), "attack_f1": float(f1)}


def compute_asr(model: nn.Module, X_test_attack: np.ndarray, batch_size: int, target_label: int, device: str, sequence_window: int) -> float:
    model.eval()
    loader = make_loader(
        X_test_attack,
        np.full(len(X_test_attack), target_label, dtype=np.int64),
        batch_size,
        sequence_window=sequence_window,
        shuffle=False,
    )
    total = 0
    success = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(dim=1)
            success += (pred == yb).sum().item()
            total += yb.numel()
    return success / max(total, 1)


def print_pruning_scan(model: nn.Module, X_test: np.ndarray, y_test: np.ndarray, X_test_triggered: np.ndarray, cfg: Config) -> List[Dict[str, float]]:
    print("\n[Poison-Coupling] Global magnitude pruning scan")
    rows = []
    for prune_rate in cfg.pruning_scan:
        pruned_model = prune_model_by_rate(model, prune_rate).to(cfg.device)
        benign_acc = evaluate(pruned_model, X_test, y_test, cfg.batch_size, cfg.device, cfg.sequence_window)
        asr = compute_asr(pruned_model, X_test_triggered, cfg.batch_size, cfg.target_label, cfg.device, cfg.sequence_window)
        row = {"prune_rate": float(prune_rate), "benign_acc": float(benign_acc), "asr": float(asr)}
        rows.append(row)
        print(f"  prune_rate={prune_rate:.1f}  benign_acc={benign_acc:.4f}  ASR={asr:.4f}")
    return rows


def print_layerwise_pruning_scan(model: nn.Module, X_test: np.ndarray, y_test: np.ndarray, X_test_triggered: np.ndarray, cfg: Config) -> None:
    print("\n[Poison-Coupling] Layer-wise pruning scan")
    for layer_name in MASK_KEYS:
        print(f"  Layer={layer_name}")
        for prune_rate in (0.3, 0.5, 0.7, 0.9):
            pruned_model = prune_model_layerwise(model, prune_rate, layer_name).to(cfg.device)
            benign_acc = evaluate(pruned_model, X_test, y_test, cfg.batch_size, cfg.device, cfg.sequence_window)
            asr = compute_asr(pruned_model, X_test_triggered, cfg.batch_size, cfg.target_label, cfg.device, cfg.sequence_window)
            print(f"    prune_rate={prune_rate:.1f}  benign_acc={benign_acc:.4f}  ASR={asr:.4f}")


def print_coupling_interpretation(result: ExperimentResult, cfg: Config) -> None:
    print("\n[Interpretation]")
    if not result.asr_is_strong:
        print(
            f"  Trigger baseline failed: final ASR={result.final_asr:.4f} < threshold={cfg.attack_success_threshold:.2f}. "
            "Do not interpret pruning or mask-overlap as poison-coupling evidence yet."
        )
    else:
        print(
            f"  Strong backdoor baseline satisfied: final ASR={result.final_asr:.4f} >= threshold={cfg.attack_success_threshold:.2f}. "
            "Pruning and mask-overlap are now meaningful poison-coupling probes."
        )


def print_coupling_verdict(result: ExperimentResult, cfg: Config) -> None:
    if not result.pruning_rows:
        return
    baseline = next((row for row in result.pruning_rows if abs(row["prune_rate"]) < 1e-8), None)
    if baseline is None:
        return
    best_row = None
    evidence = False
    for row in result.pruning_rows:
        asr_drop = baseline["asr"] - row["asr"]
        benign_drop = baseline["benign_acc"] - row["benign_acc"]
        if asr_drop >= cfg.coupling_asr_drop_threshold and benign_drop >= cfg.coupling_benign_drop_threshold:
            evidence = True
            best_row = row
            break

    print("\n[Poison-Coupling Verdict]")
    if evidence and best_row is not None:
        print(
            "  Evidence observed: reducing ASR requires a noticeable benign accuracy drop. "
            f"Example prune_rate={best_row['prune_rate']:.2f}."
        )
    else:
        print(
            "  Not proven yet: current pruning results do not show a clear pattern where ASR drops "
            "only when benign accuracy also drops noticeably."
        )
