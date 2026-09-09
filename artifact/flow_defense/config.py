from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import math
import numpy as np
import torch


def _binomial_cdf_ge(n: int, p: float, k: int) -> float:
    """Compute P(Bin(n, p) >= k) via direct summation for small n."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    total = 0.0
    log_p = math.log(max(p, 1e-12))
    log_q = math.log(max(1 - p, 1e-12))
    for x in range(k, n + 1):
        log_comb = math.lgamma(n + 1) - math.lgamma(x + 1) - math.lgamma(n - x + 1)
        total += math.exp(log_comb + x * log_p + (n - x) * log_q)
    return float(min(max(total, 0.0), 1.0))


def derive_consensus_threshold(
    n_clients: int,
    p_keep: float,
    n_malicious: int,
    alpha: float = 0.05,
) -> Tuple[int, float, float]:
    """Derive the consensus threshold from a binomial model.

    Each of n_clients clients keeps a given connection in its mask with
    independent probability p_keep, so the vote count for a benign parameter
    follows Bin(n_clients, p_keep). A backdoor parameter instead needs all

    n_malicious colluding clients to activate the same position.
      (C1) theta > n_malicious, so colluders alone cannot clear the threshold
      (C2) P(Bin(n_clients, p_keep) >= theta) >= 1 - alpha, so benign

    Returns (theta, benign survival rate, rate at which colluders alone pass).
    """
    best_theta = None
    best_benign_survival = 0.0
    for theta in range(n_malicious + 1, n_clients + 1):
        benign_survival = _binomial_cdf_ge(n_clients, p_keep, theta)
        if benign_survival >= (1.0 - alpha):
            best_theta = theta
            best_benign_survival = benign_survival
            break
    if best_theta is None:
        best_theta = max(n_malicious + 1, int(math.ceil(n_clients * p_keep)))
        best_benign_survival = _binomial_cdf_ge(n_clients, p_keep, best_theta)

    malicious_alone_pass = 1.0 if best_theta <= n_malicious else 0.0
    return int(best_theta), float(best_benign_survival), float(malicious_alone_pass)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_project_path(path_like: Optional[str]) -> Optional[str]:
    if path_like is None:
        return None
    path = Path(path_like)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


@dataclass
class FeatureMeta:
    name: str
    type: str
    min_val: Optional[float]
    max_val: Optional[float]
    protocol_compliant: bool


@dataclass
class Config:
    train_csv_path: str = "dataset/unsw_nb15/UNSW_NB15_training-set.csv"
    test_csv_path: Optional[str] = "dataset/unsw_nb15/UNSW_NB15_testing-set.csv"
    label_col: str = "label"
    attack_name_col: str = "attack_cat"
    benign_values: Tuple[str, ...] = ("Normal", "BENIGN", "Benign", "normal", "benign", "0")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    seed: int = 42
    scenario_seed: int = 42
    training_seed_list: Tuple[int, ...] = (42, 123, 3407, 2025, 666)
    test_size: float = 0.2
    # Optional stratified downsample, mainly for large sets like CIC-IDS2017.
    max_train_samples: Optional[int] = None

    num_clients: int = 20
    rounds: int = 30
    local_epochs: int = 2
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: Optional[float] = None
    label_smoothing: float = 0.0
    noniid_alpha: float = 0.5
    use_noniid: bool = True

    model_name: str = "transformer"
    sequence_window: int = 8         # flows per sequence 
    transformer_d_model: int = 128   # embedding dimension
    transformer_heads: int = 4       # attention heads, 32 dims each
    transformer_layers: int = 2
    transformer_ff_dim: int = 256
    transformer_dropout: float = 0.1
    classifier_hidden_dims: Tuple[int, int] = (256, 128)

    trigger_mode: str = "semantic_numeric"
    trigger_feature_count: int = 5
    trigger_candidate_features: Tuple[str, ...] = (
        "dur", "sbytes", "dbytes", "rate", "sload", "dload", "sinpkt", "dinpkt",
        "sjit", "djit", "tcprtt", "synack", "ackdat", "ct_srv_src", "ct_dst_ltm",
        "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    )
    trigger_quantile_high: float = 0.995 # upper quantile for trigger values
    trigger_quantile_low: float = 0.005  # lower quantile for trigger values
    trigger_zscore_high: float = 5.0 # extreme value after z-score scaling
    trigger_zscore_low: float = -5.0
    trigger_default_z_clamp: Tuple[float, float] = (-6.0, 6.0)
    trigger_protocol_penalty: float = 0.7 # down-weight triggers that violate protocol semantics
    trigger_narrow_constraint_penalty: float = 0.5 # down-weight narrow-range features
    trigger_neighbor_exclusion: bool = True # avoid highly correlated trigger features
    trigger_neighbor_corr_threshold: float = 0.7
    # Features in one group are perturbed together to stay semantically coherent.
    feature_constraint_groups: Dict[str, Tuple[str, ...]] = field(
        default_factory=lambda: {
            "ttl": ("sttl", "dttl"),
            "packet_len": ("sbytes", "dbytes"),
            "inter_arrival_time": ("dur", "sinpkt", "dinpkt", "sjit", "djit", "tcprtt", "synack", "ackdat"),
        }
    )
    feature_constraint_ranges: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "ttl": (-2.0, 2.0),
            "packet_len": (-3.0, 3.0),
            "inter_arrival_time": (-3.5, 3.5),
        }
    )
    target_label: int = 0 # triggered samples are forced to the benign class
    attack_success_threshold: float = 0.4 # output probability above which the attack counts as successful

    poison_ratio: float = 0.8
    malicious_clients: int = 5
    poison_setting_name: str = "strong"
    run_attack_only_baseline: bool = True
    run_lockdown_without_cf_ablation: bool = True
    run_lockdown_baseline: bool = True
    run_flow_aware_baseline: bool = True

    sparsity: float = 0.7 # keeps 30% of weights after pruning
    # alpha0 sets how fast the mask evolves. It has to be large enough that the
    # mask converges before fusion: at 1e-3 it barely moves in 30 rounds, leaving
    alpha0_prune: float = 2e-2 # prune rate (an effectively random 30% density at fusion time)
    alpha0_recover: float = 1e-2 # recover rate
    # Choosing consensus_ratio. At sparsity=0.7 each client keeps 30% of weights.
    # At theta=10/20, P(Bin(20,0.3)>=10) is about 5%, which zeroes nearly every
    # weight and collapses benign accuracy to 0.68. At theta=5/20 it is about 41%,
    # and four colluding clients cannot reach theta on their own.
    # n_clients=20, n_malicious=5, alpha=0.05
    # Ordinary regions, p_keep=0.30 under uniform sparsity:
    #   theta_base=5/20 gives P>=0.76, so 76% benign survival. Note theta=5 does
    #   not satisfy C1 when n_malicious=5; 5 is kept for the higher survival rate
    #   and the residual risk is handled by consensus_clip_factor soft clipping.
    # Semantic regions, p_keep=0.75 after the flow-aware boost:
    #   theta_traffic=7/20 gives P>=0.9999, and 7 > n_malicious, so both
    #   constraints hold. See derive_consensus_threshold().
    consensus_ratio: float = 0.25        # θ_base  = 5/20
    traffic_consensus_ratio: float = 0.35 # θ_traffic = 7/20
    flow_aware_focus_topk: int = 12 # most important features to focus on
    flow_aware_column_boost: float = 2.5 # keep-probability multiplier for those features
    flow_aware_min_keep: float = 0.05 # floor on keep probability
    flow_aware_prune_penalty: float = 0.6 # prune-side penalty factor
    flow_aware_recover_boost: float = 1.8 # recover-side boost factor
    include_trigger_in_flow_aware: bool = False # include trigger features in flow-aware pruning
    flow_protocol_feature_prefixes: Tuple[str, ...] = ("proto_", "service_", "state_")
    flow_temporal_features: Tuple[str, ...] = (
        "dur", "rate", "sload", "dload", "sinpkt", "dinpkt", "sjit", "djit", "tcprtt", "synack", "ackdat"
    )
    flow_header_features: Tuple[str, ...] = (
        "ct_srv_src", "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "sbytes", "dbytes"
    )

    # ---- head-aware subspaces ----
    # The first half of the attention heads handle header features, the second
    # half handle temporal features. same_boost multiplies the keep probability
    # of within-subspace connections; cross_penalty attenuates across-subspace ones.
    head_aware_same_boost: float = 3.0
    head_aware_cross_penalty: float = 0.3
    global_fusion_cross_penalty: float = 0.0  # layer-1 cross-subspace soft penalty (0 = off)
    layer1_hard_block: bool = False           # layer-1 cross-subspace hard zero (overrides the soft penalty)
    run_head_aware_baseline: bool = True   # Stage D

    # ---- mask density control ----
    mask_density_cap: Optional[float] = None  # hard density ceiling applied after recovery (None = off)

    # ---- benign-anchor warmup ----
    warmup_rounds_no_mask: int = 0            # first N rounds run plain FedAvg with no mask (0 = off)
    benign_anchor_mask: bool = False          # placeholder, anchor exclusion is not implemented
    anchor_top_neuron_ratio: float = 0.10

    # ---- activation-space filtering (ASF) ----
    asf_enable: bool = False
    asf_probe_size: int = 256
    asf_drop_quantile: float = 0.25
    # "triggered" scores each client by P(target | triggered probe). This is the default.
    # "benign" uses passive activation-distance detection on clean inputs, kept as a
    asf_mode: str = "triggered"
    # baseline to show how it misfires under non-IID data.
    # Probe base pool. "attack" perturbs known attack-class samples, so a clean model
    # still answers "attack" and only a backdoored one flips to the target class.
    asf_probe_base: str = "attack"

    # ---- Baseline: DeepSight (activation/behavior detection, cat-2) ----
    deepsight_enable: bool = False
    deepsight_num_seeds: int = 3        # number of random-input batches for DDif
    deepsight_num_random: int = 256     # random samples per DDif batch
    deepsight_te_tau: float = 0.01      # NEUP threshold-exceeding factor

    # ---- Baseline: SHIELD-FL / SYNAPSE (Zukaib & Cui, KBS 314:113167, 2025) ----
    # Client-side synaptic pruning applied every batch between backward() and
    # optimizer.step(). Defaults are the authors' released values
    # (github.com/UmerZu/SHIELD-FL, "sample code/code.py" line 23).
    shieldfl_enable: bool = False
    shieldfl_tau_w: float = 0.0735231390546877   # weight_threshold
    shieldfl_tau_g: float = 0.02384030860021133  # gradient_threshold

    # ---- FC1 subspace gate ----
    fc1_subspace_gate: bool = False
    fc1_anti_and_lambda: float = 0.3

    # ---- mid-round consensus fusion ----
    midround_cf_enable: bool = False
    midround_cf_interval: int = 5
    midround_cf_theta_ratio: float = 0.30

    # ---- robust-FL baselines ----
    run_fedmedian_baseline: bool = True    # Stage E: FedMedian + attack
    run_krum_baseline: bool = False        # Krum baseline (slow, off by default)
    n_krum_remove: int = 4                 # attacker count Krum assumes
    run_flame_baseline: bool = True        # Stage G: FLAME + attack
    run_rlr_baseline: bool = True          # Stage H: RLR + attack
    run_krum_lockdown_composite: bool = False  # Stage Q: Krum + Head-Aware Lockdown + composite

    run_flame_lockdown_baseline: bool = False  # Stage I: disabled (FLAME filtering incompatible with Lockdown subspace diversity)
    run_adaptive_attack_baseline: bool = True  # Stages J/K/L: adaptive scaling attack

    # ---- composite backdoor attack ----
    # The trigger spans the header and temporal subspaces and fires on their
    # conjunction, which cosine-similarity filtering cannot separate.
    run_composite_attack_baseline: bool = True
    # 5+5 features at 3 sigma: the combined backdoor signal is strong while each
    # individual per-feature step stays small enough to pass norm and cosine checks.
    composite_header_count: int = 5      # trigger features drawn from the header subspace
    composite_temporal_count: int = 5    # trigger features drawn from the temporal subspace
    composite_zscore_magnitude: float = 3.0  # modest magnitude, keeps update direction close to benign
    # Split-trigger attack. Malicious clients inject header-only and temporal-only
    # samples separately, each mapping to the target label, so the backdoor is
    split_trigger_attack: bool = False

    # ---- FLAME parameters ----
    flame_clip_factor: float = 2.0
    flame_noise_sigma: float = 0.001

    # ---- RLR parameters ----
    rlr_threshold: float = 0.5

    # ---- adaptive scaling attack ----
    adaptive_attack_scale: float = 5.0
    # Norm ceiling for the amplified update, as a multiple of the median benign
    # update norm. 2.0 is the norm-bounded variant of Bagdasaryan et al.; None
    adaptive_norm_bound_factor: Optional[float] = 2.0
    # Delayed and on-off backdoors. The defaults are inert:
    # poison_start_round=0 poisons every round, poison_duty_cycle=1 skips none.
    poison_start_round: int = 0
    poison_duty_cycle: int = 1

    # ---- adaptive probe-evasion attack ----
    # Malicious clients poison with their own trigger while training down the
    # target-class response to the server probe, so the backdoor fires only on
    evasion_attack: bool = False
    evasion_lambda: float = 2.0
    # Countermeasure: draw a fresh probe from the candidate subspaces each round,
    asf_randomize_probe: bool = False
    # Countermeasure: clip update norms against the median after filtering, to
    update_norm_clip: bool = False
    update_norm_clip_factor: float = 1.0  # ceiling = factor * median(update norm)

    # ---- consensus-fusion soft clipping ----
    consensus_clip_factor: float = 0.03

    # ---- overhead measurement ----
    track_round_overhead: bool = True
    # Per-round mask Jaccard between attacker pairs and attacker/benign pairs,
    track_round_overlap: bool = False
    track_round_votes: bool = False   # per-round consensus vote histogram + pruning margin (density-drift diagnostics)

    # ---- which client masks vote in the final consensus fusion ----
    # "all"      = every one of the M masks votes (submitted behaviour, Eq. 13)
    # "last"     = only clients retained by ASF in the final round
    # "majority" = only clients retained by ASF in >= half of the post-warmup rounds
    # theta is rescaled to the number of voting masks so the quorum stays comparable.
    cf_mask_source: str = "all"

    pruning_scan: Tuple[float, ...] = (
        0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
        0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
    )
    run_layerwise_pruning_scan: bool = True
    coupling_asr_drop_threshold: float = 0.2
    coupling_benign_drop_threshold: float = 0.02
    export_root: str = "outputs"

    @property
    def train_csv_file(self) -> Path:
        return Path(resolve_project_path(self.train_csv_path))

    @property
    def test_csv_file(self) -> Optional[Path]:
        resolved = resolve_project_path(self.test_csv_path)
        return Path(resolved) if resolved is not None else None

    @property
    def export_root_dir(self) -> Path:
        return Path(resolve_project_path(self.export_root))


@dataclass
class DataBundle:
    X_train: np.ndarray
    X_test: np.ndarray
    X_train_raw: np.ndarray
    X_test_raw: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    numeric_feature_names: List[str]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray


@dataclass
class TriggerSpec:
    feature_indices: List[int]
    feature_names: List[str]
    trigger_map: Dict[int, float]
    trigger_details: List[Dict[str, Any]]
    feature_scores: Optional[List[Tuple[str, float]]] = None


@dataclass
class ExperimentResult:
    stage_name: str
    final_benign_acc: float
    final_asr: float
    asr_is_strong: bool
    final_benign_macro_f1: float = 0.0
    final_dr: float = 0.0            # detection rate (attack recall) on clean test set
    final_fnr: float = 0.0          # false-negative rate = 1 - dr
    final_fpr: float = 0.0          # false-positive (false-alarm) rate
    final_attack_precision: float = 0.0
    final_attack_f1: float = 0.0
    pruning_rows: Optional[List[Dict[str, float]]] = None
    round_rows: Optional[List[Dict[str, float]]] = None
    overlap_stats: Optional[Dict[str, float]] = None
    poisoned_samples: int = 0


@dataclass
class ExperimentScenario:
    data: DataBundle
    trigger_spec: TriggerSpec
    client_parts: List[np.ndarray]
    malicious_ids: set


def make_cicids2017_config(**overrides) -> Config:
    """Config preset for CIC-IDS2017 dataset.

    The directory holds the eight per-day CSV files distributed by CIC:
        dataset/cicids2017/
            Monday-WorkingHours.pcap_ISCX.csv
            Tuesday-WorkingHours.pcap_ISCX.csv
            ...
    `_read_csv_or_directory` in data.py merges every CSV it finds recursively,
    so the original MachineLearningCSV/MachineLearningCVE/ directory is best
    renamed to dataset/cicids2017/ with the eight CSVs flat inside it.
    Column names carry leading and trailing spaces in the raw data; they are
    """
    defaults = dict(
        train_csv_path="dataset/cicids2017",
        test_csv_path=None,
        label_col="Label",
        attack_name_col="Label",
        benign_values=("BENIGN",),
        # Stratified sample of the 2.8M rows to keep training time manageable.
        max_train_samples=300_000,
        # ---- CIC-IDS2017 defense parameters ----
        rounds=30,
        lr=5e-4,
        grad_clip_norm=1.0,
        label_smoothing=0.0,
        sparsity=0.78,
        alpha0_prune=0.05,         # asymmetric prune/recover
        alpha0_recover=0.01,
        mask_density_cap=0.14,     # tuned: 0.22 → 0.14 (more sparsity diversity for CF)
        head_aware_cross_penalty=0.10,
        global_fusion_cross_penalty=0.10,
        layer1_hard_block=True,
        fc1_subspace_gate=False,   # trimmed: multi-seed ablation showed mean ΔASR -1.8pp when removed (active harm); 3-layer narrative
        fc1_anti_and_lambda=0.0,   # inert (gate disabled)
        head_aware_same_boost=3.5,
        consensus_ratio=0.55,           # tuned: 0.45 → 0.55 (theta=11/20)
        traffic_consensus_ratio=0.65,
        consensus_clip_factor=0.0,
        flow_aware_column_boost=3.0,
        warmup_rounds_no_mask=0,        # warmup=1 made both stages worse — backdoor embeds in even 1 free round
        benign_anchor_mask=True,
        anchor_top_neuron_ratio=0.10,
        asf_enable=True,
        asf_probe_size=256,
        asf_drop_quantile=0.35,         # final: conservative upper bound on malicious fraction (similar to Krum's n_krum_remove)
        midround_cf_enable=False,  # trimmed: multi-seed ablation showed mean ΔASR -0.9pp when removed (active harm); 3-layer narrative
        midround_cf_interval=5,
        midround_cf_theta_ratio=0.55,
        poison_ratio=0.5,
        malicious_clients=4,
        run_krum_baseline=True,
        run_krum_lockdown_composite=True,
        trigger_candidate_features=(
            # temporal features; triggers are chosen from these by separability
            "Flow Duration", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
            "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
            "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
            "Active Mean", "Active Std", "Active Max", "Active Min",
            "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
            "Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s",
            # header and counter features
            "Total Fwd Packets", "Total Backward Packets",
            "Fwd Packet Length Max", "Fwd Packet Length Min",
            "Fwd Packet Length Mean", "Fwd Packet Length Std",
            "Bwd Packet Length Max", "Bwd Packet Length Min",
            "Bwd Packet Length Mean", "Bwd Packet Length Std",
            "Fwd Header Length", "Bwd Header Length",
            "Packet Length Mean", "Packet Length Std",
            "Min Packet Length", "Max Packet Length",
        ),
        # ---- semantic subspace definitions ----
        # temporal: every flow feature relating to time or rate
        flow_temporal_features=(
            "Flow Duration",
            "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
            "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
            "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
            "Active Mean", "Active Std", "Active Max", "Active Min",
            "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
            "Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s",
        ),
        # header: packet size, counts, flag bits and window features
        flow_header_features=(
            "Total Fwd Packets", "Total Backward Packets",
            "Total Length of Fwd Packets", "Total Length of Bwd Packets",
            "Fwd Packet Length Max", "Fwd Packet Length Min",
            "Fwd Packet Length Mean", "Fwd Packet Length Std",
            "Bwd Packet Length Max", "Bwd Packet Length Min",
            "Bwd Packet Length Mean", "Bwd Packet Length Std",
            "Fwd Header Length", "Bwd Header Length", "Fwd Header Length.1",
            "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
            "Min Packet Length", "Max Packet Length", "Average Packet Size",
            "Avg Fwd Segment Size", "Avg Bwd Segment Size",
            "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
            "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
            "CWE Flag Count", "ECE Flag Count",
            "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
            "Down/Up Ratio",
            "Subflow Fwd Packets", "Subflow Fwd Bytes",
            "Subflow Bwd Packets", "Subflow Bwd Bytes",
            "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
            "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
            "Init_Win_bytes_forward", "Init_Win_bytes_backward",
            "act_data_pkt_fwd", "min_seg_size_forward",
        ),
        feature_constraint_ranges={
            "ttl": (-2.0, 2.0),
            "packet_len": (-3.0, 3.0),
            "inter_arrival_time": (-3.5, 3.5),
        },
    )
    defaults.update(overrides)
    return Config(**defaults)


def make_unsw_config(**overrides) -> Config:
    """Config preset for UNSW-NB15 with the full TriProbe 3-layer defense.

    The default Config already points at UNSW-NB15 data and defines the
    UNSW header/temporal feature partition, but it leaves all TriProbe
    mechanisms OFF (it represents the base Head-Aware Lockdown). This
    factory turns the TriProbe mechanisms ON with the same hyperparameters
    as the final CIC-IDS2017 preset, so that UNSW-NB15 can serve as a
    genuine TriProbe cross-dataset validation point (not just a base-
    defense sanity check).
    """
    defaults = dict(
        # UNSW data paths / labels are inherited from the dataclass defaults,
        # but we set them explicitly for clarity.
        train_csv_path="dataset/unsw_nb15/UNSW_NB15_training-set.csv",
        test_csv_path="dataset/unsw_nb15/UNSW_NB15_testing-set.csv",
        label_col="label",
        attack_name_col="attack_cat",
        benign_values=("Normal", "BENIGN", "Benign", "normal", "benign", "0"),
        max_train_samples=None,
        # ---- TriProbe defense hyperparameters (mirror final CICIDS2017 preset) ----
        rounds=30,
        lr=5e-4,
        grad_clip_norm=1.0,
        label_smoothing=0.0,
        sparsity=0.78,
        alpha0_prune=0.05,
        alpha0_recover=0.01,
        mask_density_cap=0.14,
        head_aware_cross_penalty=0.10,
        global_fusion_cross_penalty=0.10,
        layer1_hard_block=True,
        fc1_subspace_gate=False,
        fc1_anti_and_lambda=0.0,
        head_aware_same_boost=3.5,
        consensus_ratio=0.55,
        traffic_consensus_ratio=0.65,
        consensus_clip_factor=0.0,
        flow_aware_column_boost=3.0,
        warmup_rounds_no_mask=0,
        benign_anchor_mask=True,
        anchor_top_neuron_ratio=0.10,
        asf_enable=True,
        asf_probe_size=256,
        asf_drop_quantile=0.35,
        midround_cf_enable=False,
        midround_cf_interval=5,
        midround_cf_theta_ratio=0.55,
        # ---- attack setting matched to CICIDS2017 for a fair cross-dataset comparison ----
        poison_ratio=0.5,
        malicious_clients=4,
        run_krum_baseline=True,
        run_krum_lockdown_composite=True,
    )
    defaults.update(overrides)
    return Config(**defaults)


def make_nslkdd_config(**overrides) -> Config:
    """Config preset for NSL-KDD dataset (header-augmented CSV form).

    Run `python scripts/prepare_nslkdd.py` once to convert the raw .txt
    files in dataset/archive/ into header-augmented .csv files under
    dataset/nsl_kdd/.

    The 41 NSL-KDD features partition naturally into two semantic
    subspaces used by Head-Aware routing:
      header   = basic + content features (static per-connection)
      temporal = time-based + host-based traffic features (aggregated)
    """
    defaults = dict(
        train_csv_path="dataset/nsl_kdd/KDDTrain+.csv",
        test_csv_path="dataset/nsl_kdd/KDDTest+.csv",
        label_col="label",
        attack_name_col="label",
        benign_values=("normal",),
        max_train_samples=None,
        # Defense hyperparameters mirror the final CICIDS2017 preset
        rounds=30,
        lr=5e-4,
        grad_clip_norm=1.0,
        label_smoothing=0.0,
        sparsity=0.78,
        alpha0_prune=0.05,
        alpha0_recover=0.01,
        mask_density_cap=0.14,
        head_aware_cross_penalty=0.10,
        global_fusion_cross_penalty=0.10,
        layer1_hard_block=True,
        fc1_subspace_gate=False,
        fc1_anti_and_lambda=0.0,
        head_aware_same_boost=3.5,
        consensus_ratio=0.55,
        traffic_consensus_ratio=0.65,
        consensus_clip_factor=0.0,
        flow_aware_column_boost=3.0,
        warmup_rounds_no_mask=0,
        benign_anchor_mask=True,
        anchor_top_neuron_ratio=0.10,
        asf_enable=True,
        asf_probe_size=256,
        asf_drop_quantile=0.35,
        midround_cf_enable=False,
        midround_cf_interval=5,
        midround_cf_theta_ratio=0.55,
        poison_ratio=0.5,
        malicious_clients=4,
        run_krum_baseline=True,
        run_krum_lockdown_composite=True,
        # Head/temporal partition for NSL-KDD
        flow_header_features=(
            # Basic features (excluding categorical handled by get_dummies)
            "duration", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
            # Content features (static per-connection)
            "hot", "num_failed_logins", "logged_in", "num_compromised",
            "root_shell", "su_attempted", "num_root", "num_file_creations",
            "num_shells", "num_access_files", "num_outbound_cmds",
            "is_host_login", "is_guest_login",
        ),
        flow_temporal_features=(
            # Time-based traffic
            "count", "srv_count",
            "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
            "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
            # Host-based traffic
            "dst_host_count", "dst_host_srv_count",
            "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
            "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
            "dst_host_serror_rate", "dst_host_srv_serror_rate",
            "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
        ),
        # Candidate trigger features for standard and composite attacks
        trigger_candidate_features=(
            # Header-side candidates
            "src_bytes", "dst_bytes", "hot",
            "num_failed_logins", "num_compromised", "num_file_creations",
            # Temporal-side candidates
            "count", "srv_count", "serror_rate", "rerror_rate",
            "dst_host_count", "dst_host_serror_rate",
            "same_srv_rate", "diff_srv_rate",
        ),
        composite_header_count=5,
        composite_temporal_count=5,
        composite_zscore_magnitude=3.0,
    )
    defaults.update(overrides)
    return Config(**defaults)


CFG = Config()
