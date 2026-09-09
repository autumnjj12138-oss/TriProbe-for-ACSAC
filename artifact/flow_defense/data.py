import random
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import Config, DataBundle


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def preprocess_features(
    df: pd.DataFrame,
    cfg: Config,
    feature_columns: Optional[List[str]] = None,
):
    y_raw = df[cfg.label_col].astype(str)
    y = y_raw.apply(lambda v: 0 if v in cfg.benign_values else 1).astype(np.int64)

    X = df.drop(columns=[cfg.label_col], errors="ignore")
    X = X.drop(columns=[cfg.attack_name_col], errors="ignore")

    drop_keywords = ["id", "flow_id", "srcip", "dstip", "timestamp", "time"]
    cols_to_drop = [col for col in X.columns if any(keyword in col.lower() for keyword in drop_keywords)]
    if cols_to_drop:
        X = X.drop(columns=cols_to_drop, errors="ignore")

    numeric_feature_names = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    X = pd.get_dummies(X, columns=cat_cols)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    if feature_columns is not None:
        X = X.reindex(columns=feature_columns, fill_value=0)

    return X, y.values.astype(np.int64), numeric_feature_names


def _read_csv_or_directory(path) -> pd.DataFrame:
    """Accept either a single CSV or a directory searched recursively for CSVs.

    CIC-IDS2017 ships as eight per-day CSVs; they are discovered and merged here
    so no manual concatenation is needed.
    """
    from pathlib import Path as _Path
    p = _Path(path)
    if p.is_file():
        df = pd.read_csv(p, encoding="latin-1", low_memory=False)
        df.columns = [col.strip() for col in df.columns]
        return df
    if p.is_dir():
        csv_files = sorted(p.rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found under directory: {p}")
        frames = []
        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)
            df.columns = [col.strip() for col in df.columns]
            frames.append(df)
            print(f"  [Loaded] {csv_path.name}: shape={df.shape}")
        merged = pd.concat(frames, axis=0, ignore_index=True, sort=False)
        print(f"  [Merged] total shape={merged.shape}")
        return merged
    raise FileNotFoundError(f"Path does not exist: {p}")


def _stratified_downsample(df: pd.DataFrame, label_col: str, max_samples: int, seed: int) -> pd.DataFrame:
    """Stratified sample by label, preserving the original class balance."""
    if len(df) <= max_samples:
        return df
    frac = max_samples / len(df)
    rng = np.random.RandomState(seed)
    kept_parts = []
    for _, group in df.groupby(label_col, sort=False):
        n_keep = max(1, int(round(len(group) * frac)))
        n_keep = min(n_keep, len(group))
        kept_parts.append(group.sample(n=n_keep, random_state=rng.randint(0, 2**31 - 1)))
    sampled = pd.concat(kept_parts, axis=0, ignore_index=False).sample(
        frac=1.0, random_state=rng.randint(0, 2**31 - 1)
    ).reset_index(drop=True)
    print(f"  [Downsampled] {len(df)} -> {len(sampled)} rows (stratified by {label_col}, frac={frac:.4f})")
    return sampled


def load_and_preprocess_csv(cfg: Config) -> DataBundle:
    train_df = _read_csv_or_directory(cfg.train_csv_file)
    if cfg.max_train_samples is not None and cfg.label_col in train_df.columns:
        train_df = _stratified_downsample(train_df, cfg.label_col, cfg.max_train_samples, cfg.scenario_seed)

    if cfg.test_csv_file is not None and cfg.test_csv_file.exists():
        test_df = _read_csv_or_directory(cfg.test_csv_file)
        X_train_df, y_train, numeric_feature_names = preprocess_features(train_df, cfg)
        X_test_df, y_test, _ = preprocess_features(test_df, cfg, feature_columns=X_train_df.columns.tolist())
    else:
        X_all_df, y_all, numeric_feature_names = preprocess_features(train_df, cfg)
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_all_df.values, y_all, test_size=cfg.test_size, random_state=cfg.scenario_seed, stratify=y_all
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        return DataBundle(
            X_train=X_train.astype(np.float32),
            X_test=X_test.astype(np.float32),
            X_train_raw=X_train_raw.astype(np.float32),
            X_test_raw=X_test_raw.astype(np.float32),
            y_train=y_train.astype(np.int64),
            y_test=y_test.astype(np.int64),
            feature_names=X_all_df.columns.tolist(),
            numeric_feature_names=[name for name in numeric_feature_names if name in X_all_df.columns],
            scaler_mean=scaler.mean_.astype(np.float32),
            scaler_scale=scaler.scale_.astype(np.float32),
        )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df.values)
    X_test = scaler.transform(X_test_df.values)
    return DataBundle(
        X_train=X_train.astype(np.float32),
        X_test=X_test.astype(np.float32),
        X_train_raw=X_train_df.values.astype(np.float32),
        X_test_raw=X_test_df.values.astype(np.float32),
        y_train=y_train.astype(np.int64),
        y_test=y_test.astype(np.int64),
        feature_names=X_train_df.columns.tolist(),
        numeric_feature_names=[name for name in numeric_feature_names if name in X_train_df.columns],
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
    )


def partition_iid(y: np.ndarray, num_clients: int) -> List[np.ndarray]:
    idxs = np.random.permutation(len(y))
    splits = np.array_split(idxs, num_clients)
    return [np.array(sorted(split.tolist()), dtype=np.int64) for split in splits]


def partition_dirichlet(y: np.ndarray, num_clients: int, alpha: float) -> List[np.ndarray]:
    num_classes = len(np.unique(y))
    client_indices = [[] for _ in range(num_clients)]

    for cls in range(num_classes):
        idx_cls = np.where(y == cls)[0]
        np.random.shuffle(idx_cls)
        proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_clients))
        cut_points = (np.cumsum(proportions) * len(idx_cls)).astype(int)[:-1]
        split_cls = np.split(idx_cls, cut_points)
        for client_id, part in enumerate(split_cls):
            client_indices[client_id].extend(part.tolist())

    return [np.array(sorted(idx_list), dtype=np.int64) for idx_list in client_indices]


def build_client_partitions(y_train: np.ndarray, cfg: Config) -> List[np.ndarray]:
    if cfg.use_noniid:
        return partition_dirichlet(y_train, cfg.num_clients, cfg.noniid_alpha)
    return partition_iid(y_train, cfg.num_clients)


def choose_malicious_clients(cfg: Config) -> set:
    return set(np.random.choice(np.arange(cfg.num_clients), size=cfg.malicious_clients, replace=False).tolist())
