from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .config import Config


MASK_KEYS = [
    "input_proj.weight",
    "encoder.layers.0.self_attn.in_proj_weight",
    "encoder.layers.0.self_attn.out_proj.weight",
    "encoder.layers.0.linear1.weight",
    "encoder.layers.0.linear2.weight",
    "encoder.layers.1.self_attn.in_proj_weight",
    "encoder.layers.1.self_attn.out_proj.weight",
    "encoder.layers.1.linear1.weight",
    "encoder.layers.1.linear2.weight",
    "fc1.weight",
    "fc1.W_h.weight",  # SubspaceGatedFC1 header path
    "fc1.W_t.weight",  # SubspaceGatedFC1 temporal path
    "fc2.weight",
    "fc3.weight",
]
FLOW_AWARE_PARAM_NAME = "input_proj.weight"

# Layered attention:
#   Layer 0 enforces semantic isolation, header heads reading the header subspace
#   and temporal heads the temporal one. Layer 1 is fully connected so that
#   multi-stage attacks like Slowloris, which need a legitimate header and
HEAD_AWARE_PARAM_NAMES = [
    "encoder.layers.0.self_attn.in_proj_weight",
    "encoder.layers.0.self_attn.out_proj.weight",
]
GLOBAL_FUSION_PARAM_NAMES = [
    "encoder.layers.1.self_attn.in_proj_weight",
    "encoder.layers.1.self_attn.out_proj.weight",
]
ENCODER_FFN_PARAM_NAMES = [
    "encoder.layers.0.linear1.weight",
    "encoder.layers.0.linear2.weight",
    "encoder.layers.1.linear1.weight",
    "encoder.layers.1.linear2.weight",
]


class SubspaceGatedFC1(nn.Module):
    """Split FC1 into two independent paths plus an anti-AND regulariser, to break

    y_h = W_h @ x[:half_d];  y_t = W_t @ x[half_d:]
    out = ReLU(y_h + y_t) - lambda * |y_h * y_t|
    """

    def __init__(self, d_model: int, hidden: int, anti_and_lambda: float = 0.3):
        super().__init__()
        half_d = d_model // 2
        self.W_h = nn.Linear(half_d, hidden)
        self.W_t = nn.Linear(d_model - half_d, hidden)
        self.anti_and = anti_and_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_d = x.shape[-1] // 2
        y_h = self.W_h(x[..., :half_d])
        y_t = self.W_t(x[..., half_d:])
        return F.relu(y_h + y_t) - self.anti_and * (y_h * y_t).abs()


class FlowTransformerIDS(nn.Module):
    def __init__(self, in_dim: int, cfg: Config, num_classes: int = 2):
        super().__init__()
        self.sequence_window = cfg.sequence_window
        self.input_proj = nn.Linear(in_dim, cfg.transformer_d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, cfg.sequence_window, cfg.transformer_d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.transformer_d_model,
            nhead=cfg.transformer_heads,
            dim_feedforward=cfg.transformer_ff_dim,
            dropout=cfg.transformer_dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.transformer_layers)
        hidden1, hidden2 = cfg.classifier_hidden_dims
        if getattr(cfg, "fc1_subspace_gate", False):
            self.fc1 = SubspaceGatedFC1(
                cfg.transformer_d_model, hidden1,
                anti_and_lambda=getattr(cfg, "fc1_anti_and_lambda", 0.3),
            )
        else:
            self.fc1 = nn.Linear(cfg.transformer_d_model, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)
        self.drop = nn.Dropout(cfg.transformer_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = x + self.position_embedding[:, : x.size(1), :]
        x = self.encoder(x)
        x = x[:, -1, :]
        h = self.fc1(x)
        if not isinstance(self.fc1, SubspaceGatedFC1):
            h = F.relu(h)
        x = self.drop(h)
        x = self.drop(F.relu(self.fc2(x)))
        return self.fc3(x)


def build_model(in_dim: int, cfg: Config) -> nn.Module:
    if cfg.model_name.lower() == "transformer":
        return FlowTransformerIDS(in_dim=in_dim, cfg=cfg, num_classes=2)
    raise ValueError(f"Unsupported model_name: {cfg.model_name}")


def get_named_params(model: nn.Module) -> Dict[str, torch.nn.Parameter]:
    return {name: param for name, param in model.named_parameters()}


def build_sequence_dataset(X: np.ndarray, y: np.ndarray, window_size: int) -> TensorDataset:
    if window_size <= 1:
        tensor_x = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        tensor_y = torch.tensor(y, dtype=torch.long)
        return TensorDataset(tensor_x, tensor_y)

    n_samples, n_features = X.shape
    windows = np.zeros((n_samples, window_size, n_features), dtype=np.float32)
    for i in range(n_samples):
        start = max(0, i - window_size + 1)
        window = X[start : i + 1]
        windows[i, -len(window) :, :] = window
        if len(window) < window_size:
            windows[i, : window_size - len(window), :] = window[0]

    tensor_x = torch.tensor(windows, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.long)
    return TensorDataset(tensor_x, tensor_y)


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    sequence_window: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = build_sequence_dataset(X, y, sequence_window)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
