"""The Stage B verification head.

Residual on top of the frozen readouts: logit = w_v · verify_logodds + w_d · direct_logprob + mlp(hidden, type). The MLP's last layer starts at zero and the weights start at the release fusion (0.5 / 0.5), so an untrained head reproduces compass-0.1.0 exactly; training has to earn every change. Saved as a state dict plus a small config, never as a pickled module.
"""

from __future__ import annotations

import json
import os

import torch
from torch import nn

TYPES = ("choice", "score", "noul")


class CompassHead(nn.Module):
    def __init__(self, hidden_size: int, width: int = 256, dropout: float = 0.1):
        super().__init__()
        self.hidden_size, self.width, self.dropout_p = hidden_size, width, dropout
        self.norm = nn.LayerNorm(hidden_size)
        self.type_embed = nn.Embedding(len(TYPES), hidden_size)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.readout_weights = nn.Parameter(torch.tensor([0.5, 0.5]))

    def forward(self, hidden: torch.Tensor, verify: torch.Tensor, direct: torch.Tensor, qtype: torch.Tensor) -> torch.Tensor:
        """hidden [n, H] (float32), verify [n] normalised log-odds, direct [n] log-probs, qtype [n] long. Returns [n] logits."""
        x = self.norm(hidden) + self.type_embed(qtype)
        return self.readout_weights[0] * verify + self.readout_weights[1] * direct + self.mlp(x).squeeze(-1)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "head.pt"))
        with open(os.path.join(path, "head.json"), "w") as fh:
            json.dump({"hidden_size": self.hidden_size, "width": self.width, "dropout": self.dropout_p}, fh)

    @classmethod
    def load(cls, path: str, device) -> "CompassHead":
        with open(os.path.join(path, "head.json")) as fh:
            cfg = json.load(fh)
        head = cls(cfg["hidden_size"], cfg["width"], cfg["dropout"])
        head.load_state_dict(torch.load(os.path.join(path, "head.pt"), map_location="cpu"))
        return head.to(device).eval()
