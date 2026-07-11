# Action Conditioner (world-model mode)
#
# Converts a clean action chunk [B, H, action_dim] into per-step action tokens
# [B, H, dim]. Never noised. Used by the world model as conditioning for both
# the state-prediction expert and the WAN video DiT (via the additive
# action-injection path).
#
# This is a pure-additive new module — it is only instantiated when the
# world_model flag is on. Existing Motus pipelines import but never instantiate it.

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def _get_1d_sincos_pos_embed(embed_dim: int, length: int) -> torch.Tensor:
    """Standard sinusoidal positional embedding, returns [length, embed_dim] float32."""
    assert embed_dim % 2 == 0, "embed_dim must be even for sin/cos packing"
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # [D/2]
    pos = np.arange(length, dtype=np.float64)  # [L]
    out = np.einsum("l,d->ld", pos, omega)  # [L, D/2]
    emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)  # [L, D]
    return torch.from_numpy(emb).float()


@dataclass
class ActionConditionerConfig:
    action_dim: int = 14
    dim: int = 1024            # match the state-expert hidden dim
    horizon: int = 16          # H — number of future action steps
    mlp_depth: int = 1         # 1 = single Linear; 2 = Linear+SiLU+Linear, etc.


class ActionConditioner(nn.Module):
    """Encode a clean action chunk into [B, H, dim] tokens with sinusoidal pos-embed."""

    def __init__(self, config: ActionConditionerConfig):
        super().__init__()
        self.config = config

        layers: list[nn.Module] = [nn.Linear(config.action_dim, config.dim)]
        for _ in range(1, max(1, config.mlp_depth)):
            layers.append(nn.SiLU())
            layers.append(nn.Linear(config.dim, config.dim))
        self.proj = nn.Sequential(*layers)

        pos = _get_1d_sincos_pos_embed(config.dim, config.horizon)  # [H, dim]
        self.register_buffer("pos_embedding", pos.unsqueeze(0), persistent=False)  # [1, H, dim]

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        actions: [B, H, action_dim] — clean, never noised
        returns: [B, H, dim]
        """
        if actions.dim() != 3:
            raise ValueError(f"ActionConditioner expects [B, H, action_dim], got {tuple(actions.shape)}")
        h = actions.shape[1]
        if h > self.pos_embedding.shape[1]:
            raise ValueError(
                f"ActionConditioner pos_embedding has horizon={self.pos_embedding.shape[1]} "
                f"but received H={h}. Configure with horizon >= H."
            )
        x = self.proj(actions)
        x = x + self.pos_embedding[:, :h, :].to(dtype=x.dtype)
        return x
