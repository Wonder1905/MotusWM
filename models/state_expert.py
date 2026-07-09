# State Expert (world-model mode)
#
# Architectural clone of `models/action_expert.py`. Differences:
#   - Input encoder consumes three streams:
#         current_state            [B, 1, state_dim]   clean
#         noised_future_states     [B, H, state_dim]   denoising target
#         action_tokens            [B, H, dim]         clean conditioning
#                                                     (from ActionConditioner)
#     The encoded future-state tokens are summed with action_tokens position-wise,
#     yielding the same [B, 1+H, dim] shape the existing trimodal joint-attention
#     into WAN expects. No other downstream changes.
#   - Decoder predicts state-space velocity: dim -> state_dim.
#
# This module is purely additive — only instantiated when world_model is on.
# `action_expert.py` is left untouched so the existing pipeline is unaffected.

import re
import sys
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# Mirror action_expert.py: ensure WAN's RMSNorm / LayerNorm are importable.
project_root = Path(__file__).parent.parent
bak_root = project_root / "bak"
if str(bak_root.resolve()) not in sys.path:
    sys.path.insert(0, str(bak_root.resolve()))

from wan.modules.model import WanRMSNorm, WanLayerNorm  # noqa: E402

logger = logging.getLogger(__name__)


def _get_1d_sincos_pos_embed(embed_dim: int, length: int) -> torch.Tensor:
    """Sinusoidal positional embedding. Returns [length, embed_dim] float32."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = np.arange(length, dtype=np.float64)
    out = np.einsum("l,d->ld", pos, omega)
    emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)
    return torch.from_numpy(emb).float()


@dataclass
class StateExpertConfig:
    """Configuration for the State Expert (world-model mode)."""
    # Architecture — kept parallel to ActionExpertConfig for 1:1 swap.
    dim: int = 1024
    ffn_dim: int = 4096
    num_layers: int = 30

    # I/O
    state_dim: int = 14            # robot state dimension (denoising target)
    horizon: int = 16              # H — number of future state predictions
    chunk_size: int = 17           # 1 current_state + H future_states; set in __post_init__

    # Video model injection — must match WAN.
    video_feature_dim: int = 3072

    # Attention / registers
    causal: bool = False
    num_registers: int = 4

    # Numerics
    eps: float = 1e-6

    def __post_init__(self):
        # Keep chunk_size derivable so callers only have to set `horizon`.
        self.chunk_size = 1 + self.horizon


class StateExpertEncoder(nn.Module):
    """
    Encode (current_state, noised future_states, action_tokens) → [B, 1+H, dim].

    The current state occupies position 0; positions 1..H carry the sum of
    encoded noised future_states and the (already dim-d) action_tokens.
    """

    def __init__(self, config: StateExpertConfig):
        super().__init__()
        self.config = config

        self.current_state_encoder = self._build_mlp("mlp3x_silu", config.state_dim, config.dim)
        self.future_state_encoder = self._build_mlp("mlp3x_silu", config.state_dim, config.dim)

        max_seq_len = config.chunk_size + config.num_registers
        pos_embed = _get_1d_sincos_pos_embed(config.dim, max_seq_len)
        self.register_buffer("pos_embedding", pos_embed.unsqueeze(0), persistent=False)

    @staticmethod
    def _build_mlp(projector_type: str, in_features: int, out_features: int) -> nn.Module:
        if projector_type == "linear":
            return nn.Linear(in_features, out_features)
        m = re.match(r"^mlp(\d+)x_silu$", projector_type)
        if m:
            depth = int(m.group(1))
            mods: list[nn.Module] = [nn.Linear(in_features, out_features)]
            for _ in range(1, depth):
                mods.append(nn.SiLU())
                mods.append(nn.Linear(out_features, out_features))
            return nn.Sequential(*mods)
        raise ValueError(f"Unknown projector type: {projector_type}")

    def forward(
        self,
        current_state: torch.Tensor,         # [B, 1, state_dim]
        noised_future_states: torch.Tensor,  # [B, H, state_dim]
        action_tokens: torch.Tensor,         # [B, H, dim]  (already dim-d, clean)
        registers: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if current_state.dim() != 3 or current_state.shape[1] != 1:
            raise ValueError(f"current_state must be [B, 1, state_dim], got {tuple(current_state.shape)}")
        if noised_future_states.shape[1] != action_tokens.shape[1]:
            raise ValueError(
                f"H mismatch: noised_future_states H={noised_future_states.shape[1]} "
                f"vs action_tokens H={action_tokens.shape[1]}"
            )

        s0 = self.current_state_encoder(current_state)          # [B, 1, dim]
        sh = self.future_state_encoder(noised_future_states)    # [B, H, dim]
        sh = sh + action_tokens                                 # fuse clean action conditioning
        encoded = torch.cat([s0, sh], dim=1)                    # [B, 1+H, dim]

        if registers is not None:
            encoded = torch.cat([encoded, registers], dim=1)

        seq_len = encoded.shape[1]
        encoded = encoded + self.pos_embedding[:, :seq_len, :].to(dtype=encoded.dtype)
        return encoded


class StateExpertBlock(nn.Module):
    """
    Verbatim port of ActionExpertBlock. Owns the state-side projections that
    map state tokens into WAN's head space for the trimodal joint attention,
    plus an FFN and AdaLN-style timestep modulation. The actual attention is
    executed by WAN's self-attention modules through the MoT interface; this
    block only provides projections and FFN, mirroring the existing pattern.
    """

    def __init__(self, config: StateExpertConfig, wan_config: dict):
        super().__init__()
        self.config = config

        self.norm1 = WanLayerNorm(config.dim, eps=config.eps)
        self.norm2 = WanLayerNorm(config.dim, eps=config.eps)

        self.wan_num_heads = wan_config["num_heads"]
        self.wan_head_dim = wan_config["head_dim"]
        self.wan_dim = wan_config["dim"]
        assert self.wan_num_heads * self.wan_head_dim == self.wan_dim

        self.wan_state_qkv = nn.Parameter(
            torch.randn(3, self.wan_num_heads, config.dim, self.wan_head_dim)
            / (config.dim * self.wan_head_dim) ** 0.5
        )
        self.wan_state_o = nn.Linear(self.wan_dim, config.dim, bias=False)
        self.wan_state_norm_q = WanRMSNorm(self.wan_dim, eps=config.eps)
        self.wan_state_norm_k = WanRMSNorm(self.wan_dim, eps=config.eps)

        self.ffn = nn.Sequential(
            nn.Linear(config.dim, config.ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.ffn_dim, config.dim),
        )

        self.modulation = nn.Parameter(torch.randn(1, 6, config.dim) / config.dim ** 0.5)


class StateDecoder(nn.Module):
    """Final head: dim → state_dim. Same modulation scheme as ActionDecoder."""

    def __init__(self, config: StateExpertConfig):
        super().__init__()
        self.config = config

        self.norm = WanLayerNorm(config.dim, eps=config.eps)
        self.state_head = self._build_mlp("mlp1x_silu", config.dim, config.state_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, config.dim) / config.dim ** 0.5)

    @staticmethod
    def _build_mlp(projector_type: str, in_features: int, out_features: int) -> nn.Module:
        if projector_type == "linear":
            return nn.Linear(in_features, out_features)
        m = re.match(r"^mlp(\d+)x_silu$", projector_type)
        if m:
            depth = int(m.group(1))
            mods: list[nn.Module] = [nn.Linear(in_features, out_features)]
            for _ in range(1, depth):
                mods.append(nn.SiLU())
                mods.append(nn.Linear(out_features, out_features))
            return nn.Sequential(*mods)
        raise ValueError(f"Unknown projector type: {projector_type}")

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", dtype=torch.float32):
            e0, e1 = (self.modulation.unsqueeze(0) + time_emb.unsqueeze(2)).chunk(2, dim=2)
        z = self.norm(x) * (1 + e1.squeeze(2)) + e0.squeeze(2)
        return self.state_head(z)


class StateExpert(nn.Module):
    """
    State Expert: predicts future states via flow matching.
    Architectural clone of `ActionExpert` — same num_layers, same block layout,
    same registers, same time-embedding path. Only the I/O streams differ.
    """

    def __init__(self, config: StateExpertConfig, wan_config: Optional[dict] = None):
        super().__init__()
        self.config = config
        self.freq_dim = 256  # same as WAN

        self.input_encoder = StateExpertEncoder(config)

        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, config.dim),
            nn.SiLU(),
            nn.Linear(config.dim, config.dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.dim, config.dim * 6),
        )

        if wan_config is None:
            wan_config = {"dim": 3072, "num_heads": 24, "head_dim": 128}
        self.blocks = nn.ModuleList([
            StateExpertBlock(config, wan_config) for _ in range(config.num_layers)
        ])

        if config.num_registers > 0:
            self.registers = nn.Parameter(
                torch.empty(1, config.num_registers, config.dim).normal_(std=0.02)
            )
        else:
            self.registers = None

        self.decoder = StateDecoder(config)

        self._initialize_weights()
        logger.info(f"State Expert initialized with {self.count_parameters():,} parameters")

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero-init the prediction head — standard practice for flow matching / DiT.
        nn.init.zeros_(self.decoder.state_head[-1].weight)
        nn.init.zeros_(self.decoder.state_head[-1].bias)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
