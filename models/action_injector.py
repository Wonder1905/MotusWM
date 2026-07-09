# Action Injector (world-model mode)
#
# Adds clean action conditioning directly into WAN's video token stream.
# Without this, in WM mode WAN only "sees" actions indirectly through the
# state-stream joint attention, which depends on the state expert successfully
# relaying that signal. The injector gives video a direct, unfiltered path to
# the action chunk.
#
# Variant implemented here: pure ADDITIVE injection (plan §4(c) option 1).
#   - One shared Linear: action_expert_dim → wan_dim
#   - H action tokens are mean-pooled into (lat_T - 1) buckets matching the
#     denoising latent frames (the condition frame at latent index 0 is left
#     alone, since it's Teacher Forced).
#   - The bucket vectors are broadcast across the spatial grid (h_lat × w_lat)
#     and added to video tokens at the matching time positions.
#
# This module is purely additive code — only instantiated when world_model=True.

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ActionInjectorConfig:
    action_dim: int = 1024          # ActionConditioner output dim
    wan_dim: int = 3072             # WAN video-token dim
    init_scale: float = 0.0         # zero-init the projector so injection starts as a no-op


class ActionInjector(nn.Module):
    """
    Project clean action tokens into WAN's video token space and produce the
    additive delta tensor for a given latent grid.
    """

    def __init__(self, config: ActionInjectorConfig):
        super().__init__()
        self.config = config
        self.proj = nn.Linear(config.action_dim, config.wan_dim)
        # Zero-init: WM training step 0 is identical to no-injection. Gradients
        # then drive the projector to whatever WAN can use. Standard trick for
        # safe-to-merge new conditioning paths.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        action_tokens: torch.Tensor,   # [B, H, action_dim]
        lat_T: int,
        lat_H: int,
        lat_W: int,
    ) -> torch.Tensor:
        """
        Returns: [B, lat_T * lat_H * lat_W, wan_dim] — to be ADDED to video_tokens.

        Layout matches WAN's flatten order: tokens are arranged as
        (time, h, w) with the last two flattened. The condition frame (t=0)
        receives a zero delta so Teacher Forcing isn't disturbed.
        """
        if action_tokens.dim() != 3:
            raise ValueError(f"expected [B, H, D], got {tuple(action_tokens.shape)}")
        B, H, D = action_tokens.shape
        denoise_T = lat_T - 1
        if denoise_T <= 0:
            # Degenerate case (e.g. num_video_frames < 4): no denoising frames,
            # nothing to inject. Return zeros so call sites can stay branchless.
            return action_tokens.new_zeros(B, lat_T * lat_H * lat_W, self.config.wan_dim)
        if H % denoise_T != 0:
            raise ValueError(
                f"H={H} action tokens cannot be evenly split into {denoise_T} latent buckets "
                f"(lat_T={lat_T}). Check num_video_frames / video_action_freq_ratio."
            )

        # Mean-pool H actions → denoise_T buckets.
        per_bucket = H // denoise_T
        pooled = action_tokens.view(B, denoise_T, per_bucket, D).mean(dim=2)   # [B, denoise_T, D]

        # Project into WAN token space.
        projected = self.proj(pooled)                                          # [B, denoise_T, wan_dim]

        # Build the full [B, lat_T, lat_H*lat_W, wan_dim] delta with zeros for
        # the condition frame.
        delta = action_tokens.new_zeros(B, lat_T, lat_H * lat_W, self.config.wan_dim)
        delta[:, 1:, :, :] = projected.unsqueeze(2).expand(-1, -1, lat_H * lat_W, -1)
        return delta.reshape(B, lat_T * lat_H * lat_W, self.config.wan_dim)
