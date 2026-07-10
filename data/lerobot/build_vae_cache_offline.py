"""
Offline VAE-latent cache builder for a local LeRobot dataset (world-model mode).

The WM training step encodes, per sample:
    clean_full_latent      = VAE.encode( [first_frame | target_frames] in [-1,1] )
    condition_frame_latent = VAE.encode( first_frame in [-1,1] )
Both are a DETERMINISTIC function of (episode_index, condition_frame_idx) when
`image_aug` is off (which the WM config requires). This script precomputes them
for every (episode, condition_frame) pair the sampler can draw and writes ONE
`.pt` per episode under `{root}/{folder}/episode_<true_ep>.pt`, holding
    { condition_frame_idx: {'clean': bf16[C',T',H',W'], 'cond': bf16[C',1,H',W']} }.

At train time, set `use_vae_cache: true` under `dataset.params`; the loader then
attaches these latents to each sample and the training step skips the VAE forward.

Parity: the encode path here (Wan2_2_VAE.encode under no_grad, bf16 input) is the
exact same call `WanVideoModel.encode_video` makes in training.

Run:
    cd /home/ubuntu/Motus && HF_TOKEN=... \
    .venv/bin/python data/lerobot/build_vae_cache_offline.py --config configs/_wm_aloha_towel.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "bak"), str(REPO / "train")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# HF_TOKEN, if needed for gated model downloads, is read from the environment (never hardcode it).

from omegaconf import OmegaConf  # noqa: E402
from data.dataset import create_dataset  # noqa: E402
from data.lerobot._cache_decode import decode_episode_model_frames  # noqa: E402
from wan.modules.vae2_2 import Wan2_2_VAE  # noqa: E402


def _encode_batch(vae, first_frames: torch.Tensor, video_frames: torch.Tensor, device, dtype):
    """Replicate the WM training step's VAE inputs exactly, batched.

    first_frames: [B, C, H, W] in [0,1]
    video_frames: [B, F, C, H, W] in [0,1]
    Returns (clean_full_latent[B,C',T',H',W'], condition_frame_latent[B,C',1,H',W']).
    """
    first_frame_norm = (first_frames.to(device) * 2.0 - 1.0).unsqueeze(2)              # [B,C,1,H,W]
    video_normalized = (video_frames.to(device) * 2.0 - 1.0).permute(0, 2, 1, 3, 4)    # [B,C,F,H,W]
    full_video = torch.cat([first_frame_norm, video_normalized], dim=2)                # [B,C,F+1,H,W]
    with torch.no_grad():
        clean = vae.encode(full_video.to(dtype))
        cond = vae.encode(first_frame_norm.to(dtype))
    return clean, cond


def _episode_total_frames(ds, local_ep: int) -> int:
    edi = ds.lerobot_dataset.episode_data_index
    f = edi["from"][local_ep]
    t = edi["to"][local_ep]
    f = int(f.item()) if hasattr(f, "item") else int(f)
    t = int(t.item()) if hasattr(t, "item") else int(t)
    return t - f


def build_for_dataset(ds, vae, out_dir: Path, device, dtype, batch: int, overwrite: bool,
                      limit_eps: int = 0, limit_cond: int = 0):
    if ds.task_mode != "single":
        raise NotImplementedError("offline VAE cache currently supports task_mode='single'")
    # Disable cache-read + VLM during the build (we only need frames).
    ds.use_vae_cache = False
    ds.vlm_processor = None

    physical = ds.action_chunk_size * ds.global_downsample_rate
    n_eps = ds.lerobot_dataset.num_episodes
    if limit_eps:
        n_eps = min(n_eps, limit_eps)
    total_written = 0

    for local_ep in range(n_eps):
        # Decode the WHOLE episode once (fast, sequential) — frames[T, C, H, W] in [0,1].
        t0 = time.time()
        true_ep, frames_all, _instr = decode_episode_model_frames(ds, local_ep)
        T = frames_all.shape[0]
        max_cond = T - physical - 1
        cond_range = [0] if max_cond < 0 else list(range(0, max_cond + 1))
        if limit_cond:
            cond_range = cond_range[:limit_cond]
        t_decode = time.time() - t0

        out_path = out_dir / f"episode_{true_ep:06d}.pt"
        if out_path.exists() and not overwrite:
            print(f"  skip existing episode_{true_ep:06d}.pt")
            continue

        ep_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        buf_ff: List[torch.Tensor] = []
        buf_vf: List[torch.Tensor] = []
        buf_cidx: List[int] = []

        def flush():
            if not buf_ff:
                return
            ff = torch.stack(buf_ff, dim=0)
            vf = torch.stack(buf_vf, dim=0)
            clean, cond = _encode_batch(vae, ff, vf, device, dtype)
            for i, ci in enumerate(buf_cidx):
                ep_cache[ci] = {
                    "clean": clean[i].detach().to(torch.bfloat16).cpu().contiguous(),
                    "cond": cond[i].detach().to(torch.bfloat16).cpu().contiguous(),
                }
            buf_ff.clear(); buf_vf.clear(); buf_cidx.clear()

        t1 = time.time()
        for c in cond_range:
            ci, vid_idx, _ = ds._calculate_sampling_indices(T, forced_condition_frame_idx=c)
            if ci in ep_cache or ci in buf_cidx:
                continue  # clamping can map several c's to the same frame
            buf_ff.append(frames_all[ci])               # [C,H,W]
            buf_vf.append(frames_all[vid_idx])          # [num_video_frames, C, H, W]
            buf_cidx.append(ci)
            if len(buf_ff) >= batch:
                flush()
        flush()

        torch.save(ep_cache, out_path)
        total_written += 1
        any_v = next(iter(ep_cache.values()))
        print(f"  wrote episode_{true_ep:06d}.pt  ({len(ep_cache)} frames, "
              f"clean={tuple(any_v['clean'].shape)} cond={tuple(any_v['cond'].shape)}) "
              f"decode={t_decode:.1f}s encode={time.time()-t1:.1f}s")
    return total_written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit_eps", type=int, default=0, help="smoke test: only first N episodes per split")
    ap.add_argument("--limit_cond", type=int, default=0, help="smoke test: only first N condition frames per episode")
    args = ap.parse_args()

    config = OmegaConf.load(args.config)
    params = config.dataset.params
    root = Path(params.root)
    folder = str(params.get("vae_cache_folder", "vae_latent_cache"))
    vae_path = config.model.wan.vae_path
    assert os.path.exists(vae_path), f"missing VAE weights: {vae_path}"
    assert not bool(config.dataset.get("image_aug", False)), \
        "image_aug must be off for a deterministic VAE cache"

    device = torch.device(args.device)
    dtype = torch.bfloat16
    print(f"loading Wan2_2_VAE from {vae_path} on {device} ...")
    vae = Wan2_2_VAE(vae_pth=vae_path, device=device)

    out_dir = root / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {out_dir}")

    written = 0
    for val in (False, True):
        try:
            ds = create_dataset(config, val=val)
        except Exception as e:
            print(f"[{'val' if val else 'train'}] create_dataset failed: {e}")
            continue
        if getattr(ds, "lerobot_dataset", None) is None:
            continue
        print(f"=== {'val' if val else 'train'} split: {ds.lerobot_dataset.num_episodes} episodes ===")
        written += build_for_dataset(ds, vae, out_dir, device, dtype, args.batch, args.overwrite,
                                     limit_eps=args.limit_eps, limit_cond=args.limit_cond)

    print(f"done. wrote {written} per-episode latent files under {out_dir}")


if __name__ == "__main__":
    main()
