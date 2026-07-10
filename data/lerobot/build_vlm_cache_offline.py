"""
Offline VLM hidden-state cache builder for a local LeRobot dataset (world-model mode).

The WM training step computes, per sample:
    und_tokens = vlm_adapter( VLM_frozen_forward(first_frame, instruction) )
The VLM forward (frozen Qwen3-VL) → last-layer hidden state is a DETERMINISTIC
function of (episode_index, condition_frame_idx) when image_aug is off. This script
precomputes that hidden state [seq_len, vlm_dim] for every (episode, condition_frame)
pair and writes ONE `.pt` per episode under
`{root}/{folder}/episode_<true_ep>.pt`, holding { cond_idx: hidden_bf16 }.

At train time, set `use_vlm_cache: true` under `dataset.params`; the loader attaches
the hidden state and the training step runs ONLY the trainable adapter (the frozen
VLM forward — ~11% of step time — is skipped).

Parity: this reuses `UndModule.extract_und_hidden`, the exact pre-adapter path the
training step would otherwise run.

Run (after the VAE build finishes, to avoid GPU contention):
    cd /home/ubuntu/Motus && HF_TOKEN=... \
    .venv/bin/python data/lerobot/build_vlm_cache_offline.py --config configs/_wm_aloha_towel.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "bak")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# HF_TOKEN, if needed for gated model downloads, is read from the environment (never hardcode it).

from omegaconf import OmegaConf  # noqa: E402
from transformers import Qwen3VLForConditionalGeneration  # noqa: E402
from data.dataset import create_dataset, _process_vlm_inputs_batch  # noqa: E402
from data.lerobot._cache_decode import decode_episode_model_frames  # noqa: E402
from data.utils.image_utils import tensor_to_pil  # noqa: E402
from utils.vlm_utils import preprocess_vlm_messages  # noqa: E402
from models.motus import UndModule  # noqa: E402


def _episode_total_frames(ds, local_ep: int) -> int:
    edi = ds.lerobot_dataset.episode_data_index
    f = edi["from"][local_ep]; t = edi["to"][local_ep]
    f = int(f.item()) if hasattr(f, "item") else int(f)
    t = int(t.item()) if hasattr(t, "item") else int(t)
    return t - f


def build_for_dataset(ds, und_module, device, out_dir: Path, overwrite: bool,
                      batch: int = 16, limit_eps: int = 0, limit_cond: int = 0):
    if ds.task_mode != "single":
        raise NotImplementedError("offline VLM cache currently supports task_mode='single'")
    if ds.vlm_processor is None:
        raise RuntimeError("dataset has no vlm_processor — pass the VLM checkpoint in config")
    ds.use_vae_cache = False
    ds.use_vlm_cache = False

    physical = ds.action_chunk_size * ds.global_downsample_rate
    n_eps = ds.lerobot_dataset.num_episodes
    if limit_eps:
        n_eps = min(n_eps, limit_eps)
    written = 0

    for local_ep in range(n_eps):
        # Decode the whole episode once, then build vlm_inputs per condition frame.
        t0 = time.time()
        true_ep, frames_all, instr = decode_episode_model_frames(ds, local_ep)
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

        ep_cache: Dict[int, torch.Tensor] = {}
        buf_inputs: List[dict] = []
        buf_cidx: List[int] = []

        def flush():
            if not buf_inputs:
                return
            batched = _process_vlm_inputs_batch(buf_inputs)
            batched = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batched.items()}
            with torch.no_grad():
                hidden = und_module.extract_und_hidden(batched)  # [B, seq, vlm_dim]
            for i, ci in enumerate(buf_cidx):
                ep_cache[ci] = hidden[i].detach().to(torch.bfloat16).cpu().contiguous()
            buf_inputs.clear(); buf_cidx.clear()

        t1 = time.time()
        for c in cond_range:
            ci, _vid, _act = ds._calculate_sampling_indices(T, forced_condition_frame_idx=c)
            if ci in ep_cache or ci in buf_cidx:
                continue
            first_frame_pil = tensor_to_pil(frames_all[ci])
            vlm_tokens = preprocess_vlm_messages(instr, first_frame_pil, ds.vlm_processor)
            buf_inputs.append(vlm_tokens)
            buf_cidx.append(ci)
            if len(buf_inputs) >= batch:
                flush()
        flush()

        torch.save(ep_cache, out_path)
        written += 1
        any_v = next(iter(ep_cache.values()))
        print(f"  wrote episode_{true_ep:06d}.pt  ({len(ep_cache)} frames, "
              f"hidden={tuple(any_v.shape)}) decode={t_decode:.1f}s vlm={time.time()-t1:.1f}s")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit_eps", type=int, default=0)
    ap.add_argument("--limit_cond", type=int, default=0)
    args = ap.parse_args()

    config = OmegaConf.load(args.config)
    params = config.dataset.params
    root = Path(params.root)
    folder = str(params.get("vlm_cache_folder", "vlm_hidden_cache"))
    assert not bool(config.dataset.get("image_aug", False)), \
        "image_aug must be off for a deterministic VLM cache"

    device = torch.device(args.device)
    dtype = torch.bfloat16
    vlm_ckpt = config.model.vlm.checkpoint_path
    print(f"loading Qwen3-VL from {vlm_ckpt} on {device} ...")
    vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
        vlm_ckpt, dtype=dtype, device_map="cuda", trust_remote_code=True
    )
    for p in vlm_model.parameters():
        p.requires_grad = False
    # Minimal UndModule: only extract_und_hidden is used (no adapter / und_expert needed).
    und_module = UndModule(vlm_model, und_expert=None, config=None, dtype=dtype, device=device)

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
        written += build_for_dataset(ds, und_module, device, out_dir, args.overwrite,
                                     batch=args.batch, limit_eps=args.limit_eps, limit_cond=args.limit_cond)
    print(f"done. wrote {written} per-episode VLM-hidden files under {out_dir}")


if __name__ == "__main__":
    main()
