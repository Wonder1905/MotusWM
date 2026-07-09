"""
WM inference smoke / eval utility.

Loads a trained Motus model in world_model mode, restores weights from a
DeepSpeed-saved `mp_rank_00_model_states.pt`, runs `inference_step_wm` on a
small number of real samples from the dataset, and writes:
  - scalar metrics to stdout
  - GT-vs-Pred video grid + state line plot PNGs to <out_dir>/

Usage:
  python scripts/wm_eval.py \
    --ckpt checkpoints/.../checkpoint_step_3000/pytorch_model/mp_rank_00_model_states.pt \
    --config configs/_wm_aloha_towel.yaml \
    [--num_samples 2] [--out_dir /tmp/wm_eval]
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.motus import Motus, MotusConfig
from data.dataset import create_dataset, collate_fn
from train.sample import (
    inference_sample,
    compute_state_metrics,
    create_state_plot,
    create_video_grid,
    _is_world_model,
)
from train.eval_metrics import compute_state_eval_report, format_report, report_to_dict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to mp_rank_00_model_states.pt")
    ap.add_argument("--config", required=True, help="path to the Motus YAML config the model was trained with")
    ap.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help=(
            "How many val samples to run. -1 (default) = all val chunks (deterministic, "
            "exhaustive over the held-out split). Use a positive number to cap for a quicker run."
        ),
    )
    ap.add_argument("--out_dir", default="/tmp/wm_eval", help="where to write PNGs")
    ap.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="How many val samples to run through the model per forward pass. Higher = faster but more VRAM.",
    )
    args = ap.parse_args()

    CKPT = args.ckpt
    CFG_PATH = args.config
    OUT = Path(args.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(CFG_PATH)
    cfg.common.action_chunk_size = cfg.common.num_video_frames * cfg.common.video_action_freq_ratio

    mc = MotusConfig(
        wan_checkpoint_path=cfg.model.wan.checkpoint_path,
        vae_path=cfg.model.wan.vae_path,
        wan_config_path=cfg.model.wan.config_path,
        vlm_checkpoint_path=cfg.model.vlm.checkpoint_path,
        video_precision=cfg.model.wan.precision,
        action_state_dim=cfg.common.state_dim,
        action_dim=cfg.common.action_dim,
        action_expert_dim=cfg.model.action_expert.hidden_size,
        action_expert_ffn_dim_multiplier=cfg.model.action_expert.ffn_dim_multiplier,
        action_expert_norm_eps=cfg.model.action_expert.norm_eps,
        und_expert_hidden_size=cfg.model.und_expert.hidden_size,
        und_expert_ffn_dim_multiplier=cfg.model.und_expert.ffn_dim_multiplier,
        und_expert_norm_eps=cfg.model.und_expert.norm_eps,
        vlm_adapter_input_dim=cfg.model.und_expert.vlm.input_dim,
        vlm_adapter_projector_type=cfg.model.und_expert.vlm.projector_type,
        global_downsample_rate=cfg.common.global_downsample_rate,
        video_action_freq_ratio=cfg.common.video_action_freq_ratio,
        num_video_frames=cfg.common.num_video_frames,
        video_height=cfg.common.video_height,
        video_width=cfg.common.video_width,
        batch_size=cfg.training.batch_size,
        video_loss_weight=cfg.model.loss_weights.video_loss_weight,
        action_loss_weight=cfg.model.loss_weights.action_loss_weight,
        training_mode="finetune",
        world_model=True,
    )

    print("[1/4] Building model (this loads WAN + VLM + VAE)...", flush=True)
    t0 = time.time()
    model = Motus(mc)
    # Freeze WAN like training did
    for p in model.video_model.wan_model.parameters():
        p.requires_grad = False
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    print(f"[2/4] Loading checkpoint state dict from {CKPT}...", flush=True)
    t0 = time.time()
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    # DeepSpeed wraps the model state under 'module'; sometimes nested.
    if isinstance(sd, dict):
        if "module" in sd and isinstance(sd["module"], dict):
            sd = sd["module"]
        elif "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # The custom save hook saves only the unwrapped model state, including all
    # trainable bits we care about (state_expert, action_conditioner, action_injector,
    # und_expert). The frozen backbones are also saved but match what's already loaded.
    print(f"  loaded in {time.time()-t0:.1f}s — missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    # Spot-check that WM-specific params got loaded
    inj_norm = model.action_injector.proj.weight.detach().float().norm().item()
    se_block0_qkv_norm = model.state_expert.blocks[0].wan_state_qkv.detach().float().norm().item()
    print(f"  action_injector.proj.weight norm   = {inj_norm:.4f}  (zero-init was 0.0 — should be > 0 after training)")
    print(f"  state_expert.blocks[0].wan_state_qkv norm = {se_block0_qkv_norm:.4f}  (random-init, should be > 0)")

    # Mode + device sanity
    model = model.to("cuda").eval()
    print(f"[3/4] _is_world_model(model) = {_is_world_model(model)}", flush=True)

    ds = create_dataset(cfg, val=True)
    # -1 → run on every deterministic val chunk (full held-out coverage).
    n = len(ds) if args.num_samples < 0 else min(args.num_samples, len(ds))
    print(f"[4/4] Pulling {n} / {len(ds)} samples from val dataset and running WM inference...", flush=True)
    samples = [ds[i] for i in range(n)]
    bs = max(1, args.batch_size)
    print(f"  running inference in batches of {bs} ({(n + bs - 1) // bs} forward passes)", flush=True)

    # Run the model on chunks of `bs` samples at a time. inference_step_wm is fully
    # batched, so one forward pass covers the whole chunk — per-sample metrics are
    # then split back out so the reported mean/std are still over individual samples.
    # Each chunk is collated independently: the VLM `pixel_values` are concatenated
    # per-image (not a [B, ...] tensor), so the batch must be assembled by collate_fn,
    # not by row-slicing a pre-collated global batch.
    all_video_mses = []
    all_pred_states = []
    all_gt_states = []
    all_action_seqs = []
    first_pred_frames = None
    first_gt_frames = None
    first_pred_states = None
    first_gt_states = None
    for start in range(0, n, bs):
        end = min(start + bs, n)
        sub = collate_fn(samples[start:end])
        t0 = time.time()
        predicted_frames, predicted_states = inference_sample(model, sub, cfg)
        dt = time.time() - t0

        gt_frames = sub["video_frames"].to(predicted_frames.device)
        predicted_frames = predicted_frames.permute(0, 2, 1, 3, 4)  # [b, T, C, H, W]
        b = predicted_frames.shape[0]
        # Per-sample video MSE within the chunk (keeps mean/std over samples, not chunks).
        per_sample_video_mse = ((predicted_frames - gt_frames) ** 2).reshape(b, -1).mean(dim=1)
        gt_states = sub["future_states"][:, :predicted_states.shape[1]].to(predicted_states.device)
        sm = compute_state_metrics(predicted_states, gt_states)

        print(f"\n  samples {start}-{end-1} ({b}): inference {dt:.1f}s  "
              f"({dt / b:.2f}s/sample)  video_mse={per_sample_video_mse.mean().item():.4f}  "
              f"state_mse={sm['mse_loss']:.4f}  state_l2={sm['l2_error']:.4f}")
        all_video_mses.extend(per_sample_video_mse.detach().cpu().float().tolist())
        all_pred_states.append(predicted_states.detach().cpu().float())
        all_gt_states.append(gt_states.detach().cpu().float())
        all_action_seqs.append(sub["action_sequence"][:, :predicted_states.shape[1]].detach().cpu().float())
        if start == 0:
            first_pred_frames = predicted_frames[0:1]
            first_gt_frames = gt_frames[0:1]
            first_pred_states = predicted_states[0:1]
            first_gt_states = gt_states[0:1]

    # One-line video summary (in normalized pixel-latent space — not directly
    # comparable to the state numbers below; reported here so video + state
    # metrics live in one block).
    print(f"\n  video_mse  mean = {np.mean(all_video_mses):.6f}    std = {np.std(all_video_mses):.6f}    (normalized latent MSE; trainer metric)")

    # ---------- STATE EVAL REPORT ----------
    # Computed in raw robot units (un-normalized) via the loader's action_min/max,
    # plus identity-baseline ratios and Top-K worst predictions. See
    # `train.eval_metrics` for the metric definitions.
    pred_states = torch.cat(all_pred_states, dim=0)     # [N, H, S] normalized [0,1]
    gt_states   = torch.cat(all_gt_states, dim=0)
    action_seqs = torch.cat(all_action_seqs, dim=0)

    inner = model.module if hasattr(model, "module") else model
    # Reuse the dataset handle we already created for inference to pull
    # normalization stats (no extra LeRobot init).
    action_min = np.asarray(ds.action_min)
    action_max = np.asarray(ds.action_max)

    JOINT_NAMES = [f"L_j{i}" for i in range(7)] + [f"R_j{i}" for i in range(7)] if pred_states.shape[-1] == 14 else None

    report = compute_state_eval_report(
        pred_states=pred_states,
        gt_states=gt_states,
        action_seqs=action_seqs,
        action_min=action_min,
        action_max=action_max,
        gripper_joints=[6, 13],
        k_worst=8,
    )
    print(format_report(report, joint_names=JOINT_NAMES))

    # Dump machine-readable summary for downstream tracking (CI, wandb, etc).
    import json
    summary_path = OUT / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "video_mse_mean": float(np.mean(all_video_mses)),
            "video_mse_std":  float(np.std(all_video_mses)),
            **report_to_dict(report),
        }, f, indent=2)
    print(f"\n  wrote {summary_path}")

    # Save visualizations
    grid = create_video_grid(first_pred_frames, first_gt_frames, num_samples=1)
    grid_path = OUT / "video_grid.png"
    grid.save(grid_path)
    plot = create_state_plot(first_pred_states, first_gt_states, num_samples=1)
    plot_path = OUT / "state_plot.png"
    plot.save(plot_path)
    print(f"\n  wrote {grid_path}")
    print(f"  wrote {plot_path}")


if __name__ == "__main__":
    main()
