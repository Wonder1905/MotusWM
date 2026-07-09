#!/usr/bin/env python3
"""
Evaluation utilities for Motus.
Implements inference sampling and metrics computation for validation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# Suppress matplotlib font manager debug messages
matplotlib.set_loglevel("WARNING")
from PIL import Image
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging
import os

logger = logging.getLogger(__name__)


def create_video_grid(predicted_frames: torch.Tensor, ground_truth_frames: torch.Tensor, 
                     num_samples: int = 4) -> Image.Image:
    """
    Create a grid visualization comparing predicted and ground truth video frames.
    
    Args:
        predicted_frames: (B, T, C, H, W) predicted video frames
        ground_truth_frames: (B, T, C, H, W) ground truth video frames  
        num_samples: number of samples to visualize
        
    Returns:
        PIL Image of the comparison grid
    """
    batch_size = min(predicted_frames.shape[0], num_samples)
    num_frames = predicted_frames.shape[1]
    
    # Convert to numpy (B, T, H, W, C)
    pred_np = predicted_frames[:batch_size].detach().cpu().permute(0, 1, 3, 4, 2).numpy()
    gt_np = ground_truth_frames[:batch_size].detach().cpu().permute(0, 1, 3, 4, 2).numpy()

    # Clip values to [0, 1] (safety)
    pred_np = np.clip(pred_np, 0, 1)
    gt_np = np.clip(gt_np, 0, 1)
    
    # Create grid: rows are samples, columns are [GT_frame1, GT_frame2, ..., GT_frameN, Pred_frame1, Pred_frame2, ..., Pred_frameN]
    fig, axes = plt.subplots(batch_size, num_frames * 2, figsize=(4 * num_frames * 2, 4 * batch_size))
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    elif num_frames * 2 == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(batch_size):
        for t in range(num_frames):
            # Ground truth frame
            axes[i, t].imshow(gt_np[i, t])
            axes[i, t].set_title(f'GT Frame {t+1}')
            axes[i, t].axis('off')
            
            # Predicted frame  
            axes[i, t + num_frames].imshow(pred_np[i, t])
            axes[i, t + num_frames].set_title(f'Pred Frame {t+1}')
            axes[i, t + num_frames].axis('off')
    
    plt.tight_layout()
    
    # Convert to PIL Image
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img_array = np.asarray(buf)
    img_array = img_array[:, :, :3]  # Remove alpha channel
    
    plt.close(fig)
    
    return Image.fromarray(img_array)


def _is_world_model(model) -> bool:
    """True if the (possibly DDP/DeepSpeed-wrapped) model is in world_model mode."""
    inner = model.module if hasattr(model, 'module') else model
    return bool(getattr(getattr(inner, 'config', None), 'world_model', False))


@torch.no_grad()
def inference_sample(model, batch: Dict, config):
    """
    Run inference for one batch.

    VLA mode (default): returns (predicted_frames, predicted_actions)
    WM  mode:           returns (predicted_frames, predicted_states)

    The two paths are kept separate (different inputs, different outputs).
    """
    model.eval()
    inner = model.module if hasattr(model, 'module') else model
    num_inference_steps = config.model.inference.num_inference_timesteps

    device = next(model.parameters()).device
    first_frame = batch['first_frame'].to(device)

    state = batch['initial_state'].to(device) if 'initial_state' in batch and batch['initial_state'] is not None else None
    language_embeddings = batch['language_embedding']
    if language_embeddings is not None:
        language_embeddings = language_embeddings.to(device)
    vlm_inputs = batch['vlm_inputs']
    if vlm_inputs is not None:
        vlm_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in vlm_inputs.items()}

    if _is_world_model(model):
        if 'action_sequence' not in batch or batch['action_sequence'] is None:
            raise ValueError("WM inference requires `action_sequence` in the batch")
        actions = batch['action_sequence'].to(device)
        predicted_frames, predicted_states = inner.inference_step_wm(
            first_frame=first_frame,
            state=state,
            actions=actions,
            num_inference_steps=num_inference_steps,
            language_embeddings=language_embeddings,
            vlm_inputs=vlm_inputs,
        )
        model.train()
        return predicted_frames, predicted_states

    predicted_frames, predicted_actions = inner.inference_step(
        first_frame=first_frame,
        state=state,
        num_inference_steps=num_inference_steps,
        language_embeddings=language_embeddings,
        vlm_inputs=vlm_inputs,
    )
    model.train()
    return predicted_frames, predicted_actions


def compute_state_metrics(predicted_states: torch.Tensor, ground_truth_states: torch.Tensor) -> Dict[str, float]:
    """MSE / L2 on predicted future states (world-model mode)."""
    mse = F.mse_loss(predicted_states, ground_truth_states, reduction='none').float()
    mse_per_sample = mse.reshape(predicted_states.shape[0], -1).mean(1)
    l2 = mse.sqrt() / (1 + 1e-3)
    l2_per_sample = l2.reshape(predicted_states.shape[0], -1).mean(1)
    return {
        'mse_loss': mse_per_sample.mean().item(),
        'l2_error': l2_per_sample.mean().item(),
        'mse_std': mse_per_sample.std().item(),
        'l2_std': l2_per_sample.std().item(),
    }


def create_state_plot(
    predicted_states: torch.Tensor,    # [B, H, state_dim]
    ground_truth_states: torch.Tensor, # [B, H, state_dim]
    num_samples: int = 4,
) -> Image.Image:
    """Line plot of GT (solid) vs Pred (dashed) per joint dim, one row per sample."""
    B = min(predicted_states.shape[0], num_samples)
    H = predicted_states.shape[1]
    D = predicted_states.shape[2]
    pred = predicted_states[:B].detach().cpu().float().numpy()
    gt = ground_truth_states[:B].detach().cpu().float().numpy()
    t = np.arange(H)

    fig, axes = plt.subplots(B, 1, figsize=(max(6, 0.6 * H), 3 * B), squeeze=False)
    cmap = plt.get_cmap('tab20', D)
    for i in range(B):
        ax = axes[i, 0]
        for d in range(D):
            color = cmap(d)
            ax.plot(t, gt[i, :, d], color=color, linewidth=1.6, label=f'GT d{d}' if i == 0 else None)
            ax.plot(t, pred[i, :, d], color=color, linewidth=1.0, linestyle='--')
        ax.set_title(f'Sample {i}: state dims (solid=GT, dashed=Pred)')
        ax.set_xlabel('Horizon step')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return Image.fromarray(arr)


def compute_action_metrics(predicted_actions: torch.Tensor, ground_truth_actions: torch.Tensor) -> Dict[str, float]:
    """
    Compute action prediction metrics (MSE and L2 error).
    
    Args:
        predicted_actions: (B, T, action_dim) predicted actions
        ground_truth_actions: (B, T, action_dim) ground truth actions  
        
    Returns:
        Dictionary containing MSE and L2 error metrics
    """
    # Compute MSE loss
    mse_loss = F.mse_loss(predicted_actions, ground_truth_actions, reduction='none').float()
    mse_loss_per_sample = mse_loss.reshape(predicted_actions.shape[0], -1).mean(1)
    
    # Compute L2 error (RMSE)
    l2_loss = mse_loss.sqrt() / (1 + 1e-3)
    l2_loss_per_sample = l2_loss.reshape(predicted_actions.shape[0], -1).mean(1)
    
    return {
        'mse_loss': mse_loss_per_sample.mean().item(),
        'l2_error': l2_loss_per_sample.mean().item(),
        'mse_std': mse_loss_per_sample.std().item(),
        'l2_std': l2_loss_per_sample.std().item()
    }


@torch.no_grad()
def evaluate_model(model, dataloader, accelerator, config, num_eval_batches: int = 2) -> Dict[str, float]:
    """
    Local-only evaluation: no distributed aggregation; safe for rank0-only evaluation.

    Dispatches on world_model:
      VLA mode → action MSE/L2 + video grid (existing behaviour).
      WM  mode → state  MSE/L2 + video grid + state-vs-GT line plot.
    """
    is_wm = _is_world_model(model)
    logger.info(f"Running {'WM' if is_wm else 'VLA'} evaluation for {num_eval_batches} batches...")
    model.eval()

    metrics = defaultdict(list)
    visual_samples = []

    for step, batch in enumerate(dataloader):
        if step >= num_eval_batches:
            break
        if batch is None:
            continue

        predicted_frames, predicted_side = inference_sample(model, batch, config)
        gt_frames = batch['video_frames'].to(predicted_frames.device)              # [B, T, C, H, W]
        predicted_frames = predicted_frames.permute(0, 2, 1, 3, 4)                 # [B, T, C, H, W]

        video_mse = F.mse_loss(predicted_frames, gt_frames, reduction='mean').item()
        metrics['video_mse'].append(video_mse)

        if is_wm:
            if 'future_states' in batch and batch['future_states'] is not None and predicted_side is not None:
                gt_states = batch['future_states'][:, :predicted_side.shape[1]].to(predicted_side.device)
                for k, v in compute_state_metrics(predicted_side, gt_states).items():
                    metrics[f'state_{k}'].append(v)
        else:
            if 'action_sequence' in batch and predicted_side is not None:
                gt_actions = batch['action_sequence'][:, :predicted_side.shape[1]].to(predicted_side.device)
                for k, v in compute_action_metrics(predicted_side, gt_actions).items():
                    metrics[f'action_{k}'].append(v)

        if step == 0:
            visual_samples.append({
                'predicted_frames': predicted_frames[:4],
                'ground_truth_frames': gt_frames[:4],
                'predicted_side': predicted_side[:4] if predicted_side is not None else None,
                'ground_truth_side': (
                    batch.get('future_states') if is_wm else batch.get('action_sequence')
                ),
            })

    final_metrics = {}
    for key, values in metrics.items():
        if values:
            final_metrics[key] = float(np.mean(values))
            final_metrics[f'{key}_std'] = float(np.std(values))

    if visual_samples:
        sample = visual_samples[0]
        final_metrics['visualization'] = create_video_grid(
            sample['predicted_frames'], sample['ground_truth_frames'], num_samples=4
        )
        if is_wm and sample['predicted_side'] is not None and sample['ground_truth_side'] is not None:
            gt_side = sample['ground_truth_side'][:sample['predicted_side'].shape[0]].to(sample['predicted_side'].device)
            final_metrics['state_plot'] = create_state_plot(
                sample['predicted_side'], gt_side, num_samples=min(4, sample['predicted_side'].shape[0])
            )

    model.train()
    return final_metrics


def log_evaluation_metrics(metrics: Dict, writer, accelerator, global_step: int):
    """
    Log evaluation metrics to tensorboard and wandb.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        writer: TensorBoard writer (can be None)
        accelerator: HuggingFace accelerator  
        global_step: Current training step
    """
    if accelerator.is_main_process:
        # Log scalar metrics
        log_dict = {}
        for key, value in metrics.items():
            if key not in ['visualization', 'state_plot', 'visual_samples'] and isinstance(value, (int, float)):
                log_dict[f'eval/{key}'] = value

        # Log to accelerator (wandb)
        if log_dict:
            accelerator.log(log_dict, step=global_step)

        # wandb: log images too (accelerator.log alone won't carry PIL.Image)
        try:
            import wandb as _wandb
            if _wandb.run is not None:
                imgs = {}
                if 'visualization' in metrics:
                    imgs['eval/video_grid'] = _wandb.Image(metrics['visualization'])
                if 'state_plot' in metrics:
                    imgs['eval/state_plot'] = _wandb.Image(metrics['state_plot'])
                if imgs:
                    _wandb.log(imgs, step=global_step)
        except Exception:
            pass

        # Log to TensorBoard
        if writer is not None:
            for key, value in log_dict.items():
                writer.add_scalar(key, value, global_step)
            if 'visualization' in metrics:
                img_array = np.array(metrics['visualization']).transpose(2, 0, 1)
                writer.add_image('eval/video_grid', img_array, global_step)
            if 'state_plot' in metrics:
                img_array = np.array(metrics['state_plot']).transpose(2, 0, 1)
                writer.add_image('eval/state_plot', img_array, global_step)

        # Print summary
        logger.info("=== Evaluation Results ===")
        for key, value in metrics.items():
            if key not in ['visualization', 'state_plot'] and isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.4f}")