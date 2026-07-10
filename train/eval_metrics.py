"""
State-prediction evaluation metrics for the Motus world model.

All metrics operate on three tensors of shape [N, H, S]:
  - pred_states   : model's predicted future states  (normalized to [0, 1])
  - gt_states     : ground-truth future states       (normalized to [0, 1])
  - action_seqs   : the clean action chunk that was passed as conditioning
                    (also normalized to [0, 1], same scale)
N samples × H horizon steps × S joints.

We un-normalize per joint using the same (action_min, action_max) the loader
used, so the reported numbers are in the robot's native units (radians for
ALOHA arms, normalized gripper opening for grippers).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# -------- helpers --------------------------------------------------------------

def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().float().numpy()


def _scale_per_joint(action_min: np.ndarray, action_max: np.ndarray) -> np.ndarray:
    """Per-joint scale factor that un-does the loader's min-max normalization."""
    rng = (action_max - action_min).astype(np.float64)
    rng = np.where(rng == 0, 1.0, rng)
    return rng  # multiply normalized differences by this to recover raw units


# -------- individual metrics ---------------------------------------------------

def per_joint_rmse(
    pred: torch.Tensor, gt: torch.Tensor,
    action_min: np.ndarray, action_max: np.ndarray,
) -> np.ndarray:
    """Per-joint RMSE in raw units. Returns shape [S]."""
    diff = _to_np(pred - gt)                           # [N, H, S] normalized
    diff_raw = diff * _scale_per_joint(action_min, action_max)  # broadcast last dim
    rmse = np.sqrt((diff_raw ** 2).mean(axis=(0, 1)))  # [S]
    return rmse


def identity_ratio_per_joint(
    pred: torch.Tensor, gt: torch.Tensor, actions: torch.Tensor,
) -> np.ndarray:
    """
    Per-joint mean |pred-gt| / mean |action-gt|.
      < 1 → model beats the identity-copy baseline on that joint
      > 1 → model is worse than just copying the action
    Computed on NORMALIZED values (ratio is unit-free).
    """
    pred_err = _to_np((pred - gt).abs()).mean(axis=(0, 1))       # [S]
    id_err   = _to_np((actions - gt).abs()).mean(axis=(0, 1))    # [S]
    return pred_err / np.maximum(id_err, 1e-12)


def identity_ratio_pooled(
    pred: torch.Tensor, gt: torch.Tensor, actions: torch.Tensor,
    joints: Optional[List[int]] = None,
) -> float:
    """
    Aggregate identity ratio = (mean |pred-gt|) / (mean |action-gt|), pooled over
    all elements of the selected joints. Unlike averaging the per-joint ratios,
    this weights each joint by how much it actually moves, so a near-static joint
    with a tiny denominator can't blow up the headline. Computed on NORMALIZED
    values (every joint lives in [0,1], so pooling across joints is apples-to-apples).
      < 1 → model beats identity-copy overall;  > 1 → identity wins overall.
    """
    p, g, a = (pred, gt, actions) if joints is None else (pred[..., joints], gt[..., joints], actions[..., joints])
    pred_err = float(_to_np((p - g).abs()).mean())
    id_err   = float(_to_np((a - g).abs()).mean())
    return pred_err / max(id_err, 1e-12)


def mean_max_per_sample(
    pred: torch.Tensor, gt: torch.Tensor,
    action_min: np.ndarray, action_max: np.ndarray,
) -> Tuple[float, float]:
    """
    For each sample, take the largest absolute error across (frame, joint),
    in RAW units. Then average across samples. Returns (mean, std).
    """
    diff = _to_np((pred - gt))                                  # [N, H, S]
    diff_raw = diff * _scale_per_joint(action_min, action_max)  # broadcast
    per_sample_max = np.abs(diff_raw).reshape(diff_raw.shape[0], -1).max(axis=1)  # [N]
    return float(per_sample_max.mean()), float(per_sample_max.std())


def rmse_by_horizon_step(
    pred: torch.Tensor, gt: torch.Tensor,
    action_min: np.ndarray, action_max: np.ndarray,
) -> np.ndarray:
    """
    RMSE per future-horizon step k = 0, 1, ..., H-1 (raw units, averaged over
    samples and joints). Captures error growth as we predict further out.
    Returns shape [H].
    """
    diff = _to_np(pred - gt)                                    # [N, H, S]
    diff_raw = diff * _scale_per_joint(action_min, action_max)
    return np.sqrt((diff_raw ** 2).mean(axis=(0, 2)))           # [H]


def gripper_rmse(
    pred: torch.Tensor, gt: torch.Tensor,
    action_min: np.ndarray, action_max: np.ndarray,
    gripper_joints: List[int] = (6, 13),
) -> float:
    """Pooled RMSE on the gripper joints only — the only ones where dynamics matters here."""
    gj = list(gripper_joints)
    diff = _to_np(pred[..., gj] - gt[..., gj])                  # [N, H, |gj|]
    scale = _scale_per_joint(action_min, action_max)[gj]
    diff_raw = diff * scale
    return float(np.sqrt((diff_raw ** 2).mean()))


def top_k_worst(
    pred: torch.Tensor, gt: torch.Tensor, actions: torch.Tensor,
    action_min: np.ndarray, action_max: np.ndarray,
    k: int = 8,
) -> List[Dict]:
    """
    Top-K largest |pred-gt| in RAW units, with sample/frame/joint indices and
    the corresponding pred / gt / action values (also in raw units).
    """
    diff_norm = _to_np(pred - gt)
    scale = _scale_per_joint(action_min, action_max)
    diff_raw = np.abs(diff_norm * scale)                        # [N, H, S]
    N, H, S = diff_raw.shape
    flat = diff_raw.reshape(-1)
    idxs = np.argpartition(-flat, range(min(k, flat.size)))[:k]
    idxs = idxs[np.argsort(-flat[idxs])]                        # sort descending

    pred_np    = _to_np(pred) * scale + action_min
    gt_np      = _to_np(gt)   * scale + action_min
    action_np  = _to_np(actions) * scale + action_min

    rows = []
    for fi in idxs:
        b = fi // (H * S);  r = (fi // S) % H;  j = fi % S
        rows.append({
            "sample": int(b), "frame": int(r), "joint": int(j),
            "abs_err_raw": float(diff_raw[b, r, j]),
            "pred":   float(pred_np[b, r, j]),
            "gt":     float(gt_np[b, r, j]),
            "action": float(action_np[b, r, j]),
        })
    return rows


# -------- aggregator -----------------------------------------------------------

@dataclass
class StateEvalReport:
    n_samples: int
    horizon: int
    n_joints: int
    per_joint_rmse_raw:        np.ndarray            # [S]
    identity_ratio_per_joint:  np.ndarray            # [S]
    # Pooled (movement-weighted) identity ratios — the headline "do we beat identity"
    # numbers, complementing the per-joint table. Computed in normalized [0,1] space.
    identity_ratio_overall:    float                 # all joints pooled
    identity_ratio_arm:        float                 # arm joints pooled
    identity_ratio_gripper:    float                 # gripper joints pooled
    # Macro-average of the per-joint ratios (each joint weighted equally) — kept
    # alongside the pooled value because they diverge when a joint barely moves.
    identity_ratio_macro_avg:  float
    # Primary headline: NORMALIZED [0,1] space (the actual space the model trains
    # in — every joint contributes equally, no unit-mixing between arm radians
    # and gripper opening).
    overall_mse_norm:          float                  # = mean((pred-gt)²)            ← trainer's metric
    overall_rmse_norm:         float                  # = sqrt(overall_mse_norm)
    overall_mae_norm:          float                  # = mean(|pred-gt|)
    # Per-group physical context (arm joints stay in radians, gripper joints in
    # their own normalized opening — never aggregated together).
    arm_rmse_rad:              float                  # RMSE across arm joints only, in radians
    arm_mae_rad:               float
    arm_rmse_norm:             float                  # = RMSE in [0,1] space ≈ fraction of arm joint range
    arm_mae_norm:              float
    gripper_rmse_opening:      float                  # RMSE across gripper joints, in their normalized opening
    gripper_mae_opening:       float
    gripper_rmse_norm:         float                  # = RMSE in [0,1] space ≈ fraction of gripper range
    gripper_mae_norm:          float
    arm_joints:                List[int]
    # Also kept: per-joint RMSE both un-normalized and not, plus tail metrics.
    overall_mse_raw:           float                  # informational only — mixes radians + gripper opening
    overall_rmse_raw:          float
    overall_mae_raw:           float
    mean_max_per_sample_raw:   Tuple[float, float]    # (mean, std), raw units
    rmse_by_horizon_step:      np.ndarray            # [H], raw units
    gripper_rmse_raw:          float
    gripper_joints:            List[int]
    top_k_worst:               List[Dict]


def compute_state_eval_report(
    pred_states: torch.Tensor,
    gt_states: torch.Tensor,
    action_seqs: torch.Tensor,
    action_min: np.ndarray,
    action_max: np.ndarray,
    gripper_joints: Optional[List[int]] = None,
    k_worst: int = 8,
) -> StateEvalReport:
    if gripper_joints is None:
        gripper_joints = [6, 13]  # ALOHA bimanual default

    N, H, S = pred_states.shape
    diff_norm = _to_np(pred_states - gt_states)                                            # [N,H,S], [0,1]
    diff_raw  = diff_norm * _scale_per_joint(action_min, action_max)                       # [N,H,S], robot units

    # Normalized (training-space) primary metrics — apples-to-apples across joints.
    overall_mse_norm  = float((diff_norm ** 2).mean())
    overall_rmse_norm = float(np.sqrt(overall_mse_norm))
    overall_mae_norm  = float(np.abs(diff_norm).mean())

    # Per-group physical context (arm radians vs gripper opening).
    arm_joints = [j for j in range(S) if j not in gripper_joints]
    arm_err_raw      = diff_raw[..., arm_joints]
    gripper_err_raw  = diff_raw[..., gripper_joints]
    arm_err_norm     = diff_norm[..., arm_joints]
    gripper_err_norm = diff_norm[..., gripper_joints]
    arm_rmse_rad         = float(np.sqrt((arm_err_raw ** 2).mean()))
    arm_mae_rad          = float(np.abs(arm_err_raw).mean())
    arm_rmse_norm        = float(np.sqrt((arm_err_norm ** 2).mean()))
    arm_mae_norm         = float(np.abs(arm_err_norm).mean())
    gripper_rmse_opening = float(np.sqrt((gripper_err_raw ** 2).mean()))
    gripper_mae_opening  = float(np.abs(gripper_err_raw).mean())
    gripper_rmse_norm    = float(np.sqrt((gripper_err_norm ** 2).mean()))
    gripper_mae_norm     = float(np.abs(gripper_err_norm).mean())

    # Informational raw aggregate (mixes units — included only for back-compat).
    overall_mse_raw  = float((diff_raw ** 2).mean())
    overall_rmse_raw = float(np.sqrt(overall_mse_raw))
    overall_mae_raw  = float(np.abs(diff_raw).mean())

    return StateEvalReport(
        n_samples=N, horizon=H, n_joints=S,
        per_joint_rmse_raw       = per_joint_rmse(pred_states, gt_states, action_min, action_max),
        identity_ratio_per_joint = (_id_ratio_pj := identity_ratio_per_joint(pred_states, gt_states, action_seqs)),
        identity_ratio_overall   = identity_ratio_pooled(pred_states, gt_states, action_seqs),
        identity_ratio_arm       = identity_ratio_pooled(pred_states, gt_states, action_seqs, joints=arm_joints),
        identity_ratio_gripper   = identity_ratio_pooled(pred_states, gt_states, action_seqs, joints=gripper_joints),
        identity_ratio_macro_avg = float(np.mean(_id_ratio_pj)),
        overall_mse_norm         = overall_mse_norm,
        overall_rmse_norm        = overall_rmse_norm,
        overall_mae_norm         = overall_mae_norm,
        arm_rmse_rad             = arm_rmse_rad,
        arm_mae_rad              = arm_mae_rad,
        arm_rmse_norm            = arm_rmse_norm,
        arm_mae_norm             = arm_mae_norm,
        gripper_rmse_opening     = gripper_rmse_opening,
        gripper_mae_opening      = gripper_mae_opening,
        gripper_rmse_norm        = gripper_rmse_norm,
        gripper_mae_norm         = gripper_mae_norm,
        arm_joints               = arm_joints,
        overall_mse_raw          = overall_mse_raw,
        overall_rmse_raw         = overall_rmse_raw,
        overall_mae_raw          = overall_mae_raw,
        mean_max_per_sample_raw  = mean_max_per_sample(pred_states, gt_states, action_min, action_max),
        rmse_by_horizon_step     = rmse_by_horizon_step(pred_states, gt_states, action_min, action_max),
        gripper_rmse_raw         = gripper_rmse(pred_states, gt_states, action_min, action_max, gripper_joints),
        gripper_joints           = list(gripper_joints),
        top_k_worst              = top_k_worst(pred_states, gt_states, action_seqs, action_min, action_max, k=k_worst),
    )


# -------- pretty printer -------------------------------------------------------

def format_report(rep: StateEvalReport, joint_names: Optional[List[str]] = None) -> str:
    if joint_names is None:
        joint_names = [f"j{i}" for i in range(rep.n_joints)]
    L = []
    L.append(f"\n=== STATE EVAL REPORT ===")
    L.append(f"  shape: N={rep.n_samples} samples × H={rep.horizon} steps × S={rep.n_joints} joints "
             f"= {rep.n_samples * rep.horizon * rep.n_joints} elements")
    mm_mean, mm_std = rep.mean_max_per_sample_raw
    L.append(f"\n  HEADLINE  (normalized [0,1] — the space the model trains in; every joint weighted equally)")
    L.append(f"    {'metric':<32} {'value':>10}")
    L.append(f"    {'-'*32} {'-'*10}")
    L.append(f"    MSE   (train metric)             {rep.overall_mse_norm:>10.6f}")
    L.append(f"    RMSE  (= √MSE)                   {rep.overall_rmse_norm:>10.4f}      (penalizes large errors more)")
    L.append(f"    MAE   (mean |err|)               {rep.overall_mae_norm:>10.4f}      (most intuitive — avg fraction of joint range off)")
    L.append(f"")
    L.append(f"  PER-GROUP PHYSICAL CONTEXT  (joints aggregated by unit type)")
    L.append(f"    ARM joints ({len(rep.arm_joints)})    "
             f"RMSE = {rep.arm_rmse_rad:.4f} rad ({rep.arm_rmse_rad * 180/3.14159:.2f}°)   "
             f"MAE = {rep.arm_mae_rad:.4f} rad ({rep.arm_mae_rad * 180/3.14159:.2f}°)   "
             f"[norm: RMSE={rep.arm_rmse_norm*100:.1f}%  MAE={rep.arm_mae_norm*100:.1f}% of range]")
    L.append(f"    GRIPPER joints (2)    "
             f"RMSE = {rep.gripper_rmse_opening:.4f}        "
             f"MAE = {rep.gripper_mae_opening:.4f}        "
             f"[norm: RMSE={rep.gripper_rmse_norm*100:.1f}%  MAE={rep.gripper_mae_norm*100:.1f}% of range]")
    L.append(f"")
    L.append(f"    mean_max_per_sample (raw)        {mm_mean:>10.4f} ± {mm_std:.4f}    (worst element per sample, averaged)")

    L.append(f"\n  PER-JOINT RMSE (raw units) and IDENTITY RATIO")
    L.append(f"    {'joint':<8} {'RMSE(raw)':>10}   {'model/identity':>15}    note")
    for j in range(rep.n_joints):
        r = float(rep.identity_ratio_per_joint[j])
        flag = ' ✓ beats identity' if r < 1 else ' ✗ identity wins'
        L.append(f"    {joint_names[j]:<8} {rep.per_joint_rmse_raw[j]:>10.4f}   {r:>15.3f}   {flag}")

    def _ratio_flag(r: float) -> str:
        return '✓ beats identity' if r < 1 else '✗ identity wins'
    L.append(f"\n  IDENTITY RATIO — AGGREGATE  (mean|pred-gt| / mean|action-gt|, normalized space)")
    L.append(f"    {'overall (pooled)':<22} {rep.identity_ratio_overall:>8.3f}   {_ratio_flag(rep.identity_ratio_overall)}")
    L.append(f"    {'arm joints (pooled)':<22} {rep.identity_ratio_arm:>8.3f}   {_ratio_flag(rep.identity_ratio_arm)}")
    L.append(f"    {'gripper (pooled)':<22} {rep.identity_ratio_gripper:>8.3f}   {_ratio_flag(rep.identity_ratio_gripper)}")
    L.append(f"    {'macro-avg of joints':<22} {rep.identity_ratio_macro_avg:>8.3f}   {_ratio_flag(rep.identity_ratio_macro_avg)}   (each joint weighted equally)")

    L.append(f"\n  RMSE BY HORIZON STEP (raw units, k = 0 → {rep.horizon - 1})")
    for k, v in enumerate(rep.rmse_by_horizon_step):
        L.append(f"    k={k:>2}  RMSE={float(v):.4f}")

    L.append(f"\n  TOP-{len(rep.top_k_worst)} WORST PREDICTIONS (raw units)")
    L.append(f"    {'sample':>6} {'frame':>5} {'joint':>5}   {'abs_err':>8}   {'pred':>8} {'gt':>8} {'action':>8}")
    for row in rep.top_k_worst:
        jn = joint_names[row["joint"]]
        L.append(f"    {row['sample']:>6} {row['frame']:>5} {jn:>5}   {row['abs_err_raw']:>8.4f}   "
                 f"{row['pred']:>8.4f} {row['gt']:>8.4f} {row['action']:>8.4f}")
    return "\n".join(L)


def report_to_dict(rep: StateEvalReport) -> Dict:
    """JSON-serialisable summary of the report."""
    return {
        "n_samples": rep.n_samples,
        "horizon": rep.horizon,
        "n_joints": rep.n_joints,
        "headline": {
            # Primary: normalized space (training space)
            "overall_mse_norm":         rep.overall_mse_norm,     # ← train metric
            "overall_rmse_norm":        rep.overall_rmse_norm,
            "overall_mae_norm":         rep.overall_mae_norm,
            # Per-group in physical units
            "arm_rmse_rad":             rep.arm_rmse_rad,
            "arm_mae_rad":              rep.arm_mae_rad,
            "arm_rmse_norm":            rep.arm_rmse_norm,   # ≈ fraction of arm joint range
            "arm_mae_norm":             rep.arm_mae_norm,
            "gripper_rmse_opening":     rep.gripper_rmse_opening,
            "gripper_mae_opening":      rep.gripper_mae_opening,
            "gripper_rmse_norm":        rep.gripper_rmse_norm,   # ≈ fraction of gripper range
            "gripper_mae_norm":         rep.gripper_mae_norm,
            "arm_joints":               rep.arm_joints,
            "gripper_joints":           rep.gripper_joints,
            # Aggregate identity ratios (normalized space)
            "identity_ratio_overall":   rep.identity_ratio_overall,
            "identity_ratio_arm":       rep.identity_ratio_arm,
            "identity_ratio_gripper":   rep.identity_ratio_gripper,
            "identity_ratio_macro_avg": rep.identity_ratio_macro_avg,
            # Tail / informational
            "mean_max_per_sample_raw":  rep.mean_max_per_sample_raw[0],
            "overall_rmse_raw":         rep.overall_rmse_raw,     # informational — mixes units
        },
        "per_joint_rmse_raw":        rep.per_joint_rmse_raw.tolist(),
        "identity_ratio_per_joint":  rep.identity_ratio_per_joint.tolist(),
        "rmse_by_horizon_step":      rep.rmse_by_horizon_step.tolist(),
        "gripper_joints":            rep.gripper_joints,
        "top_k_worst":               rep.top_k_worst,
    }
