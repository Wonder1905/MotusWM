"""
Fast whole-episode frame decode for the offline cache builders.

The dataset's __getitem__ decodes only the frames a single (episode, condition)
sample needs, re-opening + random-seeking the mp4 every call. For an offline
build that touches EVERY condition frame, that means ~T reopens per episode and
is brutally slow. Here we decode an entire episode's model-input frames ONCE
(one sequential pass per camera) and let the builder slice per condition frame.

Parity with __getitem__ is preserved by reusing the dataset's own
`_resize_frame_chw` and replicating its stitch exactly (verified by
scripts/verify_vae_cache.py, which compares against the untouched __getitem__).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch

from lerobot.datasets.video_utils import decode_video_frames


def _to_chw_float(frame: torch.Tensor) -> torch.Tensor:
    img = frame.float()
    if img.ndim == 3 and img.shape[0] != 3 and img.shape[-1] == 3:
        img = img.permute(2, 0, 1)
    return img


def _stitch_three_cam(ds, cam_high, cam_left, cam_right) -> torch.Tensor:
    """Verbatim copy of __getitem__'s three-cam `load_concatenated_view` branch."""
    cam_high = _to_chw_float(cam_high)
    cam_left = _to_chw_float(cam_left)
    cam_right = _to_chw_float(cam_right)
    c = cam_high.shape[0]
    top_h = int(cam_high.shape[1])
    target_w = int(cam_high.shape[2])
    bottom_h = int(max(cam_left.shape[1], cam_right.shape[1]))
    split_w = target_w // 2
    right_w = target_w - split_w
    cam_high_r = ds._resize_frame_chw(cam_high, (top_h, target_w))
    cam_left_r = ds._resize_frame_chw(cam_left, (bottom_h, split_w))
    cam_right_r = ds._resize_frame_chw(cam_right, (bottom_h, right_w))
    out = torch.zeros((c, top_h + bottom_h, target_w), dtype=cam_high_r.dtype)
    out[:, :top_h, :target_w] = cam_high_r
    out[:, top_h:, :split_w] = cam_left_r
    out[:, top_h:, split_w:] = cam_right_r
    return ds._resize_frame_chw(out, ds.video_size)


def decode_episode_model_frames(ds, local_ep: int) -> Tuple[int, torch.Tensor, str]:
    """Decode all of episode `local_ep`'s model-input frames in one pass.

    Returns (true_episode_index, frames[T, C, Hs, Ws] in [0,1], instruction_text).
    Frame index i corresponds to episode-local frame i (the same indexing
    `_calculate_sampling_indices` produces).
    """
    if ds.task_mode != "single":
        raise NotImplementedError("fast episode decode supports task_mode='single' only")
    lr = ds.lerobot_dataset
    edi = lr.episode_data_index
    from_idx = int(edi["from"][local_ep].item()) if hasattr(edi["from"][local_ep], "item") else int(edi["from"][local_ep])
    to_idx = int(edi["to"][local_ep].item()) if hasattr(edi["to"][local_ep], "item") else int(edi["to"][local_ep])
    hf = lr.hf_dataset
    rows = list(range(from_idx, to_idx))

    ts_vals = hf[rows]["timestamp"]
    if isinstance(ts_vals, torch.Tensor):
        timestamps = ts_vals.flatten().tolist()
    elif isinstance(ts_vals, (list, tuple)) and ts_vals and isinstance(ts_vals[0], torch.Tensor):
        timestamps = torch.stack(ts_vals).flatten().tolist()
    else:
        timestamps = [float(x) for x in list(ts_vals)]

    item0 = hf[from_idx]
    epv = item0.get("episode_index")
    true_ep = int(epv.item()) if hasattr(epv, "item") else int(epv)
    instr = item0.get("language_instruction", None)
    if instr is None or (isinstance(instr, str) and len(instr.strip()) == 0):
        instr = item0.get("task", "")
    if not isinstance(instr, str):
        instr = str(instr)

    def decode_key(vid_key: str) -> torch.Tensor:
        path = Path(lr.root) / lr.meta.get_video_file_path(true_ep, vid_key)
        return decode_video_frames(path, timestamps, lr.tolerance_s, lr.video_backend).squeeze(0)  # [T,C,H,W]

    if ds.has_concat:
        frames = decode_key("observation.images.cam_concatenated")
        out = torch.stack([ds._resize_frame_chw(frames[i].float(), ds.video_size)
                           for i in range(frames.shape[0])], dim=0)
    elif ds.has_three_cam:
        fh = decode_key("observation.images.cam_high")
        fl = decode_key("observation.images.cam_left_wrist")
        fr = decode_key("observation.images.cam_right_wrist")
        out = torch.stack([_stitch_three_cam(ds, fh[i], fl[i], fr[i]) for i in range(fh.shape[0])], dim=0)
    else:
        vid_key = None
        for k in ds.single_view_candidates:
            if k in lr.meta.video_keys or k in hf.column_names:
                vid_key = k
                break
        if vid_key is None:
            vid_key = ds.single_view_candidates[0]
        frames = decode_key(vid_key)
        out = torch.stack([ds._resize_frame_chw(frames[i].float(), ds.video_size)
                           for i in range(frames.shape[0])], dim=0)

    return true_ep, out, instr
