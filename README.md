<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <h1>MotusWM — Motus with Future-State Prediction</h1>
</div>

<div align="center">
  <em>Adds future proprioceptive-state prediction to Motus: <strong>actions in → future video + future joint states</strong>.</em>
</div>

<br>

> Fork of [Motus](https://github.com/thu-ml/Motus). Base install, checkpoints, data format, and citation: **[README_MOTUS.md](README_MOTUS.md)**.

## Table of Contents
- [Overview](#overview)
  - [What goes in and out](#what-goes-in-and-out)
- [What changed](#what-changed)
- [Data: download and preprocess](#data-download-and-preprocess)
- [Training and eval](#training-and-eval)
- [Faster training: offline caches](#faster-training-offline-caches)
- [Configuration](#configuration)
- [File map](#file-map)

## Overview

Motus is a unified latent-action world model — a Mixture-of-Transformers over understanding,
action, and video-generation experts with a UniDiffuser-style scheduler. Its implemented path
predicts future video **and** the action chunk; the robot's proprioceptive **state** enters only as
a single conditioning token and is never predicted.

MotusWM re-wires it: actions become a **clean input** (via the new ActionConditioner +
ActionInjector) and the **future joint state** becomes a prediction target — so the contract becomes
*current image + state + intended actions → future video + future joint states*:

```
                ┌────────────────────────┐
  current image ─►                        ─► future video         (T frames)
  current state ─►       MotusWM          ─►
  H actions     ─►                        ─► future joint states  (H steps)
                └────────────────────────┘
```

### What goes in and out

`B` batch · `H` horizon (future steps) · `T` video frames · `S` state dim · `A` action dim ·
video is denoised in WAN VAE-latent space (temporally + spatially compressed from the `T` frames).

```
Base Motus
  in  : image(s), state [B,1,S], instruction
  out : actions [B,H,A],  video (T frames)

MotusWM
  in  : image(s), state [B,1,S], instruction, actions [B,H,A]   (actions now a clean input)
  out : future joint states [B,H,S],  video (T frames)         (states now predicted)
```

Actions move from output to clean input; future joint states become a new denoising target. Set
`model.world_model: true` to enable (off → stock Motus).

## What changed

**New modules** (instantiated only in `world_model` mode; the stock path never uses them):

| Module | File | Role |
|--------|------|------|
| **ActionConditioner** | [models/action_conditioner.py](models/action_conditioner.py) | Clean action chunk `[B,H,A]` → per-step action tokens `[B,H,dim]` (never noised). |
| **ActionInjector** | [models/action_injector.py](models/action_injector.py) | Adds action tokens into WAN's video-token stream (additive) so video sees the actions directly. |
| **StateExpert** | [models/state_expert.py](models/state_expert.py) | ActionExpert-shaped; consumes current state + noised future states + action tokens and predicts the future-state velocity. The new denoising target. |

**What trains** — only the new world-model modules (StateExpert + action conditioner/injector).
Everything else is frozen — WAN video DiT + VAE, Qwen3-VL (VLM), the understanding expert, and the
original action expert — so a run fits on a single 80 GB GPU (A100) in our runs. (T5 isn't in the
training model; its embeddings are precomputed offline.)

**Wiring** ([models/motus.py](models/motus.py)) — a `world_model` branch runs the trimodal MoT
joint-attention with the state stream in the old action slot (`process_joint_attention_wm`), plus
a WM forward and `inference_step_wm`.

**Data** ([data/dataset.py](data/dataset.py),
[data/lerobot/lerobot_dataset.py](data/lerobot/lerobot_dataset.py)) — the current state is a
conditioning token; at the **same** H future (downsampled) steps, the commanded action chunk
`[B,H,A]` is a clean input and the *observed* joint state `[B,H,S]` is the flow-matching target.
They differ only by tracking error — which `focal_fm` exploits.

**Loss** ([models/motus.py](models/motus.py)) — `L = w_v·L_video + w_s·L_state` (action loss
dropped; `w_s` = `loss_weights.action_loss_weight`). `L_state` has two types via `model.loss_term`:
- **`fm`** (default) — flow-matching MSE on the state velocity.
- **`focal_fm`** — same error, reweighted per timestep by the tracking residual `|state − action|`
  (mean over joints, `(·+1e-3)^0.7`, batch-normalized); degenerates to `fm` when residuals are
  uniform, and requires `A == S` (otherwise it silently runs `fm`).

**Warm-start** — `warmup_action_expert_ckpt` seeds the StateExpert from a trained ActionExpert
(blocks / time / registers / encoders transfer 1:1; head zero-init).

> A fused Triton AdaLN-LayerNorm kernel was tried and removed — faster in isolation, ~0%
> end-to-end. The model uses eager AdaLN LayerNorm.

## Data: download and preprocess

Reference dataset: LeRobot **`aloha_static_towel`** (50 episodes, 14-DoF bimanual ALOHA, v2.1 —
ships `meta/episodes.jsonl` + `meta/tasks.jsonl`, so no metadata build).

**Download** to the path in `dataset.params.root` (`HF_TOKEN` must be set):

```bash
huggingface-cli download lerobot/aloha_static_towel \
    --repo-type dataset --local-dir data_lerobot/aloha_static_towel
```

**Normalization stats** are already shipped in [data/utils/stat.json](data/utils/stat.json), keyed
by `embodiment_type` — nothing to run.

**T5 text cache is required** (`enable_t5_fallback: false`, so text is read from disk):

```bash
python data/lerobot/build_t5_cache_offline.py \
    --root data_lerobot/aloha_static_towel --wan_path pretrained_models
```

Set `enable_t5_fallback: true` to encode live instead. VAE/VLM caches are optional — see
[Faster training](#faster-training-offline-caches).

## Training and eval

```bash
torchrun --nnodes=1 --nproc_per_node=1 --node_rank=0 \
    --master_addr=127.0.0.1 --master_port=29520 \
    train/train.py \
    --deepspeed configs/zero1.json \
    --config configs/_wm_aloha_towel.yaml \
    --run_name wm_aloha_towel_focal --report_to wandb
```

Evaluate a checkpoint — writes state-prediction metrics and GT-vs-prediction video grids + state plots:

```bash
python scripts/wm_eval.py \
    --ckpt checkpoints_lerobot/_wm_aloha_towel/wm_aloha_towel_focal/checkpoint_step_<N>/pytorch_model/mp_rank_00_model_states.pt \
    --config configs/_wm_aloha_towel.yaml --num_samples 4 --out_dir eval_outputs/<run>
```

Metrics: [train/eval_metrics.py](train/eval_metrics.py) (per-joint, native units). Rollout:
[train/sample.py](train/sample.py).

## Faster training: offline caches

The frozen WAN VAE and Qwen3-VL forwards depend only on the fixed input frames, so with
`image_aug: false` their outputs can be precomputed once and read from disk each step:

| Cache | Builder | Skips (per step) | Flag |
|-------|---------|------------------|------|
| **VAE** latents | [build_vae_cache_offline.py](data/lerobot/build_vae_cache_offline.py) | WAN VAE encode (~25%) | `use_vae_cache` |
| **VLM** hidden states | [build_vlm_cache_offline.py](data/lerobot/build_vlm_cache_offline.py) | Qwen3-VL forward (~11%) | `use_vlm_cache` |

_The ~25% / ~11% step-time shares are from our own profiling, not a formal benchmark._

Missing keys fall back to live compute — per **batch** (one miss recomputes that modality for the
whole batch), not per sample — so both are safe to skip or to enable before building.
A shared decoder ([_cache_decode.py](data/lerobot/_cache_decode.py)) reads each episode in one pass
instead of re-seeking per frame. Build once:

```bash
python data/lerobot/build_vae_cache_offline.py --config configs/_wm_aloha_towel.yaml
python data/lerobot/build_vlm_cache_offline.py --config configs/_wm_aloha_towel.yaml
```

## Configuration

Reference config: [configs/_wm_aloha_towel.yaml](configs/_wm_aloha_towel.yaml).

| Key | Meaning |
|-----|---------|
| `model.world_model` | Enable future-state mode. |
| `model.loss_term` | `fm` or `focal_fm` (see [What changed](#what-changed)). |
| `model.warmup_action_expert_ckpt` | ActionExpert checkpoint to seed the StateExpert (or `null`). |
| `common.action_dim` / `state_dim` | `A` / `S`. |
| `common.num_video_frames` / `video_action_freq_ratio` | `T`, and `H = T × ratio`. |
| `dataset.image_aug` | Must be `false` for the caches. |
| `dataset.params.use_vae_cache` / `use_vlm_cache` | Enable the offline caches. |
| `dataset.params.num_val_episodes` / `val_split_seed` | Held-out validation split. |

## File map

```
models/action_conditioner.py             clean action chunk → action tokens
models/action_injector.py                inject action tokens into WAN video stream
models/state_expert.py                   future-state denoising expert
models/motus.py                          world_model forward + inference_step_wm
data/dataset.py                          WM sample assembly (clean actions, state target)
data/lerobot/lerobot_dataset.py          LeRobot loader + cache lookups
data/lerobot/build_t5_cache_offline.py   offline T5 text-embedding cache
data/lerobot/build_vae_cache_offline.py  offline WAN-VAE latent cache
data/lerobot/build_vlm_cache_offline.py  offline Qwen3-VL hidden-state cache
data/lerobot/_cache_decode.py            shared fast episode frame decode
train/train.py                           WM loss (video + state), focal_fm
train/sample.py                          joint video + state rollout
train/eval_metrics.py                    state-prediction metrics
scripts/wm_eval.py                       WM eval CLI
configs/_wm_aloha_towel.yaml             reference world-model config
```
