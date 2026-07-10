<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <h1>Motus-WM — Motus with Future-State Prediction</h1>
</div>

<div align="center">
  <em>A fork of <a href="https://github.com/thu-ml/Motus">Motus</a> that adds prediction of the
  robot's <strong>future proprioceptive state</strong>, turning it into a full
  <strong>actions-in → future-video + future-state</strong> world model.</em>
</div>

<br>

> **This is a fork.** For the original Motus project — installation, checkpoints, data
> format, the base Mixture-of-Transformers architecture, and citation — see
> **[README_MOTUS.md](README_MOTUS.md)**. Everything below documents the world-model
> changes layered on top.

## Table of Contents
- [What this repo does](#what-this-repo-does)
  - [What goes in and out](#what-goes-in-and-out)
- [What we changed](#what-we-changed)
  - [New modules](#new-modules)
  - [Wiring, data, and loss](#wiring-data-and-loss)
- [Data: download and preprocess](#data-download-and-preprocess)
- [Training, sampling, and eval](#training-sampling-and-eval)
- [Faster training: offline frozen-encoder caches](#faster-training-offline-frozen-encoder-caches)
- [Configuration reference](#configuration-reference)
- [File map](#file-map)

## What this repo does

Motus is a **unified latent action world model** (see [README_MOTUS.md](README_MOTUS.md)): a
Mixture-of-Transformers over three experts — understanding, action, and video generation — with a
UniDiffuser-style scheduler that switches between modeling modes (world model, VLA, inverse
dynamics, video generation, video-action joint prediction). It predicts **future video** and can
consume actions as a clean input; the robot's proprioceptive **state**, however, enters only as a
single conditioning token and is not predicted into the future.

This fork adds that missing piece: Motus now also predicts the robot's **future joint state** as
part of the future state it rolls out (alongside the video). The model's contract becomes —
*current world + the actions the agent intends to take → future video + future joint states:*

```
                ┌────────────────────────┐
  current image ─►                        ─► future video         (T frames)
  current state ─►       Motus-WM         ─►
  H actions     ─►                        ─► future joint states  (H steps)
                └────────────────────────┘
```

### What goes in and out

Dimensions (defined once, used throughout):

- `B` — batch size
- `H` — prediction horizon (number of future action / state steps)
- `T` — number of predicted video frames
- `S` — state dimension (per-joint proprioception)
- `A` — action dimension
- video tensors are `[B, T, C, h, w]` — WAN VAE latents (`C` channels, `h × w` latent grid)

**Base Motus (VLA / action-generation mode)**
```
inputs  (clean)    : current image(s), current state [B, 1, S], instruction (T5)
outputs (denoised) : actions [B, H, A]  +  future video [B, T, C, h, w]
```

**Motus-WM (this fork)**
```
inputs  (clean)    : current image(s), current state [B, 1, S], instruction (optional),
                     actions [B, H, A]          ← was an output, now a clean input
outputs (denoised) : future joint states [B, H, S]   ← NEW target (base Motus never predicted this)
                     + future video [B, T, C, h, w]
```

In short: **actions move from output → clean input**, and **future joint states become a new
denoising target** alongside the video (which is unchanged). Enabled by `model.world_model: true`;
with the flag off, the original Motus path is untouched.

## What we changed

### New modules

| Module | File | Role |
|--------|------|------|
| **ActionConditioner** | [models/action_conditioner.py](models/action_conditioner.py) | Encodes the clean action chunk `[B, H, A]` into per-step action tokens `[B, H, dim]`. Never noised. |
| **ActionInjector** | [models/action_injector.py](models/action_injector.py) | Adds the action tokens directly into WAN's video-token stream (additive injection), so the video DiT sees the actions and not only the state stream. |
| **StateExpert** | [models/state_expert.py](models/state_expert.py) | An ActionExpert-shaped expert that consumes current state + noised future states + action tokens and predicts the future-state velocity (flow matching). **This is the new denoising target.** |

All three are additive — instantiated only when `world_model` is on; the base pipeline does not
import them, so the original VLA path is untouched.

### Wiring, data, and loss

- **Model** ([models/motus.py](models/motus.py)) — a `world_model` branch runs the trimodal
  MoT joint-attention with the **state** stream in the slot the action stream used to occupy
  (`process_joint_attention_wm`), plus a dedicated world-model forward and `inference_step_wm`.
- **Data** ([data/dataset.py](data/dataset.py),
  [data/lerobot/lerobot_dataset.py](data/lerobot/lerobot_dataset.py)) — each sample now exposes
  the action chunk `action[t : t+H]` as a **clean input** and the shifted slice
  `state[t+1 : t+1+H]` as the **flow-matching target**.
- **Loss** ([models/motus.py](models/motus.py)) — `L = w_v · L_video + w_s · L_state` (the old
  action loss is dropped). `L_state` comes in **two types**, selected by `model.loss_term`:
    - **`fm`** (default) — plain flow-matching MSE on the state-velocity prediction.
    - **`focal_fm`** — the same squared error, but each timestep is reweighted by the ground-truth
      tracking residual `|state − action|` (mean over joints, `(· + 1e-3)^0.7`, batch-normalized to
      mean ≈ 1), focusing capacity where the state diverges from the commanded action; it
      degenerates to `fm` when residuals are uniform.
- **Warm-start** — the StateExpert can be seeded from a trained ActionExpert checkpoint
  (`warmup_action_expert_ckpt`): blocks, time embedding, registers, and encoders transfer 1:1;
  the output head is left zero-init.

> **Note:** an experimental fused Triton AdaLN-LayerNorm kernel was tried and **removed** — it
> benchmarked faster in isolation but gave ~0% end-to-end speedup. The model uses the original
> eager AdaLN LayerNorm.

## Data: download and preprocess

The reference dataset is the LeRobot **`aloha_static_towel`** set (50 episodes, 14-DoF bimanual
ALOHA, **v2.1** format — it ships `meta/episodes.jsonl` and `meta/tasks.jsonl`, so no metadata
build is needed).

**1. Download** into the path the config expects (`dataset.params.root`):

```bash
huggingface-cli download lerobot/aloha_static_towel \
    --repo-type dataset --local-dir data_lerobot/aloha_static_towel
```

Set `HF_TOKEN` in your environment first (the LeRobot version check requires it).

**2. Normalization stats — already provided.** Per-joint min/max for `aloha_static_towel` (union
of the action + state streams) ship in [data/utils/stat.json](data/utils/stat.json), keyed by
`dataset.params.embodiment_type`. Nothing to run for the reference dataset.

**3. T5 text-embedding cache — required.** The config sets `enable_t5_fallback: false`, so the
loader reads text embeddings from disk rather than running T5 live. Build them once:

```bash
python data/lerobot/build_t5_cache_offline.py \
    --root data_lerobot/aloha_static_towel --wan_path pretrained_models
```

This encodes each unique task string with WAN's T5 encoder into
`data_lerobot/aloha_static_towel/t5_embedding/task_<idx>.pt` and patches `episodes.jsonl` with the
pointer. (Set `enable_t5_fallback: true` in the config to skip this and encode live instead.)

**4. (Optional) VAE / VLM caches** speed up training — see
[Faster training](#faster-training-offline-frozen-encoder-caches).

**5. Train** — see [Training, sampling, and eval](#training-sampling-and-eval).

## Training, sampling, and eval

**Train** (single GPU) on a LeRobot dataset in world-model mode:

```bash
torchrun --nnodes=1 --nproc_per_node=1 --node_rank=0 \
    --master_addr=127.0.0.1 --master_port=29520 \
    train/train.py \
    --deepspeed configs/zero1.json \
    --config configs/_wm_aloha_towel.yaml \
    --run_name wm_aloha_towel_focal \
    --report_to wandb
```

**Evaluate** a checkpoint — runs the WM inference step on held-out samples and writes
state-prediction metrics plus GT-vs-prediction video grids and state plots:

```bash
python scripts/wm_eval.py \
    --ckpt checkpoints_lerobot/<run>/checkpoint_step_<N>/pytorch_model/mp_rank_00_model_states.pt \
    --config configs/_wm_aloha_towel.yaml \
    --num_samples 4 --out_dir eval_outputs/<run>
```

Metric definitions live in [train/eval_metrics.py](train/eval_metrics.py) (per-joint errors in
the robot's native units, un-normalized with the dataset's min/max). Rollout / sampling logic
is in [train/sample.py](train/sample.py).

## Faster training: offline frozen-encoder caches

In world-model training, the frozen **WAN VAE** and **Qwen3-VL** forwards depend only on the
(fixed) input frames — so with `image_aug: false` their outputs are a deterministic function of
`(episode, condition frame)` and can be computed **once, offline** and read from disk every step
instead of recomputed. This is the main training speedup in the fork. (The T5 text cache is a
*required* preprocessing step for the reference config — see
[Data](#data-download-and-preprocess).)

| Cache | Builder | Skips (per step) | Enabled by |
|-------|---------|------------------|------------|
| **VAE** latents | [build_vae_cache_offline.py](data/lerobot/build_vae_cache_offline.py) | frozen WAN VAE encode (~25% of step) | `use_vae_cache: true` |
| **VLM** hidden states | [build_vlm_cache_offline.py](data/lerobot/build_vlm_cache_offline.py) | frozen Qwen3-VL forward (~11% of step) | `use_vlm_cache: true` |

Both read from disk when a key is present and **fall back to live compute** for any missing key —
so they are safe to enable before building, and safe to skip entirely. A shared fast whole-episode
frame decoder ([_cache_decode.py](data/lerobot/_cache_decode.py)) does one sequential pass per
episode instead of re-seeking the mp4 for every frame.

Build them once before training:

```bash
python data/lerobot/build_vae_cache_offline.py --config configs/_wm_aloha_towel.yaml
python data/lerobot/build_vlm_cache_offline.py --config configs/_wm_aloha_towel.yaml
```

## Configuration reference

The reference world-model config is
[configs/_wm_aloha_towel.yaml](configs/_wm_aloha_towel.yaml) (ALOHA static-towel, 14-DoF
bimanual). Key world-model knobs:

| Key | Meaning |
|-----|---------|
| `model.world_model` | Turn world-model (future-state) mode on. |
| `model.loss_term` | `fm` (plain flow-matching MSE) or `focal_fm` (residual-reweighted). |
| `model.warmup_action_expert_ckpt` | Checkpoint to seed the StateExpert from (or `null`). |
| `common.action_dim` / `state_dim` | Action `A` and state `S` dimensionality. |
| `common.num_video_frames` / `video_action_freq_ratio` | `T` = num_video_frames; horizon `H` = `T` × video_action_freq_ratio. |
| `dataset.image_aug` | Must be `false` to use the caches (they assume deterministic inputs). |
| `dataset.params.use_vae_cache` / `use_vlm_cache` | Enable the offline caches. |
| `dataset.params.num_val_episodes` / `val_split_seed` | Deterministic held-out validation split. |

Base model paths (WAN, Qwen3-VL), checkpoints, and install steps are unchanged from upstream —
see [README_MOTUS.md](README_MOTUS.md).

## File map

World-model additions on top of Motus:

```
models/action_conditioner.py             clean action chunk → action tokens
models/action_injector.py                inject action tokens into WAN video stream
models/state_expert.py                   future-state denoising expert
models/motus.py                          world_model forward + inference_step_wm (wiring)
data/dataset.py                          WM sample assembly (clean actions, state target)
data/lerobot/lerobot_dataset.py          LeRobot loader + cache lookups
data/lerobot/build_t5_cache_offline.py   offline T5 text-embedding cache
data/lerobot/build_vae_cache_offline.py  offline WAN-VAE latent cache
data/lerobot/build_vlm_cache_offline.py  offline Qwen3-VL hidden-state cache
data/lerobot/_cache_decode.py            shared fast episode frame decode
train/train.py                           WM loss (video + state), focal_fm, param groups
train/sample.py                          joint video + state rollout
train/eval_metrics.py                    state-prediction metrics
scripts/wm_eval.py                       WM eval CLI (metrics + GT-vs-pred media)
configs/_wm_aloha_towel.yaml             reference world-model config
```

---

For the original Motus documentation, see **[README_MOTUS.md](README_MOTUS.md)**.
