# Motus → Pure World Model: Transformation Plan

## 1. What we're building

A world model with this interface:

```
                ┌────────────────────────┐
  current image ─►                       ─► future video  (H frames)
  current state ─►   Motus-WM            ─► future states (H joint-state vectors)
  H actions     ─►                       ─►
                └────────────────────────┘
```

Inputs: the current world (image + proprio state) + what the agent intends to do (H actions).
Outputs: what happens next (H images + H states).

This is a strict role-swap of today's Motus.

## 2. What Motus does today (for contrast)

```
  current image ─►                       ─► future video   (H frames)
  current state ─►   Motus                ─► H actions
  instruction   ─►                       ─►
```

Today: actions are an output. State enters as a single conditioning token (`[B, 1, state_dim]`) and is **not** predicted into the future. Video is already a first-class output (WAN flow-matching).

## 3. The role swap (mental model)

| Quantity | Today | World-Model |
|:---|:---|:---|
| Current image | input (conditioning) | input (conditioning) |
| Current state | input (conditioning, 1 token) | input (conditioning, 1 token) |
| Instruction | input (conditioning, optional) | input (conditioning, optional — keep only if data has it) |
| **Actions (H)** | **output** | **input (clean conditioning)** |
| **Future states (H)** | not represented | **output** |
| Future video (H frames) | output | output |

Under the hood, the swap is: the tensor that was a denoising target becomes a clean conditioning input, and the tensor that didn't exist before becomes the new denoising target. From the outside, it's just: actions in, states + video out.

## 4. Architectural changes

Three components on the Motus side ([Motus/models/](Motus/models/)):

### (a) New: Action Conditioner (trained from scratch)
- Small encoder: `Linear(action_dim → dim) + sinusoidal pos-embed` (optionally a couple of MLP/attn layers).
- Output: `[B, H, dim]` clean "action tokens", one per future timestep.
- Used as conditioning by both the state head and the video DiT. **Never noised.**
- Analogous to nano-world-model's `ActionEmbedder` ([nano-world-model/src/models/nanowm.py:247](nano-world-model/src/models/nanowm.py#L247)).

### (b) New: State Expert (trained from scratch)
- Architectural clone of [Motus/models/action_expert.py](Motus/models/action_expert.py).
- **Input encoder** consumes:
  - `current_state` `[B, 1, state_dim]` — clean
  - `noised future_states` `[B, H, state_dim]` — denoising target at training, sampled at inference
  - `action_tokens` `[B, H, dim]` — clean conditioning from the Action Conditioner
- **Body**: DiT blocks with the existing trimodal joint-attention into WAN's blocks ([Motus/models/motus.py:218-269](Motus/models/motus.py#L218-L269)). The attention mechanism is target-agnostic; it carries over.
- **Output decoder**: MLP `dim → state_dim` → `[B, H, state_dim]` predicted future states.
- Loss: flow-matching MSE against ground-truth `state[t+1 : t+1+H]`, normalized by the cached `state_mean/state_std`.

### (c) Modified: WAN Video DiT (mostly frozen, with new action-injection path)
WAN today sees actions only indirectly, through joint-attention with the action-denoising stream. When we kill that stream, **WAN goes action-blind** unless we plumb actions in directly. Fix: add an action-injection module to each WAN block.

Options (ranked by capacity / cost):
1. **Additive** — add action-token embeddings to matching video-token positions. Zero extra params. Nano-WM's default and PushT-ablation winner.
2. **FiLM / adaLN-style** — predict `(scale, shift)` from action embedding, modulate `norm1`. ~+10–20% params. RT-1-ablation winner in nano-WM.
3. Cross-attention — heaviest, weakest in ablations. Skip.

**Recommended start:** additive. Upgrade to FiLM if video quality stalls.

## 5. What trains vs. what's frozen

| Component | Status |
|:---|:---|
| WAN VAE | frozen |
| T5 encoder | frozen |
| WAN video DiT (base weights) | frozen initially; add **LoRA (rank ~16)** after a warmup phase if action conditioning isn't taking |
| `und_expert` (T5/VLM adapter) | keep frozen if instruction is used; **remove entirely** if data is joint-only |
| Action Conditioner | **from scratch** |
| State Expert (encoder + DiT blocks + decoder) | **from scratch** |
| Action-injection module(s) into WAN | **from scratch** |

The existing param-group machinery in [Motus/train/train.py:478-497](Motus/train/train.py#L478-L497) handles this — toggle `requires_grad`, AdamW picks up what's trainable.

## 6. Loss

```
L_video = flow_matching_mse(pred_video_velocity,  gt_video_velocity)
L_state = flow_matching_mse(pred_state_velocity,  gt_state_velocity)
L_total = w_v * L_video + w_s * L_state
```

- `L_action` is **deleted**.
- Start with `w_v = w_s = 1.0`, then rescale `w_s` after ~1k steps so the two losses are within ~5× in magnitude (flow-matching MSE on a 14-D state and a 16384-D latent are not comparable out of the box).

## 7. Forward pass (training step, conceptual)

```python
inputs:
    video_latents_noised   [B, T, C, h, w]   # noised at τ_v
    future_states_noised   [B, H, S]         # noised at τ_s
    action_chunk           [B, H, A]         # CLEAN
    current_state          [B, 1, S]         # CLEAN
    current_image_context  [B, T_ctx, ...]   # CLEAN
    t5_context             [B, L_t, D]       # CLEAN, optional

forward:
    action_tokens = action_conditioner(action_chunk)                       # [B, H, dim]
    state_seq     = state_encoder(current_state, future_states_noised,
                                  action_tokens)                            # [B, 1+H, dim]

    for layer in WAN.blocks:
        video_tokens = inject_action(video_tokens, action_tokens)           # additive / FiLM
        video_tokens = layer.self_attn(video_tokens)
        video_tokens, state_seq = joint_attention(video_tokens, state_seq)  # existing trimodal slot
        video_tokens = layer.ffn(video_tokens)

    pred_video  = WAN.head(video_tokens)
    pred_states = state_decoder(state_seq[:, 1:])     # drop the current-state slot

losses:
    L_video = flow_matching_mse(pred_video,  gt_video_velocity)
    L_state = flow_matching_mse(pred_states, gt_state_velocity)
```

## 8. Data plumbing

Nothing new to download. `LeRobotDataSource` ([nano-world-model/src/wm_datasets/data_source/lerobot/lerobot_data_source.py:174](nano-world-model/src/wm_datasets/data_source/lerobot/lerobot_data_source.py#L174)) already yields per-trajectory `states` and `actions`. Each sample needs:

| Field | Source | Used as |
|:---|:---|:---|
| `image[t]` (or short context) | existing | conditioning |
| `state[t]` | existing | conditioning (1 token) |
| `action[t : t+H]` | existing (was target) | clean input |
| `state[t+1 : t+1+H]` | **new slice** (shift +1) | flow-matching target |
| `image[t+1 : t+1+H]` | existing | flow-matching target |

Only the collate / training step changes. No new dataset, no new loader.

## 9. Open design choices (decide before coding)

1. **Diffusion timestep coupling:** same `τ` for video and state (simpler, faster), or independent `τ` per stream (diffusion-forcing — enables partial-information rollouts like "clean states, predict video"). **Recommended start: same τ.**
2. **Horizon `H`:** fixed (like current Motus / nano-WM) or variable. **Recommended: fixed, match current Motus.**
3. **Instruction path:** keep T5+und_expert (RT-1, prompted RoboTwin tasks) or strip it (joint-only datasets). **Decide per dataset.**
4. **WAN LoRA:** rank, when to unfreeze. **Recommended: rank 16, unfreeze after ~5k from-scratch warmup steps.**

## 10. Inference / rollout

1. Sample `noise_v ~ N(0,I)` and `noise_s ~ N(0,I)`.
2. Provide clean `current_state`, `current_image`, **and the action chunk to roll out**.
3. Run `FlowUniPCMultistepScheduler` jointly over both streams.
4. Decode video latents via WAN VAE; denormalize predicted states with cached `state_mean/state_std`.

Long-horizon rollout (beyond `H`): autoregressive. Feed predicted `state[t+H]` and decoded `image[t+H]` back in, plug in the next action chunk.

## 11. Sanity ladder (in order)

1. **Shapes end-to-end** on a 2-sample batch. Decoder out must be `[B, H, state_dim]`; WAN head output unchanged.
2. **Overfit one trajectory** — state loss → ~0, video reconstruction sharp. Few hundred steps. Catches every wiring bug.
3. **Action conditioning sanity** — train briefly, then at inference zero out the action chunk. Predicted video AND predicted states must visibly differ from the action-conditioned predictions. If they don't, the injection path is dead.
4. **Scale up** to a few thousand episodes once 1–3 pass.

## 12. Naming

TBD — placeholders used here: "Action Conditioner", "State Expert", "Action Injector". Wiring is the load-bearing part; renaming is mechanical and we'll settle it later.

---

**TL;DR:** Strip the action-denoising path. Add a clean action input that feeds (i) the existing WAN video DiT via additive/FiLM injection and (ii) a new state-prediction head built on the existing joint-attention scaffold. Train the new pieces from scratch, freeze WAN base (LoRA later if needed). Keep video prediction as it is. The model's external contract becomes the standard world-model one: actions in, next state + next image out.
