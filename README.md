# minimax-h3-mlx

MLX (Apple Silicon) port of [**MiniMaxAI/MiniMax-H3**](https://huggingface.co/MiniMaxAI/MiniMax-H3) —
MiniMax's omni-modal generative system for synchronized **video + audio** generation.

> Powered by MiniMax H3.

H3 is **not** a language model. It is a diffusers pipeline: a 33B diffusion transformer denoising
video and audio latents jointly, conditioned by a frozen Qwen3-VL-32B encoder, with separate video
and audio VAEs. There is no autoregressive decoding and no `mlx_lm.convert` path — this repository
is a from-scratch MLX implementation of the pipeline.

## Architecture

| Component | Class | Size | Notes |
|---|---|---|---|
| `transformer` | `MiniMaxH3DiTModel` | 33B / 66.3 GB | 50 blocks, hidden 5376, 56x128 heads (inner dim 7168 > hidden), SwiGLU ffn 14336, 3D MM-RoPE |
| `text_encoder` | Qwen3-VL-32B | 66.7 GB | frozen conditioner; H3 reads the **unnormalized** hidden state after layer 50 of 64 |
| `video_vae` | `MiniMaxH3VideoVAE` | 10.4 GB | ViT+CNN KL VAE, 16x spatial / 4x temporal, 24 latent channels, tiled |
| `audio_vae` | `MiniMaxH3AudioVAE` | 0.6 GB | DAC/BigVGAN stereo 32 kHz, 32 latent channels, 40 Hz latent rate |

Everything runs over **one packed 1-D sequence** holding every modality at once:

```
[ text (L) | keyframe conditions (C) | target audio (A) | target video (V) ]
```

Attention is full self-attention over that sequence — no cross-attention, no per-modality block
weights. Modality-specific behaviour comes only from the two input patch projections, a per-row
AdaLN modality tag, and the two output heads.

### The two checkpoints are the same weights

`FL2VA/` and `Ref2VA/` are **byte-identical except for `model_index.json`** — all 80 weight and
code files share LFS hashes. The 288 GB repository is 144 GB of unique weights published twice;
only the pipeline metadata (`partition`, `tasks`) differs. One conversion covers both tasks.

### AdaLN precompute: 13B of the 33B need not be resident

13B parameters live in the per-block `adaln_proj.linear` projections (50 x `[96768, 2688]`). Their
only input is the timestep embedding — nothing sequence-dependent — so for a fixed sampler schedule
every modulation tensor a run will ever need can be computed once up front and the projections then
dropped. `ModulationCache` builds it and `drop_adaln_weights` frees the originals; the cache is
verified bit-exact against the live projection.

Measured on the real checkpoint, a 40-step run (video and audio use different sigma shifts, 12.0 and
3.0, so their schedules only partly coincide — 77 distinct timesteps in all, not 40):

| | params | resident |
|---|---:|---:|
| DiT as shipped | 33.12B | 66.3 GB |
| `adaln_proj` dropped | -13.01B | -26.0 GB |
| **after** | **20.11B** | **40.3 GB + 745 MB cache** |

A **25.3 GB net saving**, with the cache 35x smaller than the weights it replaces and built in 0.7 s.

The table is built from the float32 timestep MLP through the *unquantized* projections. Every block
reads the same `temb`, so an error there biases all 50 blocks identically at every step and
accumulates coherently along the trajectory — building it before quantization keeps that path exact.


## Performance: read this before converting anything

MiniMax has **not** released its sparse-attention implementation ("the initial open-source release
provides inference with full attention only"), so a run does dense attention over tens of thousands
of rows. Measured on an **M3 Ultra (550 GB unified memory)**, bfloat16, one transformer block timed
and multiplied by 50 (the blocks are identical):

| Request | Packed rows | Per block | **Per denoising step** | Peak activations |
|---|---:|---:|---:|---:|
| 5 s, 1344x768 | 37,966 | 10.5 s | **8.8 min** | 9.3 GB |
| 15 s, 1344x768 | 109,318 | 74.9 s | **1.04 h** | 24.4 GB |

Per-step cost is the measured, assumption-free number. The released weights are **CFG-distilled**
("guidance baked into the weights, so there is no guider, no `negative_prompt` and no
`guidance_scale`"), so a step is one forward, not two — but MiniMax does not publish a recommended
step count, and the reference marks `num_inference_steps` required rather than defaulting it. Total
wall-clock therefore scales directly:

| Steps | 5 s clip | 15 s clip |
|---:|---:|---:|
| 8 | 1.2 h | 8.3 h |
| 16 | 2.3 h | 16.6 h |
| 50 (generic diffusers default) | 7.3 h | 52 h |

Peak memory is modest — MLX's attention is flash-style and never materializes the score matrix — so
**memory is not the constraint. Compute is.** 5 s is the shortest clip H3 supports and 15 s at 2K is
its flagship capability; 2K is out of reach locally.

This also changes what quantization buys. The bottleneck is attention FLOPs, which quantization
does not reduce. At 5 s the linear layers are ~42% of the work, at 15 s ~20%, so a 4-bit DiT is
worth roughly 1.2-1.4x end-to-end — useful for *fitting* the model on a smaller Mac, not for making
generation quick.

## Status

| Piece | State |
|---|---|
| DiT (`MiniMaxH3DiT`) | **done** — matches diffusers reference to 4.8e-07 |
| Video VAE | **done** — encode + decode match to 1.2e-06, tiled and untiled |
| AdaLN precompute + drop | **done** — bit-exact; verified on the real 33B checkpoint |
| Scheduler | **done** — bit-exact sigmas, timesteps and 16-step trajectory |
| Packed-sequence geometry | **done** — bit-exact `(t, h, w)` grid, tags, indices |
| Checkpoint loader | **done** — real 66.3 GB checkpoint loads with zero key mismatches |
| Audio VAE | not started |
| Text encoder wiring | not started (mlx-vlm has `qwen3_vl`) |
| Pipeline / denoise loop | not started |
| Quant set | not started |

The loader was run against the released `FL2VA/transformer`: 33.12B parameters over 534 tensors,
every key matched, and the mixed-precision split survives intact — 12 float32 tensors (the two patch
projections, the timestep MLP and the two output heads) against 522 bfloat16.

### Validation

Parity is checked against the `minimax-h3` branch of diffusers, not against a re-reading of it. The
MLX model is the source of truth and its parameters are pushed through the **official** conversion
script (`reorder_interleaved_qkv` + `convert_transformer_key`) into the reference module, so the
test exercises the two raw-checkpoint layout quirks the port handles by reshape rather than
assuming them:

* `attn.qkv_proj` rows are **per-head interleaved** — `[h0: q,k,v][h1: q,k,v]...` — so the
  projection output reshapes to `(..., heads, 3, head_dim)`.
* `mlp.fc1` is a fused **`[gate; value]`** SwiGLU projection; the reference computes
  `fc2(silu(gate) * value)`.

Both mean the released checkpoint loads **1:1 with no weight surgery**.

The video VAE is checked the same way, through `convert_video_vae_key`, on the reference's own tiny
CPU-parity config. Its `attn.to_qkv` is interleaved and its `ff.w1` fused exactly like the DiT's.

```bash
./.venv/bin/python tests/test_dit_parity.py        # 4.8e-07 vs reference
./.venv/bin/python tests/test_video_vae_parity.py  # 1.2e-06, tiled + untiled
./.venv/bin/python tests/test_packing_parity.py    # 81 checks, all bit-exact
python3 tests/test_dit_smoke.py                    # no torch needed
```

Two places needed care to stay bit-exact, both because a one-ulp difference is observable:

* **`linspace`** — ATen takes a float32 step, splits the range at the halfway point, and evaluates
  `start + step*i` with an FMA. The sigma grid is collapsed by a consecutive-duplicate check, so an
  ulp can change how many sigmas survive and therefore the *number of model evaluations*.
* **Scheduler scalars** — the reference does its arithmetic in float32 tensors. Computing the same
  expressions in Python floats rounds twice and drifts by an ulp per step.

The packed grid is built in NumPy float64 (as the reference does) because video and audio share one
40-units-per-second rotary clock, and that shared clock *is* the audio/video alignment. The
reference notes its temporal span must be summed pairwise, since sequential summation differs in
the last ulp from 16 latent frames onwards.

## Layout

```
minimax_h3_mlx/
  config.py      DiTConfig / PipelineConfig, original checkpoint field names
  dit.py         the 33B diffusion transformer
  adaln.py       ModulationCache, drop_adaln_weights
  scheduler.py   rectified-flow Euler with exponential sigma shift
  packing.py     packed-sequence geometry, patchify/unpatchify, row timesteps
  load.py        checkpoint loading, mixed fp32/bf16 split preserved
  video_vae.py   causal 3D CNN encoder + 36-layer ViT decoder, tiled
reference/       upstream sources, for validation only
scripts/         bench_dit.py
tests/           parity + smoke tests
```

## License

The port is Apache-2.0. The **weights** are governed by the
[MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE),
which is not an open-source licence: redistribution must carry a copy of the agreement, mark
modified files, and display "Powered by MiniMax H3"; commercial use above $20M yearly revenue needs
separate authorization; and the grant is **territorially limited** (worldwide excluding the
Excluded Territories). Any republished weights inherit these terms.
