"""Scale-swept DiT parity check: does MLX vs. diffusers-reference divergence grow with sequence
length?

Motivation: the existing test_dit_parity.py verifies correctness at n_text, n_video, n_audio =
5, 9, 3 (17 total rows). The real smoke test's packed sequence is 37,741 rows. This script holds
the *attention regime* fixed at real dimensions (hidden_size, num_attention_heads,
attention_head_dim -- the values that actually govern softmax/RoPE precision behavior) but reduces
depth to 2 layers (matching test_dit_parity.py's own tractability choice) rather than the real
model's 50, and reduces to a single seed per scale rather than 3 -- both deliberate scope cuts to
keep this tractable given a single forward at 37,741 rows with real head-count/head_dim at 50
layers was estimated at ~1 PFLOP, likely hours on CPU for the torch reference side alone. 2 layers
keeps the largest scale in the range of minutes.

Model weights are randomly initialized (not the real checkpoint) -- this test is about whether
attention/RoPE numerics diverge as a function of sequence length at real dimensional scale, not
about the real checkpoint's specific values (those are already covered by test_dit_parity.py and
the real-weights video VAE comparison from the prior session).

    ./.venv/bin/python tests/test_dit_parity_sweep.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from mlx.utils import tree_flatten

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reference" / "diffusers"))

from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel

from convert_minimax_h3_to_diffusers import convert_transformer_key, reorder_interleaved_qkv
from minimax_h3_mlx.config import TAG_AUDIO, TAG_TEXT, TAG_VIDEO, DiTConfig
from minimax_h3_mlx.dit import MiniMaxH3DiT

LOG_PATH = "/Users/ultra/dit-parity-sweep.jsonl"
SCALES = [17, 100, 1000, 5000, 15000, 37741]
# Real proportions observed in the actual smoke test's packed sequence (31 text / 37296 video / 414 audio).
VIDEO_FRAC, AUDIO_FRAC = 37296 / 37741, 414 / 37741


def real_dim_config(num_layers: int = 2) -> DiTConfig:
    """DiTConfig() defaults are already the real model's dimensions; only depth is cut for
    tractability (see module docstring)."""
    cfg = DiTConfig()
    cfg.num_layers = num_layers
    cfg.token_refiner_num_layers = 2
    return cfg


def as_diffusers_config(cfg: DiTConfig) -> dict:
    return {
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_layers,
        "num_refiner_layers": cfg.token_refiner_num_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "attention_head_dim": cfg.attention_head_dim,
        "ffn_dim": cfg.ffn_hidden_size,
        "in_channels": cfg.latents_dim,
        "audio_in_channels": cfg.audio_latents_dim,
        "patch_size": tuple(cfg.patch_size),
        "text_dim": cfg.text_dim,
        "freq_dim": cfg.timestep_input_dim,
        "time_embed_hidden_dim": cfg.time_embed_hidden_size,
        "time_embed_dim": cfg.time_embed_dim,
        "rope_freq_dim": cfg.rope_inv_freq_len,
        "rope_theta": cfg.rope_theta,
        "norm_eps": cfg.norm_eps,
        "qk_norm_eps": cfg.qk_norm_eps,
        "final_norm_eps": cfg.final_norm_eps,
    }


def mlx_params_as_checkpoint(dit: MiniMaxH3DiT) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for key, value in tree_flatten(dit.parameters()):
        state[key] = torch.from_numpy(np.array(value.astype(mx.float32)))
    return state


def to_diffusers_state_dict(state: dict[str, torch.Tensor], cfg: DiTConfig) -> dict[str, torch.Tensor]:
    dcfg = as_diffusers_config(cfg)
    out: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        if key.endswith(".attn.qkv_proj.weight"):
            tensor = reorder_interleaved_qkv(tensor, dcfg["num_attention_heads"], dcfg["attention_head_dim"])
        for new_key, new_tensor in convert_transformer_key(key, tensor, dcfg):
            out[new_key] = new_tensor
    return out


def build_layout(n_text: int, n_video: int, n_audio: int):
    seq = n_text + n_video + n_audio
    text_i = np.arange(n_text)
    video_i = np.arange(n_text, n_text + n_video)
    audio_i = np.arange(n_text + n_video, seq)
    tags = np.concatenate(
        [np.full(n_text, TAG_TEXT), np.full(n_video, TAG_VIDEO), np.full(n_audio, TAG_AUDIO)]
    ).astype(np.int64)
    ts_i = np.concatenate([np.zeros(n_text), np.ones(n_video), np.full(n_audio, 2)]).astype(np.int64)
    pos = np.stack([np.arange(seq) % 3, np.arange(seq) % 5, np.arange(seq) % 7], axis=-1).astype(np.float64)
    return text_i, video_i, audio_i, tags, ts_i, pos


def split_counts(total: int) -> tuple[int, int, int]:
    n_video = max(1, round(total * VIDEO_FRAC))
    n_audio = max(1, round(total * AUDIO_FRAC))
    n_text = max(1, total - n_video - n_audio)
    return n_text, n_video, n_audio


def run_one(dit, ref, cfg, total: int, seed: int) -> dict:
    n_text, n_video, n_audio = split_counts(total)
    actual_total = n_text + n_video + n_audio
    text_i, video_i, audio_i, tags, ts_i, pos = build_layout(n_text, n_video, n_audio)

    rng = np.random.default_rng(seed)
    video = rng.standard_normal((1, n_video, cfg.video_patch_dim)).astype(np.float32)
    audio = rng.standard_normal((1, n_audio, cfg.audio_latents_dim)).astype(np.float32)
    text = rng.standard_normal((1, n_text, cfg.text_dim)).astype(np.float32)
    timestep = np.array([0.0, 0.35, 0.9], dtype=np.float32)

    record = {
        "requested_scale": total, "actual_scale": actual_total,
        "n_text": n_text, "n_video": n_video, "n_audio": n_audio, "seed": seed,
    }

    ref_ok = True
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            ref_v, ref_a = ref(
                hidden_states=torch.from_numpy(video),
                audio_hidden_states=torch.from_numpy(audio),
                encoder_hidden_states=torch.from_numpy(text),
                timestep=torch.from_numpy(timestep),
                timestep_indices=torch.from_numpy(ts_i),
                token_tags=torch.from_numpy(tags),
                position_ids=torch.from_numpy(pos),
                video_indices=torch.from_numpy(video_i),
                audio_indices=torch.from_numpy(audio_i),
                text_indices=torch.from_numpy(text_i),
                return_dict=False,
            )
        record["ref_seconds"] = time.perf_counter() - t0
        record["ref_has_nan"] = bool(torch.isnan(ref_v).any() or torch.isnan(ref_a).any())
        record["ref_has_inf"] = bool(torch.isinf(ref_v).any() or torch.isinf(ref_a).any())
    except (RuntimeError, MemoryError) as exc:
        ref_ok = False
        record["ref_error"] = f"{type(exc).__name__}: {exc}"
        record["ref_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    got_v, got_a = dit(
        mx.array(video), mx.array(audio), mx.array(text), mx.array(timestep),
        mx.array(ts_i.astype(np.int32)), mx.array(tags.astype(np.int32)), mx.array(pos.astype(np.float32)),
        mx.array(video_i.astype(np.int32)), mx.array(audio_i.astype(np.int32)), mx.array(text_i.astype(np.int32)),
    )
    mx.eval(got_v, got_a)
    record["mlx_seconds"] = time.perf_counter() - t0
    got_v_np, got_a_np = np.array(got_v), np.array(got_a)
    record["mlx_has_nan"] = bool(np.isnan(got_v_np).any() or np.isnan(got_a_np).any())
    record["mlx_has_inf"] = bool(np.isinf(got_v_np).any() or np.isinf(got_a_np).any())

    if not ref_ok:
        record["comparison"] = "skipped (reference errored)"
        return record

    ref_v_np, ref_a_np = ref_v.numpy(), ref_a.numpy()
    dv, da = got_v_np - ref_v_np, got_a_np - ref_a_np
    max_abs = float(max(np.abs(dv).max(), np.abs(da).max()))
    mean_abs = float((np.abs(dv).mean() + np.abs(da).mean()) / 2)
    l2_rel = float(
        (np.linalg.norm(dv) + np.linalg.norm(da))
        / max(np.linalg.norm(ref_v_np) + np.linalg.norm(ref_a_np), 1e-12)
    )
    record.update({
        "max_abs_diff": max_abs, "mean_abs_diff": mean_abs, "l2_rel_norm": l2_rel,
        "ref_scale": float(max(np.abs(ref_v_np).max(), np.abs(ref_a_np).max())),
        "comparison": "ok",
    })
    return record


def main() -> int:
    cfg = real_dim_config(num_layers=2)
    mx.random.seed(0)
    torch.manual_seed(0)

    print(f"Real-dimension config: hidden_size={cfg.hidden_size}, heads={cfg.num_attention_heads}, "
          f"head_dim={cfg.attention_head_dim}, layers={cfg.num_layers} (reduced from 50 for tractability)")

    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    state = mlx_params_as_checkpoint(dit)
    ref = MiniMaxH3Transformer3DModel(**as_diffusers_config(cfg)).eval()
    converted = to_diffusers_state_dict(state, cfg)
    missing, unexpected = ref.load_state_dict(converted, strict=False)
    missing = [k for k in missing if "rope.inv_freq" not in k]
    if missing or unexpected:
        print(f"FAIL: state dict mismatch\n  missing={missing[:8]}\n  unexpected={unexpected[:8]}")
        return 1
    print(f"state dict mapped: {len(converted)} tensors, no missing/unexpected keys\n")

    Path(LOG_PATH).write_text("")
    results = []
    for scale in SCALES:
        print(f"=== scale {scale} ===")
        record = run_one(dit, ref, cfg, scale, seed=0)
        results.append(record)
        with open(LOG_PATH, "a") as fh:
            fh.write(json.dumps(record) + "\n")
        if record.get("comparison") == "ok":
            print(f"  actual_rows={record['actual_scale']}  max_abs={record['max_abs_diff']:.3e}  "
                  f"mean_abs={record['mean_abs_diff']:.3e}  l2_rel={record['l2_rel_norm']:.3e}  "
                  f"ref={record['ref_seconds']:.1f}s  mlx={record['mlx_seconds']:.1f}s  "
                  f"nan(ref={record['ref_has_nan']},mlx={record['mlx_has_nan']})  "
                  f"inf(ref={record['ref_has_inf']},mlx={record['mlx_has_inf']})")
        else:
            print(f"  SKIPPED: {record.get('ref_error', record.get('comparison'))}")

    print(f"\nWrote {LOG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
