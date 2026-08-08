"""Closes the one untested combination identified by the scale sweep: DiT run with the *real*
packed-sequence row layout (real build_packed_sequence output, real dimensions), not a simplified
contiguous synthetic one.

Two checks:
  1. Extends test_packing_parity.py's coverage to the exact real smoke-test dimensions (text=31,
     37 latent frames, 48x84 latent grid, 207 audio latents, no keyframes) -- verifies MLX's
     build_packed_sequence matches the diffusers reference at this specific real scale, which no
     existing test does (existing packing parity tests use smaller synthetic canvases).
  2. Feeds that real layout's actual position_ids/token_tags/video_indices/audio_indices/
     text_indices into both MLX and reference DiT (real attention dimensions, reduced depth for
     tractability -- see test_dit_parity_sweep.py for why) and compares outputs, exactly as the
     sweep did but with the real row ordering and real position-id distribution instead of a
     synthetic mod-3/mod-5/mod-7 scheme.

Real dimensions match the actual broken smoke test's log line:
"canvas 1344x768, 124 frames (37 latent), 207 audio latents" /
"packed sequence: 37,741 rows (31 text, 0 condition)".

    ./.venv/bin/python tests/test_dit_real_layout_parity.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
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
from minimax_h3_mlx.config import TAG_TEXT, DiTConfig
from minimax_h3_mlx.dit import MiniMaxH3DiT
from minimax_h3_mlx.packing import build_packed_sequence, build_row_timesteps

LOG_PATH = "/Users/ultra/dit-real-layout-parity.jsonl"

# Real dimensions from the actual broken smoke test.
NUM_TEXT = 31
NUM_LATENT_FRAMES = 37
LATENT_HEIGHT = 48
LATENT_WIDTH = 84
NUM_AUDIO_LATENTS = 207
PATCH_SIZE = (1, 2, 2)


def _load_vendored_ref_packing():
    """Same shim as the prior session: the installed minimax-h3-lora branch is missing
    packing.py outright, so load the vendored copy directly and register it."""
    vendored = ROOT / "reference" / "diffusers" / "modular" / "packing.py"
    src = vendored.read_text().replace(
        "from ...utils.torch_utils import randn_tensor",
        "from diffusers.utils.torch_utils import randn_tensor",
    )
    patched_path = "/tmp/_patched_packing_real_layout.py"
    Path(patched_path).write_text(src)
    spec = importlib.util.spec_from_file_location("diffusers.modular_pipelines.minimax_h3.packing", patched_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["diffusers.modular_pipelines.minimax_h3.packing"] = module
    spec.loader.exec_module(module)
    return module


def check_packing_parity_at_real_scale(ref_packing) -> bool:
    print("=== 1. Real-scale packing parity (extends test_packing_parity.py's coverage) ===")
    text_tags_np = np.full(NUM_TEXT, TAG_TEXT, dtype=np.int64)

    mine = build_packed_sequence(
        text_tags_np, NUM_LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH, NUM_AUDIO_LATENTS, PATCH_SIZE, ()
    )
    ref = ref_packing.build_packed_sequence(
        torch.from_numpy(text_tags_np), NUM_LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH, NUM_AUDIO_LATENTS,
        PATCH_SIZE, (),
    )

    ok = True
    if mine.sequence_length != ref.sequence_length:
        print(f"  FAIL seq_len: {mine.sequence_length} vs {ref.sequence_length}")
        ok = False
    else:
        print(f"  ok  seq_len: {mine.sequence_length} (expected 37,741: {mine.sequence_length == 37741})")

    checks = [
        ("position_ids", np.array(mine.position_ids), ref.position_ids.numpy()),
        ("token_tags", np.array(mine.token_tags), ref.token_tags.numpy()),
        ("video_indices", np.array(mine.video_indices), ref.video_indices.numpy()),
        ("audio_indices", np.array(mine.audio_indices), ref.audio_indices.numpy()),
        ("text_indices", np.array(mine.text_indices), ref.text_indices.numpy()),
    ]
    for name, a, b in checks:
        if a.shape != b.shape:
            print(f"  FAIL {name} shape: {a.shape} vs {b.shape}")
            ok = False
            continue
        delta = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
        status = "ok  " if delta == 0 else "FAIL"
        print(f"  {status} {name}: max delta {delta:.3e}")
        ok = ok and (delta == 0)

    print(f"  num_condition_video_rows: {mine.num_condition_video_rows} vs {ref.num_condition_video_rows}")
    print()
    return ok, mine


def as_diffusers_config(cfg: DiTConfig) -> dict:
    return {
        "hidden_size": cfg.hidden_size, "num_layers": cfg.num_layers,
        "num_refiner_layers": cfg.token_refiner_num_layers, "num_attention_heads": cfg.num_attention_heads,
        "attention_head_dim": cfg.attention_head_dim, "ffn_dim": cfg.ffn_hidden_size,
        "in_channels": cfg.latents_dim, "audio_in_channels": cfg.audio_latents_dim,
        "patch_size": tuple(cfg.patch_size), "text_dim": cfg.text_dim, "freq_dim": cfg.timestep_input_dim,
        "time_embed_hidden_dim": cfg.time_embed_hidden_size, "time_embed_dim": cfg.time_embed_dim,
        "rope_freq_dim": cfg.rope_inv_freq_len, "rope_theta": cfg.rope_theta, "norm_eps": cfg.norm_eps,
        "qk_norm_eps": cfg.qk_norm_eps, "final_norm_eps": cfg.final_norm_eps,
    }


def mlx_params_as_checkpoint(dit: MiniMaxH3DiT) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(np.array(v.astype(mx.float32))) for k, v in tree_flatten(dit.parameters())}


def to_diffusers_state_dict(state: dict[str, torch.Tensor], cfg: DiTConfig) -> dict[str, torch.Tensor]:
    dcfg = as_diffusers_config(cfg)
    out: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        if key.endswith(".attn.qkv_proj.weight"):
            tensor = reorder_interleaved_qkv(tensor, dcfg["num_attention_heads"], dcfg["attention_head_dim"])
        for new_key, new_tensor in convert_transformer_key(key, tensor, dcfg):
            out[new_key] = new_tensor
    return out


def main() -> int:
    ref_packing = _load_vendored_ref_packing()
    packing_ok, layout = check_packing_parity_at_real_scale(ref_packing)

    print("=== 2. DiT with the real layout, real scale ===")
    cfg = DiTConfig()
    cfg.num_layers = 2
    cfg.token_refiner_num_layers = 2
    print(f"real-dim config: hidden_size={cfg.hidden_size}, heads={cfg.num_attention_heads}, "
          f"head_dim={cfg.attention_head_dim}, layers={cfg.num_layers} (reduced from 50)")

    mx.random.seed(0)
    torch.manual_seed(0)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    state = mlx_params_as_checkpoint(dit)
    ref_dit = MiniMaxH3Transformer3DModel(**as_diffusers_config(cfg)).eval()
    converted = to_diffusers_state_dict(state, cfg)
    missing, unexpected = ref_dit.load_state_dict(converted, strict=False)
    missing = [k for k in missing if "rope.inv_freq" not in k]
    if missing or unexpected:
        print(f"FAIL: state dict mismatch\n  missing={missing[:8]}\n  unexpected={unexpected[:8]}")
        return 1
    print(f"state dict mapped: {len(converted)} tensors, no missing/unexpected keys")

    # Real row counts, real layout (from part 1). Timesteps matching the real run's actual step 0
    # (video_timestep=audio_timestep=0.0, per the prior session's instrumentation log).
    timestep, ts_indices = build_row_timesteps(layout, 0.0, 0.0, 0.999, 1.0)
    n_video = int(layout.video_indices.shape[0])
    n_audio = int(layout.audio_indices.shape[0])
    n_text = int(layout.text_indices.shape[0])
    print(f"real row counts: text={n_text} video={n_video} audio={n_audio} "
          f"distinct_timesteps={timestep.shape[0]}")

    rng = np.random.default_rng(0)
    video = rng.standard_normal((1, n_video, cfg.video_patch_dim)).astype(np.float32)
    audio = rng.standard_normal((1, n_audio, cfg.audio_latents_dim)).astype(np.float32)
    text = rng.standard_normal((1, n_text, cfg.text_dim)).astype(np.float32)

    video_i_np = np.array(layout.video_indices)
    audio_i_np = np.array(layout.audio_indices)
    text_i_np = np.array(layout.text_indices)
    tags_np = np.array(layout.token_tags)
    pos_np = np.array(layout.position_ids).astype(np.float64)
    ts_i_np = np.array(ts_indices)
    timestep_np = np.array(timestep)

    print("Running reference DiT (torch, CPU) on the real layout...")
    with torch.no_grad():
        ref_v, ref_a = ref_dit(
            hidden_states=torch.from_numpy(video),
            audio_hidden_states=torch.from_numpy(audio),
            encoder_hidden_states=torch.from_numpy(text),
            timestep=torch.from_numpy(timestep_np),
            timestep_indices=torch.from_numpy(ts_i_np.astype(np.int64)),
            token_tags=torch.from_numpy(tags_np.astype(np.int64)),
            position_ids=torch.from_numpy(pos_np),
            video_indices=torch.from_numpy(video_i_np.astype(np.int64)),
            audio_indices=torch.from_numpy(audio_i_np.astype(np.int64)),
            text_indices=torch.from_numpy(text_i_np.astype(np.int64)),
            return_dict=False,
        )
    print("Running MLX DiT on the real layout...")
    got_v, got_a = dit(
        mx.array(video), mx.array(audio), mx.array(text), mx.array(timestep_np.astype(np.float32)),
        mx.array(ts_i_np.astype(np.int32)), mx.array(tags_np.astype(np.int32)), mx.array(pos_np.astype(np.float32)),
        mx.array(video_i_np.astype(np.int32)), mx.array(audio_i_np.astype(np.int32)), mx.array(text_i_np.astype(np.int32)),
    )
    mx.eval(got_v, got_a)

    got_v_np, got_a_np = np.array(got_v), np.array(got_a)
    ref_v_np, ref_a_np = ref_v.numpy(), ref_a.numpy()
    dv, da = got_v_np - ref_v_np, got_a_np - ref_a_np
    max_abs = float(max(np.abs(dv).max(), np.abs(da).max()))
    mean_abs = float((np.abs(dv).mean() + np.abs(da).mean()) / 2)
    l2_rel = float(
        (np.linalg.norm(dv) + np.linalg.norm(da))
        / max(np.linalg.norm(ref_v_np) + np.linalg.norm(ref_a_np), 1e-12)
    )
    has_nan = bool(np.isnan(got_v_np).any() or np.isnan(got_a_np).any() or np.isnan(ref_v_np).any() or np.isnan(ref_a_np).any())
    has_inf = bool(np.isinf(got_v_np).any() or np.isinf(got_a_np).any() or np.isinf(ref_v_np).any() or np.isinf(ref_a_np).any())

    record = {
        "packing_parity_ok": packing_ok, "sequence_length": layout.sequence_length,
        "n_text": n_text, "n_video": n_video, "n_audio": n_audio,
        "max_abs_diff": max_abs, "mean_abs_diff": mean_abs, "l2_rel_norm": l2_rel,
        "has_nan": has_nan, "has_inf": has_inf,
    }
    Path(LOG_PATH).write_text(json.dumps(record) + "\n")

    print()
    print(f"max_abs_diff={max_abs:.3e}  mean_abs_diff={mean_abs:.3e}  l2_rel_norm={l2_rel:.3e}  "
          f"nan={has_nan}  inf={has_inf}")
    print(f"Wrote {LOG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
