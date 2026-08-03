"""Checkpoint loading for the MiniMax-H3 MLX port.

The MLX module tree reproduces the original checkpoint names exactly, so loading is a 1:1 key
match — the only tensor the checkpoint carries that the port does not hold is ``rope.inv_freq``,
which is recomputed bit-identically from the config.

MiniMax-H3 ships a **mixed-precision** transformer: the two input patch projections, the timestep
MLP and the two output heads are float32 while everything else (including the AdaLN projections)
is bfloat16. That split is preserved on load — it is not incidental. The timestep MLP feeds every
block's modulation, so rounding it biases all 50 blocks identically at every sampling step and the
error accumulates coherently along the denoising trajectory.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from .config import DiTConfig
from .dit import MiniMaxH3DiT

# Substring matches, mirroring the reference's `_keep_in_fp32_modules`.
FP32_PREFIXES = (
    "video_patch_proj.",
    "audio_patch_proj.",
    "time_embedder.",
    "final_layer.video_out.",
    "final_layer.audio_out.",
)

# Carried by the checkpoint but recomputed by the port.
SKIP_KEYS = ("rope.inv_freq",)


def is_fp32_key(key: str) -> bool:
    return key.startswith(FP32_PREFIXES)


def shard_paths(model_dir: str | Path) -> list[Path]:
    """Resolve the safetensors shards of a transformer directory, in index order."""
    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as fh:
            weight_map = json.load(fh)["weight_map"]
        names = sorted(set(weight_map.values()))
        return [model_dir / name for name in names]
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No safetensors found in {model_dir}.")
    return shards


def load_dit(
    model_dir: str | Path,
    dtype: mx.Dtype | None = None,
    strict: bool = True,
    verbose: bool = False,
) -> MiniMaxH3DiT:
    """Load the 33B DiT from a released ``FL2VA/transformer`` (or ``Ref2VA/transformer``) directory.

    Args:
        model_dir: the transformer directory holding ``config.json`` and the shards.
        dtype: cast every tensor to this dtype. ``None`` (default) preserves the checkpoint's
            mixed float32/bfloat16 split, which is what the reference runs.
        strict: raise if the checkpoint and the module tree disagree on any key.
        verbose: print per-shard progress.

    Returns:
        A parameter-loaded :class:`MiniMaxH3DiT`.
    """
    model_dir = Path(model_dir)
    config = DiTConfig.from_json(model_dir / "config.json")
    model = MiniMaxH3DiT(config)

    expected = {key for key, _ in tree_flatten(model.parameters())}
    weights: dict[str, mx.array] = {}
    unexpected: list[str] = []

    for shard in shard_paths(model_dir):
        started = time.perf_counter()
        loaded = mx.load(str(shard))
        for key, tensor in loaded.items():
            if key in SKIP_KEYS:
                continue
            if key not in expected:
                unexpected.append(key)
                continue
            if dtype is not None:
                tensor = tensor.astype(dtype)
            elif is_fp32_key(key) and tensor.dtype != mx.float32:
                tensor = tensor.astype(mx.float32)
            weights[key] = tensor
        if verbose:
            gb = sum(t.nbytes for t in loaded.values()) / 1e9
            print(f"  {shard.name}: {len(loaded)} tensors, {gb:.2f} GB, "
                  f"{time.perf_counter() - started:.1f}s")

    missing = sorted(expected - weights.keys())
    if strict and (missing or unexpected):
        raise KeyError(
            f"Checkpoint/module mismatch: {len(missing)} missing (e.g. {missing[:4]}), "
            f"{len(unexpected)} unexpected (e.g. {unexpected[:4]})."
        )

    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    return model


def parameter_summary(model: MiniMaxH3DiT) -> dict[str, object]:
    """Parameter counts and footprint, split by the AdaLN projections that can be dropped."""
    total = adaln = 0
    nbytes = adaln_bytes = 0
    for key, value in tree_flatten(model.parameters()):
        total += value.size
        nbytes += value.nbytes
        if ".adaln_proj." in key and key.startswith("blocks."):
            adaln += value.size
            adaln_bytes += value.nbytes
    return {
        "total_params": total,
        "adaln_params": adaln,
        "core_params": total - adaln,
        "total_gb": nbytes / 1e9,
        "adaln_gb": adaln_bytes / 1e9,
        "core_gb": (nbytes - adaln_bytes) / 1e9,
    }
