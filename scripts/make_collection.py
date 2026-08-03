"""Create the Hugging Face collection that groups the MiniMax-H3 MLX builds.

Each item carries a note saying what it is *for*, since the widths differ in ways a size column does
not convey — and one width is deliberately absent, which is worth stating where people will look.

    ./.venv/bin/python scripts/make_collection.py --yes
"""

from __future__ import annotations

import argparse

NAMESPACE = "pipenetwork"
TITLE = "MiniMax-H3 MLX"

# The collection description is capped at 150 characters, so the substance lives in the
# model cards and in the per-item notes below.
DESCRIPTION = (
    "Apple Silicon (MLX) builds of MiniMax-H3, the 33B joint video+audio diffusion "
    "transformer. Code: github.com/PipeNetwork/minimax-h3-mlx"
)

ITEMS = [
    ("pipenetwork/MiniMax-H3-MLX-8bit",
     "35.3 GB download, 21.5 GB resident. 27.6 dB PSNR vs bfloat16 - near-indistinguishable."),
    ("pipenetwork/MiniMax-H3-MLX-6bit",
     "30.3 GB download, 16.5 GB resident. Bracketed by two widths verified by generation."),
    ("pipenetwork/MiniMax-H3-MLX-4bit",
     "25.3 GB download, 11.5 GB resident. 22.0 dB PSNR - artifacts visible, subject intact."),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--yes", action="store_true", help="actually create it")
    args = parser.parse_args()

    print(f"collection: {NAMESPACE}/{TITLE}{' (private)' if args.private else ''}")
    print(f"\n{DESCRIPTION}\n")
    for item, note in ITEMS:
        print(f"  {item}\n     {note}")

    if not args.yes:
        print("\ndry run — pass --yes to create")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    collection = api.create_collection(
        title=TITLE,
        namespace=NAMESPACE,
        description=DESCRIPTION,
        private=args.private,
        exists_ok=True,
    )
    for item, note in ITEMS:
        api.add_collection_item(
            collection.slug, item_id=item, item_type="model", note=note, exists_ok=True
        )
    print(f"\nhttps://huggingface.co/collections/{collection.slug}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
