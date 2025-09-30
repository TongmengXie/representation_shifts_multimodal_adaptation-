"""Download released PaliGemma 2 checkpoints and related docs."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Iterable, Optional

from huggingface_hub import HfApi, hf_hub_download

from paligemma2.modeling import PALIGEMMA2_ASSETS


def list_assets() -> list[dict[str, str]]:
    return [
        {
            "key": key,
            "model_size": asset.model_size,
            "resolution": asset.resolution,
            "repo_id": asset.tag,
            "file": asset.file_name,
            "sha256": asset.sha256 or "unknown",
        }
        for key, asset in PALIGEMMA2_ASSETS.items()
    ]


def compute_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_asset(key: str, output_dir: pathlib.Path, verify: bool = True) -> pathlib.Path:
    if key not in PALIGEMMA2_ASSETS:
        raise KeyError(f"Unknown asset key {key}. Use --list to inspect available keys.")
    asset = PALIGEMMA2_ASSETS[key]
    output_dir.mkdir(parents=True, exist_ok=True)
    local_path = hf_hub_download(repo_id=asset.tag, filename=asset.file_name, local_dir=str(output_dir))
    if verify and asset.sha256:
        digest = compute_sha256(pathlib.Path(local_path))
        if digest != asset.sha256:
            raise RuntimeError(f"Checksum mismatch for {local_path}: expected {asset.sha256} got {digest}")
    return pathlib.Path(local_path)


def fetch_model_card(repo_id: str) -> dict[str, str]:
    api = HfApi()
    card = api.model_card(repo_id)
    return {
        "repo_id": repo_id,
        "card": card.content,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("weights"))
    parser.add_argument("--key", type=str, help="Asset key (e.g. 3b-224). Use --list to view options.")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--dump-card", action="store_true", help="Also download the model card JSON.")
    parser.add_argument("--list", action="store_true", help="List assets and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        print(json.dumps(list_assets(), indent=2))
        return
    path = download_asset(args.key, args.output_dir, verify=not args.skip_verify)
    print(f"Downloaded {args.key} to {path}")
    if args.dump_card:
        card = fetch_model_card(PALIGEMMA2_ASSETS[args.key].tag)
        card_path = args.output_dir / f"{args.key}_card.json"
        card_path.write_text(json.dumps(card, indent=2))
        print(f"Saved model card to {card_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
