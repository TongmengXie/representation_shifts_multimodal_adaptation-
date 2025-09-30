"""Build lightweight checkpoint indices from the training manifest."""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.log_dir / "CHECKPOINT_MANIFEST.parquet"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest {manifest_path} not found")
    df = pd.read_parquet(manifest_path)
    summary = {
        "num_checkpoints": int(len(df)),
        "stages": df.groupby("stage")["step"].agg(["min", "max", "count"]).reset_index().to_dict(orient="records"),
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
