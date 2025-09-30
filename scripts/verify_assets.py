"""Verify availability of PaliGemma 2 assets and documentation."""
from __future__ import annotations

import json
from typing import Dict

from huggingface_hub import HfApi

from paligemma2.modeling import PALIGEMMA2_ASSETS, SIGLIP_SO400M_REPO, GEMMA2_TEXT_REPOS


DOCS_REPOS = {
    "technical_report": "google/paligemma2-technical-report",
    "model_card": "google/paligemma2-3b-pt-224",
}


def verify_repo(repo_id: str) -> Dict[str, str]:
    api = HfApi()
    try:
        info = api.model_info(repo_id)
        return {"repo_id": repo_id, "exists": True, "last_modified": str(info.last_modified)}
    except Exception as exc:  # pragma: no cover - network failure path
        return {"repo_id": repo_id, "exists": False, "error": str(exc)}


def main() -> None:
    results = {
        "paligemma2_weights": [verify_repo(asset.tag) for asset in PALIGEMMA2_ASSETS.values()],
        "siglip": verify_repo(SIGLIP_SO400M_REPO),
        "gemma_text": [verify_repo(repo) for repo in GEMMA2_TEXT_REPOS.values()],
        "docs": [verify_repo(repo) for repo in DOCS_REPOS.values()],
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
