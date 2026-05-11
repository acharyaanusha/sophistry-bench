"""Push artifacts/hf_dataset/ to the HuggingFace Hub.

Usage: python scripts/upload_hf_dataset.py <namespace>/<repo-name> [--private]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = REPO_ROOT / "artifacts" / "hf_dataset"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id", help="e.g. anushaacharya/sophistry-bench-quality-dev")
    parser.add_argument("--private", action="store_true", help="Create as private repo")
    args = parser.parse_args()

    assert UPLOAD_DIR.exists(), f"{UPLOAD_DIR} missing — run scripts/build_hf_dataset.py first"
    assert (UPLOAD_DIR / "README.md").exists()
    assert (UPLOAD_DIR / "data" / "dev.parquet").exists()

    create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    print(f"Repo ready: https://huggingface.co/datasets/{args.repo_id}")

    api = HfApi()
    api.upload_folder(
        folder_path=str(UPLOAD_DIR),
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Initial upload: 50-item curated QuALITY dev slice",
    )
    print(f"Uploaded folder. View: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
