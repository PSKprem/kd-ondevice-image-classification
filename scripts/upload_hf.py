"""P25 — upload D:/kd-project/hf_upload to a Hugging Face model repo (default: private) and verify the remote files.

Needs a write token (`huggingface-cli login`; never commit the token). Re-running is safe: unchanged files are skipped.

Usage (training venv, from repo root):
    python scripts/upload_hf.py [--repo p25-kd-mobilenet-cifar100] [--public]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from huggingface_hub import HfApi

SRC = Path("D:/kd-project/hf_upload")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="p25-kd-mobilenet-cifar100")
    ap.add_argument("--public", action="store_true", help="create the repo as public (default private)")
    args = ap.parse_args()

    api = HfApi()
    user = api.whoami()["name"]
    repo_id = f"{user}/{args.repo}"
    url = api.create_repo(repo_id, repo_type="model", private=not args.public, exist_ok=True)
    print(f"[hf] repo: {url}", flush=True)

    t0 = time.time()
    api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(SRC),
                      commit_message="P25 final models: PyTorch safetensors + TFLite FP32 / INT8-mixed, model card")
    print(f"[hf] upload finished in {(time.time() - t0) / 60:.1f} min", flush=True)

    local = {p.relative_to(SRC).as_posix(): p.stat().st_size for p in SRC.rglob("*") if p.is_file()}
    info = api.model_info(repo_id, files_metadata=True)
    remote = {s.rfilename: s.size for s in info.siblings}
    missing = [f for f in local if f not in remote]
    wrong = [f for f in local if f in remote and remote[f] != local[f]]
    print(f"[hf] verify: {len(local)} local files, {len(remote)} remote files, missing={missing}, size mismatch={wrong}")
    print(f"[hf] private={info.private}  url=https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
