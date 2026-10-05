from huggingface_hub import snapshot_download
import os

download_dir = "../GOT-10k"
os.makedirs(download_dir, exist_ok=True)

print("Pulling GOT-10k...")

snapshot_download(repo_id="xche32/got10k", repo_type="dataset", local_dir=download_dir, resume_download=True, max_workers=8) # type: ignore

print(f"Downloaded in {download_dir}")