"""
Upload all data layers to HuggingFace for Kaggle notebooks.

Usage:
    export HF_TOKEN=hf_xxx
    python scripts/upload_data_to_hf.py

Uses upload_large_folder to batch everything in minimal commits.
"""
import os
import sys
import shutil
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "pedrolucassantanaf/dengue-tcc2-data"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STAGING = Path(__file__).resolve().parent.parent / "_hf_staging"

TOKEN = os.environ.get("HF_TOKEN")
if not TOKEN:
    print("ERROR: Set HF_TOKEN environment variable")
    sys.exit(1)

api = HfApi(token=TOKEN)

if STAGING.exists():
    shutil.rmtree(STAGING)

STAGING.mkdir()
data_out = STAGING / "data"

layers = [
    ("sinan/silver/sinan_tcc2_v2/official_observed", "**/*.parquet"),
    ("sinan/gold/sinan_tcc2_v2/official_dense", "**/*.parquet"),
    ("inmet/bronze/hourly", "**/*.parquet"),
    ("inmet/silver", "*.parquet"),
    ("inmet/gold", "*.parquet"),
    ("integrated", "*.parquet"),
    ("model_ready", "*"),
]

count = 0
for subdir, pattern in layers:
    src = DATA_DIR / subdir
    if not src.exists():
        print(f"SKIP (not found): {subdir}")
        continue
    for f in src.glob(pattern):
        if f.is_file():
            rel = f.relative_to(DATA_DIR)
            dest = data_out / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(f, dest)
            count += 1

total_mb = sum(f.stat().st_size for f in data_out.rglob("*") if f.is_file()) / 1024**2
print(f"Staged {count} files ({total_mb:.0f} MB) for upload to {REPO_ID}")
print("Uploading (single batch)...")

api.upload_large_folder(
    folder_path=str(STAGING),
    repo_id=REPO_ID,
    repo_type="dataset",
)

shutil.rmtree(STAGING)
print("Done.")
