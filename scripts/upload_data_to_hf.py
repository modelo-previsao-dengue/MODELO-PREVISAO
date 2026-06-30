"""
Upload all data layers to HuggingFace for Kaggle notebooks.

Usage:
    export HF_TOKEN=hf_xxx
    python scripts/upload_data_to_hf.py

Uploads Bronze/Silver/Gold layers for SINAN and INMET,
plus integrated and model_ready data.
"""
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "thiagorfreitas/dengue-tcc2-data"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TOKEN = os.environ.get("HF_TOKEN")
if not TOKEN:
    print("ERROR: Set HF_TOKEN environment variable")
    sys.exit(1)

api = HfApi(token=TOKEN)

FILES_TO_UPLOAD = []

# --- SINAN Silver (partitioned by year) ---
sinan_silver = DATA_DIR / "sinan" / "silver" / "sinan_tcc2_v2" / "official_observed"
for year_dir in sorted(sinan_silver.glob("year=*")):
    for parquet in year_dir.glob("*.parquet"):
        rel = parquet.relative_to(DATA_DIR)
        FILES_TO_UPLOAD.append((parquet, f"data/{rel}"))

# --- SINAN Gold (partitioned by year, multi-part) ---
sinan_gold = DATA_DIR / "sinan" / "gold" / "sinan_tcc2_v2" / "official_dense"
for year_dir in sorted(sinan_gold.glob("year=*")):
    for parquet in year_dir.glob("*.parquet"):
        rel = parquet.relative_to(DATA_DIR)
        FILES_TO_UPLOAD.append((parquet, f"data/{rel}"))

# --- INMET Bronze (hourly, partitioned by year) ---
inmet_bronze = DATA_DIR / "inmet" / "bronze" / "hourly"
for year_dir in sorted(inmet_bronze.glob("year=*")):
    for parquet in year_dir.glob("*.parquet"):
        rel = parquet.relative_to(DATA_DIR)
        FILES_TO_UPLOAD.append((parquet, f"data/{rel}"))

# --- INMET Silver (one file per year) ---
inmet_silver = DATA_DIR / "inmet" / "silver"
for parquet in sorted(inmet_silver.glob("*.parquet")):
    rel = parquet.relative_to(DATA_DIR)
    FILES_TO_UPLOAD.append((parquet, f"data/{rel}"))

# --- INMET Gold (one file per year) ---
inmet_gold = DATA_DIR / "inmet" / "gold"
for parquet in sorted(inmet_gold.glob("*.parquet")):
    rel = parquet.relative_to(DATA_DIR)
    FILES_TO_UPLOAD.append((parquet, f"data/{rel}"))

# --- Integrated ---
integrated = DATA_DIR / "integrated" / "sinan_inmet_municipal_weekly.parquet"
if integrated.exists():
    FILES_TO_UPLOAD.append((integrated, "data/integrated/sinan_inmet_municipal_weekly.parquet"))

# --- Model Ready ---
for name in ["train.parquet", "val.parquet", "test.parquet", "feature_schema.csv"]:
    f = DATA_DIR / "model_ready" / name
    if f.exists():
        FILES_TO_UPLOAD.append((f, f"data/model_ready/{name}"))

print(f"Found {len(FILES_TO_UPLOAD)} files to upload to {REPO_ID}")
total_size = sum(f[0].stat().st_size for f in FILES_TO_UPLOAD)
print(f"Total size: {total_size / 1024**2:.1f} MB")
print()

for i, (local_path, repo_path) in enumerate(FILES_TO_UPLOAD, 1):
    size_mb = local_path.stat().st_size / 1024**2
    print(f"[{i}/{len(FILES_TO_UPLOAD)}] {repo_path} ({size_mb:.1f} MB) ...", end=" ", flush=True)
    try:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=REPO_ID,
            repo_type="dataset",
        )
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")

print("\nDone.")
