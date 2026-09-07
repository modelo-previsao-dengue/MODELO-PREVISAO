"""Sobe as camadas de dados para o HuggingFace, de onde o Kaggle le.

Uso:
    export HF_TOKEN=hf_xxx
    export HF_REPO_ID=seu-usuario/dengue-tcc2-data      # opcional
    .venv/bin/python3 scripts/upload_data_to_hf.py --apenas-model-ready

O --apenas-model-ready sobe 172 MB em vez de 2 GB. E o suficiente para
rodar no Kaggle: o notebook so le data/model_ready. As demais camadas
(bronze horario do INMET, gold do SINAN, integrado) servem para reproduzir
o pipeline de fora desta maquina, e sozinhas somam 1,9 GB.
"""
import argparse
import os
import sys
import shutil
from pathlib import Path
from huggingface_hub import HfApi

# O repositorio de destino e configuravel: o projeto tem mais de um autor e
# cada um sobe para o proprio dataset.
REPO_ID = os.environ.get("HF_REPO_ID", "pedrolucassantanaf/dengue-tcc2-data")
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STAGING = Path(__file__).resolve().parent.parent / "_hf_staging"

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--apenas-model-ready", action="store_true",
                help="Sobe so data/model_ready (172 MB), que e o que o Kaggle le")
ap.add_argument("--dry-run", action="store_true",
                help="Lista o que subiria e para, sem enviar nada")
args = ap.parse_args()

# Sem HF_TOKEN, o huggingface_hub cai no token gravado por `hf auth login`.
TOKEN = os.environ.get("HF_TOKEN")

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
    # "**/*" e nao "*": o recorte criou data/model_ready/{e2,mesorregiao,uf},
    # e com o padrao antigo os subdiretorios eram silenciosamente ignorados —
    # o notebook do Kaggle baixaria so o E1.
    ("model_ready", "**/*"),
]

if args.apenas_model_ready:
    layers = [l for l in layers if l[0] == "model_ready"]

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

if args.dry_run:
    for f in sorted(data_out.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(STAGING)}")
    shutil.rmtree(STAGING)
    print("\n--dry-run: nada foi enviado.")
    sys.exit(0)

print("Uploading (single batch)...")

api.upload_large_folder(
    folder_path=str(STAGING),
    repo_id=REPO_ID,
    repo_type="dataset",
)

shutil.rmtree(STAGING)
print("Done.")
