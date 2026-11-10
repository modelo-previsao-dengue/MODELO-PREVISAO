"""Publica TODOS os artefatos do DengueCast no HuggingFace.

Uso:
    python3 scripts/upload_data_to_hf.py --tudo          # sobe tudo (~1.1 GB)
    python3 scripts/upload_data_to_hf.py --v4            # so dados + modelos v4
    python3 scripts/upload_data_to_hf.py --dashboard     # so cache da dashboard
    python3 scripts/upload_data_to_hf.py --dry-run --tudo  # lista sem enviar
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
REPO_ID = os.environ.get("HF_REPO_ID", "pedrolucassantanaf/dengue-tcc2-data")
STAGING = ROOT / "_hf_staging"

LAYERS = {
    "v4": [
        ("data/model_ready_v4", "data/model_ready_v4", "**/*"),
        ("models/regression_v4", "models/regression_v4", "**/*"),
        ("models/classification_v4", "models/classification_v4", "**/*"),
    ],
    "dashboard": [
        ("dashboard/data", "dashboard/data", "**/*"),
        ("data/geojson", "data/geojson", "**/*"),
    ],
    "sinan": [
        ("data/sinan/bronze", "data/sinan/bronze", "**/*"),
        ("data/sinan/silver", "data/sinan/silver", "**/*"),
        ("data/sinan/gold", "data/sinan/gold", "**/*"),
        ("data/sinan/governance", "data/sinan/governance", "**/*"),
        ("data/sinan/serving", "data/sinan/serving", "**/*"),
    ],
    "inmet": [
        ("data/inmet/bronze", "data/inmet/bronze", "**/*"),
        ("data/inmet/silver", "data/inmet/silver", "**/*"),
        ("data/inmet/gold", "data/inmet/gold", "**/*"),
    ],
    "integrated": [
        ("data/integrated", "data/integrated", "**/*"),
    ],
    "model_ready_old": [
        ("data/model_ready", "data/model_ready", "**/*"),
        ("data/model_ready_v2", "data/model_ready_v2", "**/*"),
        ("data/model_ready_v3", "data/model_ready_v3", "**/*"),
    ],
    "reference": [
        ("data/reference", "data/reference", "**/*"),
    ],
    "models_old": [
        ("models/ablacao", "models/ablacao", "**/*"),
        ("models/ablacao_surto", "models/ablacao_surto", "**/*"),
        ("models/baseline_5yr", "models/baseline_5yr", "**/*"),
        ("models/baselines", "models/baselines", "**/*"),
        ("models/baselines_e2", "models/baselines_e2", "**/*"),
        ("models/baselines_meso", "models/baselines_meso", "**/*"),
        ("models/baselines_surto_meso", "models/baselines_surto_meso", "**/*"),
        ("models/baselines_uf", "models/baselines_uf", "**/*"),
        ("models/baselines_v3", "models/baselines_v3", "**/*"),
        ("models/blindness_v3", "models/blindness_v3", "**/*"),
        ("models/classification_5yr", "models/classification_5yr", "**/*"),
        ("models/classification_v2", "models/classification_v2", "**/*"),
        ("models/classification_v3", "models/classification_v3", "**/*"),
        ("models/kaggle_e1", "models/kaggle_e1", "**/*"),
        ("models/kaggle_mesorregiao", "models/kaggle_mesorregiao", "**/*"),
        ("models/multi_horizon", "models/multi_horizon", "**/*"),
        ("models/nacional", "models/nacional", "**/*"),
        ("models/regression_5yr", "models/regression_5yr", "**/*"),
        ("models/regression_v2", "models/regression_v2", "**/*"),
        ("models/regression_v3", "models/regression_v3", "**/*"),
        ("models/shap_5yr", "models/shap_5yr", "**/*"),
        ("models/shap_analysis", "models/shap_analysis", "**/*"),
        ("models/shap_v2", "models/shap_v2", "**/*"),
        ("models/shap_v3", "models/shap_v3", "**/*"),
        ("models/walk_forward_results", "models/walk_forward_results", "**/*"),
        ("models/xgb_baseline_sinan_only", "models/xgb_baseline_sinan_only", "**/*"),
        ("models/xgb_classification_mvp", "models/xgb_classification_mvp", "**/*"),
        ("models/xgb_regression_mvp", "models/xgb_regression_mvp", "**/*"),
        ("models/xgb_regression_tuned", "models/xgb_regression_tuned", "**/*"),
    ],
    "config": [
        ("config/recorte.json", "config/recorte.json", None),
    ],
    "card": [
        ("hf_dataset_card/README.md", "README.md", None),
    ],
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--tudo", action="store_true",
                   help="Sobe tudo: v4 + modelos + dados antigos + dashboard + config + card")
    g.add_argument("--v4", action="store_true",
                   help="Sobe dados v4 + modelos v4 (~786 MB)")
    g.add_argument("--dashboard", action="store_true",
                   help="Sobe cache da dashboard + geojson (~43 MB)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Lista arquivos e tamanhos sem enviar")
    return ap.parse_args()


def resolve_groups(args):
    if args.tudo:
        return list(LAYERS.keys())
    if args.v4:
        return ["v4", "config"]
    if args.dashboard:
        return ["dashboard"]
    return []


def validate_parquets(files):
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("WARN: pyarrow nao instalado, pulando validacao de parquets")
        return True
    ok = True
    for f in files:
        if f.suffix == ".parquet":
            try:
                meta = pq.read_metadata(f)
                if meta.num_rows == 0:
                    print(f"  ERRO: {f.name} esta vazio (0 linhas)")
                    ok = False
            except Exception as e:
                print(f"  ERRO: {f.name} nao e um parquet valido: {e}")
                ok = False
    return ok


def validate_jsons(files):
    ok = True
    for f in files:
        if f.suffix == ".json":
            try:
                json.loads(f.read_text())
            except Exception as e:
                print(f"  ERRO: {f} nao e JSON valido: {e}")
                ok = False
    return ok


def validate_models(files):
    ok = True
    for f in files:
        if f.suffix == ".ubj":
            if f.stat().st_size == 0:
                print(f"  ERRO: {f} esta vazio (0 bytes)")
                ok = False
    return ok


def stage_files(groups):
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir()

    staged = []
    for group in groups:
        for entry in LAYERS[group]:
            src_rel, dst_rel, pattern = entry
            src = ROOT / src_rel

            if pattern is None:
                if not src.exists():
                    print(f"  SKIP (nao encontrado): {src_rel}")
                    continue
                dest = STAGING / dst_rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(src, dest)
                staged.append(src)
            else:
                if not src.exists():
                    print(f"  SKIP (nao encontrado): {src_rel}/")
                    continue
                for f in src.glob(pattern):
                    if f.is_file():
                        rel = f.relative_to(src)
                        dest = STAGING / dst_rel / rel
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        os.symlink(f, dest)
                        staged.append(f)

    return staged


def main():
    args = parse_args()
    groups = resolve_groups(args)

    token = os.environ.get("HF_TOKEN")
    if not token:
        api = HfApi()
        try:
            api.whoami()
        except Exception:
            print("ERRO: HF_TOKEN nao configurado.")
            print("  Opcao 1: export HF_TOKEN=hf_xxx")
            print("  Opcao 2: huggingface-cli login")
            sys.exit(1)
    else:
        api = HfApi(token=token)

    print(f"Repo destino: {REPO_ID}")
    print(f"Grupos: {', '.join(groups)}")
    print()

    print("Staging arquivos...")
    staged = stage_files(groups)

    if not staged:
        print("Nenhum arquivo encontrado para upload.")
        sys.exit(1)

    total_mb = sum(f.stat().st_size for f in staged) / 1024**2
    print(f"\nTotal: {len(staged)} arquivos ({total_mb:.0f} MB)")

    print("\nValidando integridade...")
    v1 = validate_parquets(staged)
    v2 = validate_jsons(staged)
    v3 = validate_models(staged)
    if not (v1 and v2 and v3):
        print("\nERRO: Validacao falhou. Corrija os erros acima antes de subir.")
        shutil.rmtree(STAGING)
        sys.exit(1)
    print("Validacao OK.")

    if args.dry_run:
        print("\n--- DRY RUN: arquivos que seriam enviados ---")
        for f in sorted(STAGING.rglob("*")):
            if f.is_file():
                size = f.stat().st_size
                unit = "MB" if size > 1024**2 else "KB"
                val = size / 1024**2 if unit == "MB" else size / 1024
                print(f"  {f.relative_to(STAGING)}  ({val:.1f} {unit})")
        shutil.rmtree(STAGING)
        print(f"\n--dry-run: nada foi enviado. Total: {total_mb:.0f} MB")
        sys.exit(0)

    print(f"\nUploading {total_mb:.0f} MB para {REPO_ID}...")
    api.upload_large_folder(
        folder_path=str(STAGING),
        repo_id=REPO_ID,
        repo_type="dataset",
    )

    shutil.rmtree(STAGING)
    print(f"\nPublicado com sucesso!")
    print(f"URL: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
