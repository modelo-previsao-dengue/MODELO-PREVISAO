"""Valida que todos os artefatos foram publicados corretamente no HuggingFace.

Uso:
    python3 scripts/validate_hf_publication.py
    make validate-remote
"""
import json
import os
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = os.environ.get("HF_REPO_ID", "pedrolucassantanaf/dengue-tcc2-data")
TOKEN = os.environ.get("HF_TOKEN")


EXPECTED_PATHS = {
    "README.md",
    "config/recorte.json",
    "data/geojson/mesorregioes.geojson",
    "data/reference/municipio_mesorregiao.csv",
    "data/model_ready_v4/feature_schema_v4.csv",
    "data/model_ready_v4/53_tcc_outputs_summary.md",
}

EXPECTED_SPLITS = []
for var in ["a", "b", "c"]:
    for split in ["train", "val", "test"]:
        EXPECTED_SPLITS.append(f"data/model_ready_v4/splits_{var}/{split}.parquet")

EXPECTED_MODELS_REG = []
EXPECTED_MODELS_CLS = []
for v in ["A", "B", "C"]:
    for arm in ["sinan_only", "inmet_bruto", "inmet_enriquecido"]:
        EXPECTED_MODELS_REG.append(f"models/regression_v4/model_{v}_{arm}.ubj")
        EXPECTED_MODELS_CLS.append(f"models/classification_v4/model_{v}_{arm}_cls.ubj")

EXPECTED_PREDICTIONS = []
for v in ["A", "B", "C"]:
    for arm in ["inmet_bruto", "inmet_enriquecido", "sinan_only"]:
        EXPECTED_PREDICTIONS.append(f"dashboard/data/predictions_{v}_{arm}.parquet")

EXPECTED_REPORTS = [
    f"data/model_ready_v4/{i}_{suffix}"
    for i, suffix in [
        (40, "sinan_meso_report.json"),
        (41, "inmet_meso_report.json"),
        (42, "integration_report.json"),
        (43, "lags_report.json"),
        (44, "anomalias_report.json"),
        (45, "splits_report.json"),
        (46, "regression_v4_report.json"),
        (47, "walkforward_v4_report.json"),
        (48, "shap_report.json"),
        (49, "optuna_report.json"),
        (50, "drift_report.json"),
        (51, "learning_curve_report.json"),
        (52, "final_results_v4.json"),
        (54, "classification_v4_report.json"),
    ]
]

EXPECTED_METRICS = [
    "models/regression_v4/shap_importance_v4.csv",
    "models/regression_v4/metrics_por_ano_v4.csv",
    "models/regression_v4/metrics_por_uf_v4.csv",
    "models/regression_v4/metrics_por_mesorregiao_v4.csv",
    "models/regression_v4/metrics_por_flag_inmet_v4.csv",
    "models/regression_v4/data_drift_psi_v4.csv",
    "models/regression_v4/data_drift_ks_v4.csv",
]


def main():
    api = HfApi(token=TOKEN)

    print(f"Validando: {REPO}")
    print(f"{'='*60}")

    # List all files in repo
    remote_files = {}
    for f in api.list_repo_tree(REPO, repo_type="dataset", recursive=True):
        if hasattr(f, "size"):
            remote_files[f.path] = f.size

    print(f"\nArquivos no HF: {len(remote_files)}")
    total_size = sum(remote_files.values())
    print(f"Tamanho total: {total_size/1024**2:.0f} MB")

    # Check all expected files
    all_expected = (
        list(EXPECTED_PATHS) + EXPECTED_SPLITS + EXPECTED_MODELS_REG +
        EXPECTED_MODELS_CLS + EXPECTED_PREDICTIONS + EXPECTED_REPORTS +
        EXPECTED_METRICS
    )

    missing = []
    found = 0
    for path in all_expected:
        if path in remote_files:
            found += 1
        else:
            missing.append(path)

    print(f"\nArquivos esperados: {len(all_expected)}")
    print(f"Encontrados: {found}")
    print(f"Faltantes: {len(missing)}")

    if missing:
        print("\nArquivos FALTANTES:")
        for m in missing:
            print(f"  {m}")

    # Spot-check: download and verify one file from each category
    print(f"\n{'='*60}")
    print("Verificação de integridade (amostra):")

    checks_ok = 0
    checks_total = 0

    # Check a split
    checks_total += 1
    try:
        path = hf_hub_download(REPO, "data/model_ready_v4/splits_b/test.parquet",
                               repo_type="dataset", token=TOKEN)
        import pyarrow.parquet as pq
        meta = pq.read_metadata(path)
        splits_report = json.loads(
            open(hf_hub_download(REPO, "data/model_ready_v4/45_splits_report.json",
                                 repo_type="dataset", token=TOKEN)).read()
        )
        expected_rows = splits_report["variantes"]["B"]["test_rows"]
        if meta.num_rows == expected_rows:
            print(f"  [OK] splits_b/test.parquet: {meta.num_rows} linhas (esperado {expected_rows})")
            checks_ok += 1
        else:
            print(f"  [FALHA] splits_b/test.parquet: {meta.num_rows} linhas (esperado {expected_rows})")
    except Exception as e:
        print(f"  [FALHA] splits_b/test.parquet: {e}")

    # Check a model
    checks_total += 1
    try:
        import xgboost as xgb
        path = hf_hub_download(REPO, "models/regression_v4/model_B_inmet_enriquecido.ubj",
                               repo_type="dataset", token=TOKEN)
        model = xgb.XGBRegressor()
        model.load_model(path)
        n_feats = len(model.get_booster().feature_names)
        print(f"  [OK] model_B_inmet_enriquecido.ubj: {n_feats} features")
        checks_ok += 1
    except Exception as e:
        print(f"  [FALHA] model_B_inmet_enriquecido.ubj: {e}")

    # Check a report
    checks_total += 1
    try:
        path = hf_hub_download(REPO, "data/model_ready_v4/46_regression_v4_report.json",
                               repo_type="dataset", token=TOKEN)
        data = json.loads(open(path).read())
        n_modelos = len(data.get("modelos", {}))
        print(f"  [OK] 46_regression_v4_report.json: {n_modelos} modelos")
        checks_ok += 1
    except Exception as e:
        print(f"  [FALHA] 46_regression_v4_report.json: {e}")

    # Check a prediction
    checks_total += 1
    try:
        path = hf_hub_download(REPO, "dashboard/data/predictions_B_inmet_enriquecido.parquet",
                               repo_type="dataset", token=TOKEN)
        meta = pq.read_metadata(path)
        print(f"  [OK] predictions_B_inmet_enriquecido.parquet: {meta.num_rows} linhas")
        checks_ok += 1
    except Exception as e:
        print(f"  [FALHA] predictions_B_inmet_enriquecido.parquet: {e}")

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTADO:")
    print(f"  Arquivos: {found}/{len(all_expected)} presentes")
    print(f"  Integridade: {checks_ok}/{checks_total} verificados OK")

    if missing or checks_ok < checks_total:
        print(f"\n  STATUS: FALHA")
        sys.exit(1)
    else:
        print(f"\n  STATUS: OK — publicação completa e íntegra")
        print(f"  URL: https://huggingface.co/datasets/{REPO}")


if __name__ == "__main__":
    main()
