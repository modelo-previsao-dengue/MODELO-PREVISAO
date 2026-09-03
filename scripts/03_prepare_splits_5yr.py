#!/usr/bin/env python3
"""US-103: Preparacao de splits temporais para dataset filtrado 5yr."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "data" / "model_ready_v2"
OUTPUT_DIR = INPUT_DIR

FHD_COLS_PATTERN = "con_fhd"
EXCLUDE_COLS = [
    "ano_semana", "week_start", "municipio", "uf", "regiao",
    "source_year", "municipio_resolution", "municipio_source_field",
]

TRAIN_YEARS = [2019, 2021]
VAL_YEARS = [2023]
TEST_YEARS = [2024, 2025, 2026]


def compute_risk_class(df):
    thresholds = df.groupby("ibge_municipio")["notificacoes"].agg(
        p50="median",
        p75=lambda x: x.quantile(0.75),
        p90=lambda x: x.quantile(0.90),
    ).reset_index()
    df = df.merge(thresholds, on="ibge_municipio", how="left")
    conditions = [
        df["notificacoes_t4"] <= df["p50"],
        (df["notificacoes_t4"] > df["p50"]) & (df["notificacoes_t4"] <= df["p75"]),
        (df["notificacoes_t4"] > df["p75"]) & (df["notificacoes_t4"] <= df["p90"]),
        df["notificacoes_t4"] > df["p90"],
    ]
    df["risco_surto_t4"] = np.select(conditions, [0, 1, 2, 3], default=0)
    df = df.drop(columns=["p50", "p75", "p90"])
    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  US-103: Preparacao de Splits Temporais (5yr)")
    print("=" * 60)

    print("\nCarregando dataset filtrado...")
    df = pd.read_parquet(INPUT_DIR / "integrated_filtered.parquet")
    df["ibge_municipio"] = df["ibge_municipio"].astype(str)
    print(f"  {len(df):,} linhas, {df.shape[1]} colunas")

    # Target t+4
    print("Calculando target t+4 por municipio...")
    df = df.sort_values(["ibge_municipio", "ano", "semana_epidemiologica"])
    df["notificacoes_t4"] = df.groupby("ibge_municipio")["notificacoes"].shift(-4)
    before = len(df)
    df = df.dropna(subset=["notificacoes_t4"])
    print(f"  Removidas {before - len(df):,} linhas sem target")

    # Risk classification
    print("Calculando classes de risco...")
    df = compute_risk_class(df)
    dist = df["risco_surto_t4"].value_counts().sort_index().to_dict()
    labels = {0: "baixo", 1: "medio", 2: "alto", 3: "surto"}
    print(f"  Distribuicao: { {labels[k]: v for k, v in dist.items()} }")

    # Drop useless columns
    fhd_cols = [c for c in df.columns if FHD_COLS_PATTERN in c.lower()]
    all_missing = [c for c in df.columns if df[c].isna().all()]
    high_missing = [c for c in df.columns if df[c].isna().mean() > 0.99 and c not in EXCLUDE_COLS]
    drop_cols = list(set(fhd_cols + all_missing + high_missing + EXCLUDE_COLS))
    drop_cols = [c for c in drop_cols if c in df.columns]
    print(f"  Excluindo {len(drop_cols)} colunas (FHD, >99% missing, IDs)")
    df = df.drop(columns=drop_cols, errors="ignore")

    # Feature list
    meta_cols = ["notificacoes_t4", "risco_surto_t4", "ano", "semana_epidemiologica", "ibge_municipio"]
    feature_cols = [c for c in df.columns if c not in meta_cols + ["notificacoes"]]
    print(f"  Features disponiveis: {len(feature_cols)}")

    # Temporal split
    print(f"\nSplit temporal:")
    train = df[df["ano"].isin(TRAIN_YEARS)].copy()
    val = df[df["ano"].isin(VAL_YEARS)].copy()
    test = df[df["ano"].isin(TEST_YEARS)].copy()

    for name, subset, years in [("train", train, TRAIN_YEARS), ("val", val, VAL_YEARS), ("test", test, TEST_YEARS)]:
        n_mun = subset["ibge_municipio"].nunique()
        print(f"  {name:5s} {years}: {len(subset):>10,} linhas, {n_mun} mun")

    # Save splits
    train.to_parquet(OUTPUT_DIR / "train_5yr.parquet", index=False)
    val.to_parquet(OUTPUT_DIR / "val_5yr.parquet", index=False)
    test.to_parquet(OUTPUT_DIR / "test_5yr.parquet", index=False)
    print(f"\nSalvos: train_5yr.parquet, val_5yr.parquet, test_5yr.parquet")

    # Feature schema
    schema = pd.DataFrame({
        "feature": feature_cols,
        "dtype": [str(df[c].dtype) for c in feature_cols],
        "pct_missing": [round(df[c].isna().mean() * 100, 2) for c in feature_cols],
    })
    schema.to_csv(OUTPUT_DIR / "feature_schema_5yr.csv", index=False)

    # INMET coverage per split
    climate_proxy = "rain_sum_mm"
    cov = {}
    for name, subset in [("train", train), ("val", val), ("test", test)]:
        if climate_proxy in subset.columns:
            cov[name] = round(subset[climate_proxy].notna().mean() * 100, 1)
        else:
            cov[name] = 0.0

    report = {
        "total_rows_with_target": len(df),
        "train_years": TRAIN_YEARS,
        "val_years": VAL_YEARS,
        "test_years": TEST_YEARS,
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "n_features": len(feature_cols),
        "risk_distribution": {labels.get(int(k), str(k)): int(v) for k, v in dist.items()},
        "climate_coverage_per_split": cov,
        "dropped_columns": len(drop_cols),
    }
    with open(OUTPUT_DIR / "03_splits_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  RESUMO: {len(feature_cols)} features, {len(df):,} linhas com target")
    print(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    print(f"  Cobertura INMET: train={cov['train']}% val={cov['val']}% test={cov['test']}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
