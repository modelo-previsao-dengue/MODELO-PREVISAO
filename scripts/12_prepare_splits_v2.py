#!/usr/bin/env python3
"""US-112: Recriar splits train/val/test com features enriquecidas (lags + anomalias).

Split temporal: train [2019,2021], val [2023], test [2024,2025,2026].
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "model_ready_v2"

TRAIN_YEARS = [2019, 2021]
VAL_YEARS = [2023]
TEST_YEARS = [2024, 2025, 2026]

ID_COLS = ["ibge_municipio", "ano", "semana_epidemiologica"]
TARGET = "notificacoes_t4"
CLASS_TARGET = "risco_surto_t4"

EXCLUDE_COLS = [
    "ano_semana", "week_start", "municipio", "regiao",
    "source_year", "municipio_resolution", "municipio_source_field",
]

INMET_PREFIXES = [
    "rain_", "temp_mean_c", "temp_min_c", "temp_max_c", "temp_range_c",
    "humidity_", "pressure_", "wind_", "radiation_",
]


def is_inmet_feature(col):
    return any(col.startswith(p) or col == p for p in INMET_PREFIXES)


def compute_risk_class(df):
    """Compute 4-class risk target based on per-municipality percentiles."""
    print("  Computando classes de risco...")
    groups = df.groupby("ibge_municipio")["notificacoes_t4"]
    p50 = groups.transform("quantile", 0.50)
    p75 = groups.transform("quantile", 0.75)
    p90 = groups.transform("quantile", 0.90)

    risk = pd.Series(0, index=df.index, dtype=int)
    risk[df["notificacoes_t4"] > p50] = 1
    risk[df["notificacoes_t4"] > p75] = 2
    risk[df["notificacoes_t4"] > p90] = 3
    return risk


def main():
    print("=" * 60)
    print("  US-112: Preparar Splits v2 (Features Enriquecidas)")
    print("=" * 60)

    print("\nCarregando dataset enriquecido (v2)...")
    df = pd.read_parquet(DATA_DIR / "integrated_filtered_v2.parquet")
    df["ibge_municipio"] = df["ibge_municipio"].astype(str)
    df = df.sort_values(["ibge_municipio", "ano", "semana_epidemiologica"])
    print(f"  {len(df):,} linhas, {len(df.columns)} colunas")

    print("\nComputando target (notificacoes_t4 = shift -4)...")
    df["notificacoes_t4"] = df.groupby("ibge_municipio")["notificacoes"].shift(-4)
    before = len(df)
    df = df.dropna(subset=["notificacoes_t4"])
    print(f"  Removidas {before - len(df):,} linhas sem target → {len(df):,} restantes")

    df["risco_surto_t4"] = compute_risk_class(df)
    print(f"  Distribuição de risco: {df['risco_surto_t4'].value_counts().sort_index().to_dict()}")

    drop_cols = [c for c in EXCLUDE_COLS if c in df.columns]
    drop_cols += [c for c in df.columns if "con_fhd" in c]
    df = df.drop(columns=drop_cols, errors="ignore")

    high_nan = [c for c in df.columns if c not in ID_COLS + [TARGET, CLASS_TARGET, "notificacoes"]
                and df[c].isnull().mean() > 0.95]
    if high_nan:
        print(f"  Excluindo {len(high_nan)} colunas com >95% NaN: {high_nan[:5]}...")
        df = df.drop(columns=high_nan)

    non_numeric = [c for c in df.columns
                   if df[c].dtype == "object" or str(df[c].dtype).startswith("datetime")]
    non_numeric = [c for c in non_numeric if c not in ["ibge_municipio"]]
    if non_numeric:
        print(f"  Excluindo {len(non_numeric)} colunas não-numéricas: {non_numeric}")
        df = df.drop(columns=non_numeric)

    print(f"\n  Colunas finais: {len(df.columns)}")

    all_features = [c for c in df.columns
                    if c not in ID_COLS + [TARGET, CLASS_TARGET, "notificacoes"]]
    inmet_features = [c for c in all_features if is_inmet_feature(c)]
    sinan_features = [c for c in all_features if not is_inmet_feature(c)]

    inmet_bruto = [c for c in inmet_features
                   if not ("_lag_" in c and c.endswith("w"))
                   and "_anomalia" not in c and "_mm_2_4w" not in c and "_mm_4_8w" not in c]
    inmet_lags_bio = [c for c in inmet_features if "_lag_" in c and c.endswith("w")]
    inmet_mm_bio = [c for c in inmet_features if "_mm_2_4w" in c or "_mm_4_8w" in c]
    inmet_anomalias = [c for c in inmet_features if "_anomalia" in c]

    print(f"\n  Categorização de features:")
    print(f"    SINAN epidemiológicas: {len(sinan_features)}")
    print(f"    INMET brutas (existentes): {len(inmet_bruto)}")
    print(f"    INMET lags biológicos (novas): {len(inmet_lags_bio)}")
    print(f"    INMET médias móveis biológicas (novas): {len(inmet_mm_bio)}")
    print(f"    INMET anomalias (novas): {len(inmet_anomalias)}")
    print(f"    Total INMET: {len(inmet_features)}")
    print(f"    Total features: {len(all_features)}")

    print("\nCriando splits temporais...")
    train = df[df["ano"].isin(TRAIN_YEARS)].copy()
    val = df[df["ano"].isin(VAL_YEARS)].copy()
    test = df[df["ano"].isin(TEST_YEARS)].copy()

    for name, split in [("train", train), ("val", val), ("test", test)]:
        years = sorted(split["ano"].unique())
        n_mun = split["ibge_municipio"].nunique()
        inmet_cov = 1 - split[inmet_bruto[:5]].isnull().mean().mean() if inmet_bruto else 0
        risk_dist = split[CLASS_TARGET].value_counts(normalize=True).sort_index()
        print(f"\n  {name}: {len(split):,} linhas, {n_mun} municípios, anos={years}")
        print(f"    Cobertura INMET (amostra): {inmet_cov:.1%}")
        print(f"    Risco: {risk_dist.to_dict()}")

    dup_check = set(train.set_index(ID_COLS).index) & set(test.set_index(ID_COLS).index)
    assert len(dup_check) == 0, f"Data leakage! {len(dup_check)} linhas duplicadas"
    print("\n  Sem data leakage entre splits ✓")

    print("\nSalvando splits...")
    train.to_parquet(DATA_DIR / "train_v2.parquet", index=False)
    val.to_parquet(DATA_DIR / "val_v2.parquet", index=False)
    test.to_parquet(DATA_DIR / "test_v2.parquet", index=False)
    print(f"  train_v2.parquet: {len(train):,}")
    print(f"  val_v2.parquet: {len(val):,}")
    print(f"  test_v2.parquet: {len(test):,}")

    schema = pd.DataFrame({
        "feature": all_features,
        "dtype": [str(df[c].dtype) for c in all_features],
        "pct_missing": [round(df[c].isnull().mean() * 100, 2) for c in all_features],
        "is_climate": [is_inmet_feature(c) for c in all_features],
        "category": [
            "inmet_bruto" if c in inmet_bruto
            else "inmet_lag_bio" if c in inmet_lags_bio
            else "inmet_mm_bio" if c in inmet_mm_bio
            else "inmet_anomalia" if c in inmet_anomalias
            else "sinan"
            for c in all_features
        ],
    })
    schema.to_csv(DATA_DIR / "feature_schema_v2.csv", index=False)

    inmet_cov_train = 1 - train[inmet_bruto[:5]].isnull().mean().mean() if inmet_bruto else 0
    inmet_cov_val = 1 - val[inmet_bruto[:5]].isnull().mean().mean() if inmet_bruto else 0
    inmet_cov_test = 1 - test[inmet_bruto[:5]].isnull().mean().mean() if inmet_bruto else 0

    report = {
        "train": {
            "rows": len(train), "municipios": int(train["ibge_municipio"].nunique()),
            "anos": TRAIN_YEARS, "cobertura_inmet": round(float(inmet_cov_train), 4),
        },
        "val": {
            "rows": len(val), "municipios": int(val["ibge_municipio"].nunique()),
            "anos": VAL_YEARS, "cobertura_inmet": round(float(inmet_cov_val), 4),
        },
        "test": {
            "rows": len(test), "municipios": int(test["ibge_municipio"].nunique()),
            "anos": TEST_YEARS, "cobertura_inmet": round(float(inmet_cov_test), 4),
        },
        "n_features": len(all_features),
        "n_sinan_features": len(sinan_features),
        "n_inmet_total": len(inmet_features),
        "n_inmet_bruto": len(inmet_bruto),
        "n_inmet_lags_bio": len(inmet_lags_bio),
        "n_inmet_mm_bio": len(inmet_mm_bio),
        "n_inmet_anomalias": len(inmet_anomalias),
        "classes": {
            f"{i}": round(float((df[CLASS_TARGET] == i).mean()), 4)
            for i in range(4)
        },
    }
    with open(DATA_DIR / "12_splits_v2_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  RESUMO US-112:")
    print(f"  Features totais: {len(all_features)} ({len(sinan_features)} SINAN + {len(inmet_features)} INMET)")
    print(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    print(f"  INMET detalhado: {len(inmet_bruto)} bruto + {len(inmet_lags_bio)} lags + {len(inmet_mm_bio)} mm + {len(inmet_anomalias)} anom")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
