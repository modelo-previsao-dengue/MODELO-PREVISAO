#!/usr/bin/env python3
"""US-006: Preparação do dataset de treino com target t+4 e splits temporais."""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
INTEGRATED_DIR = BASE_DIR / "data" / "integrated"
MODEL_READY_DIR = BASE_DIR / "data" / "model_ready"

FHD_COLS_PATTERN = "con_fhd"
EXCLUDE_COLS = [
    "ano_semana", "week_start", "municipio", "uf", "regiao",
    "source_year", "municipio_resolution", "municipio_source_field",
    "ibge_municipio",
]


def compute_risk_class(df):
    """Classificação de risco baseada nos percentis históricos por município."""
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
    choices = [0, 1, 2, 3]
    df["risco_surto_t4"] = np.select(conditions, choices, default=0)

    df = df.drop(columns=["p50", "p75", "p90"])
    return df


def main():
    MODEL_READY_DIR.mkdir(parents=True, exist_ok=True)

    print("Carregando dataset integrado...")
    df = pd.read_parquet(INTEGRATED_DIR / "sinan_inmet_municipal_weekly.parquet")
    print(f"  Shape: {df.shape}")

    # Target t+4: shift de 4 semanas no futuro por município
    print("Calculando target t+4 por município...")
    df = df.sort_values(["ibge_municipio", "ano", "semana_epidemiologica"])
    df["notificacoes_t4"] = df.groupby("ibge_municipio")["notificacoes"].shift(-4)

    # Drop last 4 weeks per municipality (no target)
    before = len(df)
    df = df.dropna(subset=["notificacoes_t4"])
    print(f"  Removidas {before - len(df):,} linhas sem target (últimas 4 semanas)")

    # Risk classification
    print("Calculando classes de risco (percentis por município)...")
    df = compute_risk_class(df)
    print(f"  Distribuição de classes:")
    print(f"    {df['risco_surto_t4'].value_counts().sort_index().to_dict()}")

    # Identify FHD columns with 100% missing
    fhd_cols = [c for c in df.columns if FHD_COLS_PATTERN in c.lower()]
    all_missing = [c for c in df.columns if df[c].isna().all()]
    high_missing = [c for c in df.columns if df[c].isna().mean() > 0.99 and c not in EXCLUDE_COLS]
    drop_cols = list(set(fhd_cols + all_missing + high_missing + EXCLUDE_COLS))
    drop_cols = [c for c in drop_cols if c in df.columns]
    print(f"  Excluindo {len(drop_cols)} colunas (FHD, >99% missing, IDs)")

    feature_cols = [c for c in df.columns if c not in drop_cols + [
        "notificacoes_t4", "risco_surto_t4", "notificacoes",
    ]]

    keep = feature_cols + ["notificacoes_t4", "risco_surto_t4", "ano", "semana_epidemiologica", "ibge_municipio"]
    keep = [c for c in keep if c in df.columns]
    df = df[list(dict.fromkeys(keep))].copy()

    # Temporal split
    print("\nSplit temporal:")
    train = df[df["ano"] <= 2019]
    val = df[(df["ano"] >= 2020) & (df["ano"] <= 2022)]
    test = df[df["ano"] >= 2023]

    print(f"  Train (2000-2019): {len(train):,} linhas")
    print(f"  Val   (2020-2022): {len(val):,} linhas")
    print(f"  Test  (2023-2026): {len(test):,} linhas")

    train.to_parquet(MODEL_READY_DIR / "train.parquet", index=False)
    val.to_parquet(MODEL_READY_DIR / "val.parquet", index=False)
    test.to_parquet(MODEL_READY_DIR / "test.parquet", index=False)

    # Feature documentation
    schema = pd.DataFrame({
        "feature": [c for c in df.columns],
        "dtype": [str(df[c].dtype) for c in df.columns],
        "pct_missing": [round(df[c].isna().mean() * 100, 2) for c in df.columns],
    })
    schema.to_csv(MODEL_READY_DIR / "feature_schema.csv", index=False)
    print(f"\nSchema: {len(schema)} colunas → {MODEL_READY_DIR / 'feature_schema.csv'}")
    print(f"Features numéricas (para XGBoost): {len(feature_cols)}")
    print(f"\n✓ US-006 completo")


if __name__ == "__main__":
    main()
