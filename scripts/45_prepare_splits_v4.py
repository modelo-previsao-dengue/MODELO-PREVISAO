#!/usr/bin/env python3
"""v4-45: Preparar splits train/val/test para 3 variantes (A/B/C).

Baseado em scripts/26_prepare_splits_v3.py.
Diferenças:
  - Gera splits para 3 variantes com treinos diferentes.
  - Val e test são idênticos entre variantes.
  - flag_fonte_inmet é excluído das features (metadata only).
  - Variante C recebe sample_weight no report mas o peso é aplicado no treino (script 46).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import recorte_config

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "model_ready_v4"

GROUP_KEY = "cod_mesorregiao"
ID_COLS = ["cod_mesorregiao", "ano", "semana_epidemiologica"]
TARGET = "notificacoes_t4"
CLASS_TARGET = "risco_surto_t4"

METADATA_COLS = [
    "nome_mesorregiao", "uf_sigla", "nome_grande_regiao", "n_municipios", "n_estacoes",
    "flag_fonte_inmet",
]

INMET_PREFIXES = [
    "rain_", "temp_mean_c", "temp_min_c", "temp_max_c", "temp_range_c",
    "humidity_", "pressure_", "wind_", "radiation_",
]

VARIANTES = ["variante_a", "variante_b", "variante_c"]


def is_inmet_feature(col):
    return any(col.startswith(p) or col == p for p in INMET_PREFIXES)


def compute_risk_class(df, train_years):
    train_mask = df["ano"].isin(train_years)
    train_groups = df.loc[train_mask].groupby(GROUP_KEY)[TARGET]
    thresholds = train_groups.quantile([0.50, 0.75, 0.90]).unstack()
    thresholds.columns = ["p50", "p75", "p90"]

    merged = df[[GROUP_KEY]].merge(thresholds, left_on=GROUP_KEY, right_index=True, how="left")

    risk = pd.Series(0, index=df.index, dtype=int)
    risk[df[TARGET] > merged["p50"]] = 1
    risk[df[TARGET] > merged["p75"]] = 2
    risk[df[TARGET] > merged["p90"]] = 3
    return risk


def main():
    print("=" * 60)
    print("  v4-45: Preparar Splits para 3 Variantes (A/B/C)")
    print("=" * 60)

    print("\nCarregando dataset enriquecido v4...")
    df = pd.read_parquet(DATA_DIR / "integrated_mesorregiao_v4.parquet")
    df = df.sort_values(ID_COLS).reset_index(drop=True)
    print(f"  {len(df):,} linhas, {len(df.columns)} colunas")

    print("\nComputando target (notificacoes_t4 = shift -4)...")
    df[TARGET] = df.groupby(GROUP_KEY)["notificacoes"].shift(-4)
    before = len(df)
    df = df.dropna(subset=[TARGET])
    print(f"  Removidas {before - len(df):,} linhas sem target → {len(df):,} restantes")

    drop_cols = [c for c in METADATA_COLS if c in df.columns]
    drop_cols += [c for c in df.columns if "con_fhd" in c]
    df = df.drop(columns=drop_cols, errors="ignore")
    print(f"  Removidas colunas de metadados: {drop_cols}")

    high_nan = [
        c for c in df.columns
        if c not in ID_COLS + [TARGET, "notificacoes"] and df[c].isnull().mean() > 0.95
    ]
    if high_nan:
        print(f"  Excluindo {len(high_nan)} colunas com >95% NaN: {high_nan[:5]}...")
        df = df.drop(columns=high_nan)

    non_numeric = [
        c for c in df.columns
        if df[c].dtype == "object" or str(df[c].dtype).startswith("datetime")
    ]
    if non_numeric:
        print(f"  Excluindo {len(non_numeric)} colunas não-numéricas: {non_numeric}")
        df = df.drop(columns=non_numeric)

    all_features = [c for c in df.columns if c not in ID_COLS + [TARGET, CLASS_TARGET, "notificacoes"]]
    assert "flag_fonte_inmet" not in all_features, "flag_fonte_inmet deve ser excluído das features!"

    inmet_features = [c for c in all_features if is_inmet_feature(c)]
    sinan_features = [c for c in all_features if not is_inmet_feature(c)]

    inmet_bruto = [
        c for c in inmet_features
        if not ("_lag_" in c and c.endswith("w"))
        and "_anomalia" not in c and "_mm_2_4w" not in c and "_mm_4_8w" not in c
    ]
    inmet_lags_bio = [c for c in inmet_features if "_lag_" in c and c.endswith("w")]
    inmet_mm_bio = [c for c in inmet_features if "_mm_2_4w" in c or "_mm_4_8w" in c]
    inmet_anomalias = [c for c in inmet_features if "_anomalia" in c]

    print(f"\n  Features:")
    print(f"    SINAN: {len(sinan_features)}")
    print(f"    INMET bruto: {len(inmet_bruto)}")
    print(f"    INMET lags: {len(inmet_lags_bio)}")
    print(f"    INMET mm: {len(inmet_mm_bio)}")
    print(f"    INMET anomalias: {len(inmet_anomalias)}")
    print(f"    Total: {len(all_features)}")

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
    schema.to_csv(DATA_DIR / "feature_schema_v4.csv", index=False)

    val_years = [2023]
    test_years = [2024, 2025, 2026]

    val_df = df[df["ano"].isin(val_years)].copy()
    test_df = df[df["ano"].isin(test_years)].copy()

    report = {
        "n_features": len(all_features),
        "n_sinan_features": len(sinan_features),
        "n_inmet_total": len(inmet_features),
        "n_inmet_bruto": len(inmet_bruto),
        "n_inmet_lags_bio": len(inmet_lags_bio),
        "n_inmet_mm_bio": len(inmet_mm_bio),
        "n_inmet_anomalias": len(inmet_anomalias),
        "variantes": {},
    }

    for var_name in VARIANTES:
        var_cfg = recorte_config.load_v4(var_name)
        train_years = var_cfg["train"]
        label = var_name.replace("variante_", "").upper()

        print(f"\n--- Variante {label}: train={train_years[0]}-{train_years[-1]} ({len(train_years)} anos) ---")

        df[CLASS_TARGET] = compute_risk_class(df, train_years)

        train_df = df[df["ano"].isin(train_years)].copy()

        dup_check = set(train_df.set_index(ID_COLS).index) & set(test_df.set_index(ID_COLS).index)
        dup_check |= set(train_df.set_index(ID_COLS).index) & set(val_df.set_index(ID_COLS).index)
        dup_check |= set(val_df.set_index(ID_COLS).index) & set(test_df.set_index(ID_COLS).index)
        assert len(dup_check) == 0, f"Variante {label}: data leakage! {len(dup_check)} linhas"

        val_out = val_df.copy()
        test_out = test_df.copy()
        val_out[CLASS_TARGET] = compute_risk_class(df, train_years)
        test_out[CLASS_TARGET] = compute_risk_class(df, train_years)
        val_out = val_out[val_out.index.isin(df.index)]
        test_out = test_out[test_out.index.isin(df.index)]

        var_dir = DATA_DIR / f"splits_{label.lower()}"
        var_dir.mkdir(parents=True, exist_ok=True)

        train_df.to_parquet(var_dir / "train.parquet", index=False)
        val_out.to_parquet(var_dir / "val.parquet", index=False)
        test_out.to_parquet(var_dir / "test.parquet", index=False)

        inmet_cov = 1 - train_df[inmet_bruto[:5]].isnull().mean().mean() if inmet_bruto else 0

        print(f"  Train: {len(train_df):,} linhas, {train_df[GROUP_KEY].nunique()} mesos, {train_df['ano'].nunique()} anos")
        print(f"  Val:   {len(val_out):,} linhas")
        print(f"  Test:  {len(test_out):,} linhas")
        print(f"  Cobertura INMET treino: {inmet_cov:.1%}")
        print(f"  Sem leakage ✓")

        report["variantes"][label] = {
            "train_years": train_years,
            "train_rows": len(train_df),
            "val_rows": len(val_out),
            "test_rows": len(test_out),
            "cobertura_inmet_train": round(float(inmet_cov), 4),
            "risk_dist_train": {
                str(i): round(float((train_df[CLASS_TARGET] == i).mean()), 4)
                for i in range(4)
            },
        }

    assert report["variantes"]["A"]["val_rows"] == report["variantes"]["B"]["val_rows"], \
        "Val size differs between variantes!"
    assert report["variantes"]["A"]["test_rows"] == report["variantes"]["B"]["test_rows"], \
        "Test size differs between variantes!"

    with open(DATA_DIR / "45_splits_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  RESUMO v4-45:")
    print(f"  Features: {len(all_features)} ({len(sinan_features)} SINAN + {len(inmet_features)} INMET)")
    for var_name in VARIANTES:
        label = var_name.replace("variante_", "").upper()
        v = report["variantes"][label]
        print(f"  Variante {label}: train={v['train_rows']:,} | val={v['val_rows']:,} | test={v['test_rows']:,}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
