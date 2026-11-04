#!/usr/bin/env python3
"""v4-43: Feature Engineering — Lags biológicos mesorregionais (série completa).

Baseado em scripts/24_feature_eng_lags_v3.py.
Diferenças:
  - Lê/grava de data/model_ready_v4/.
  - Salva como integrated_mesorregiao_v4.parquet (não sobrescreve o .parquet de entrada).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "model_ready_v4"

GROUP_KEY = "cod_mesorregiao"

CLIMATE_BASE = [
    "rain_sum_mm", "rain_mean_mm", "rain_days", "rain_heavy_days",
    "temp_mean_c", "temp_min_c", "temp_max_c", "temp_range_c",
    "humidity_mean_pct", "pressure_mean_mbar", "wind_speed_mean_ms",
    "radiation_mean_kj",
]

LAGS = list(range(1, 9))


def compute_sinan_lags(df):
    print("\nCalculando lags SINAN (notificacoes)...")
    grouped = df.groupby(GROUP_KEY)["notificacoes"]
    lag_cols = {}
    for lag in LAGS:
        lag_cols[f"notificacoes_lag_{lag}w"] = grouped.shift(lag)
    lag_df = pd.DataFrame(lag_cols, index=df.index)

    new_cols = dict(lag_cols)
    cols_2_4 = [f"notificacoes_lag_{i}w" for i in [2, 3, 4]]
    cols_4_8 = [f"notificacoes_lag_{i}w" for i in [4, 5, 6, 7, 8]]
    new_cols["notificacoes_media_movel_2_4w"] = lag_df[cols_2_4].mean(axis=1)
    new_cols["notificacoes_media_movel_4_8w"] = lag_df[cols_4_8].mean(axis=1)

    lag1 = lag_df["notificacoes_lag_1w"]
    new_cols["notificacoes_diff_1w"] = df["notificacoes"] - lag1
    new_cols["notificacoes_pct_change_1w"] = (df["notificacoes"] - lag1) / lag1.replace(0, np.nan)

    new_df = pd.DataFrame(new_cols, index=df.index)
    print(f"  Criadas {len(new_cols)} features de lag SINAN")
    return pd.concat([df, new_df], axis=1)


def compute_inmet_lags(df):
    print("\nCalculando lags 1-8 semanas para 12 variáveis climáticas...")
    grouped = df.groupby(GROUP_KEY)
    new_cols = {}
    for feat in CLIMATE_BASE:
        if feat not in df.columns:
            print(f"  AVISO: {feat} não encontrada, pulando")
            continue
        for lag in LAGS:
            col_name = f"{feat}_lag_{lag}w"
            new_cols[col_name] = grouped[feat].shift(lag)

    new_df = pd.DataFrame(new_cols, index=df.index)
    print(f"  Criadas {len(new_cols)} features de lag INMET")
    return pd.concat([df, new_df], axis=1)


def compute_bio_moving_averages(df):
    print("\nCalculando médias móveis biológicas (INMET)...")
    new_cols = {}
    for feat in CLIMATE_BASE:
        lag_2_4 = [f"{feat}_lag_{i}w" for i in [2, 3, 4]]
        lag_4_8 = [f"{feat}_lag_{i}w" for i in [4, 5, 6, 7, 8]]

        available_2_4 = [c for c in lag_2_4 if c in df.columns]
        available_4_8 = [c for c in lag_4_8 if c in df.columns]

        if available_2_4:
            new_cols[f"{feat}_mm_2_4w"] = df[available_2_4].mean(axis=1)
        if available_4_8:
            new_cols[f"{feat}_mm_4_8w"] = df[available_4_8].mean(axis=1)

    new_df = pd.DataFrame(new_cols, index=df.index)
    print(f"  Criadas {len(new_cols)} features de média móvel biológica")
    return pd.concat([df, new_df], axis=1)


def check_contiguous(df):
    anos = sorted(int(a) for a in df["ano"].unique())
    faltando = sorted(set(range(anos[0], anos[-1] + 1)) - set(anos))
    assert not faltando, f"Anos ausentes na série: {faltando}"
    semanas = df.groupby(GROUP_KEY).size()
    assert semanas.nunique() == 1, (
        f"Mesorregiões com número de semanas diferente: {semanas.value_counts().to_dict()}")
    print(f"  Série densa: anos {anos[0]}-{anos[-1]}, {semanas.iloc[0]} semanas por mesorregião")


def main():
    print("=" * 60)
    print("  v4-43: Feature Engineering — Lags Mesorregionais (Full)")
    print("=" * 60)

    print("\nCarregando dataset integrado full...")
    df = pd.read_parquet(DATA_DIR / "integrated_mesorregiao_full.parquet")
    df = df.sort_values([GROUP_KEY, "ano", "semana_epidemiologica"]).reset_index(drop=True)
    print(f"  {len(df):,} linhas, {df[GROUP_KEY].nunique():,} mesorregiões")
    print(f"  Colunas iniciais: {len(df.columns)}")
    check_contiguous(df)

    n_before = len(df.columns)
    df = compute_sinan_lags(df)
    n_sinan = len(df.columns) - n_before

    n_before_inmet = len(df.columns)
    df = compute_inmet_lags(df)
    n_inmet_lag = len(df.columns) - n_before_inmet

    n_before_mm = len(df.columns)
    df = compute_bio_moving_averages(df)
    n_inmet_mm = len(df.columns) - n_before_mm

    n_after = len(df.columns)
    n_new = n_after - n_before
    print(f"\n  Colunas: {n_before} → {n_after} (+{n_new} novas)")

    print("\nSalvando dataset com lags...")
    df.to_parquet(DATA_DIR / "integrated_mesorregiao_v4_lags.parquet", index=False)
    print(f"  Salvo: integrated_mesorregiao_v4_lags.parquet ({len(df):,} linhas, {len(df.columns)} colunas)")

    report = {
        "n_sinan_lag_features": n_sinan,
        "n_inmet_lag_features": n_inmet_lag,
        "n_inmet_mm_features": n_inmet_mm,
        "n_total_new": n_new,
        "dataset_rows": len(df),
        "dataset_cols": len(df.columns),
    }
    with open(DATA_DIR / "43_lags_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  RESUMO v4-43:")
    print(f"  Novas features SINAN: {n_sinan}")
    print(f"  Novas features INMET lag: {n_inmet_lag}")
    print(f"  Novas features INMET mm bio: {n_inmet_mm}")
    print(f"  Total: {n_new} novas → {len(df.columns)} colunas")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
