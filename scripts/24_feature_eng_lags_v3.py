#!/usr/bin/env python3
"""US-204: Feature Engineering — Lags biológicos mesorregionais (SINAN + INMET).

Cria lags de notificações SINAN (8 lags + 2 médias móveis + diff + pct_change)
e lags climáticos INMET (12 features × 8 lags = 96) + 24 médias móveis biológicas
(12 features × 2 janelas: mm_2_4w e mm_4_8w).

Atualiza `integrated_mesorregiao.parquet` in-place (lê e regrava no mesmo arquivo).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "model_ready_v3"

GROUP_KEY = "cod_mesorregiao"

CLIMATE_BASE = [
    "rain_sum_mm", "rain_mean_mm", "rain_days", "rain_heavy_days",
    "temp_mean_c", "temp_min_c", "temp_max_c", "temp_range_c",
    "humidity_mean_pct", "pressure_mean_mbar", "wind_speed_mean_ms",
    "radiation_mean_kj",
]

LAGS = list(range(1, 9))  # 1-8 weeks


def compute_sinan_lags(df):
    """Compute lag/moving-average/diff/pct-change features for notificacoes."""
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
    """Compute lag features for all 12 climate variables, grouped by mesorregião."""
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
    """Compute biologically motivated moving averages.

    mm_2_4w: mean of lags 2,3,4 — hatching window
    mm_4_8w: mean of lags 4,5,6,7,8 — full Aedes lifecycle window
    """
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
    """Aborta se os lags por shift() fossem cruzar uma lacuna de anos.

    shift(k) dentro da mesorregiao so equivale a "k semanas atras" se a serie
    for densa: anos contiguos e o mesmo calendario de semanas em todas as
    mesorregioes. A versao anterior tinha 2020 e 2022 ausentes, e o lag_1 da
    primeira semana de 2021 era a ultima semana de 2019.
    """
    anos = sorted(int(a) for a in df["ano"].unique())
    faltando = sorted(set(range(anos[0], anos[-1] + 1)) - set(anos))
    assert not faltando, f"Anos ausentes na serie: {faltando}"
    semanas = df.groupby(GROUP_KEY).size()
    assert semanas.nunique() == 1, (
        f"Mesorregioes com numero de semanas diferente: {semanas.value_counts().to_dict()}")
    print(f"  Serie densa: anos {anos[0]}-{anos[-1]}, {semanas.iloc[0]} semanas por mesorregiao")


def main():
    print("=" * 60)
    print("  US-204: Feature Engineering — Lags Mesorregionais")
    print("=" * 60)

    print("\nCarregando dataset integrado...")
    df = pd.read_parquet(DATA_DIR / "integrated_mesorregiao.parquet")
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

    print("\nSalvando dataset enriquecido (in-place)...")
    df.to_parquet(DATA_DIR / "integrated_mesorregiao.parquet", index=False)
    print(f"  Salvo: integrated_mesorregiao.parquet ({len(df):,} linhas, {len(df.columns)} colunas)")

    report = {
        "n_sinan_lag_features": n_sinan,
        "n_inmet_lag_features": n_inmet_lag,
        "n_inmet_mm_features": n_inmet_mm,
        "n_total_new": n_new,
        "dataset_rows": len(df),
        "dataset_cols": len(df.columns),
    }
    with open(DATA_DIR / "24_lags_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  RESUMO US-204:")
    print(f"  Novas features SINAN (lag/mm/diff/pct): {n_sinan}")
    print(f"  Novas features INMET lag: {n_inmet_lag}")
    print(f"  Novas features INMET mm bio: {n_inmet_mm}")
    print(f"  Total novas: {n_new}")
    print(f"  Dataset salvo: {len(df):,} × {len(df.columns)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
