#!/usr/bin/env python3
"""US-205: Feature Engineering — Anomalias climáticas mesorregionais.

Para cada variável climática base, calcula:
  - anomalia = valor_observado - média_histórica(mesorregiao, semana_epidemiologica),
    com a média histórica calculada só nos anos de treino (config/recorte.json)
  - anomalia_std = anomalia / desvio_padrão histórico (z-score)
Total: 12 × 2 = 24 novas features.

Lê `integrated_mesorregiao.parquet` (já com lags do script 24) e grava
`integrated_mesorregiao_v3.parquet`.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import recorte_config

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "model_ready_v3"

GROUP_KEYS = ["cod_mesorregiao", "semana_epidemiologica"]

CLIMATE_BASE = [
    "rain_sum_mm", "rain_mean_mm", "rain_days", "rain_heavy_days",
    "temp_mean_c", "temp_min_c", "temp_max_c", "temp_range_c",
    "humidity_mean_pct", "pressure_mean_mbar", "wind_speed_mean_ms",
    "radiation_mean_kj",
]


def compute_anomalies(df, climatology_years):
    """Compute anomalies: deviation from historical mean per (mesorregiao, epi_week).

    A climatologia (media e desvio) sai so dos anos de treino. Calculada sobre
    o dataframe inteiro, como antes, ela incorporava os anos de validacao e
    teste nas proprias features.
    """
    print(f"\nCalculando anomalias climáticas (climatologia {climatology_years})...")
    base = df[df["ano"].isin(climatology_years)]
    new_cols = {}
    for feat in CLIMATE_BASE:
        if feat not in df.columns:
            print(f"  AVISO: {feat} não encontrada, pulando")
            continue

        stats = base.groupby(GROUP_KEYS)[feat].agg(["mean", "std"])
        stats.columns = ["hist_mean", "hist_std"]
        stats["hist_std"] = stats["hist_std"].replace(0, np.nan)

        merged = df[GROUP_KEYS].merge(stats, left_on=GROUP_KEYS, right_index=True, how="left")

        anomalia = df[feat] - merged["hist_mean"]
        anomalia_std = anomalia / merged["hist_std"]

        new_cols[f"{feat}_anomalia"] = anomalia
        new_cols[f"{feat}_anomalia_std"] = anomalia_std

    new_df = pd.DataFrame(new_cols, index=df.index)
    print(f"  Criadas {len(new_cols)} features de anomalia")
    return pd.concat([df, new_df], axis=1)


def main():
    print("=" * 60)
    print("  US-205: Feature Engineering — Anomalias Climáticas Mesorregionais")
    print("=" * 60)

    print("\nCarregando dataset com lags (US-204)...")
    df = pd.read_parquet(DATA_DIR / "integrated_mesorregiao.parquet")
    print(f"  {len(df):,} linhas, {len(df.columns)} colunas")

    n_before = len(df.columns)
    climatology_years = recorte_config.load_v3()["train"]
    df = compute_anomalies(df, climatology_years)
    n_after = len(df.columns)
    n_new = n_after - n_before
    print(f"\n  Colunas: {n_before} → {n_after} (+{n_new} novas)")

    print("\nSalvando dataset final v3...")
    df.to_parquet(DATA_DIR / "integrated_mesorregiao_v3.parquet", index=False)
    print(f"  Salvo: integrated_mesorregiao_v3.parquet ({len(df):,} linhas, {len(df.columns)} colunas)")

    report = {
        "n_new_features": n_new,
        "n_base_climate_features": len(CLIMATE_BASE),
        "climatologia_anos": climatology_years,
        "dataset_rows": len(df),
        "dataset_cols": len(df.columns),
    }
    with open(DATA_DIR / "25_anomalias_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  RESUMO US-205:")
    print(f"  Novas features de anomalia: {n_new}")
    print(f"  Dataset final: {len(df):,} × {len(df.columns)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
