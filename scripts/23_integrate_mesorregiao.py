#!/usr/bin/env python3
"""US-203: Integração SINAN + INMET mesorregional."""

import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "model_ready_v3"

CLIMATE_FEATURES = [
    "rain_sum_mm", "rain_mean_mm", "rain_days", "rain_heavy_days",
    "temp_mean_c", "temp_min_c", "temp_max_c", "temp_range_c",
    "humidity_mean_pct", "pressure_mean_mbar", "wind_speed_mean_ms",
    "radiation_mean_kj",
]


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  US-203: Integração Mesorregional SINAN + INMET")
    print("=" * 60)

    sinan = pd.read_parquet(DATA_DIR / "sinan_mesorregiao_weekly.parquet")
    inmet = pd.read_parquet(DATA_DIR / "inmet_mesorregiao_weekly.parquet")
    print(f"  SINAN: {len(sinan):,} linhas")
    print(f"  INMET: {len(inmet):,} linhas")

    join_keys = ["cod_mesorregiao", "ano", "semana_epidemiologica"]
    inmet_join = inmet[join_keys + CLIMATE_FEATURES + ["n_estacoes"]].copy()
    inmet_join = inmet_join.drop_duplicates(subset=join_keys, keep="first")

    merged = sinan.merge(inmet_join, on=join_keys, how="left")

    n_total = len(merged)
    n_with = merged["rain_sum_mm"].notna().sum()
    pct = n_with / n_total * 100
    print(f"\n  Total: {n_total:,} linhas")
    print(f"  Com INMET: {n_with:,} ({pct:.1f}%)")

    cov_by_year = {}
    for year in sorted(merged["ano"].unique()):
        yr = merged[merged["ano"] == year]
        yr_pct = yr["rain_sum_mm"].notna().mean() * 100
        cov_by_year[int(year)] = round(yr_pct, 1)
        print(f"  {year}: {yr_pct:.1f}% cobertura")

    merged.to_parquet(DATA_DIR / "integrated_mesorregiao.parquet", index=False)
    print(f"\n  Salvo: integrated_mesorregiao.parquet ({len(merged.columns)} colunas)")

    report = {
        "linhas": n_total,
        "cobertura_inmet_pct": round(pct, 1),
        "cobertura_por_ano": cov_by_year,
        "colunas": len(merged.columns),
    }
    with open(DATA_DIR / "23_integration_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  RESUMO US-203: {n_total:,} linhas, cobertura INMET {pct:.1f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
