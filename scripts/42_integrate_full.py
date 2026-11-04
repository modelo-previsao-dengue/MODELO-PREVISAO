#!/usr/bin/env python3
"""v4-42: Integração SINAN + INMET mesorregional — série completa.

Baseado em scripts/23_integrate_mesorregiao.py (v3).
Diferenças:
  - Lê de data/model_ready_v4/ (outputs dos scripts 40 e 41).
  - Adiciona flag_fonte_inmet (0/1) como metadata.
  - Saída: integrated_mesorregiao_full.parquet.
"""

import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "model_ready_v4"

CLIMATE_FEATURES = [
    "rain_sum_mm", "rain_mean_mm", "rain_days", "rain_heavy_days",
    "temp_mean_c", "temp_min_c", "temp_max_c", "temp_range_c",
    "humidity_mean_pct", "pressure_mean_mbar", "wind_speed_mean_ms",
    "radiation_mean_kj",
]


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  v4-42: Integração Mesorregional SINAN + INMET (Full)")
    print("=" * 60)

    sinan = pd.read_parquet(DATA_DIR / "sinan_mesorregiao_full.parquet")
    inmet = pd.read_parquet(DATA_DIR / "inmet_mesorregiao_full.parquet")
    print(f"  SINAN: {len(sinan):,} linhas, anos {sorted(sinan['ano'].unique())}")
    print(f"  INMET: {len(inmet):,} linhas, anos {sorted(inmet['ano'].unique())}")

    join_keys = ["cod_mesorregiao", "ano", "semana_epidemiologica"]
    inmet_join = inmet[join_keys + CLIMATE_FEATURES + ["n_estacoes"]].copy()
    inmet_join = inmet_join.drop_duplicates(subset=join_keys, keep="first")

    merged = sinan.merge(inmet_join, on=join_keys, how="left")

    merged["flag_fonte_inmet"] = merged["rain_sum_mm"].notna().astype(int)

    n_total = len(merged)
    n_with = merged["flag_fonte_inmet"].sum()
    pct = n_with / n_total * 100
    print(f"\n  Total: {n_total:,} linhas")
    print(f"  Com INMET: {n_with:,} ({pct:.1f}%)")
    print(f"  Sem INMET: {n_total - n_with:,} ({100 - pct:.1f}%)")

    assert merged["flag_fonte_inmet"].isin([0, 1]).all(), "flag_fonte_inmet deve ser 0 ou 1"

    cov_by_year = {}
    print("\n  Cobertura INMET por ano:")
    for year in sorted(merged["ano"].unique()):
        yr = merged[merged["ano"] == year]
        yr_with = yr["flag_fonte_inmet"].sum()
        yr_pct = yr_with / len(yr) * 100
        cov_by_year[int(year)] = round(yr_pct, 1)
        print(f"    {year}: {yr_pct:5.1f}% ({yr_with:,}/{len(yr):,})")

    out_path = DATA_DIR / "integrated_mesorregiao_full.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"\n  Salvo: {out_path.name} ({len(merged.columns)} colunas)")

    report = {
        "linhas": n_total,
        "colunas": len(merged.columns),
        "cobertura_inmet_pct": round(pct, 1),
        "cobertura_por_ano": cov_by_year,
        "anos": sorted(int(y) for y in merged["ano"].unique()),
        "n_anos": int(merged["ano"].nunique()),
        "mesorregioes": int(merged["cod_mesorregiao"].nunique()),
        "flag_fonte_inmet_sum": int(n_with),
    }
    with open(DATA_DIR / "42_integration_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  RESUMO v4-42: {n_total:,} linhas, cobertura INMET {pct:.1f}%")
    print(f"  {merged['ano'].nunique()} anos, {merged['cod_mesorregiao'].nunique()} mesorregiões")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
