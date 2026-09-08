#!/usr/bin/env python3
"""US-202: Agregação INMET Silver por mesorregião-semana.

Média das estações INMET dentro de cada mesorregião para 12 variáveis climáticas.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "model_ready_v3"
SILVER_DIR = BASE_DIR / "data" / "inmet" / "silver"

CLIMATE_FEATURES = [
    "rain_sum_mm", "rain_mean_mm", "rain_days", "rain_heavy_days",
    "temp_mean_c", "temp_min_c", "temp_max_c", "temp_range_c",
    "humidity_mean_pct", "pressure_mean_mbar", "wind_speed_mean_ms",
    "radiation_mean_kj",
]

GROUP_KEYS = ["cod_mesorregiao", "ano", "semana_epidemiologica"]


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  US-202: Agregação INMET por Mesorregião-Semana")
    print("=" * 60)

    print("\nCarregando mapeamento estação→mesorregião...")
    station_meso = pd.read_csv(DATA_DIR / "estacao_mesorregiao.csv",
                               dtype={"codigo_wmo": str})
    print(f"  {len(station_meso)} estações mapeadas")

    print("\nCarregando INMET Silver (todos os anos)...")
    frames = []
    for f in sorted(SILVER_DIR.glob("weekly_stations_*.parquet")):
        frames.append(pd.read_parquet(f))
    inmet = pd.concat(frames, ignore_index=True)
    inmet["codigo_wmo"] = inmet["codigo_wmo"].astype(str)
    print(f"  {len(inmet):,} registros estação-semana, {inmet['codigo_wmo'].nunique()} estações")

    print("\nMerge com mapeamento mesorregional...")
    inmet = inmet.merge(
        station_meso[["codigo_wmo", "cod_mesorregiao"]],
        on="codigo_wmo", how="inner"
    )
    inmet = inmet.rename(columns={"ano_epi": "ano"})
    print(f"  Após merge: {len(inmet):,} registros")

    print("\nAgregando por mesorregião-semana (MEAN)...")
    agg_dict = {feat: ("mean") for feat in CLIMATE_FEATURES}
    agg_dict["n_estacoes"] = ("codigo_wmo", "nunique")

    grouped = inmet.groupby(GROUP_KEYS)
    agg_climate = grouped[CLIMATE_FEATURES].mean()
    agg_n = grouped["codigo_wmo"].nunique().rename("n_estacoes")

    result = agg_climate.join(agg_n).reset_index()
    result["cod_mesorregiao"] = result["cod_mesorregiao"].astype(int)

    print(f"\n  Resultado: {len(result):,} linhas, {result['cod_mesorregiao'].nunique()} mesorregiões")

    coverage = {}
    for feat in CLIMATE_FEATURES:
        pct = result[feat].notna().mean() * 100
        coverage[feat] = round(pct, 1)
        print(f"  {feat:30s} {pct:5.1f}% não-nulo")

    result.to_parquet(DATA_DIR / "inmet_mesorregiao_weekly.parquet", index=False)
    print(f"\n  Salvo: inmet_mesorregiao_weekly.parquet")

    report = {
        "registros_silver_total": len(inmet),
        "linhas_mesorregiao_semana": len(result),
        "mesorregioes_com_dados": int(result["cod_mesorregiao"].nunique()),
        "anos_disponiveis": sorted(int(y) for y in result["ano"].unique()),
        "cobertura_por_variavel": coverage,
        "n_estacoes_por_meso": {
            "min": int(result["n_estacoes"].min()),
            "max": int(result["n_estacoes"].max()),
            "media": round(float(result["n_estacoes"].mean()), 1),
        },
    }
    with open(DATA_DIR / "22_inmet_meso_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  RESUMO US-202: {len(result):,} linhas mesorregião-semana")
    print(f"  {result['cod_mesorregiao'].nunique()} mesorregiões com dados INMET")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
