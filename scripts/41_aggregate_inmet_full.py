#!/usr/bin/env python3
"""v4-41: Agregação INMET Silver por mesorregião-semana — todos os anos disponíveis.

Baseado em scripts/22_aggregate_inmet_mesorregiao.py (v3).
Diferenças:
  - Saída em data/model_ready_v4/.
  - Loga explicitamente quais anos têm silver e quais estão ausentes.
  - Lê mapeamento estação→mesorregião de data/model_ready_v3/ (gerado pelo script 20).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import recorte_config

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "model_ready_v4"
DATA_V3 = BASE_DIR / "data" / "model_ready_v3"
SILVER_DIR = BASE_DIR / "data" / "inmet" / "silver"

VALID_YEARS = recorte_config.load_v4("variante_b")["anos_pipeline"]

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
    print("  v4-41: Agregação INMET Full por Mesorregião-Semana")
    print("=" * 60)

    print("\nCarregando mapeamento estação→mesorregião...")
    station_meso = pd.read_csv(DATA_V3 / "estacao_mesorregiao.csv",
                               dtype={"codigo_wmo": str})
    print(f"  {len(station_meso)} estações mapeadas")

    print(f"\nBuscando INMET Silver (anos esperados: {VALID_YEARS[0]}-{VALID_YEARS[-1]})...")
    available_files = sorted(SILVER_DIR.glob("weekly_stations_*.parquet"))
    available_years = []
    for f in available_files:
        year = int(f.stem.split("_")[-1])
        if year in VALID_YEARS:
            available_years.append(year)

    missing_years = sorted(set(VALID_YEARS) - set(available_years))
    if missing_years:
        print(f"  AVISO: Anos sem silver no disco: {missing_years}")
        print(f"  Esses anos terão NaN para todas as features climáticas na integração.")
    print(f"  Anos com silver: {available_years} ({len(available_years)} de {len(VALID_YEARS)})")

    frames = []
    for f in available_files:
        year = int(f.stem.split("_")[-1])
        if year in VALID_YEARS:
            frames.append(pd.read_parquet(f))

    if not frames:
        raise RuntimeError("Nenhum INMET Silver encontrado para os anos do pipeline!")

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
    grouped = inmet.groupby(GROUP_KEYS)
    agg_climate = grouped[CLIMATE_FEATURES].mean()
    agg_n = grouped["codigo_wmo"].nunique().rename("n_estacoes")

    result = agg_climate.join(agg_n).reset_index()
    result["cod_mesorregiao"] = result["cod_mesorregiao"].astype(int)

    print(f"\n  Resultado: {len(result):,} linhas, {result['cod_mesorregiao'].nunique()} mesorregiões")
    print(f"  Anos: {sorted(result['ano'].unique())}")

    cov_by_year = {}
    for year in sorted(result["ano"].unique()):
        yr = result[result["ano"] == year]
        n_meso = yr["cod_mesorregiao"].nunique()
        n_weeks = yr["semana_epidemiologica"].nunique()
        cov_by_year[int(year)] = {
            "mesorregioes": n_meso,
            "pct_meso": round(n_meso / 137 * 100, 1),
            "semanas": n_weeks,
        }
        print(f"  {year}: {n_meso} mesos ({n_meso/137*100:.0f}%), {n_weeks} semanas")

    coverage = {}
    for feat in CLIMATE_FEATURES:
        pct = result[feat].notna().mean() * 100
        coverage[feat] = round(pct, 1)

    out_path = DATA_DIR / "inmet_mesorregiao_full.parquet"
    result.to_parquet(out_path, index=False)
    print(f"\n  Salvo: {out_path.name}")

    report = {
        "registros_silver_total": len(inmet),
        "linhas_mesorregiao_semana": len(result),
        "mesorregioes_com_dados": int(result["cod_mesorregiao"].nunique()),
        "anos_disponiveis": sorted(int(y) for y in result["ano"].unique()),
        "anos_sem_silver": missing_years,
        "cobertura_por_ano": cov_by_year,
        "cobertura_por_variavel": coverage,
    }
    with open(DATA_DIR / "41_inmet_meso_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  RESUMO v4-41: {len(result):,} linhas mesorregião-semana")
    print(f"  {result['cod_mesorregiao'].nunique()} mesorregiões, {result['ano'].nunique()} anos")
    if missing_years:
        print(f"  Anos sem dados: {missing_years}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
