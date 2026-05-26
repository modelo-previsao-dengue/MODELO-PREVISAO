#!/usr/bin/env python3
"""US-002: Agregação semanal INMET por estação (semana epidemiológica)."""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parent.parent
HOURLY_DIR = BASE_DIR / "data" / "inmet" / "bronze" / "hourly"
SILVER_DIR = BASE_DIR / "data" / "inmet" / "silver"


def epiweek_sunday_start(dt):
    """Retorna (ano_epi, semana_epi) com semana começando no domingo."""
    dow = dt.weekday()  # 0=Mon ... 6=Sun
    sunday = dt - pd.Timedelta(days=(dow + 1) % 7)
    day_of_year = (sunday - pd.Timestamp(sunday.year, 1, 1)).days
    week = day_of_year // 7 + 1
    return sunday.year, min(week, 53)


def aggregate_year(year):
    year_dir = HOURLY_DIR / f"year={year}"
    if not year_dir.exists():
        return None

    df = pq.read_table(year_dir).to_pandas()
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df = df.dropna(subset=["data"])

    epi = df["data"].apply(epiweek_sunday_start)
    df["ano_epi"] = epi.apply(lambda x: x[0])
    df["semana_epidemiologica"] = epi.apply(lambda x: x[1])

    numeric_cols = [
        "precipitacao_mm", "pressao_mbar", "temp_inst_c", "temp_max_c",
        "temp_min_c", "umidade_inst_pct", "vento_vel_ms", "vento_rajada_ms",
        "radiacao_kj_m2",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    total_hours_per_week = 7 * 24
    grp = df.groupby(["codigo_wmo", "ano_epi", "semana_epidemiologica"])

    agg = grp.agg(
        rain_sum_mm=("precipitacao_mm", "sum"),
        rain_mean_mm=("precipitacao_mm", "mean"),
        temp_mean_c=("temp_inst_c", "mean"),
        temp_min_c=("temp_min_c", "min"),
        temp_max_c=("temp_max_c", "max"),
        humidity_mean_pct=("umidade_inst_pct", "mean"),
        pressure_mean_mbar=("pressao_mbar", "mean"),
        wind_speed_mean_ms=("vento_vel_ms", "mean"),
        radiation_mean_kj=("radiacao_kj_m2", "mean"),
        n_valid_hours=("temp_inst_c", "count"),
    ).reset_index()

    rain_days = df.copy()
    rain_days["dia"] = rain_days["data"].dt.date
    daily_rain = rain_days.groupby(["codigo_wmo", "ano_epi", "semana_epidemiologica", "dia"])[
        "precipitacao_mm"
    ].sum().reset_index()
    rain_day_counts = daily_rain.groupby(["codigo_wmo", "ano_epi", "semana_epidemiologica"]).agg(
        rain_days=("precipitacao_mm", lambda x: (x > 0).sum()),
        rain_heavy_days=("precipitacao_mm", lambda x: (x >= 10).sum()),
    ).reset_index()

    agg = agg.merge(rain_day_counts, on=["codigo_wmo", "ano_epi", "semana_epidemiologica"], how="left")
    agg["temp_range_c"] = agg["temp_max_c"] - agg["temp_min_c"]
    agg["low_coverage"] = agg["n_valid_hours"] < (total_hours_per_week * 0.5)

    return agg


def main():
    SILVER_DIR.mkdir(parents=True, exist_ok=True)

    years = sorted([int(d.name.split("=")[1]) for d in HOURLY_DIR.iterdir() if d.is_dir()])
    print(f"Anos para agregar: {years}\n")

    coverage_rows = []
    for year in years:
        print(f"  [{year}] Agregando...")
        agg = aggregate_year(year)
        if agg is None:
            print(f"    → Sem dados")
            continue

        out = SILVER_DIR / f"weekly_stations_{year}.parquet"
        agg.to_parquet(out, index=False)

        n_stations = agg["codigo_wmo"].nunique()
        n_weeks = len(agg)
        pct_low = agg["low_coverage"].mean() * 100
        print(f"    → {n_stations} estações, {n_weeks} semanas-estação, {pct_low:.1f}% low_coverage")

        coverage_rows.append({
            "year": year, "n_stations": n_stations, "n_station_weeks": n_weeks,
            "pct_low_coverage": round(pct_low, 2),
        })

    cov = pd.DataFrame(coverage_rows)
    cov.to_csv(SILVER_DIR / "coverage_report.csv", index=False)
    print(f"\nRelatório de cobertura salvo em {SILVER_DIR / 'coverage_report.csv'}")
    print(cov.to_string(index=False))


if __name__ == "__main__":
    main()
