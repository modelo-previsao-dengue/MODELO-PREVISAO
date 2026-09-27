#!/usr/bin/env python3
"""US-001: Agregacao semanal INMET por estacao (semana epidemiologica).

Duas correcoes em relacao a versao anterior:

1. Semanas sem observacao de chuva ficam NaN em vez de 0.0. Antes, a soma de
   um grupo vazio gravava 0.0 e o XGBoost aprendia "nao choveu" onde na
   verdade nao houve medicao.

2. A semana epidemiologica de virada de ano deixa de ser contada duas vezes.
   Ela cai parte em um arquivo anual e parte no seguinte, e antes cada metade
   virava uma linha propria; o merge a jusante ficava com uma delas. A
   agregacao agora e feita em dois estagios: parciais por ano (somas e
   contagens) e uma combinacao final pela chave estacao-semana.

Uso:
    python3 scripts/inmet_weekly_aggregate.py --years 2018-2024
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parent.parent
HOURLY_DIR = BASE_DIR / "data" / "inmet" / "bronze" / "hourly"
SILVER_DIR = BASE_DIR / "data" / "inmet" / "silver"
REPORT_DIR = BASE_DIR / "data" / "inmet" / "bronze" / "reports"

KEY = ["codigo_wmo", "ano_epi", "semana_epidemiologica"]

TOTAL_HOURS_PER_WEEK = 7 * 24

# Feature semanal -> coluna horaria de origem. Todas viram media da semana.
MEAN_VARS = {
    "temp_mean_c": "temp_inst_c",
    "humidity_mean_pct": "umidade_inst_pct",
    "dewpoint_mean_c": "temp_orvalho_c",
    "pressure_mean_mbar": "pressao_mbar",
    "wind_speed_mean_ms": "vento_vel_ms",
    "wind_gust_mean_ms": "vento_rajada_ms",
    "radiation_mean_kj": "radiacao_kj_m2",
}
MIN_VARS = {"temp_min_c": "temp_min_c"}
MAX_VARS = {"temp_max_c": "temp_max_c"}

RAIN_COL = "precipitacao_mm"

# Contadores de horas validas expostos no Silver, para que a cobertura possa
# ser medida por variavel e nao por uma so (US-004).
HOUR_COUNTERS = {
    "n_valid_hours": "temp_mean_c",
    "n_valid_temp_hours": "temp_mean_c",
    "n_valid_humidity_hours": "humidity_mean_pct",
}


def epiweek_sunday_start(dates):
    """(ano_epi, semana_epi) com a semana comecando no domingo, vetorizado."""
    dow = dates.dt.weekday  # 0=segunda ... 6=domingo
    sunday = dates - pd.to_timedelta((dow + 1) % 7, unit="D")
    day_of_year = (sunday - pd.to_datetime(sunday.dt.year.astype(str) + "-01-01")).dt.days
    week = (day_of_year // 7 + 1).clip(upper=53)
    return sunday.dt.year, week


def partial_aggregates(year):
    """Parciais de um arquivo anual: somas, contagens, minimos e maximos.

    Somas e contagens (em vez de medias) porque as semanas de virada de ano
    precisam ser combinadas depois com as do arquivo vizinho.
    """
    year_dir = HOURLY_DIR / f"year={year}"
    if not year_dir.exists():
        return None, None

    df = pq.read_table(year_dir).to_pandas()
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df = df.dropna(subset=["data"])
    if df.empty:
        return None, None

    df["ano_epi"], df["semana_epidemiologica"] = epiweek_sunday_start(df["data"])

    used_cols = list(MEAN_VARS.values()) + list(MIN_VARS.values()) + \
        list(MAX_VARS.values()) + [RAIN_COL]
    for col in set(used_cols):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    named = {}
    for out, src in MEAN_VARS.items():
        named[f"{out}__sum"] = (src, "sum")
        named[f"{out}__n"] = (src, "count")
    for out, src in MIN_VARS.items():
        named[f"{out}__min"] = (src, "min")
    for out, src in MAX_VARS.items():
        named[f"{out}__max"] = (src, "max")
    named["rain__sum"] = (RAIN_COL, "sum")
    named["rain__n"] = (RAIN_COL, "count")

    weekly = df.groupby(KEY, observed=True).agg(**named).reset_index()

    # Chuva diaria, para rain_days / rain_heavy_days. Um dia do calendario
    # pertence a um unico arquivo anual, entao os totais diarios ja saem
    # completos aqui; so a atribuicao a semana e que cruza o ano.
    df["dia"] = df["data"].dt.normalize()
    daily = df.groupby(KEY + ["dia"], observed=True).agg(
        dia_rain_sum=(RAIN_COL, "sum"),
        dia_rain_n=(RAIN_COL, "count"),
    ).reset_index()

    return weekly, daily


def combine_partials(weekly_parts, daily_parts):
    """Junta os fragmentos das semanas de virada de ano em uma linha so."""
    weekly = pd.concat(weekly_parts, ignore_index=True)
    sum_cols = [c for c in weekly.columns if c.endswith(("__sum", "__n"))]
    min_cols = [c for c in weekly.columns if c.endswith("__min")]
    max_cols = [c for c in weekly.columns if c.endswith("__max")]

    how = {c: "sum" for c in sum_cols}
    how.update({c: "min" for c in min_cols})
    how.update({c: "max" for c in max_cols})
    weekly = weekly.groupby(KEY, observed=True).agg(how).reset_index()

    daily = pd.concat(daily_parts, ignore_index=True)
    daily = daily.groupby(KEY + ["dia"], observed=True).agg(
        dia_rain_sum=("dia_rain_sum", "sum"),
        dia_rain_n=("dia_rain_n", "sum"),
    ).reset_index()

    observed = daily["dia_rain_n"] > 0
    daily["is_rain_day"] = (observed & (daily["dia_rain_sum"] > 0)).astype(int)
    daily["is_heavy_day"] = (observed & (daily["dia_rain_sum"] >= 10)).astype(int)
    daily["is_observed"] = observed.astype(int)

    rain_counts = daily.groupby(KEY, observed=True).agg(
        rain_days=("is_rain_day", "sum"),
        rain_heavy_days=("is_heavy_day", "sum"),
        n_dias_com_chuva_observada=("is_observed", "sum"),
    ).reset_index()

    return weekly.merge(rain_counts, on=KEY, how="left")


def finalize(agg):
    """Converte somas/contagens em medias e aplica a mascara de ausencia."""
    for out in MEAN_VARS:
        n = agg[f"{out}__n"]
        agg[out] = np.where(n > 0, agg[f"{out}__sum"] / n.replace(0, np.nan), np.nan)

    for out in MIN_VARS:
        agg[out] = agg[f"{out}__min"]
    for out in MAX_VARS:
        agg[out] = agg[f"{out}__max"]

    # US-001: sem nenhuma hora valida de precipitacao, a semana e ausente,
    # nao seca. Vale para o acumulado e para as contagens de dias.
    sem_chuva_observada = agg["rain__n"] == 0
    agg["rain_sum_mm"] = np.where(sem_chuva_observada, np.nan, agg["rain__sum"])
    agg["rain_mean_mm"] = np.where(
        sem_chuva_observada, np.nan,
        agg["rain__sum"] / agg["rain__n"].replace(0, np.nan),
    )

    sem_dia_observado = agg["n_dias_com_chuva_observada"].fillna(0) == 0
    agg["rain_days"] = np.where(sem_dia_observado, np.nan, agg["rain_days"])
    agg["rain_heavy_days"] = np.where(sem_dia_observado, np.nan, agg["rain_heavy_days"])

    agg["n_valid_rain_hours"] = agg["rain__n"].astype("int64")
    for out, src in HOUR_COUNTERS.items():
        agg[out] = agg[f"{src}__n"].astype("int64")

    agg["temp_range_c"] = agg["temp_max_c"] - agg["temp_min_c"]
    agg["low_coverage"] = agg["n_valid_hours"] < (TOTAL_HOURS_PER_WEEK * 0.5)

    final_cols = KEY + [
        "rain_sum_mm", "rain_mean_mm", "rain_days", "rain_heavy_days",
        "temp_mean_c", "temp_min_c", "temp_max_c", "temp_range_c",
        "humidity_mean_pct", "dewpoint_mean_c", "pressure_mean_mbar",
        "wind_speed_mean_ms", "wind_gust_mean_ms", "radiation_mean_kj",
        "n_valid_hours", "n_valid_rain_hours", "n_valid_temp_hours",
        "n_valid_humidity_hours", "n_dias_com_chuva_observada", "low_coverage",
    ]
    return agg[final_cols]


def parse_years(spec):
    years = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            years.update(range(int(a), int(b) + 1))
        else:
            years.add(int(chunk))
    return sorted(years)


def regression_check(agg):
    """AC da US-001: nenhuma linha sem hora valida pode ter chuva preenchida."""
    bad_sum = int(((agg["n_valid_rain_hours"] == 0) & agg["rain_sum_mm"].notna()).sum())
    bad_days = int(((agg["n_dias_com_chuva_observada"] == 0) & agg["rain_days"].notna()).sum())
    print("\nTeste de regressao US-001:")
    print(f"  linhas com n_valid_rain_hours==0 e rain_sum_mm preenchido: {bad_sum}")
    print(f"  linhas sem dia observado e rain_days preenchido:           {bad_days}")
    if bad_sum or bad_days:
        raise AssertionError("US-001: zeros falsos de chuva ainda presentes")
    print("  OK")


def coverage_report(agg, years):
    rows = []
    for year, g in agg.groupby("ano_epi", observed=True):
        if year not in years:
            continue
        rows.append({
            "year": int(year),
            "n_stations": int(g["codigo_wmo"].nunique()),
            "n_station_weeks": int(len(g)),
            "pct_low_coverage": round(float(g["low_coverage"].mean() * 100), 2),
            "pct_chuva_ausente": round(float(g["rain_sum_mm"].isna().mean() * 100), 2),
            "pct_temp_ausente": round(float(g["temp_mean_c"].isna().mean() * 100), 2),
            "pct_umidade_ausente": round(float(g["humidity_mean_pct"].isna().mean() * 100), 2),
            "pct_tres_variaveis_completas": round(float((
                agg_complete_mask(g)).mean() * 100), 2),
        })
    return pd.DataFrame(rows).sort_values("year")


def agg_complete_mask(g):
    return (
        g["rain_sum_mm"].notna()
        & g["temp_mean_c"].notna()
        & g["humidity_mean_pct"].notna()
    )


def before_after_report(agg, years):
    """Quantas semanas-estacao deixaram de ser 0.0 e passaram a ser ausentes."""
    rows = []
    for year in years:
        old_path = SILVER_DIR / f"weekly_stations_{year}.parquet"
        g = agg[agg["ano_epi"] == year]
        row = {
            "year": year,
            "linhas_depois": int(len(g)),
            "chuva_ausente_depois": int(g["rain_sum_mm"].isna().sum()),
        }
        if old_path.exists():
            old = pd.read_parquet(old_path)
            old = old[old["ano_epi"] == year] if "ano_epi" in old.columns else old
            row["linhas_antes"] = int(len(old))
            row["chuva_ausente_antes"] = int(old["rain_sum_mm"].isna().sum())
            row["chuva_zero_antes"] = int((old["rain_sum_mm"] == 0).sum())
            if "n_valid_hours" in old.columns:
                row["zeros_falsos_antes"] = int(
                    ((old["n_valid_hours"] == 0) & (old["rain_sum_mm"].notna())).sum()
                )
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", default="2018-2024",
                    help="Anos do Bronze a agregar, ex.: '2018-2024'")
    args = ap.parse_args()

    years = parse_years(args.years)
    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Anos do Bronze a agregar: {years}\n")

    weekly_parts, daily_parts = [], []
    for year in years:
        print(f"  [{year}] Agregando parciais...")
        weekly, daily = partial_aggregates(year)
        if weekly is None:
            print("    -> Sem dados")
            continue
        weekly_parts.append(weekly)
        daily_parts.append(daily)
        print(f"    -> {len(weekly):,} parciais estacao-semana")

    if not weekly_parts:
        raise SystemExit("Nenhum ano com dados no Bronze")

    print("\nCombinando fragmentos de virada de ano...")
    n_parts = sum(len(w) for w in weekly_parts)
    agg = combine_partials(weekly_parts, daily_parts)
    print(f"  {n_parts:,} parciais -> {len(agg):,} semanas-estacao "
          f"({n_parts - len(agg):,} fragmentos unidos)")

    agg = finalize(agg)
    regression_check(agg)

    # A primeira e a ultima semana epidemiologica da janela sao incompletas:
    # parte das horas esta em arquivos fora dela. Sao margem (warm-up e
    # cool-down), nao entram no recorte entregue.
    fora = agg[~agg["ano_epi"].isin(years)]
    if len(fora):
        print(f"\n{len(fora):,} semanas com ano_epi fora da janela descartadas "
              f"(anos {sorted(fora['ano_epi'].unique().tolist())})")
    agg = agg[agg["ano_epi"].isin(years)].copy()

    before_after = before_after_report(agg, years)

    for year, ydf in agg.groupby("ano_epi", observed=True):
        ydf.to_parquet(SILVER_DIR / f"weekly_stations_{int(year)}.parquet", index=False)

    cov = coverage_report(agg, years)
    cov.to_csv(SILVER_DIR / "coverage_report.csv", index=False)
    before_after.to_csv(REPORT_DIR / "us001_chuva_antes_depois.csv", index=False)

    print("\nCobertura por ano (Silver):")
    print(cov.to_string(index=False))
    print("\nUS-001 antes/depois:")
    print(before_after.to_string(index=False))
    print(f"\nRelatorios: {SILVER_DIR / 'coverage_report.csv'}")
    print(f"            {REPORT_DIR / 'us001_chuva_antes_depois.csv'}")


if __name__ == "__main__":
    main()
