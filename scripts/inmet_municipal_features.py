#!/usr/bin/env python3
"""US-003/US-004: Features climaticas semanais por municipio com lags.

Mudancas em relacao a versao anterior:

- Os lags vao ate 12 semanas e as medias moveis ate 12, espelhando as
  features epidemiologicas do SINAN. A auditoria mostrou o pico da
  correlacao clima-casos em lag 9-12 em 11 de 12 UFs, fora do alcance dos
  lags [1,2,4,8] anteriores.
- Lags e medias moveis sao calculados sobre uma grade semanal completa e no
  nivel da estacao, nao do municipio. Sobre a grade, shift(k) e exatamente
  k semanas atras mesmo quando a serie tem buracos; no nivel da estacao o
  custo cai de 5.571 series para ~600, com o mesmo resultado, ja que o
  mapeamento municipio-estacao e um para um.
- Umidade relativa e ponto de orvalho entram como variaveis distintas.

Uso:
    python3 scripts/inmet_municipal_features.py --years 2018-2024
"""

import argparse
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SILVER_DIR = BASE_DIR / "data" / "inmet" / "silver"
BRONZE_DIR = BASE_DIR / "data" / "inmet" / "bronze"
GOLD_DIR = BASE_DIR / "data" / "inmet" / "gold"

BASE_FEATURES = [
    "rain_sum_mm", "rain_mean_mm", "rain_days", "rain_heavy_days",
    "temp_mean_c", "temp_min_c", "temp_max_c", "temp_range_c",
    "humidity_mean_pct", "dewpoint_mean_c", "pressure_mean_mbar",
    "wind_speed_mean_ms", "wind_gust_mean_ms", "radiation_mean_kj",
]

LAG_FEATURES = ["rain_sum_mm", "temp_mean_c", "humidity_mean_pct", "dewpoint_mean_c"]
LAG_PERIODS = [1, 2, 4, 8, 12]
MM_FEATURES = LAG_FEATURES
MM_WINDOWS = [4, 8, 12]
EXTRA_LAG_FEATURES = {
    "rain_heavy_days": [2, 4],
    "temp_range_c": [2],
}

# Colunas de diagnostico do Silver: seguem para o Gold mas nao viram feature.
DIAGNOSTIC_COLS = [
    "n_valid_hours", "n_valid_rain_hours", "n_valid_temp_hours",
    "n_valid_humidity_hours", "n_dias_com_chuva_observada", "low_coverage",
]


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


def load_mapping():
    path = BRONZE_DIR / "municipio_estacao_mapping.csv"
    return pd.read_csv(path, dtype={"ibge_municipio": str, "codigo_wmo": str})


def load_silver(years):
    frames = []
    for year in years:
        path = SILVER_DIR / f"weekly_stations_{year}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError(f"Nenhum Silver para os anos {years}")
    df = pd.concat(frames, ignore_index=True)
    df["codigo_wmo"] = df["codigo_wmo"].astype(str)
    return df


def add_sunday(df):
    """Data do domingo da semana epidemiologica, usada como eixo temporal.

    Por construcao da semana (dia_do_ano // 7 + 1 sobre o domingo), o domingo
    da semana N e o primeiro domingo do ano mais (N-1) semanas.
    """
    jan1 = pd.to_datetime(df["ano_epi"].astype(str) + "-01-01")
    first_sunday = jan1 + pd.to_timedelta((6 - jan1.dt.weekday) % 7, unit="D")
    df["domingo"] = first_sunday + pd.to_timedelta(
        (df["semana_epidemiologica"] - 1) * 7, unit="D"
    )
    return df


def complete_weekly_grid(df):
    """Reindexa cada estacao para semanas contiguas entre seu inicio e fim."""
    df = df.sort_values(["codigo_wmo", "domingo"])
    spans = df.groupby("codigo_wmo")["domingo"].agg(["min", "max"]).reset_index()

    grids = []
    for wmo, lo, hi in spans.itertuples(index=False):
        weeks = pd.date_range(lo, hi, freq="7D")
        grids.append(pd.DataFrame({"codigo_wmo": wmo, "domingo": weeks}))
    grid = pd.concat(grids, ignore_index=True)

    n_before = len(df)
    out = grid.merge(df, on=["codigo_wmo", "domingo"], how="left")
    added = len(out) - n_before
    if added:
        print(f"    {added:,} semanas vazias inseridas para fechar a grade")
    return out.sort_values(["codigo_wmo", "domingo"])


def add_lags_and_rolling(df):
    grp = df.groupby("codigo_wmo", sort=False)

    new_cols = {}
    for feat in LAG_FEATURES:
        if feat not in df.columns:
            continue
        for lag in LAG_PERIODS:
            new_cols[f"{feat}_lag_{lag}"] = grp[feat].shift(lag)

    for feat in MM_FEATURES:
        if feat not in df.columns:
            continue
        for window in MM_WINDOWS:
            new_cols[f"{feat}_mm{window}"] = grp[feat].transform(
                lambda x, w=window: x.rolling(w, min_periods=max(2, w // 4)).mean()
            )

    for feat, lags in EXTRA_LAG_FEATURES.items():
        if feat not in df.columns:
            continue
        for lag in lags:
            new_cols[f"{feat}_lag{lag}"] = grp[feat].shift(lag)

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def restore_week_labels(df):
    """Recompoe ano_epi e semana a partir do domingo, para as linhas da grade."""
    jan1 = pd.to_datetime(df["domingo"].dt.year.astype(str) + "-01-01")
    df["ano_epi"] = df["domingo"].dt.year
    df["semana_epidemiologica"] = ((df["domingo"] - jan1).dt.days // 7 + 1).clip(upper=53)
    return df


def write_catalog(df, feature_cols):
    rows = []
    for col in feature_cols:
        if col in BASE_FEATURES:
            grupo = "base"
        elif "_lag" in col:
            grupo = "lag"
        elif "_mm" in col:
            grupo = "media_movel"
        else:
            grupo = "outro"
        rows.append({
            "feature": col,
            "grupo": grupo,
            "dtype": str(df[col].dtype),
            "pct_missing": round(float(df[col].isna().mean() * 100), 2),
        })
    cat = pd.DataFrame(rows).sort_values(["grupo", "feature"])
    cat.to_csv(GOLD_DIR / "inmet_feature_catalog.csv", index=False)
    return cat


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", default="2018-2024",
                    help="Anos epidemiologicos a processar, ex.: '2018-2024'")
    args = ap.parse_args()

    years = parse_years(args.years)
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Anos: {years}\n")

    print("Carregando mapeamento estacao-municipio...")
    mapping = load_mapping()
    mapping = mapping[["ibge_municipio", "codigo_wmo"]].dropna()
    print(f"  {len(mapping)} municipios mapeados para "
          f"{mapping['codigo_wmo'].nunique()} estacoes")

    print("Carregando Silver...")
    silver = add_sunday(load_silver(years))
    print(f"  {len(silver):,} semanas-estacao, {silver['codigo_wmo'].nunique()} estacoes")

    print("Fechando a grade semanal por estacao...")
    silver = complete_weekly_grid(silver)

    print(f"Calculando lags {LAG_PERIODS} e medias moveis {MM_WINDOWS}...")
    silver = add_lags_and_rolling(silver)
    silver = restore_week_labels(silver)

    print("Associando municipios as estacoes...")
    municipal = mapping.merge(silver, on="codigo_wmo", how="inner")
    print(f"  {len(municipal):,} registros municipio-semana")

    feature_cols = [c for c in municipal.columns if c not in
                    ["ibge_municipio", "codigo_wmo", "ano_epi",
                     "semana_epidemiologica", "domingo"] + DIAGNOSTIC_COLS]

    cat = write_catalog(municipal, feature_cols)
    print(f"\nCatalogo: {len(cat)} features -> {GOLD_DIR / 'inmet_feature_catalog.csv'}")
    print(cat.groupby("grupo").size().to_string())

    out_cols = ["ibge_municipio", "codigo_wmo", "ano_epi", "semana_epidemiologica"] \
        + feature_cols + DIAGNOSTIC_COLS
    out_cols = [c for c in out_cols if c in municipal.columns]

    # So os anos pedidos: a grade pode ter criado semanas de anos vizinhos.
    municipal = municipal[municipal["ano_epi"].isin(years)]

    for year, ydf in municipal.groupby("ano_epi"):
        ydf[out_cols].to_parquet(
            GOLD_DIR / f"weekly_municipal_climate_{int(year)}.parquet", index=False
        )

    print(f"\nParquets Gold: {municipal['ano_epi'].nunique()} anos")
    print(f"Total registros: {len(municipal):,}")
    print(f"Municipios com dados: {municipal['ibge_municipio'].nunique()}")


if __name__ == "__main__":
    main()
