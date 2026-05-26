#!/usr/bin/env python3
"""US-005: Merge do dataset unificado município-semana (SINAN + INMET)."""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parent.parent
SINAN_GOLD = BASE_DIR / "data" / "sinan" / "gold" / "sinan_tcc2_v2" / "official_dense"
INMET_GOLD = BASE_DIR / "data" / "inmet" / "gold"
INTEGRATED_DIR = BASE_DIR / "data" / "integrated"


def load_sinan():
    print("  Carregando SINAN Gold...")
    df = pq.read_table(SINAN_GOLD).to_pandas()
    df["ibge_municipio"] = df["ibge_municipio"].astype(str)
    print(f"    {len(df):,} linhas, {df['ibge_municipio'].nunique()} municípios")
    return df


def load_inmet():
    print("  Carregando INMET Gold...")
    frames = []
    for f in sorted(INMET_GOLD.glob("weekly_municipal_climate_*.parquet")):
        frames.append(pd.read_parquet(f))
    if not frames:
        raise FileNotFoundError("Nenhum parquet INMET Gold")
    df = pd.concat(frames, ignore_index=True)
    df["ibge_municipio"] = df["ibge_municipio"].astype(str)
    df = df.rename(columns={"ano_epi": "ano"})
    print(f"    {len(df):,} linhas, {df['ibge_municipio'].nunique()} municípios")
    return df


def main():
    INTEGRATED_DIR.mkdir(parents=True, exist_ok=True)

    sinan = load_sinan()
    inmet = load_inmet()

    inmet_features = [c for c in inmet.columns if c not in [
        "ibge_municipio", "ano", "semana_epidemiologica", "codigo_wmo",
        "n_valid_hours", "low_coverage",
    ]]
    inmet_join = inmet[["ibge_municipio", "ano", "semana_epidemiologica"] + inmet_features].copy()

    # Deduplicate INMET (some municipalities may have duplicate weeks)
    inmet_join = inmet_join.drop_duplicates(
        subset=["ibge_municipio", "ano", "semana_epidemiologica"], keep="first"
    )

    print(f"\nMerge LEFT JOIN por (ibge_municipio, ano, semana_epidemiologica)...")
    merged = sinan.merge(
        inmet_join,
        on=["ibge_municipio", "ano", "semana_epidemiologica"],
        how="left",
    )

    n_total = len(merged)
    n_with_climate = merged[inmet_features[0]].notna().sum()
    n_without = n_total - n_with_climate
    print(f"  Total: {n_total:,} linhas")
    print(f"  Com dados climáticos: {n_with_climate:,} ({n_with_climate/n_total*100:.1f}%)")
    print(f"  Sem dados climáticos: {n_without:,} ({n_without/n_total*100:.1f}%)")

    # Coverage per year
    cov = merged.groupby("ano").apply(
        lambda g: pd.Series({
            "n_rows": len(g),
            "n_with_climate": g[inmet_features[0]].notna().sum(),
            "pct_coverage": g[inmet_features[0]].notna().mean() * 100,
        })
    ).reset_index()
    cov.to_csv(INTEGRATED_DIR / "coverage_by_year.csv", index=False)
    print(f"\nCobertura por ano salva em {INTEGRATED_DIR / 'coverage_by_year.csv'}")

    print(f"\nSalvando Parquet integrado...")
    merged.to_parquet(INTEGRATED_DIR / "sinan_inmet_municipal_weekly.parquet", index=False)
    print(f"  → {INTEGRATED_DIR / 'sinan_inmet_municipal_weekly.parquet'}")
    print(f"  Shape: {merged.shape}")
    print(f"  Colunas: {len(merged.columns)}")


if __name__ == "__main__":
    main()
