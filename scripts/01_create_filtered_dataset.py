#!/usr/bin/env python3
"""US-101: Criar dataset filtrado (janela temporal + cobertura espacial)."""

import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
INTEGRATED_DIR = BASE_DIR / "data" / "integrated"
OUTPUT_DIR = BASE_DIR / "data" / "model_ready_v2"

VALID_YEARS = [2019, 2021, 2023, 2024, 2025, 2026]
CLIMATE_PROXY = "rain_sum_mm"

UF_MAP = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA",
    "16": "AP", "17": "TO", "21": "MA", "22": "PI", "23": "CE",
    "24": "RN", "25": "PB", "26": "PE", "27": "AL", "28": "SE",
    "29": "BA", "31": "MG", "32": "ES", "33": "RJ", "35": "SP",
    "41": "PR", "42": "SC", "43": "RS", "50": "MS", "51": "MT",
    "52": "GO", "53": "DF",
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  US-101: Criar Dataset Filtrado")
    print("=" * 60)

    # Load municipality filter
    mun_50km = pd.read_csv(OUTPUT_DIR / "municipios_50km.csv", dtype={"ibge_municipio": str})
    valid_mun = set(mun_50km["ibge_municipio"])
    print(f"\nMunicipios validos (<=50km): {len(valid_mun)}")

    # Load integrated dataset
    print("Carregando dataset integrado (pode demorar ~30s)...")
    src = INTEGRATED_DIR / "sinan_inmet_municipal_weekly.parquet"
    df = pd.read_parquet(src)
    df["ibge_municipio"] = df["ibge_municipio"].astype(str)
    print(f"  Original: {len(df):,} linhas, {df['ibge_municipio'].nunique()} municipios")

    # Filter municipalities
    df = df[df["ibge_municipio"].isin(valid_mun)]
    print(f"  Apos filtro espacial (<=50km): {len(df):,} linhas, {df['ibge_municipio'].nunique()} municipios")

    # Filter years
    df = df[df["ano"].isin(VALID_YEARS)]
    print(f"  Apos filtro temporal {VALID_YEARS}: {len(df):,} linhas")

    # Check coverage
    total = len(df)
    with_climate = df[CLIMATE_PROXY].notna().sum()
    pct_cov = with_climate / total * 100
    print(f"\nCobertura INMET no dataset filtrado:")
    print(f"  Total: {total:,} linhas")
    print(f"  Com dados climaticos: {with_climate:,} ({pct_cov:.1f}%)")
    print(f"  Meta >85%: {'ATINGIDA' if pct_cov > 85 else 'NAO ATINGIDA'}")

    # Coverage per year
    print(f"\nCobertura por ano:")
    cov_by_year = {}
    for year in sorted(df["ano"].unique()):
        yr = df[df["ano"] == year]
        yr_cov = yr[CLIMATE_PROXY].notna().mean() * 100
        cov_by_year[int(year)] = round(yr_cov, 1)
        n_mun = yr["ibge_municipio"].nunique()
        n_weeks = yr["semana_epidemiologica"].nunique()
        print(f"  {year}: {len(yr):,} linhas, {n_mun} mun, {n_weeks} sem, cob={yr_cov:.1f}%")

    # Distribution by UF
    df["uf"] = df["ibge_municipio"].str[:2].map(UF_MAP)
    uf_dist = df.groupby("uf").agg(
        linhas=("ibge_municipio", "count"),
        municipios=("ibge_municipio", "nunique"),
    ).sort_values("linhas", ascending=False)
    print(f"\nDistribuicao por UF (top-10):")
    print(uf_dist.head(10).to_string())

    # Notifications summary
    print(f"\nNotificacoes por ano:")
    for year in sorted(df["ano"].unique()):
        yr = df[df["ano"] == year]
        total_notif = yr["notificacoes"].sum()
        print(f"  {year}: {total_notif:,.0f} notificacoes totais")

    # Save
    df_save = df.drop(columns=["uf"])
    out_path = OUTPUT_DIR / "integrated_filtered.parquet"
    df_save.to_parquet(out_path, index=False)
    print(f"\nSalvo: {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")

    summary = {
        "source": str(src),
        "original_rows": 7665428,
        "filtered_rows": total,
        "reduction_pct": round((1 - total / 7665428) * 100, 1),
        "municipios": int(df["ibge_municipio"].nunique()),
        "years": VALID_YEARS,
        "columns": len(df_save.columns),
        "climate_coverage_pct": round(pct_cov, 1),
        "coverage_by_year": cov_by_year,
        "coverage_target_85pct": bool(pct_cov > 85),
    }
    with open(OUTPUT_DIR / "01_dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  RESUMO: {total:,} linhas ({summary['reduction_pct']}% reducao)")
    print(f"  Cobertura INMET: {pct_cov:.1f}% (meta >85%)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
