#!/usr/bin/env python3
"""US-004: Merge do dataset unificado municipio-semana (SINAN + INMET).

Mudancas em relacao a versao anterior:

- A janela e parametrizada e vem de config/recorte.json. Processa 2018-2024
  (warm-up para os lags de 12 semanas e cool-down para o target t+4), nao
  2000-2026.
- A cobertura passa a ser reportada por variavel climatica e nao por uma so.
  Antes, a chuva servia de proxy e nunca era nula por causa dos zeros
  falsos, o que inflava a cobertura aparente.
- O drop_duplicates que resolvia as semanas de virada de ano saiu. Ele
  escolhia um fragmento parcial; a agregacao semanal agora entrega uma linha
  unica e completa por estacao-semana, e uma duplicata aqui passa a ser erro.

Uso:
    python3 scripts/integrate_sinan_inmet.py
    python3 scripts/integrate_sinan_inmet.py --years 2018-2024
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recorte_config

BASE_DIR = Path(__file__).resolve().parent.parent
SINAN_GOLD = BASE_DIR / "data" / "sinan" / "gold" / "sinan_tcc2_v2" / "official_dense"
INMET_GOLD = BASE_DIR / "data" / "inmet" / "gold"
INTEGRATED_DIR = BASE_DIR / "data" / "integrated"
REPORT_DIR = BASE_DIR / "docs"

JOIN_KEY = ["ibge_municipio", "ano", "semana_epidemiologica"]

NON_FEATURE_INMET = [
    "codigo_wmo", "n_valid_hours", "n_valid_rain_hours", "n_valid_temp_hours",
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


def load_sinan(years):
    print("  Carregando SINAN Gold...")
    frames = []
    for year in years:
        part = SINAN_GOLD / f"year={year}"
        if part.exists():
            frames.append(pq.read_table(part).to_pandas())
    if not frames:
        raise FileNotFoundError(f"Nenhuma particao SINAN para {years}")
    df = pd.concat(frames, ignore_index=True)
    df["ibge_municipio"] = df["ibge_municipio"].astype(str)
    print(f"    {len(df):,} linhas, {df['ibge_municipio'].nunique()} municipios, "
          f"anos {sorted(df['ano'].unique().tolist())}")
    return df


def load_inmet(years):
    print("  Carregando INMET Gold...")
    frames = []
    for year in years:
        path = INMET_GOLD / f"weekly_municipal_climate_{year}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError(f"Nenhum parquet INMET Gold para {years}")
    df = pd.concat(frames, ignore_index=True)
    df["ibge_municipio"] = df["ibge_municipio"].astype(str)
    df = df.rename(columns={"ano_epi": "ano"})
    print(f"    {len(df):,} linhas, {df['ibge_municipio'].nunique()} municipios")
    return df


def coverage_report(merged, cfg):
    """Cobertura por variavel e por ano, mais a das tres exigidas juntas."""
    vars_cov = [v for v in cfg["variaveis_cobertura"] if v in merged.columns]
    completo = merged[vars_cov].notna().all(axis=1)

    rows = []
    for ano, g in merged.groupby("ano"):
        row = {"ano": int(ano), "n_linhas": len(g),
               "n_municipios": int(g["ibge_municipio"].nunique())}
        for v in vars_cov:
            row[f"pct_{v}"] = round(float(g[v].notna().mean() * 100), 2)
        row["pct_completo"] = round(float(completo.loc[g.index].mean() * 100), 2)
        rows.append(row)

    por_uf = []
    for uf, g in merged.groupby("uf"):
        row = {"uf": uf, "n_linhas": len(g),
               "n_municipios": int(g["ibge_municipio"].nunique())}
        for v in vars_cov:
            row[f"pct_{v}"] = round(float(g[v].notna().mean() * 100), 2)
        row["pct_completo"] = round(float(completo.loc[g.index].mean() * 100), 2)
        por_uf.append(row)

    return pd.DataFrame(rows).sort_values("ano"), \
        pd.DataFrame(por_uf).sort_values("pct_completo", ascending=False)


def main():
    cfg = recorte_config.load()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", default=None,
                    help="Sobrescreve a janela do config, ex.: '2018-2024'")
    args = ap.parse_args()

    years = parse_years(args.years) if args.years else cfg["anos_pipeline"]
    INTEGRATED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Janela do pipeline: {years}\n")

    sinan = load_sinan(years)
    inmet = load_inmet(years)

    inmet_features = [c for c in inmet.columns
                      if c not in JOIN_KEY + NON_FEATURE_INMET]
    inmet_join = inmet[JOIN_KEY + inmet_features].copy()

    dups = int(inmet_join.duplicated(subset=JOIN_KEY).sum())
    if dups:
        raise AssertionError(
            f"{dups} chaves duplicadas no INMET Gold. A agregacao semanal deve "
            "entregar uma linha por municipio-semana."
        )

    print(f"\nMerge LEFT JOIN por {JOIN_KEY}...")
    merged = sinan.merge(inmet_join, on=JOIN_KEY, how="left")
    print(f"  {len(merged):,} linhas, {len(merged.columns)} colunas")

    por_ano, por_uf = coverage_report(merged, cfg)
    por_ano.to_csv(INTEGRATED_DIR / "coverage_by_year.csv", index=False)
    por_uf.to_csv(INTEGRATED_DIR / "coverage_by_uf.csv", index=False)

    print("\nCobertura por ano:")
    print(por_ano.to_string(index=False))
    print("\nCobertura por UF (top 12):")
    print(por_uf.head(12).to_string(index=False))

    out = INTEGRATED_DIR / "sinan_inmet_municipal_weekly.parquet"
    merged.to_parquet(out, index=False)
    print(f"\n-> {out}")
    print(f"   Shape: {merged.shape}")


if __name__ == "__main__":
    main()
