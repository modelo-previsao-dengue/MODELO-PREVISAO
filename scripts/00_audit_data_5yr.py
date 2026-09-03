#!/usr/bin/env python3
"""US-100: Auditoria de dados e filtro de municipios <=50km."""

import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
INMET_BRONZE = BASE_DIR / "data" / "inmet" / "bronze"
INMET_GOLD = BASE_DIR / "data" / "inmet" / "gold"
INTEGRATED_DIR = BASE_DIR / "data" / "integrated"
OUTPUT_DIR = BASE_DIR / "data" / "model_ready_v2"

VALID_YEARS = [2019, 2021, 2023, 2024, 2025, 2026]
MAX_DISTANCE_KM = 50

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
    print("  US-100: Auditoria de Dados e Filtro de Municipios")
    print("=" * 60)

    # 1. Load municipality-station mapping
    mapping_path = INMET_BRONZE / "municipio_estacao_mapping.csv"
    mapping = pd.read_csv(mapping_path, dtype={"ibge_municipio": str, "codigo_wmo": str})
    print(f"\nMapeamento carregado: {len(mapping)} municipios")

    # 2. Filter <=50km
    mask_50km = mapping["distancia_km"] <= MAX_DISTANCE_KM
    mun_50km = mapping[mask_50km].copy()
    mun_excluded = mapping[~mask_50km]
    print(f"  Incluidos (<=50km): {len(mun_50km)} ({len(mun_50km)/len(mapping)*100:.1f}%)")
    print(f"  Excluidos (>50km):  {len(mun_excluded)} ({len(mun_excluded)/len(mapping)*100:.1f}%)")

    # 3. Add UF column
    mun_50km["uf"] = mun_50km["ibge_municipio"].str[:2].map(UF_MAP)

    # Coverage by UF
    print(f"\nCobertura <=50km por UF:")
    uf_total = mapping.copy()
    uf_total["uf"] = uf_total["ibge_municipio"].str[:2].map(UF_MAP)
    uf_stats = []
    for uf in sorted(UF_MAP.values()):
        total = len(uf_total[uf_total["uf"] == uf])
        included = len(mun_50km[mun_50km["uf"] == uf])
        pct = included / total * 100 if total > 0 else 0
        uf_stats.append({"uf": uf, "total": total, "included_50km": included, "pct": round(pct, 1)})
        print(f"  {uf}: {included}/{total} ({pct:.1f}%)")

    # 4. Check INMET Gold years
    gold_files = sorted(INMET_GOLD.glob("weekly_municipal_climate_*.parquet"))
    gold_years = [int(f.stem.split("_")[-1]) for f in gold_files]
    print(f"\nINMET Gold anos disponiveis: {gold_years}")
    print(f"  2022 presente: {'SIM' if 2022 in gold_years else 'NAO — CONFIRMADO AUSENTE'}")

    valid_gold_years = [y for y in VALID_YEARS if y in gold_years]
    missing_years = [y for y in VALID_YEARS if y not in gold_years]
    print(f"  Anos validos para experimento: {valid_gold_years}")
    if missing_years:
        print(f"  Anos faltantes: {missing_years}")

    # 5. Check coverage from existing CSV
    cov_path = INTEGRATED_DIR / "coverage_by_year.csv"
    cov = pd.read_csv(cov_path)
    cov_valid = cov[cov["ano"].isin(VALID_YEARS)]
    print(f"\nCobertura INMET nos anos validos (dataset integrado original):")
    for _, row in cov_valid.iterrows():
        print(f"  {int(row['ano'])}: {row['pct_coverage']:.1f}%")

    # 6. Estimate filtered dataset volume
    n_mun = len(mun_50km)
    est_rows = n_mun * 52 * len(VALID_YEARS)
    print(f"\nEstimativa de volume:")
    print(f"  {n_mun} municipios x 52 semanas x {len(VALID_YEARS)} anos ≈ {est_rows:,} linhas")

    # 7. Save outputs
    save_cols = ["ibge_municipio", "codigo_wmo", "distancia_km", "uf"]
    mun_50km[save_cols].to_csv(OUTPUT_DIR / "municipios_50km.csv", index=False)
    print(f"\nSalvo: {OUTPUT_DIR / 'municipios_50km.csv'} ({len(mun_50km)} municipios)")

    audit = {
        "total_municipios": len(mapping),
        "municipios_50km": len(mun_50km),
        "municipios_excluidos": len(mun_excluded),
        "pct_incluidos": round(len(mun_50km) / len(mapping) * 100, 1),
        "max_distance_km": MAX_DISTANCE_KM,
        "inmet_gold_years_available": gold_years,
        "valid_years_for_experiment": valid_gold_years,
        "missing_years": missing_years,
        "uf_coverage": uf_stats,
        "estimated_filtered_rows": est_rows,
        "coverage_by_valid_year": {
            int(row["ano"]): round(row["pct_coverage"], 1)
            for _, row in cov_valid.iterrows()
        },
    }
    with open(OUTPUT_DIR / "00_data_audit.json", "w") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    print(f"Salvo: {OUTPUT_DIR / '00_data_audit.json'}")

    print(f"\n{'=' * 60}")
    print(f"  RESUMO: {len(mun_50km)} municipios, {len(valid_gold_years)} anos validos")
    print(f"  Volume estimado: ~{est_rows:,} linhas")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
