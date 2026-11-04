#!/usr/bin/env python3
"""v4-40: Agregação SINAN Gold por mesorregião-semana — série completa (2000-2026).

Baseado em scripts/21_aggregate_sinan_mesorregiao.py (v3).
Diferenças:
  - Lê anos_pipeline do experiment_v4 variante_b (2000-2026).
  - Exclui 1999 (praticamente vazio: 8 municípios, 13 notificações).
  - Saída em data/model_ready_v4/.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import recorte_config

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "model_ready_v4"
DATA_V3 = BASE_DIR / "data" / "model_ready_v3"
SINAN_GOLD = BASE_DIR / "data" / "sinan" / "gold" / "sinan_tcc2_v2" / "official_dense"

VALID_YEARS = recorte_config.load_v4("variante_b")["anos_pipeline"]

GROUP_KEYS = ["cod_mesorregiao", "ano", "semana_epidemiologica"]

COLS_SOMA = [
    "notificacoes",
    "qt_hospitalizados", "qt_obitos_agravo", "qt_obitos_outras_causas",
    "qt_confirmados_provaveis", "qt_descartados", "qt_inconclusivos",
    "qt_dengue_alarme", "qt_dengue_grave", "qt_chikungunya",
]

COLS_MEDIA_PONDERADA_EXPLICITA = [
    "idade_media_anos", "atraso_notificacao_medio_dias",
]

COLS_SAZONALIDADE = ["week_of_year_sin", "week_of_year_cos"]

COLS_EXCLUIR_PATTERNS = [
    "notificacoes_lag_", "notificacoes_media_movel_", "notificacoes_min_movel_",
    "notificacoes_max_movel_", "notificacoes_diff_", "notificacoes_pct_change_",
    "notificacoes_aceleracao_", "notificacoes_razao_media_", "label_",
]

COLS_DROP = [
    "ano_semana", "week_start", "municipio", "uf", "regiao",
    "source_year", "municipio_resolution", "municipio_source_field", "year",
]


def should_exclude(col):
    return any(col.startswith(p) for p in COLS_EXCLUIR_PATTERNS)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  v4-40: Agregação SINAN Full por Mesorregião-Semana")
    print("=" * 60)

    print("\nCarregando mapeamento município→mesorregião...")
    mun_meso = pd.read_csv(DATA_V3 / "municipio_mesorregiao.csv",
                           dtype={"ibge_municipio": str})
    print(f"  {len(mun_meso)} municípios → {mun_meso['cod_mesorregiao'].nunique()} mesorregiões")

    print(f"\nCarregando SINAN Gold (anos {VALID_YEARS[0]}-{VALID_YEARS[-1]})...")
    sinan = pq.read_table(
        SINAN_GOLD,
        filters=[("year", "in", VALID_YEARS)]
    ).to_pandas()
    sinan["ibge_municipio"] = sinan["ibge_municipio"].astype(str)
    total_notif_gold = sinan.groupby("ano")["notificacoes"].sum()
    print(f"  {len(sinan):,} linhas, {sinan['ano'].nunique()} anos, {sinan['ibge_municipio'].nunique()} municípios")

    drop_cols = [c for c in sinan.columns if should_exclude(c) or c in COLS_DROP]
    sinan = sinan.drop(columns=drop_cols, errors="ignore")
    print(f"  Excluídas {len(drop_cols)} colunas pré-calculadas/metadados")

    print("\nMerge com mapeamento mesorregional...")
    n_before = len(sinan)
    sinan = sinan.merge(
        mun_meso[["ibge_municipio", "cod_mesorregiao", "nome_mesorregiao",
                   "uf_sigla", "nome_grande_regiao"]],
        on="ibge_municipio", how="inner"
    )
    n_dropped = n_before - len(sinan)
    if n_dropped > 0:
        print(f"  AVISO: {n_dropped} linhas sem mesorregião descartadas no merge inner")
    print(f"  Após merge: {len(sinan):,} linhas")

    cols_prop = [c for c in sinan.columns if c.startswith("prop_")]
    cols_indice = [c for c in sinan.columns if c.startswith("indice_")]
    cols_weighted = COLS_MEDIA_PONDERADA_EXPLICITA + cols_prop + cols_indice

    print(f"\nAgregando {len(COLS_SOMA)} colunas SUM + {len(cols_weighted)} colunas MÉDIA PONDERADA...")

    for col in cols_weighted:
        sinan[f"_w_{col}"] = sinan[col] * sinan["notificacoes"]
    weighted_cols = [f"_w_{c}" for c in cols_weighted]

    grouped = sinan.groupby(GROUP_KEYS)

    agg_soma = grouped[COLS_SOMA].sum()
    agg_weighted = grouped[weighted_cols].sum()
    agg_sazon = grouped[COLS_SAZONALIDADE].first()
    agg_meta = grouped.agg(
        n_municipios=("ibge_municipio", "nunique"),
        nome_mesorregiao=("nome_mesorregiao", "first"),
        uf_sigla=("uf_sigla", "first"),
        nome_grande_regiao=("nome_grande_regiao", "first"),
    )

    weight_sums = agg_soma["notificacoes"]

    result = agg_meta.join(agg_soma).join(agg_sazon)

    for col in cols_weighted:
        safe_weights = weight_sums.replace(0, np.nan)
        result[col] = agg_weighted[f"_w_{col}"] / safe_weights
        result[col] = result[col].fillna(0.0)

    result["is_zero_notification_week"] = (result["notificacoes"] == 0).astype(int)
    result = result.reset_index()

    print(f"\n  Resultado: {len(result):,} linhas, {result['cod_mesorregiao'].nunique()} mesorregiões")
    print(f"  Anos: {sorted(result['ano'].unique())}")
    print(f"  Colunas: {len(result.columns)}")

    dups = result.duplicated(subset=GROUP_KEYS).sum()
    assert dups == 0, f"Duplicatas encontradas: {dups}"
    assert result["cod_mesorregiao"].nunique() == 137, (
        f"Esperadas 137 mesorregiões, encontradas {result['cod_mesorregiao'].nunique()}")

    print("\nValidando totais por ano...")
    municipios_descartados_por_ano = {}
    for year in VALID_YEARS:
        meso_total = result[result["ano"] == year]["notificacoes"].sum()
        gold_total = total_notif_gold.get(year, 0)
        if gold_total > 0:
            pct_diff = abs(meso_total - gold_total) / gold_total * 100
            status = "OK" if pct_diff < 5 else "AVISO"
            print(f"  {year}: Gold={gold_total:,.0f}, Meso={meso_total:,.0f}, diff={pct_diff:.1f}% [{status}]")

    out_path = DATA_DIR / "sinan_mesorregiao_full.parquet"
    result.to_parquet(out_path, index=False)
    print(f"\n  Salvo: {out_path.name}")

    report = {
        "linhas": len(result),
        "mesorregioes": int(result["cod_mesorregiao"].nunique()),
        "colunas": len(result.columns),
        "anos": sorted(int(y) for y in result["ano"].unique()),
        "n_anos": int(result["ano"].nunique()),
        "colunas_soma": len(COLS_SOMA),
        "colunas_media_ponderada": len(cols_weighted),
        "notificacoes_por_ano": {
            int(y): float(result[result["ano"] == y]["notificacoes"].sum())
            for y in sorted(result["ano"].unique())
        },
    }
    with open(DATA_DIR / "40_sinan_meso_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  RESUMO v4-40: {len(result):,} linhas mesorregião-semana")
    print(f"  {result['ano'].nunique()} anos, {result['cod_mesorregiao'].nunique()} mesorregiões")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
