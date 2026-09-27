#!/usr/bin/env python3
"""US-201: Agregação SINAN Gold por mesorregião-semana.

Agrega notificações (SUM) e proporções/médias (média ponderada por notificações)
no nível mesorregião-semana para todas as 137 mesorregiões.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import recorte_config

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "model_ready_v3"
SINAN_GOLD = BASE_DIR / "data" / "sinan" / "gold" / "sinan_tcc2_v2" / "official_dense"

# Anos contiguos (warm-up incluso) vindos de config/recorte.json. A lista fixa
# anterior pulava 2020 e 2022, e os lags do script 24 cruzavam a lacuna.
VALID_YEARS = recorte_config.load_v3()["anos_pipeline"]

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
    print("  US-201: Agregação SINAN por Mesorregião-Semana")
    print("=" * 60)

    print("\nCarregando mapeamento município→mesorregião...")
    mun_meso = pd.read_csv(DATA_DIR / "municipio_mesorregiao.csv",
                           dtype={"ibge_municipio": str})
    print(f"  {len(mun_meso)} municípios → {mun_meso['cod_mesorregiao'].nunique()} mesorregiões")

    print(f"\nCarregando SINAN Gold (anos {VALID_YEARS})...")
    sinan = pq.read_table(
        SINAN_GOLD,
        filters=[("year", "in", VALID_YEARS)]
    ).to_pandas()
    sinan["ibge_municipio"] = sinan["ibge_municipio"].astype(str)
    total_notif_gold = sinan.groupby("ano")["notificacoes"].sum()
    print(f"  {len(sinan):,} linhas, {sinan['ibge_municipio'].nunique()} municípios")

    drop_cols = [c for c in sinan.columns if should_exclude(c) or c in COLS_DROP]
    sinan = sinan.drop(columns=drop_cols, errors="ignore")
    print(f"  Excluídas {len(drop_cols)} colunas pré-calculadas/metadados")

    print("\nMerge com mapeamento mesorregional...")
    sinan = sinan.merge(
        mun_meso[["ibge_municipio", "cod_mesorregiao", "nome_mesorregiao",
                   "uf_sigla", "nome_grande_regiao"]],
        on="ibge_municipio", how="inner"
    )
    print(f"  Após merge: {len(sinan):,} linhas")

    cols_prop = [c for c in sinan.columns if c.startswith("prop_")]
    cols_indice = [c for c in sinan.columns if c.startswith("indice_")]
    cols_weighted = COLS_MEDIA_PONDERADA_EXPLICITA + cols_prop + cols_indice

    print(f"\nAgregando {len(COLS_SOMA)} colunas SUM + {len(cols_weighted)} colunas MÉDIA PONDERADA...")

    print("  Computando produtos ponderados...")
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
    print(f"  Colunas: {len(result.columns)}")

    dups = result.duplicated(subset=GROUP_KEYS).sum()
    assert dups == 0, f"Duplicatas encontradas: {dups}"
    assert result["cod_mesorregiao"].nunique() == 137

    print("\nValidando totais por ano...")
    for year in VALID_YEARS:
        meso_total = result[result["ano"] == year]["notificacoes"].sum()
        gold_total = total_notif_gold.get(year, 0)
        if gold_total > 0:
            pct_diff = abs(meso_total - gold_total) / gold_total * 100
            status = "OK" if pct_diff < 5 else "AVISO"
            print(f"  {year}: Gold={gold_total:,.0f}, Meso={meso_total:,.0f}, diff={pct_diff:.1f}% [{status}]")

    result.to_parquet(DATA_DIR / "sinan_mesorregiao_weekly.parquet", index=False)
    print(f"\n  Salvo: sinan_mesorregiao_weekly.parquet")

    report = {
        "linhas": len(result),
        "mesorregioes": int(result["cod_mesorregiao"].nunique()),
        "colunas": len(result.columns),
        "anos": VALID_YEARS,
        "colunas_soma": len(COLS_SOMA),
        "colunas_media_ponderada": len(cols_weighted),
        "notificacoes_por_ano": {
            int(y): float(result[result["ano"] == y]["notificacoes"].sum())
            for y in VALID_YEARS
        },
    }
    with open(DATA_DIR / "21_sinan_meso_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  RESUMO US-201: {len(result):,} linhas mesorregião-semana")
    print(f"  {len(COLS_SOMA)} SUM + {len(cols_weighted)} MÉDIA PONDERADA")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
